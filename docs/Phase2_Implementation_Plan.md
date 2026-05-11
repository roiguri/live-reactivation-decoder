# Phase 2 Implementation Plan

> Quick implementation plan for the active stateful micro-batch architecture. For rationale and trade-offs, see [../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md](../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md). For the maintained backend contract, see [backend_architecture.md](backend_architecture.md).

Last design update on **2026-05-11**.

## Architecture Outline

```text
AppSession.build_live_stream_session(...)
  -> constructs LSLReceiver, OnlinePreprocessor, LiveInferenceEngine,
     StreamWorker, and optional PredictionLogger
  -> returns LiveStreamSession

LiveStreamSession.start()/stop()
  -> owns lifecycle and cleanup

StreamWorker.run()
  -> LSLReceiver -> batch accumulator -> OnlinePreprocessor
     -> LiveInferenceEngine -> prediction_ready signal -> UI/logger
```

- Input stream: 1000 Hz EEG from LSL, expected as 64 EEG channels plus one trigger channel.
- Runtime unit: 40-sample micro-batches, approximately 40 ms at 1000 Hz.
- Model-facing output: features at the locked Phase 1 target rate; prediction
  count per micro-batch depends on the target rate.
- Critical invariant: filtering state and decimation phase persist across batches.
- Lifecycle invariant: `StreamWorker` has dependency references but does not create,
  start/stop, or close the receiver/logger. `LiveStreamSession` owns that lifecycle.
- Obsolete approach: do not reintroduce `RingBuffer` or full-window reprocessing.

## Status Legend

- `[x]` Done
- `[~]` Partial
- `[ ]` TODO
- `[!]` Blocking decision

## Current Status

- `[x]` Config schema and `SettingsManager` are implemented.
- `[x]` `OfflinePreprocessor` is implemented and unit-tested.
- `[x]` `ModelEvaluator` is implemented and unit-tested.
- `[x]` `LSLReceiver` is implemented and unit-tested.
- `[~]` Replay, smoke-test, and opt-in integration tooling exists.
- `[x]` `OfflinePreprocessor` exports `online_state` including `interp_weights` and `pre_whitener`.
- `[x]` `DecoderPipelineArtifact` loader exists and unwraps the Phase 1 artifact.
- `[x]` `OnlinePreprocessor` is implemented and fully tested (76 tests).
- `[x]` `LiveInferenceEngine` receives unwrapped models/metadata and predicts positive-class probabilities.
- `[x]` `PredictionLogger` CSV sink is implemented.
- `[x]` `StreamWorker` is implemented as the injected-dependency micro-batch loop.
- `[ ]` Session-level composition needs refactor to the target API: `AppSession.build_live_stream_session(...) -> LiveStreamSession`.
- `[x]` Phase 1 sample rate locked: configurable via `resample.target_rate` in YAML (default 256 Hz). `online_state` schema locked.

## Component Plan

### `LSLReceiver` - Input Boundary

**Status:** `[x]` Implemented in `online_decoder/src/backend/online_phase/lsl_receiver.py`.

**Responsibilities:**
- Manage optional `LSLProxy.exe` subprocess lifecycle.
- Discover and connect to the selected LSL stream.
- Validate stream shape before use: 1000 Hz, 65 channels.
- Pull all currently pending samples since the previous call.
- Split the trigger channel from EEG data.
- Decode NeurOne trigger values with `(int(raw_value) >> 8) & 0xFF`.
- Emit non-zero trigger edges only, avoiding duplicate markers while a trigger is held high.

**Current output contract:**
- `timestamps`: NumPy array, shape `(n_samples,)`.
- `eeg_chunk`: NumPy array, shape `(n_samples, 64)`.
- `markers`: list of `(timestamp, code)` tuples detected during the pull.

**Cleanup TODOs:**
- `[x]` Reconcile docs/scripts/tests that still expect `start()` to return `True` or `False`; current code returns `None` and raises on failure.
- `[x]` Add richer marker output: marker code plus LSL timestamp.
- `[ ]` Run final validation on the real lab stream with `LSLProxy.exe` on the decoding machine.
- `[ ]` Decide whether malformed chunks should always be skipped or escalated to the UI in live mode.

### `OnlinePreprocessor` - Stateful Cleaner

**Status:** `[x]` Implemented in `src/backend/online_phase/online_preprocessor.py`. 76 tests passing.

**Pipeline order (causal, mirrors offline):**
1. Bandpass + notch filter — causal IIR via `sosfilt` with persistent `zi`
2. Decimate to target rate — FIR anti-alias + phase-tracked subsampling
3. Interpolate bad channels — fixed weight matrix from Phase 1
4. Average reference
5. Apply ICA — delta formula with `pre_whitener`

**Key implementation notes:**
- Target rate is fully configurable via `preprocessing_settings["resample"]["target_rate"]` — not hardcoded. Default 256 Hz comes from `experiment_config.yaml`.
- Constructor cross-validates `online_state["sfreq_offline"]` against `target_rate` to catch Phase 1/2 config drift.
- `online_state` must include: `bad_channels`, `interp_weights`, `ch_names`, `ica_unmixing`, `ica_mixing`, `ica_pca_components`, `ica_pca_mean`, `ica_exclude`, `pre_whitener`, `sfreq_offline`.
- Benchmark: 0.21 ms mean per batch (default config), 0.53 ms worst-case (64 ch, 40 ICA comp) — well within 40 ms budget.

**Tests before live use:**
- `[x]` Chunked filtering equals one continuous causal filtering call for the same data.
- `[x]` Irregular batch sizes preserve decimation alignment and timestamp alignment.
- `[x]` Shape validation rejects unexpected channel counts and timestamp lengths.
- `[x]` Parametrized over 7 target rates (100–512 Hz).

### `DecoderPipelineArtifact` Loader - Startup Boundary

**Status:** `[x]` Implemented in `online_decoder/src/backend/online_phase/artifact_loader.py`.

**Responsibilities:**
- Load the Phase 1 `decoder_pipeline.joblib` artifact with `joblib`.
- Validate only the top-level artifact envelope: required keys, non-empty
  `models` dict, and dict `metadata`.
- Return `models`, `online_state`, and `metadata` as separate fields.
- Keep `online_state` opaque; its internal schema belongs to `OnlinePreprocessor`.

**Composition flow:**
```python
artifact = load_decoder_pipeline_artifact("decoder_pipeline.joblib")

preprocessor = OnlinePreprocessor(
    preprocessing_settings=preprocessing_settings,
    online_state=artifact.online_state,
)

engine = LiveInferenceEngine(
    models=artifact.models,
    metadata=artifact.metadata,
)
```

### `LiveInferenceEngine` - Decoder Runtime

**Status:** `[x]` Implemented in `online_decoder/src/backend/online_phase/live_inference.py`.

**Detailed plan:** See [LiveInferenceEngine_Implementation_Plan.md](LiveInferenceEngine_Implementation_Plan.md).

**Inputs:**
- Unwrapped trained sklearn-compatible decoder models.
- Model-facing metadata such as `feature_width` and optional global
  `positive_class`.
- `model_features` from `OnlinePreprocessor`.

**Responsibilities:**
- Validate model runtime compatibility.
- Validate feature width before prediction.
- Run `predict_proba()` for each configured decoder task.
- Select the positive-class probability. Phase 1 is expected to train every
  one-vs-other decoder with `0 = other` and `1 = target`, so the default
  positive label is `1`.
- Return probabilities for every feature row produced by the batch.

**Boundary rule:**
- `LiveInferenceEngine` does not load joblib artifacts.
- `LiveInferenceEngine` does not receive, expose, or interpret `online_state`.

**Output contract:**
- Dictionary mapping task name to probability array, aligned to the preprocessor output timestamps.

**Tests before live use:**
- `[x]` Loader unwraps a fixture pipeline and returns opaque `online_state`.
- `[x]` Predicts all configured tasks.
- `[x]` Validates feature width before prediction.
- `[x]` Selects positive-class probability columns.

### `StreamWorker` - Online Orchestrator

**Status:** `[x]` Implemented in `online_decoder/src/backend/online_phase/stream_worker.py`.

**Responsibilities:**
- Own the online run loop and batch accumulator in a background thread.
- Keep references to injected `LSLReceiver`, `OnlinePreprocessor`, and `LiveInferenceEngine` dependencies for use inside `run()`.
- Call `LSLReceiver.pull_new_data()` repeatedly.
- Accumulate variable-size LSL chunks into 40-sample batches.
- Keep leftover samples for the next batch.
- Call `OnlinePreprocessor.process_batch()`.
- Call `LiveInferenceEngine.predict()`.
- Emit all predictions, aligned timestamps, and markers via `prediction_ready`.
- Stop the run loop when `stop()` is requested.

**Non-responsibilities:**
- Does not create the receiver, preprocessor, inference engine, or logger.
- Does not load `decoder_pipeline.joblib`.
- Does not start/stop the LSL connection.
- Does not close log files.
- Does not know about frontend plots or buffers.

**Boundary rule:**
- Batching belongs in `StreamWorker`, not in `LSLReceiver`.
- Lifecycle belongs in `LiveStreamSession`, not in `StreamWorker`.

**Tests before live use:**
- `[x]` Variable-size chunks produce stable 40-sample batches.
- `[x]` Leftover samples are preserved across pulls.
- `[x]` Marker timestamps are included/deferred according to batch boundaries.
- `[x]` Prediction timestamps remain aligned to the emitted probability rows.
- `[ ]` Receiver, preprocessing, and inference errors are surfaced instead of silently swallowed.

### `PredictionLogger` - CSV Sink

**Status:** `[x]` Implemented in `online_decoder/src/backend/online_phase/prediction_logger.py`.

**Responsibilities:**
- Consume `prediction_ready` signal payloads.
- Write one CSV row per prediction timestamp.
- Match marker timestamps to nearest prediction rows within `0.5 / target_sfreq`.
- Flush after each batch and close idempotently.

**Boundary rule:**
- `PredictionLogger` is a signal consumer, not part of the worker loop.
- `LiveStreamSession` closes the logger during shutdown.

### `LiveStreamSession` - Online Lifecycle

**Status:** `[ ]` Target design; refactor current session composition to this API.

**Responsibilities:**
- Represent one composed live decoding run.
- Expose `prediction_ready` by forwarding the underlying worker signal.
- Start in order: `receiver.start()` then `worker.start()`.
- Stop in order: `worker.stop()`, `worker.wait()`, `logger.close()` if present, then `receiver.stop()`.
- Make `start()` and `stop()` idempotent.

**Boundary rule:**
- `LiveStreamSession` owns lifecycle, but does not implement the micro-batch loop.
- `StreamWorker` has references to runtime dependencies, but `LiveStreamSession` owns the start/stop/cleanup sequence.

### `AppSession.build_live_stream_session(...)` - Composition Boundary

**Status:** `[ ]` Target API; replace the current `session.online`/handle shape.

**Responsibilities:**
- Keep `AppSession` as the only backend class imported by the frontend.
- Load `DecoderPipelineArtifact`.
- Construct `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine`, `StreamWorker`, and optional `PredictionLogger`.
- Connect logger to `worker.prediction_ready` when `log_path` is provided.
- Return a `LiveStreamSession` without starting it.

**Target frontend usage:**
```python
live = session.build_live_stream_session(decoder_pipeline_path, log_path)
live.prediction_ready.connect(probability_buffer.on_predictions)
live.start()
live.stop()
```

**Boundary rule:**
- Do not introduce `OnlinePhase` or expose `session.online`.
- Do not make `StreamWorker` construct or own the whole online runtime.

## Resolved Phase 1 Decisions

Previously blocking decisions — all now resolved.

### `[x]` Model-Facing Sample Rate

Locked at **configurable via `resample.target_rate`** in `experiment_config.yaml` (default 256 Hz, not hardcoded). `OnlinePreprocessor` reads it from `preprocessing_settings` and cross-validates against `online_state["sfreq_offline"]`.

### `[x]` `online_state` Schema

`OfflinePreprocessor.export_online_state()` now exports:
- `bad_channels`, `interp_weights` (precomputed spherical-spline weight matrix, or None)
- `ch_names`, `sfreq_offline`
- `ica_unmixing`, `ica_mixing`, `ica_pca_components`, `ica_pca_mean`, `ica_exclude`
- `pre_whitener` (per-channel-type rescaling factor from MNE ICA fitting)

### `[x]` Artifact envelope

```python
{"models": {...}, "online_state": {...}, "metadata": {...}}
```

The artifact loader treats `online_state` as opaque. `LiveInferenceEngine` never receives it.

## Implementation Order

1. `[x]` Clean up `LSLReceiver.start()` expectations in docs, smoke script, and integration test.
2. `[x]` Implement artifact loader and `LiveInferenceEngine`.
3. `[x]` Lock the Phase 1 sample rate, artifact envelope, and `online_state` schema.
4. `[x]` Implement `OnlinePreprocessor` with full test suite and benchmark.
5. `[x]` Implement `StreamWorker`.
6. `[x]` Implement `PredictionLogger`.
7. `[ ]` Refactor session composition to `LiveStreamSession` and `AppSession.build_live_stream_session(...)`.
8. `[ ]` Add latency logging to `StreamWorker`.
9. `[ ]` Run replay-based dry run.
10. `[ ]` Validate with the real lab LSL stream.

## Test Plan

- Normal tests: from `online_decoder/`, run `.venv/bin/python -m pytest tests/ -q`.
- Loader/engine tests: `.venv/bin/python -m pytest tests/online_phase/test_artifact_loader.py tests/online_phase/test_live_inference.py -q`.
- Replay integration: run only when requested with `RUN_LSL_INTEGRATION=1`.
- Preprocessor state tests must pass before any live experiment.
- Real lab validation remains a final checklist item; home replay does not complete it.

## Assumptions

- `online_decoder/docs/backend_architecture.md` remains the maintained backend source of truth.
- `RingBuffer` remains obsolete for this app.
- The online pipeline accepts the known training/inference mismatch: Phase 1 can use offline cleaning, while Phase 2 uses causal streaming preprocessing.
