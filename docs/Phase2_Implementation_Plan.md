# Phase 2 Implementation Plan

> Quick implementation plan for the active stateful micro-batch architecture. For rationale and trade-offs, see [../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md](../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md). For the maintained backend contract, see [backend_architecture.md](backend_architecture.md).

Last reconciled with code on **2026-05-09**.

## Architecture Outline

```text
LSLReceiver -> StreamWorker -> OnlinePreprocessor -> LiveInferenceEngine -> logs/UI
```

- Input stream: 1000 Hz EEG from LSL, expected as 64 EEG channels plus one trigger channel.
- Runtime unit: 40-sample micro-batches, approximately 40 ms at 1000 Hz.
- Model-facing output: features at the locked Phase 1 target rate; prediction
  count per micro-batch depends on the target rate.
- Critical invariant: filtering state and decimation phase persist across batches.
- Obsolete approach: do not reintroduce `RingBuffer` or full-window reprocessing.

## Status Legend

- `[x]` Done
- `[~]` Partial
- `[ ]` TODO
- `[!]` Blocking decision

## Current Status

- `[x]` Config schema and `SettingsManager` are implemented.
- `[x]` `LSLReceiver` is implemented and unit-tested.
- `[~]` Replay, smoke-test, and opt-in integration tooling exists.
- `[~]` `OfflinePreprocessor` exists and exports an initial `online_state`.
- `[x]` `DecoderPipelineArtifact` loader exists and unwraps the Phase 1 artifact.
- `[ ]` `OnlinePreprocessor` is missing.
- `[x]` `LiveInferenceEngine` receives unwrapped models/metadata and predicts positive-class probabilities.
- `[ ]` `StreamWorker` is missing.
- `[!]` Phase 1 sample rate, final artifact envelope, and `online_state` schema must be locked before `OnlinePreprocessor` implementation.

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
- `markers`: list of integer marker codes detected during the pull.

**Cleanup TODOs:**
- `[x]` Reconcile docs/scripts/tests that still expect `start()` to return `True` or `False`; current code returns `None` and raises on failure.
- `[ ]` Add richer marker output if needed: marker code plus timestamp and/or sample index.
- `[ ]` Run final validation on the real lab stream with `LSLProxy.exe` on the decoding machine.
- `[ ]` Decide whether malformed chunks should always be skipped or escalated to the UI in live mode.

### `OnlinePreprocessor` - Stateful Cleaner

**Status:** `[ ]` Not implemented. This is the next critical backend class.

**Inputs:**
- `eeg_batch_1000hz`: NumPy array, shape `(n_samples, n_channels)`.
- `timestamps`: NumPy array aligned to input samples.
- `preprocessing_settings`: validated config settings.
- `online_state`: fixed Phase 1 state exported with the trained decoder artifact.

**Outputs:**
- `model_features`: NumPy array aligned to model-facing samples.
- `output_timestamps`: NumPy array aligned to `model_features` rows.

**Required behavior:**
- Use `scipy.signal.sosfilt` with persistent `zi` state for causal bandpass filtering.
- Use persistent `zi` state for notch filtering when notch filtering is enabled.
- Apply the fixed bad-channel policy from Phase 1.
- Apply average reference after bad-channel handling.
- Apply the fixed ICA transform from Phase 1.
- Resample/decimate from 1000 Hz to the locked target rate with persistent
  state so irregular batch sizes preserve alignment.
- Provide `reset_state()` for a new run.

**Tests before live use:**
- `[ ]` Chunked filtering equals one continuous causal filtering call for the same data.
- `[ ]` Irregular batch sizes preserve decimation alignment and timestamp alignment.
- `[ ]` Shape validation rejects unexpected channel counts and timestamp lengths.

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

**Status:** `[ ]` Not implemented.

**Responsibilities:**
- Own the online run loop in a background thread.
- Call `LSLReceiver.pull_new_data()` repeatedly.
- Accumulate variable-size LSL chunks into 40-sample batches.
- Keep leftover samples for the next batch.
- Call `OnlinePreprocessor.process_batch()`.
- Call `LiveInferenceEngine.predict()`.
- Emit or log all predictions, not only the latest prediction.
- Surface errors cleanly to the UI.

**Boundary rule:**
- Batching belongs in `StreamWorker`, not in `LSLReceiver`.

**Tests before live use:**
- `[ ]` Variable-size chunks produce stable 40-sample batches.
- `[ ]` Leftover samples are preserved across pulls.
- `[ ]` Receiver, preprocessing, and inference errors are surfaced instead of silently swallowed.
- `[ ]` Prediction timestamps remain aligned to the emitted probability rows.

## Blocking Phase 1 Decisions

Before implementing `OnlinePreprocessor`, lock the Phase 1 target rate,
`online_state` schema, and saved artifact contract. The artifact loader unwraps
the saved envelope; `LiveInferenceEngine` consumes only the unwrapped models and
model-facing metadata.

### `[!]` Model-Facing Sample Rate

Current Phase 2 docs previously assumed `250 Hz`. The new Phase 1 config default
is `256 Hz`. Do not finalize `OnlinePreprocessor` resampling behavior until this
is discussed with the Phase 1 developer and locked.

Default artifact shape:

```python
{
    "models": {...},
    "online_state": {...},
    "metadata": {...},
}
```

Required contents:
- Decoder task names.
- Trained sklearn-compatible models.
- Final artifact metadata, including model feature width and expected row layout.
- Preprocessing settings required by Phase 2.

Current `OfflinePreprocessor.export_online_state()` output:
- `bad_channels`
- `ica_unmixing`
- `ica_mixing`
- `ica_pca_components`
- `ica_pca_mean`
- `ica_exclude`
- `ch_names`
- `sfreq_offline`

Still-open `online_state` / artifact items:
- final model-facing sample rate
- bad-channel handling policy for live data
- feature layout and feature width metadata
- final channel-order naming convention
- final `decoder_pipeline.joblib` metadata envelope

The artifact loader must treat `online_state` as an opaque payload: it returns
it for `OnlinePreprocessor`, but does not validate its keys or matrix contents.
`LiveInferenceEngine` must never receive `online_state`.

## Implementation Order

Build complete, self-contained classes first. The artifact loader can treat
`online_state` as opaque while the Phase 1 artifact schema is still evolving.
`OnlinePreprocessor` should wait because bad-channel handling and ICA/spatial
transform behavior depend directly on that schema.

1. `[x]` Clean up `LSLReceiver.start()` expectations in docs, smoke script, and integration test.
2. `[x]` Implement artifact loader and `LiveInferenceEngine`.
3. `[ ]` Implement `StreamWorker`.
4. `[!]` Lock the Phase 1 sample rate, artifact envelope, and `online_state` schema.
5. `[ ]` Implement `OnlinePreprocessor`.
6. `[ ]` Add stateful filtering and decimation tests.
7. `[ ]` Add latency logging.
8. `[ ]` Run replay-based dry run.
9. `[ ]` Validate with the real lab LSL stream.

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
