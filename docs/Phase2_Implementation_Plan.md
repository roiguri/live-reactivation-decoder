# Phase 2 Implementation Plan

> Quick implementation plan for the active stateful micro-batch architecture. For rationale and trade-offs, see [../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md](../../knowledge_base/01_timeline/03_online_stage_design/Phase2_Architecture_Discussion.md). For the maintained backend contract, see [backend_architecture.md](backend_architecture.md).

Last reconciled with code on **2026-05-09**.

## Architecture Outline

```text
LSLReceiver -> StreamWorker -> OnlinePreprocessor -> LiveInferenceEngine -> logs/UI
```

- Input stream: 1000 Hz EEG from LSL, expected as 64 EEG channels plus one trigger channel.
- Runtime unit: 40-sample micro-batches, approximately 40 ms at 1000 Hz.
- Model-facing output: 250 Hz features, approximately 10 predictions per 40 ms batch.
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
- `[ ]` `OnlinePreprocessor` is missing.
- `[ ]` `LiveInferenceEngine` is missing.
- `[ ]` `StreamWorker` is missing.
- `[!]` Phase 1 artifact and `online_state` schema must be locked before `OnlinePreprocessor` implementation.

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
- `clean_features_250hz`: NumPy array aligned to decimated samples.
- `output_timestamps`: NumPy array aligned to `clean_features_250hz` rows.

**Required behavior:**
- Use `scipy.signal.sosfilt` with persistent `zi` state for causal bandpass filtering.
- Use persistent `zi` state for notch filtering when notch filtering is enabled.
- Apply the fixed bad-channel policy from Phase 1.
- Apply average reference after bad-channel handling.
- Apply the fixed ICA transform from Phase 1.
- Decimate from 1000 Hz to 250 Hz with a persistent sample counter so irregular batch sizes preserve phase.
- Provide `reset_state()` for a new run.

**Tests before live use:**
- `[ ]` Chunked filtering equals one continuous causal filtering call for the same data.
- `[ ]` Irregular batch sizes preserve decimation alignment and timestamp alignment.
- `[ ]` Shape validation rejects unexpected channel counts and timestamp lengths.

### `LiveInferenceEngine` - Decoder Runtime

**Status:** `[ ]` Not implemented.

**Inputs:**
- Path to Phase 1 `decoder_pipeline.joblib`.
- `clean_features_250hz` from `OnlinePreprocessor`.

**Responsibilities:**
- Load trained sklearn-compatible decoder models.
- Expose the loaded `online_state` for `OnlinePreprocessor`.
- Validate feature width before prediction.
- Run `predict_proba()` for each configured decoder task.
- Return probabilities for every feature row produced by the batch.

**Output contract:**
- Dictionary mapping task name to probability array, aligned to the preprocessor output timestamps.

**Tests before live use:**
- `[ ]` Loads a fixture pipeline and exposes `online_state`.
- `[ ]` Predicts all configured tasks.
- `[ ]` Rejects wrong feature width with a clear error.

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

## Blocking Schema Decision

Before implementing `OnlinePreprocessor`, lock the Phase 1 saved artifact contract.

Default artifact shape:

```python
{
    "models": {...},
    "online_state": {...},
    "metadata": {...},
}
```

Required contents:
- Channel names and channel order used during training.
- Bad channels and the online bad-channel handling policy.
- ICA matrix and excluded component metadata.
- Model feature width and expected row layout.
- Decoder task names.
- Trained sklearn-compatible models.
- Preprocessing settings required by Phase 2.

## Implementation Order

1. `[!]` Lock the Phase 1 artifact / `online_state` schema.
2. `[~]` Clean up `LSLReceiver.start()` expectations in docs, smoke script, and integration test.
3. `[ ]` Implement `OnlinePreprocessor`.
4. `[ ]` Add stateful filtering and decimation tests.
5. `[ ]` Implement `LiveInferenceEngine`.
6. `[ ]` Implement `StreamWorker`.
7. `[ ]` Add latency logging.
8. `[ ]` Run replay-based dry run.
9. `[ ]` Validate with the real lab LSL stream.

## Test Plan

- Normal tests: from `online_decoder/`, run `.venv/bin/python -m pytest tests/ -q`.
- Replay integration: run only when requested with `RUN_LSL_INTEGRATION=1`.
- Preprocessor state tests must pass before any live experiment.
- Real lab validation remains a final checklist item; home replay does not complete it.

## Assumptions

- `online_decoder/docs/backend_architecture.md` remains the maintained backend source of truth.
- `RingBuffer` remains obsolete for this app.
- The online pipeline accepts the known training/inference mismatch: Phase 1 can use offline cleaning, while Phase 2 uses causal streaming preprocessing.
