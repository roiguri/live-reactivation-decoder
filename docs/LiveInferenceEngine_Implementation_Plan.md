# LiveInferenceEngine Implementation Plan

> Detailed, test-driven plan for the Phase 2 live decoder runtime. This plan is
> intentionally commit-organized so implementation can proceed in small verified
> steps.

## Summary

Build `LiveInferenceEngine` as a complete runtime class before implementing
`StreamWorker`. The engine loads the planned Phase 1 decoder artifact envelope,
owns decoder models and inference metadata, exposes `online_state` for
`OnlinePreprocessor`, and predicts probabilities from already-preprocessed
model-facing feature rows.

`predict()` must not perform preprocessing and must not interpret `online_state`.

## Role Boundaries

`LiveInferenceEngine` is responsible for:

- loading the planned `decoder_pipeline.joblib` artifact
- validating artifact shape, model compatibility, and feature width
- storing sklearn-compatible decoder models by task name
- exposing `online_state` for startup code to pass into `OnlinePreprocessor`
- selecting each decoder's positive-class probability
- predicting probabilities for all decoder tasks for every feature row

`LiveInferenceEngine` is not responsible for:

- EEG preprocessing, filtering, ICA, bad-channel handling, or referencing
- buffering, micro-batch assembly, or timestamp alignment
- threshold decisions, trigger emission, logging, or UI updates
- interpreting the contents of `online_state`

## `online_state`

`online_state` is the Phase 1 preprocessing state needed by
`OnlinePreprocessor` to reproduce the training-time feature space during live
processing. Phase 1 currently exports this state directly; the final
`decoder_pipeline.joblib` envelope that bundles it with models and metadata is
still planned.

Phase 1 now has an initial `OfflinePreprocessor.export_online_state()` surface
with these observed keys:

- `bad_channels`
- `ica_unmixing`
- `ica_mixing`
- `ica_pca_components`
- `ica_pca_mean`
- `ica_exclude`
- `ch_names`
- `sfreq_offline`

Those keys are useful current context, but they are not treated here as the
final locked Phase 1 contract. The inference engine must load and return the
`online_state` object without validating its internal keys, matrix shapes,
sample-rate policy, bad-channel policy, feature layout, or feature width.
Missing or still-open items include bad-channel policy, feature layout, final
model feature width metadata, final artifact metadata, and the model-facing
sample-rate decision.

The inference engine loads and returns `online_state` because the Phase 1
artifact bundles setup data for the whole live pipeline. Prediction itself uses
only already-preprocessed model-facing feature rows, loaded models, and
inference metadata.

Expected startup flow:

```python
engine = LiveInferenceEngine("decoder_pipeline.joblib")
online_state = engine.load_pipeline()

preprocessor = OnlinePreprocessor(
    preprocessing_settings=preprocessing_settings,
    online_state=online_state,
)

features, timestamps = preprocessor.process_batch(eeg_batch, timestamps)
probabilities = engine.predict(features)
```

## Public Contract

Add `backend.online_phase.live_inference.LiveInferenceEngine`.

Planned artifact shape:

```python
{
    "models": {"red decoder": fitted_model, ...},
    "online_state": {...},
    "metadata": {
        "feature_width": 64,
        "positive_class": 1,
        "task_positive_classes": {"red decoder": 1},
    },
}
```

Required API:

- `__init__(pipeline_filepath: str | Path)`
- `load_pipeline() -> dict`
- `predict(model_features: np.ndarray) -> dict[str, np.ndarray]`

Behavior:

- `load_pipeline()` loads and validates the top-level artifact envelope, stores
  models and metadata, and returns the exact `online_state` object from the
  artifact without inspecting its internal schema.
- `predict()` requires a loaded pipeline and a 2D feature array.
- `predict()` returns one 1D positive-class probability vector per decoder task.
- Each probability vector has shape `(n_feature_rows,)`.
- `LiveInferenceEngine` must not assume `250 Hz` or `256 Hz`; it validates
  feature width only.

Errors:

- missing artifact: `FileNotFoundError`
- malformed artifact, model, metadata, or features: `ValueError`
- `predict()` before `load_pipeline()`: `RuntimeError`

## Commit Plan

### Commit 0: Add This Plan

- [ ] Create `online_decoder/docs/LiveInferenceEngine_Implementation_Plan.md`.
- [ ] Link this file from `online_decoder/docs/Phase2_Implementation_Plan.md`.
- [ ] Keep historical docs unchanged.
- [ ] Test gate: no runtime tests required.
- [ ] Suggested commit: `docs: add live inference engine implementation plan`

### Commit 1: Artifact Loading Tests And Loader

- [ ] Add `online_decoder/tests/online_phase/test_live_inference.py`.
- [ ] Write tests first for:
  - [ ] valid artifact loads successfully
  - [ ] `load_pipeline()` returns the exact `online_state`
  - [ ] engine stores models and metadata
  - [ ] missing artifact raises `FileNotFoundError`
  - [ ] non-dict artifact raises `ValueError`
  - [ ] missing `models`, `online_state`, or `metadata` raises `ValueError`
  - [ ] empty `models` raises `ValueError`
  - [ ] model without `predict_proba` raises `ValueError`
- [ ] Implement `online_decoder/src/backend/online_phase/live_inference.py`.
- [ ] Keep loading explicit: constructor stores path, `load_pipeline()` reads the file.
- [ ] Do not validate `online_state` internals in this commit; only require the
  top-level `online_state` key to be present.
- [ ] Test gate:
  - [ ] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [ ] `.venv/bin/python -m pytest tests/online_phase -q`
- [ ] Suggested commit: `feat: load live inference artifacts`

### Commit 2: Feature Width And Metadata Validation

- [ ] Add tests for:
  - [ ] `feature_width` from `metadata["feature_width"]`
  - [ ] fallback to model `n_features_in_` when metadata omits `feature_width`
  - [ ] inconsistent model feature widths raise `ValueError`
  - [ ] non-integer or non-positive `feature_width` raises `ValueError`
- [ ] Implement feature-width derivation and validation.
- [ ] Expose read-only properties if useful:
  - [ ] `models`
  - [ ] `metadata`
  - [ ] `feature_width`
  - [ ] `online_state`
- [ ] Test gate:
  - [ ] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [ ] `.venv/bin/python -m pytest tests/online_phase -q`
- [ ] Suggested commit: `feat: validate live inference feature metadata`

### Commit 3: Prediction Tests And Prediction Logic

- [ ] Add tests for:
  - [ ] `predict()` before load raises `RuntimeError`
  - [ ] non-2D input raises `ValueError`
  - [ ] wrong feature width raises `ValueError`
  - [ ] empty feature batch returns empty vectors for every task
  - [ ] multiple decoders return one vector per task
  - [ ] output vector length equals input row count
- [ ] Implement `predict()`.
- [ ] Ensure `predict()` uses only preprocessed feature rows, models, and metadata.
- [ ] Ensure `predict()` does not read or interpret `online_state`.
- [ ] Test gate:
  - [ ] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [ ] `.venv/bin/python -m pytest tests/online_phase -q`
- [ ] Suggested commit: `feat: predict live decoder probabilities`

### Commit 4: Positive-Class Selection

- [ ] Add tests for positive-class selection:
  - [ ] per-task `metadata["task_positive_classes"][task_name]`
  - [ ] global `metadata["positive_class"]`
  - [ ] fallback to class label `1`
  - [ ] fallback to class label `True`
  - [ ] missing identifiable positive class raises `ValueError`
  - [ ] probability matrix with wrong row count raises `ValueError`
  - [ ] probability matrix with too few columns raises `ValueError`
- [ ] Implement deterministic positive-column selection from `model.classes_`.
- [ ] Return only the selected positive-class vector, not the full probability matrix.
- [ ] Test gate:
  - [ ] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [ ] `.venv/bin/python -m pytest tests/online_phase -q`
- [ ] Suggested commit: `feat: select positive decoder probabilities`

### Commit 5: Package Export And Maintained Docs

- [ ] Export `LiveInferenceEngine` from `backend.online_phase.__init__`.
- [ ] Add a small import test if package-level export is added.
- [ ] Update maintained docs:
  - [ ] mark `LiveInferenceEngine` as implemented in `Phase2_Implementation_Plan.md`
  - [ ] document that `online_state` is exposed for `OnlinePreprocessor`, not used by `predict()`
  - [ ] document positive-vector output
- [ ] Do not update `backend_plan.md`.
- [ ] Test gate:
  - [ ] `.venv/bin/python -m pytest tests/online_phase -q`
  - [ ] `.venv/bin/python -m pytest tests -q`
- [ ] Suggested commit: `docs: document live inference engine contract`

## Final Acceptance

- [ ] Every commit passes its listed test gate before committing.
- [ ] Final full test suite passes from `online_decoder/`.
- [ ] `LiveInferenceEngine` is complete without depending on `OnlinePreprocessor`.
- [ ] `online_state` remains opaque inside the engine.
- [ ] Final `git status --short` is clean.

## Assumptions

- The engine targets only the planned `decoder_pipeline.joblib` dict, not legacy monorepo `.pkl` decoder files.
- `joblib` is used for artifact loading.
- Input to `predict()` is already preprocessed, decimated, model-facing data.
- Timestamp alignment, buffering, thresholding, triggers, logging, and UI emission are out of scope for this class.
