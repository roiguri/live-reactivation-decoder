# LiveInferenceEngine Implementation Plan

> Detailed, test-driven plan for the Phase 2 live decoder runtime. This plan is
> intentionally commit-organized so implementation can proceed in small verified
> steps.

## Summary

Build `LiveInferenceEngine` as a complete runtime class before implementing
`StreamWorker`. Artifact unwrapping lives in a dedicated loader; the engine
receives already-unwrapped decoder models plus model-facing metadata and
predicts probabilities from already-preprocessed model-facing feature rows.

`predict()` must not perform preprocessing and must not interpret `online_state`.

## Role Boundaries

`DecoderPipelineArtifact` loader is responsible for:

- loading the planned `decoder_pipeline.joblib` artifact
- validating artifact-envelope shape
- returning unwrapped `models`, `online_state`, and `metadata`
- leaving `online_state` opaque

`LiveInferenceEngine` is responsible for:

- storing sklearn-compatible decoder models by task name
- validating model compatibility and feature width
- selecting each decoder's positive-class probability
- predicting probabilities for all decoder tasks for every feature row

`LiveInferenceEngine` is not responsible for:

- artifact file loading or `joblib`
- EEG preprocessing, filtering, ICA, bad-channel handling, or referencing
- buffering, micro-batch assembly, or timestamp alignment
- threshold decisions, trigger emission, logging, or UI updates
- receiving, exposing, or interpreting `online_state`

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
final locked Phase 1 contract. The artifact loader must return the
`online_state` object without validating its internal keys, matrix shapes,
sample-rate policy, bad-channel policy, feature layout, or feature width.
Missing or still-open items include bad-channel policy, feature layout, final
model feature width metadata, final artifact metadata, and the model-facing
sample-rate decision.

Prediction itself uses only already-preprocessed model-facing feature rows,
loaded models, and inference metadata.

Expected startup flow:

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

features, timestamps = preprocessor.process_batch(eeg_batch, timestamps)
probabilities = engine.predict(features)
```

## Public Contract

Add:

- `backend.online_phase.artifact_loader.DecoderPipelineArtifact`
- `backend.online_phase.artifact_loader.load_decoder_pipeline_artifact`
- `backend.online_phase.live_inference.LiveInferenceEngine`

Planned artifact shape:

```python
{
    "models": {"red decoder": fitted_model, ...},
    "online_state": {...},
    "metadata": {
        "feature_width": 64,
        "positive_class": 1,
    },
}
```

Required API:

- `load_decoder_pipeline_artifact(path: str | Path) -> DecoderPipelineArtifact`
- `LiveInferenceEngine(models: dict[str, Any], metadata: dict[str, Any] | None = None)`
- `predict(model_features: np.ndarray) -> dict[str, np.ndarray]`

Behavior:

- `load_decoder_pipeline_artifact()` validates the top-level artifact envelope
  and returns the exact unwrapped parts from the loaded payload.
- `LiveInferenceEngine` validates model-facing runtime compatibility.
- `predict()` requires a 2D feature array.
- `predict()` returns one 1D positive-class probability vector per decoder task.
  Phase 1 is expected to train every one-vs-other decoder with `0 = other` and
  `1 = target`, so `positive_class` defaults to `1`.
- Each probability vector has shape `(n_feature_rows,)`.
- `LiveInferenceEngine` must not assume `250 Hz` or `256 Hz`; it validates
  feature width only.

Errors:

- missing artifact: `FileNotFoundError` from the loader
- malformed artifact envelope: `ValueError` from the loader
- malformed model, metadata, features, or probability outputs: `ValueError` from the engine

## Commit Plan

When committing this implementation, keep the work split into these chunks:

### Commit 1: Artifact Loader Boundary

- Add `artifact_loader.py` with `DecoderPipelineArtifact` and
  `load_decoder_pipeline_artifact()`.
- Add `test_artifact_loader.py`.
- Export the loader and artifact dataclass from `backend.online_phase`.
- Test gate: `.venv/bin/python -m pytest tests/online_phase/test_artifact_loader.py -q`.
- Suggested commit: `feat: load decoder pipeline artifacts`

### Commit 2: Injected Live Inference Engine

- Break the old path-based `LiveInferenceEngine` API.
- Make the engine accept unwrapped `models` and model-facing `metadata`.
- Implement model validation, feature-width validation, `predict()`, probability
  output validation, and positive-class selection.
- Update `test_live_inference.py` for the injected API.
- Test gate: `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`.
- Suggested commit: `feat: predict with injected live decoders`

### Commit 3: Maintained Docs

- Update `Phase2_Implementation_Plan.md` and `backend_architecture.md` to show
  loader-owned artifact unwrapping and engine-owned runtime validation.
- Keep `backend_plan.md` unchanged because it is legacy.
- Test gate: `.venv/bin/python -m pytest tests/online_phase -q`.
- Suggested commit: `docs: document phase 2 artifact loading boundary`

## Historical Commit Notes

The original checklist below was written before the artifact-loader boundary was
split out of `LiveInferenceEngine`. The maintained contract above and
`Phase2_Implementation_Plan.md` are the source of truth: artifact loading now
belongs to `artifact_loader.py`, while `LiveInferenceEngine` receives unwrapped
models and metadata.

### Commit 0: Add This Plan

- [x] Create `online_decoder/docs/LiveInferenceEngine_Implementation_Plan.md`.
- [x] Link this file from `online_decoder/docs/Phase2_Implementation_Plan.md`.
- [x] Keep historical docs unchanged.
- [x] Test gate: no runtime tests required.
- [x] Suggested commit: `docs: add live inference engine implementation plan`

### Commit 1: Artifact Loading Tests And Loader

- [x] Add `online_decoder/tests/online_phase/test_live_inference.py`.
- [x] Write tests first for:
  - [x] valid artifact loads successfully
  - [x] `load_pipeline()` returns the exact `online_state`
  - [x] engine stores models and metadata
  - [x] missing artifact raises `FileNotFoundError`
  - [x] non-dict artifact raises `ValueError`
  - [x] missing `models`, `online_state`, or `metadata` raises `ValueError`
  - [x] empty `models` raises `ValueError`
  - [x] model without `predict_proba` raises `ValueError`
- [x] Implement `online_decoder/src/backend/online_phase/live_inference.py`.
- [x] Keep loading explicit: constructor stores path, `load_pipeline()` reads the file.
- [x] Do not validate `online_state` internals in this commit; only require the
  top-level `online_state` key to be present.
- [x] Test gate:
  - [x] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [x] `.venv/bin/python -m pytest tests/online_phase -q`
- [x] Suggested commit: `feat: load live inference artifacts`

### Commit 2: Feature Width And Metadata Validation

- [x] Add tests for:
  - [x] `feature_width` from `metadata["feature_width"]`
  - [x] fallback to model `n_features_in_` when metadata omits `feature_width`
  - [x] inconsistent model feature widths raise `ValueError`
  - [x] non-integer or non-positive `feature_width` raises `ValueError`
- [x] Implement feature-width derivation and validation.
- [x] Expose read-only properties if useful:
  - [x] `models`
  - [x] `metadata`
  - [x] `feature_width`
  - [x] `online_state`
- [x] Test gate:
  - [x] `.venv/bin/python -m pytest tests/online_phase/test_live_inference.py -q`
  - [x] `.venv/bin/python -m pytest tests/online_phase -q`
- [x] Suggested commit: `feat: validate live inference feature metadata`

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
  - [ ] fallback to class label `1`
  - [ ] optional global `metadata["positive_class"]`
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

- The loader targets only the planned `decoder_pipeline.joblib` dict, not legacy monorepo `.pkl` decoder files.
- `joblib` is used by the artifact loader, not by `LiveInferenceEngine`.
- Input to `predict()` is already preprocessed, decimated, model-facing data.
- Timestamp alignment, buffering, thresholding, triggers, logging, and UI emission are out of scope for this class.
