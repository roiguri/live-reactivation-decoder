# online_decoder — Claude Guidelines

`online_decoder/` is the **standalone app root**. It must remain self-contained and portable — no imports from the parent monorepo.

## Authority

Code under `src/` is the source of truth.

Use [docs/backend_architecture.md](docs/backend_architecture.md) for the maintained backend summary and [docs/Phase2_Implementation_Plan.md](docs/Phase2_Implementation_Plan.md) for the active Phase 2 implementation checklist. If the docs and code disagree, follow the code and update the docs.

## Directory Layout

```text
online_decoder/
├── scripts/            — Characterization, replay, and smoke-test helpers
├── src/backend/
│   ├── core/           — SettingsManager and Pydantic config models
│   ├── offline_phase/  — utils, OfflinePreprocessor, ModelEvaluator, ModelTrainer, OfflineOrchestrator
│   └── online_phase/   — LSLReceiver and online inference package scaffold
├── tests/
│   ├── core/           — Config validation tests
│   ├── notebooks/      — Manual validation notebooks
│   ├── offline_phase/  — Offline phase unit tests (preprocessor, evaluator, trainer)
│   └── online_phase/   — LSLReceiver unit and opt-in replay integration tests
├── tools/lslproxy/     — LSLProxy.exe and Windows DLLs (hardware interface)
└── docs/               — Backend status and architecture notes
```

## Current Backend Scope

- Phase 2 backend surface in the current branch: `LSLReceiver`, `DecoderPipelineArtifact` loader, `OnlinePreprocessor`, `LiveInferenceEngine`, `StreamWorker`, and `PredictionLogger`.
- Phase 2 session API: `AppSession.build_live_stream_session(...) -> LiveStreamSession`. `AppSession` remains the app-level composition boundary; do not introduce `OnlinePhase` or expose `session.online`.
- `StreamWorker` owns only the injected-dependency micro-batch loop. It keeps references to receiver/preprocessor/inference objects for `run()`, but `LiveStreamSession` owns start/stop/cleanup for the receiver, worker, and optional logger.
- Committed Phase 1 surface: config models, `SettingsManager`, `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, shared `utils.py` (`build_classifier`, `get_task_data`), `OfflineOrchestrator` (Phase 1 state machine, owns file I/O and `decoder_pipeline.joblib` export), and `AppSession` (`src/backend/session.py` — the single frontend entry point; owns `SettingsManager` lifetime and exposes `session.offline` for Phase 1).


## Dependency Management

- `requirements.txt` — runtime deps only
- `requirements-dev.txt` — `-r requirements.txt` plus test tooling
- `src/` code never imports test libraries such as `pytest`

## Running Tests

```bash
# From online_decoder/ root
pytest tests/
pytest tests/ -v --cov=src
RUN_LSL_INTEGRATION=1 pytest tests/online_phase/test_lsl_receiver_integration.py -q
python scripts/characterize_lsl.py --duration 10
python scripts/smoke_test_lsl_receiver.py --duration 5
python scripts/smoke_stream_worker.py --pipeline /path/to/decoder_pipeline.joblib --duration 5 --log /tmp/smoke.csv
```

## Config Schema

The experiment config lives in `experiment_config.yaml`. Its schema is defined in `src/backend/core/config_models.py` using Pydantic v2. When the YAML schema changes, update the Pydantic models.

## When to Update This File

Update `CLAUDE.md` when:
- The committed backend surface changes in a way that affects repo navigation
- The Phase 2 architecture direction changes materially
- A new top-level workflow directory is added
- A new project-wide convention is established
- The config schema structure changes significantly

Do not update it for every individual file.
