# online_decoder — Claude Guidelines

`online_decoder/` is the **standalone app root**. It must remain self-contained and portable — no imports from the parent monorepo.

## Authority

Code under `src/` is the source of truth.

Use [docs/backend_architecture.md](docs/backend_architecture.md) for the maintained backend summary, [docs/Phase2_Implementation_Plan.md](docs/Phase2_Implementation_Plan.md) for the Phase 2 backend checklist, and [docs/Phase2_UI_Plan_M1.md](docs/Phase2_UI_Plan_M1.md) for the completed M1 UI plan. If the docs and code disagree, follow the code and update the docs.

## Directory Layout

```text
online_decoder/
├── scripts/            — Characterization, replay, and smoke-test helpers
├── src/backend/
│   ├── core/           — SettingsManager and Pydantic config models
│   ├── offline_phase/  — utils, OfflinePreprocessor, ModelEvaluator, ModelTrainer, OfflineOrchestrator
│   └── online_phase/   — LSLReceiver, OnlinePreprocessor, LiveInferenceEngine, StreamWorker, LiveSessionLogger
├── src/frontend/
│   ├── screens/        — Phase1Screen, Phase2Screen
│   ├── widgets/        — Phase 1 widgets + LiveProbabilityChart (pyqtgraph)
│   ├── widgets/phase2/ — Phase2Header, Phase2SettingsPanel, StartHaltButton
│   └── debug/          — Debug entry points (--phase2 quick-jump)
├── tests/
│   ├── core/           — Config validation tests
│   ├── notebooks/      — Manual validation notebooks
│   ├── offline_phase/  — Offline phase unit tests (preprocessor, evaluator, trainer)
│   ├── online_phase/   — LSLReceiver unit and opt-in replay integration tests
│   └── frontend/       — Headless UI tests (Phase 2 lifecycle, EvaluationView)
├── tools/lslproxy/     — LSLProxy.exe and Windows DLLs (hardware interface)
└── docs/               — Backend status and architecture notes
```

## Current Backend Scope

- Phase 2 backend surface: `LSLReceiver`, `DecoderPipelineArtifact` loader, `OnlinePreprocessor`, `LiveInferenceEngine`, `StreamWorker`, and `LiveSessionLogger` (run sink → `predictions.csv` + `markers.csv` + `manifest.json` + `predictions.npz`; `export_session_npz` for recovery).
- Phase 2 session API: `AppSession.build_live_stream_session(...) -> LiveStreamSession`. `AppSession` remains the app-level composition boundary; do not introduce `OnlinePhase` or expose `session.online`.
- `StreamWorker` owns only the injected-dependency micro-batch loop. It keeps references to receiver/preprocessor/inference objects for `run()`, but `LiveStreamSession` owns start/stop/cleanup for the receiver, worker, and optional logger.
- Phase 1 surface: config models, `SettingsManager`, `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, shared `utils.py` (`build_classifier`, `get_task_data`), `OfflineOrchestrator` (Phase 1 state machine, owns file I/O and `decoder_pipeline.joblib` export), and `AppSession` (`src/backend/session.py` — the single frontend entry point; owns `SettingsManager` lifetime and exposes `session.offline` for Phase 1).

## Current Frontend Scope

- Phase 2 live-inference UI: `Phase2Screen` (layout glue + lifecycle), `LiveProbabilityChart` (pyqtgraph, ring-buffered), `Phase2Header`, `Phase2SettingsPanel`, `StartHaltButton`.
- `pyqtgraph>=0.13` is a runtime dependency scoped to Phase 2 (Phase 1 uses matplotlib).
- Phase 2 screen is the only frontend consumer of `LiveStreamSession`. It imports only `AppSession` — no direct imports of backend internals.

## Known Conventions

- **LSL unit scaling (lab validation needed)**: The `lsl_to_si_scale` parameter was removed from `OnlinePreprocessor`. VHDR replay via `PlayerLSL` delivers data in SI volts (MNE converts on load), so no scaling is needed for replay-based validation. Whether NeurOne's LSL proxy outputs µV or V has not been verified in the lab — if it outputs µV, a scaling mechanism will need to be re-introduced.
- **Stream sources vs. the receiver**: `LSLReceiver` is a pure consumer (resolve + pull). Publishing a stream onto the network is a `StreamSource` (`src/backend/online_phase/stream_source.py`): `LslProxySource` wraps `tools/lslproxy/LSLProxy.exe` (Windows-only), and Phase 2 replay will add a `ReplaySource` sibling. `AppSession` owns the active source — started during `discover_streams()` and reused by the subsequent run (no proxy relaunch), stopped via `stop_stream_source()`. All live-LSL testing must happen on Windows.
- `debug_snapshots/` is git-ignored. Re-run `scripts/demo_seed_debug_snapshots.py` when joining a new machine.


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

The `preprocessing:` block follows the new reference pipeline (see `docs/Preprocessing_Migration_Plan.md`): `resample_filter_stage` (`early`|`late`), `channel_hygiene`, `highpass`, `notch`, `ica` (incl. `iclabel`), `epochs`, `lowpass`, `final_resample`. The old `bandpass`/`resample`/`reject_criteria` sections are gone. Both offline and online phases now consume the same positional `online_state` schema (`eeg_chunk_indices`, `bad_indices`, ICA matrices, interp weights, pre_whitener — no channel names).

## When to Update This File

Update `CLAUDE.md` when:
- The committed backend surface changes in a way that affects repo navigation
- The Phase 2 architecture direction changes materially
- A new top-level workflow directory is added
- A new project-wide convention is established
- The config schema structure changes significantly

Do not update it for every individual file.
