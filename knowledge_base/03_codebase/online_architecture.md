# Online Stage Architecture

Back to [Codebase Overview](README.md) or [Project Index](../START_HERE.md).

## Status

This repo is the standalone online-stage app, but it is only **partially implemented**.

As reconciled against the active Phase 2 design on **2026-05-11**, the app currently contains:
- The config/schema layer
- The `LSLReceiver` input boundary
- `OfflinePreprocessor` (Phase 1 preprocessing pipeline)
- `ModelEvaluator` (Phase 1 cross-validation)
- `ModelTrainer` (Phase 1 final training, spatial patterns)
- Shared offline utilities (`utils.py` — `build_classifier`, `get_task_data`)
- `OfflineOrchestrator` (Phase 1 façade — single frontend entry point, owns file I/O, state, and `decoder_pipeline.joblib` export)
- `DecoderPipelineArtifact` loader and `LiveInferenceEngine` (Phase 2 inference boundary)
- `OnlinePreprocessor`, `StreamWorker`, and `PredictionLogger` for the Phase 2 micro-batch backend
- Replay, characterization, and smoke-test scripts
- Unit tests plus an opt-in replay integration test; validation notebooks

It does **not** yet contain the PyQt6 frontend or final real-lab validation.
It does **not** yet contain the committed live decoding stack end-to-end. The PyQt6 frontend is **partially implemented** — Phase 1 UI Steps 1–4 of 12 are committed (shell, journey panel, settings view with shared widgets).

## Authority

For the online app, the source of truth is the code in `src/`.

When documents disagree:
- Code beats docs
- The current implementation docs in `docs/` summarize the maintained online-app implementation view
- Historical timeline documents remain useful context, but they are not the active implementation contract

## Overview

This is a **standalone, self-contained application** for the online stage. It has no dependency on the parent `reactivation-decoder` repo.

## Conceptual Phase Structure

The app is conceptually organized into two phases:

### Phase 1: Offline Training

Older docs sometimes called this "offline." It is the same Phase 1 concept.

Phase 1 is the subject-break training side of the app. Its job is to:
- preprocess the recorded localizer EEG
- evaluate decoder performance across time
- choose or confirm the inference timepoint
- train the final decoder artifacts
- export the state Phase 2 will need for live preprocessing and inference

Conceptually, the main Phase 1 pieces are:
- **OfflinePreprocessor** — ✅ committed in `src/backend/offline_phase/`
- **ModelEvaluator** — ✅ committed in `src/backend/offline_phase/`
- **ModelTrainer** — ✅ committed in `src/backend/offline_phase/`
- **OfflineOrchestrator** — ✅ committed in `src/backend/offline_phase/`; the single frontend-facing entry point for Phase 1, owning file I/O, state management, bundling, and `decoder_pipeline.joblib` export. During raw load it also decodes parallel-port triggers from the analog trigger channel (see [`02_reference/parallel_port_trigger_decoding.md`](../02_reference/parallel_port_trigger_decoding.md))
- **Phase 1 UI** — 🔲 partially implemented (Steps 1–4 of 12); PyQt6 5-node training pipeline. See [frontend_layout.md](../../docs/architecture/frontend_layout.md) and [Phase1_UI_Plan.md](../../docs/old/phase1_ui_plan.md)

Phase 2 depends on the `decoder_pipeline.joblib` produced by the orchestrator.

### Phase 2: Live Inference

Phase 2 is the live path that consumes the outputs of Phase 1 and runs:
- live EEG ingestion
- causal preprocessing
- decimation
- model inference
- downstream logging and UI updates

The active Phase 2 direction is a **stateful micro-batch pipeline**:
- `AppSession.build_live_stream_session(...)` composes the backend objects and returns `LiveStreamSession`
- `LiveStreamSession` owns lifecycle: receiver start, worker start/stop/join, optional logger close, receiver stop
- `StreamWorker` owns only the background batch loop and uses injected `LSLReceiver`, `OnlinePreprocessor`, and `LiveInferenceEngine` dependencies
- `StreamWorker.prediction_ready` feeds UI/logging consumers

`RingBuffer` is **obsolete** in this app and should not be treated as the current design.

## Current Implementation Surface

The current committed backend is centered on:
- `SettingsManager` and the Pydantic config models
- `LSLReceiver` for stream ingestion and trigger decoding
- `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, `OfflineOrchestrator`, and shared `utils.py` for Phase 1
- `DecoderPipelineArtifact` loader, `OnlinePreprocessor`, `LiveInferenceEngine`, `StreamWorker`, and `PredictionLogger` for Phase 2 backend processing

## Directory Structure

```text
repository/
├── src/
│   ├── backend/
│   │   ├── core/           — SettingsManager and Pydantic config models
│   │   ├── offline_phase/  — utils, OfflinePreprocessor, ModelEvaluator, ModelTrainer, OfflineOrchestrator
│   │   └── online_phase/   — LSLReceiver, OnlinePreprocessor, LiveInferenceEngine, StreamWorker, PredictionLogger, artifact loader
│   └── frontend/           — PyQt6 UI (partially committed: screens/, views/, widgets/, styles/)
├── scripts/                — Characterization, replay, and smoke-test helpers
├── tests/
│   ├── core/               — Config validation tests
│   ├── notebooks/          — Manual validation notebooks
│   ├── offline_phase/      — OfflinePreprocessor, ModelEvaluator, ModelTrainer, OfflineOrchestrator tests
│   └── online_phase/       — LSLReceiver unit and opt-in replay integration tests; artifact loader and LiveInferenceEngine tests
├── tools/lslproxy/         — LSLProxy.exe and Windows DLLs
└── docs/                   — Current backend and frontend notes
```

Not currently finalized:
- `src/frontend/` — PyQt6 UI (planned; see [frontend_architecture.md](frontend_architecture.md))
- Real lab validation of the full Phase 2 stream against the NeurOne/LSLProxy setup

## Current Components

### Core (`src/backend/core/`)
- **Config models**: Pydantic v2 schema for `experiment_config.yaml`
- **SettingsManager**: Validates the YAML and exposes preprocessing, decoder, and event-mapping sections

### Online Phase (`src/backend/online_phase/`)
- **LSLReceiver**: Resolves an LSL stream, optionally launches `LSLProxy.exe`, strips the trigger channel, decodes marker edges, validates the stream layout, and returns aggregated chunks
- **OnlinePreprocessor**: Causal online preprocessing with persistent filter/decimation state and fixed Phase 1 spatial transforms
- **LiveInferenceEngine**: Runs trained decoder models and returns per-task positive-class probabilities
- **StreamWorker**: Background `QThread` micro-batch loop. It keeps references to injected runtime dependencies but does not own their lifecycle.
- **PredictionLogger**: Optional CSV sink connected to `prediction_ready`

### Support Tooling
- **`scripts/characterize_lsl.py`**: Measures chunk sizes, inter-arrival timing, and effective sample rate
- **`scripts/replay_xdf_to_lsl.py`**: Replays recorded `.xdf` data into an LSL stream
- **`scripts/smoke_test_lsl_receiver.py`**: Manual connectivity and data-flow check
- **`scripts/smoke_stream_worker.py`**: Headless StreamWorker + logger smoke check using a Phase 1 pipeline artifact

## Phase 2 Session API

The active session-level API is:
- **AppSession** remains the only backend class imported by the frontend.
- **AppSession.build_live_stream_session(...)** constructs the receiver, preprocessor, inference engine, worker, and optional logger.
- **LiveStreamSession** exposes `prediction_ready`, `start()`, and `stop()`.
- Do **not** introduce `OnlinePhase` or `session.online`.
- Do **not** make `StreamWorker` construct or own the whole online runtime.

The intended live path is:
1. Frontend calls `live = session.build_live_stream_session(...)`
2. Frontend connects `live.prediction_ready`
3. `live.start()` starts the receiver and worker
4. `StreamWorker` pulls pending EEG samples and markers, accumulates about 40 ms of samples, preprocesses, predicts, and emits all predictions
5. `live.stop()` stops the worker, waits for it, closes the optional logger, and stops the receiver

## What This Document Doesn't Cover

- Full historical design debates: see the `01_timeline/03_online_stage_design/` documents
- Detailed LSL hardware notes: see [../01_timeline/03_online_stage_design/Lab Equipment & LSL.md](../01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md)
- Parent-repo semester-A architecture: see [offline_architecture.md](offline_architecture.md)
- Fine-grained backend status and interfaces: see [Backend Architecture](../../docs/architecture/backend_architecture.md)
- Frontend UI design: see [frontend_layout.md](../../docs/architecture/frontend_layout.md) and [Phase1_UI_Plan.md](../../docs/old/phase1_ui_plan.md)
