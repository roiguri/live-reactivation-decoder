# Frontend Architecture

Back to [Codebase Overview](README.md) or [Project Index](../START_HERE.md).

## Status

The frontend (`src/frontend/`) is **planned, not committed**.

The full Phase 1 UI implementation contract lives in [../../docs/Phase1_UI_Plan.md](../../docs/old/phase1_ui_plan.md). This document is the knowledge-base summary.

---

## Scope

The first frontend to be built is the **Phase 1 training pipeline screen** — a 5-node linear journey that takes the researcher from experiment configuration to a saved `decoder_pipeline.joblib` artifact:

1. **Pipeline Settings** — load experiment config YAML, pick output directory, review parsed settings
2. **Load Data** — pick data directory, load `.vhdr` file
3. **Preprocessing** — run ICA, review components manually, finish pipeline
4. **Evaluation** — run temporal generalization CV, inspect AUC curves and TGMs, select decoding timepoint
5. **Train & Save** — train final decoders, inspect spatial topomaps

The WelcomeScreen (experiment/subject setup) and Phase 2 live inference UI are out of scope for the first iteration.

---

## Layout Philosophy

`Phase1Screen` uses a two-panel split:

- **Left panel** (`QStackedWidget`): the dynamic workspace. Each node swaps the visible page.
- **Right panel** (320 px fixed): the journey panel. Five `JourneyNode` widgets stacked vertically with a connecting trail line.

The journey panel acts as the app's state machine. Only one node is active at a time. A node's action button is disabled until that node is active. Advancing a node:
1. Animates the trail segment filling from the completed node to the next (500 ms, `InOutCubic`).
2. Switches the workspace page on the left.

This enforces the linear pipeline: the researcher cannot skip or re-run a step out of order.

---

## Technology Decisions

| Concern | Decision | Rationale |
|---|---|---|
| UI framework | PyQt6 | Native Python, Windows-compatible, required for Phase 2 `QThread` streaming |
| Charts | matplotlib via `FigureCanvasQTAgg` | MNE topomaps render natively in matplotlib; no conversion layer needed |
| Threading | Worker-object pattern (`QObject` on `QThread`) | Keeps the main UI thread responsive during 10-60 s backend operations |
| Animation | `QPropertyAnimation` on a custom `fill_progress` property | Smooth trail fill without a separate animation library |

---

## Threading Model

Every blocking backend call (preprocessing, evaluation, training) runs on a background thread:

```
BaseWorker(QObject)
├── Signals: started(), progress(str), result_ready(object), error_occurred(str)
└── run() — overridden per worker, calls one OfflineOrchestrator method
```

UI pattern:
1. Disable controls, show indeterminate `QProgressBar`.
2. Move worker onto `QThread`; connect `thread.started → worker.run`.
3. On `result_ready` → advance journey panel, populate charts.
4. On `error_occurred` → `QMessageBox.critical(...)`, re-enable controls.

---

## Backend Integration

The only backend class the frontend imports directly is `AppSession` (`src/backend/session.py`). It is created inside `Phase1Screen` when Node 1 ("Pipeline Settings") completes — after the user picks the config YAML and output directory. `main.py` has no backend imports.

`AppSession` exposes `session.offline` (`OfflineOrchestrator`) for all Node 2–5 calls. The orchestrator holds all intermediate state (raw data, ICA object, epochs, evaluation results) between user-triggered steps, so the UI never stores EEG objects directly.

---

## Charting Widgets

Three embedded matplotlib canvases cover all visualization needs:

- `AUCChart` — time-series plot of decoder AUC across epoch time; click handler selects the inference timepoint.
- `TGMChart` — `imshow` of the temporal generalization matrix with ms axis labels and colorbar.
- `TopomapWidget` — `mne.viz.plot_topomap` of trained decoder spatial patterns; only shown after training completes (Node 5).

ICA component cards (Node 3) also embed a two-axis matplotlib figure (topomap + time series) per component.

---

## What This Document Doesn't Cover

- Full file structure and widget hierarchy: [../../docs/Phase1_UI_Plan.md](../../docs/old/phase1_ui_plan.md)
- Backend class interfaces: [../../docs/backend_architecture.md](../../docs/architecture/backend_architecture.md)
- Operator workflow intent: [../01_timeline/03_online_stage_design/](../01_timeline/03_online_stage_design/)
