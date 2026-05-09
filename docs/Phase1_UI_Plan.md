# Phase 1 UI — Implementation Plan

Back to [Backend Architecture](backend_architecture.md) or [Docs Index](README.md).

---

## Status

This is the **active implementation contract** for the Phase 1 PyQt6 frontend.

No frontend code is committed yet. When implementation begins, `src/frontend/` is the target directory.

Design reference: React mockup at `https://github.com/roiguri/decoder_gui` (`Phase1Screen.jsx`).

---

## Scope

This plan covers the **Phase 1 training pipeline only**: a 5-node linear journey that guides the researcher through configuring the experiment, loading data, preprocessing with ICA review, evaluating decoder performance, and training final models.

Nodes:
1. **Pipeline Settings** — load experiment config YAML, pick output directory, review parsed settings
2. **Load Data** — pick data directory, load `.vhdr` file
3. **Preprocessing** — run ICA, review components, finish pipeline
4. **Evaluation** — run temporal generalization CV, inspect AUC/TGMs, select timepoint
5. **Train & Save** — train final decoders, inspect spatial topomaps

Out of scope for this plan:
- WelcomeScreen (experiment / subject setup)
- Phase 2 live inference UI

---

## Charting Library

**Chosen: matplotlib via `FigureCanvasQTAgg`.**

Rationale:
- MNE's `plot_topomap` and ICA component plots are matplotlib-native; embedding in PyQt6 requires no conversion layer.
- The offline phase has no real-time rendering requirements, so chart latency is irrelevant.
- One library covers all three chart types needed: AUC time curves, TGM heatmaps, and spatial topomaps.

---

## File Structure

```
online_decoder/src/frontend/
├── __init__.py
├── main.py                        # QApplication entry point + bootstrap dialog
├── main_window.py                 # QMainWindow with QStackedWidget (one page per screen)
├── screens/
│   ├── __init__.py
│   └── phase1_screen.py           # QWidget: QHBoxLayout(workspace | journey panel)
├── widgets/
│   ├── __init__.py
│   ├── journey_panel.py           # JourneyNode + JourneyPanel (right sidebar, 320 px)
│   ├── ica_component_card.py      # Card: matplotlib topomap + waveform + keep/reject toggle
│   └── charts/
│       ├── __init__.py
│       ├── auc_chart.py           # FigureCanvas: AUC curves + clickable timepoint marker
│       ├── tgm_chart.py           # FigureCanvas: TGM imshow + colorbar
│       └── topomap_widget.py      # FigureCanvas: mne.viz.plot_topomap wrapper
├── views/                         # QStackedWidget pages — one per node or sub-step
│   ├── __init__.py
│   ├── settings_view.py           # Node 1: config file picker + output dir + read-only settings display
│   ├── load_data_view.py          # Node 2: data directory picker + load action
│   ├── preprocessing_view.py      # Node 3: inner QStackedWidget (4 sub-steps)
│   ├── evaluation_view.py         # Node 4: inner QStackedWidget (2 sub-steps)
│   └── train_view.py              # Node 5: inner QStackedWidget (2 sub-steps)
├── workers/
│   ├── __init__.py
│   ├── base_worker.py             # BaseWorker(QObject) with standard signal set
│   ├── load_worker.py             # Wraps orchestrator.load_raw_data()
│   ├── preprocessing_worker.py    # Step1Worker + Step2Worker
│   ├── evaluation_worker.py       # Wraps orchestrator.run_evaluation()
│   └── training_worker.py         # Wraps orchestrator.run_training(timepoint)
└── styles/
    ├── __init__.py
    └── theme.py                   # Color constants + global QSS stylesheet
```

---

## App Bootstrap

`main.py` has **no file dialogs and no backend imports**. It only:
1. Creates `QApplication`, applies global QSS stylesheet
2. Instantiates `Phase1Screen()` (session starts as `None`)
3. Adds it to `MainWindow` and calls `show()`

`AppSession` is created inside `Phase1Screen` when Node 1 ("Pipeline Settings") completes:
1. User clicks "Load Config File" → `QFileDialog.getOpenFileName` → YAML path stored
2. User clicks "Select Output Directory" → `QFileDialog.getExistingDirectory` → dir stored
3. Read-only parsed settings are displayed; "Continue" becomes enabled
4. User clicks "Continue" → `self.session = AppSession(config_path, output_dir)` → `journey_panel.advance(1)`

`AppSession` is the **only** backend class the frontend imports directly. It owns `SettingsManager` internally and exposes `session.offline` (`OfflineOrchestrator`) for Nodes 2–5. Phase 2 will add `session.build_stream_worker(...)`.

`Phase1Screen.__init__(self)` starts with `self.session = None`; all nodes from Node 2 onwards access `self.session.offline`.

**Running the app:** always run from the `online_decoder/` root:

```bash
PYTHONPATH=src python -m frontend.main
```

`main.py` must not manipulate `sys.path` itself. The `PYTHONPATH=src` prefix makes both `frontend.*` and `backend.*` importable as top-level packages. Backend imports in frontend code therefore look like:

```python
from backend.session import AppSession
```

---

## Backend API Used by the UI

The frontend imports only `AppSession` (`src/backend/session.py`). All Phase 1 calls go through `session.offline` (an `OfflineOrchestrator`):

| UI action | Call | Returns |
|---|---|---|
| Pick data dir | `orchestrator.set_file_path(data_dir)` | `None` |
| Load Data button | `orchestrator.load_raw_data()` | `None` |
| Run Preprocessing | `orchestrator.run_step1_prepare_ica()` | `(ica_obj, suggested_components: list[int])` |
| Confirm ICA | `orchestrator.run_step2_finish_pipeline(excluded)` | `{"n_epochs": int}` |
| Run Evaluation | `orchestrator.run_evaluation()` | `{times, suggested_timepoint, average_peak_auc, tasks}` |
| Run Training | `orchestrator.run_training(timepoint)` | `{"model_filepath": Path, "spatial_patterns": dict, "mne_info": Info}` |

---

## Serial Build Plan

Build the frontend from the outside in. Each step is **independently runnable and testable** — no step requires any future step to work.

---

### Step 1 — App shell + theme

**Create:**
- `styles/__init__.py`, `styles/theme.py` — color constants + QSS string
- `main_window.py` — `QMainWindow` with a `QStackedWidget` (one slot per screen)
- `main.py` — `QApplication` + stylesheet → instantiates `Phase1Screen()` → shows `MainWindow` (no dialogs, no backend imports)

**Stub:** `Phase1Screen` is an empty `QWidget` with `BG_LIGHT` background and `session = None`.

**Test:** `python -m frontend.main` — window opens immediately (no dialogs) with the correct background color.

---

### Step 2 — Journey panel (static) + Phase1Screen layout

**Create:**
- `screens/phase1_screen.py` — `QHBoxLayout`: left = stretching placeholder `QWidget`, right = journey panel (320 px fixed)
- `widgets/__init__.py`, `widgets/journey_panel.py` — `JourneyNode` + `JourneyPanel`: 5 nodes; trail drawn in `paintEvent`; `QPropertyAnimation` fills trail segment on `advance()`; Node 1 starts active

**Behaviour at this step:** clicking a node's action button calls `journey_panel.advance()` directly (no backend, no workspace change). All 5 nodes cycle through on click.

**Test:** run the app → click node buttons → trail segment fills with animation → node circle changes state (active → complete → next active).

---

### Step 3 — Stub workspace views + left–right wiring

**Create:**
- `views/__init__.py`
- `views/settings_view.py` — stub `QLabel("Node 1: Pipeline Settings")`
- `views/load_data_view.py` — stub `QLabel("Node 2: Load Data")`
- `views/preprocessing_view.py` — stub
- `views/evaluation_view.py` — stub
- `views/train_view.py` — stub

**Wire:** replace the left placeholder with a `QStackedWidget` holding the 5 stubs; connect `journey_panel.node_changed(int)` → `workspace_stack.setCurrentIndex(int)`.

**Test:** clicking node buttons fills the trail **and** changes the left panel label.

---

### Step 4 — Node 1: Pipeline Settings (backend wired)

**Create:**
- `views/settings_view.py` (replace stub) — two file-picker rows + read-only settings display:
  - "Load Config File" `QPushButton` → `QFileDialog.getOpenFileName` filter `"Config (*.yaml *.yml)"` → stores `_config_path`; shows a checkmark label on success
  - "Select Output Directory" `QPushButton` → `QFileDialog.getExistingDirectory` → stores `_output_dir`; shows selected path
  - Read-only `QGroupBox` sections: **Preprocessing** (bandpass, resample, ICA, epochs) and **Model Evaluation** (model, CV folds, decoders) populated from `SettingsManager` after config load
  - "Continue" button (disabled until both paths are set); on click: `self.session = AppSession(config_path, output_dir)` → `journey_panel.advance(1)`

**Wire:** Node 1 action button is the "Load Config File" trigger. "Continue" is a separate button inside the view (enabled only when both paths are filled).

On `AppSession` construction error: `QMessageBox.critical(...)`, paths cleared, user can retry.

**Test:** pick the real `experiment_config.yaml` + a writable output dir → settings sections populate with correct values → "Continue" becomes enabled → click → trail animates to Node 2.

---

### Step 5 — Node 2: Load Data (backend wired)

**Create:**
- `workers/__init__.py`, `workers/base_worker.py` — `BaseWorker(QObject)` with signals
- `workers/load_worker.py` — calls `orchestrator.load_raw_data()`
- `views/load_data_view.py` (replace stub) — dir picker `QPushButton` + path label + "Load Data" button (disabled until dir selected) + indeterminate `QProgressBar`

**Wire:** "Load Data" → `orchestrator.set_file_path(data_dir)` → start `LoadWorker` on `QThread` → `result_ready` → `journey_panel.advance(2)`.

**Test with a real `.vhdr` file:** pick dir → click "Load Data" → progress bar shows → trail animates to Node 3. Window stays responsive during load.

---

### Step 6 — Node 3, page 0: Preprocessing Step 1 + progress

**Create:**
- `workers/preprocessing_worker.py` — `PreprocessingStep1Worker` calls `orchestrator.run_step1_prepare_ica()`
- `views/preprocessing_view.py` — inner `QStackedWidget`; Page 0 = indeterminate `QProgressBar` + status label; Pages 1–3 are stubs

**Wire:** Node 3 action button → start `PreprocessingStep1Worker` → `result_ready(ica_obj, suggested_components)` → advance inner stack to Page 1 stub.

**Test:** after Node 2, click Node 3 → progress bar runs for the real preprocessing → transitions to the stub ICA page. No UI freeze during EEG computation.

---

### Step 7 — Node 3, page 1: ICA component review

**Create:**
- `widgets/ica_component_card.py` — `ICAComponentCard(QWidget)`: embedded `FigureCanvas` with two axes (left: `mne.viz.plot_topomap(ica.get_components()[:, i], ica.info)`, right: `ica.get_sources(raw).get_data()[i, :500]`); color badge; keep/reject `QPushButton` (checkable)
- Update `views/preprocessing_view.py` Page 1 — `QScrollArea` → 4-column grid of `ICAComponentCard` built from `(ica_obj, suggested_components)`; "Confirm" button below grid

**Wire:** "Confirm" → collect rejected indices → start `PreprocessingStep2Worker` placeholder (immediately emits `result_ready({})`) → Page 2 stub.

**Test:** after Step 6, the real ICA grid renders; toggling keep/reject changes button color; clicking Confirm transitions forward.

---

### Step 8 — Node 3, pages 2–3: Preprocessing Step 2 + complete

**Create:**
- Add `PreprocessingStep2Worker` to `workers/preprocessing_worker.py` — calls `orchestrator.run_step2_finish_pipeline(excluded_components)`
- Update `views/preprocessing_view.py`:
  - Page 2 = indeterminate `QProgressBar` + status label
  - Page 3 = stats labels (epochs retained, components removed); calls `journey_panel.advance(3)` on display

**Wire:** Confirm (Step 7) → real `PreprocessingStep2Worker` → progress (Page 2) → stats (Page 3) → `journey_panel.advance(3)`.

**Test:** full Node 3 flow: preprocessing → ICA grid → confirm → step 2 progress → epoch stats → trail to Node 4.

---

### Step 9 — Chart widgets (isolated)

**Create:**
- `widgets/charts/__init__.py`
- `widgets/charts/auc_chart.py` — `AUCChart(FigureCanvas)`: plots one colored line per decoder (`diagonal_auc` vs `times`); vertical dashed line at `suggested_timepoint`; `mpl_connect("button_press_event")` snaps to nearest timepoint → emits `timepoint_selected(float)`
- `widgets/charts/tgm_chart.py` — `TGMChart(FigureCanvas)`: `imshow(tgm_matrix)` with ms axis labels and colorbar

**Test in isolation:** write `scripts/test_charts.py` — create fake numpy arrays matching the evaluation result shape, instantiate `AUCChart` and `TGMChart` in a minimal `QApplication`, verify `timepoint_selected` fires on click. Delete after confirming.

---

### Step 10 — Node 4: Evaluation (progress + results view)

**Create:**
- `workers/evaluation_worker.py` — calls `orchestrator.run_evaluation()`
- Update `views/evaluation_view.py`:
  - Page 0 = indeterminate `QProgressBar` + `EvaluationWorker`
  - Page 1 = `QTabWidget`:
    - Summary tab: `AUCChart` (all decoders) + stats panel (selected time, per-decoder AUC) + "Confirm Timepoint" button (disabled until timepoint selected)
    - Per-decoder tabs: individual `AUCChart` + `TGMChart`

**Wire:** Node 4 button → `EvaluationWorker` → `result_ready` → build charts → Page 1. Timepoint click → update stats panel. "Confirm Timepoint" → `Phase1Screen._selected_timepoint = t` → `journey_panel.advance(4)`.

**Test:** after Node 3, click "Run Evaluation" → progress → AUC chart with all decoder lines; click different timepoints → stats panel updates; confirm → trail to Node 5.

---

### Step 11 — Topomap widget (isolated)

**Create:**
- `widgets/charts/topomap_widget.py` — `TopomapWidget(FigureCanvas)`: calls `mne.viz.plot_topomap(pattern, info, axes=ax, show=False)`

**Test in isolation:** write `scripts/test_topomap.py` — load a saved `.fif` or fabricate `mne.Info`, instantiate `TopomapWidget` in a minimal `QApplication`, confirm it renders without error. Delete after confirming.

---

### Step 12 — Node 5: Train & Save (progress + complete)

**Create:**
- `workers/training_worker.py` — calls `orchestrator.run_training(selected_timepoint)`
- Update `views/train_view.py`:
  - Page 0 = indeterminate `QProgressBar` + `TrainingWorker`
  - Page 1 = read-only `QLineEdit` showing `model_filepath`; grid of `TopomapWidget` (one per decoder task using `spatial_patterns[task]` + `mne_info`)

**Wire:** Node 5 button → `TrainingWorker(orchestrator, Phase1Screen._selected_timepoint)` → `result_ready` → build topomap grid → Page 1.

**Test (full pipeline):** run all 12 steps end-to-end: settings → load → preprocess → ICA review → evaluate → confirm timepoint → train → topomaps render; confirm `decoder_pipeline.joblib` exists in output dir; confirm no UI freeze at any stage.

---

## Threading Model

Every blocking backend call runs in a `QThread` using the **worker-object pattern** (move a `QObject` subclass onto a `QThread` rather than subclassing `QThread`).

```python
# workers/base_worker.py
from PyQt6.QtCore import QObject, pyqtSignal as Signal

class BaseWorker(QObject):
    started        = Signal()
    progress       = Signal(str)     # free-form status text for UI label
    result_ready   = Signal(object)  # payload type varies per subclass
    error_occurred = Signal(str)

    def run(self) -> None:
        ...  # override in each subclass
```

PyQt6 exports `pyqtSignal`, not `Signal`. All frontend code uses the alias `from PyQt6.QtCore import pyqtSignal as Signal` so the plan's `Signal(...)` notation matches the implementation directly.

**Standard UI lifecycle when a node button is clicked:**
1. Disable the button; show an indeterminate `QProgressBar`.
2. Instantiate `Worker(orchestrator, ...)` and `QThread()`.
3. Move worker to thread; connect `thread.started → worker.run`.
4. Connect `worker.result_ready` → handler (advances sub-step or calls `journey_panel.advance()`).
5. Connect `worker.error_occurred` → `QMessageBox.critical(...)`.
6. Connect cleanup: `worker.finished → thread.quit`, `thread.finished → thread.deleteLater`.
7. `thread.start()`.

---

## Journey Panel

**Layout:** right sidebar, fixed 320 px wide, `QVBoxLayout` with 5 `JourneyNode` widgets and a vertical trail line.

### `JourneyNode(QWidget)`
- Numbered circle: inactive = white + gray border, active = `PRIMARY_BLUE` fill, complete = green checkmark.
- Title + short description labels below the circle.
- Action button at the bottom; disabled when node is not active.
- Emits `action_clicked()` signal.

### `JourneyPanel(QWidget)`
- Draws a vertical trail between node centers in `paintEvent`.
- **Trail fill animation:** on `advance(node_index)`, a `QPropertyAnimation` on a custom `fill_progress` float property (0.0 → 1.0) drives a repaint that fills the trail segment from the just-completed node to the next node in `PRIMARY_BLUE`. Duration 500 ms, easing `QEasingCurve.Type.InOutCubic`.
- `advance(node_index)` marks node complete, starts animation, activates next node, emits `node_changed(int)`.

`Phase1Screen` connects `journey_panel.node_changed` to `workspace_stack.setCurrentIndex(node_index)`.

---

## Node Detail

### Node 1 — Pipeline Settings (`settings_view.py`)

- "Load Config File" `QPushButton` → `QFileDialog.getOpenFileName` filter `"Config (*.yaml *.yml)"` → displays a "✓ Config loaded" label on success; shows read-only parsed settings in two `QGroupBox` sections:
  - **Preprocessing**: bandpass, resample rate, ICA components/method, epoch tmin/tmax
  - **Model Evaluation**: model type, CV folds, decoder task list
- "Select Output Directory" `QPushButton` → `QFileDialog.getExistingDirectory` → shows selected path.
- "Continue" button (disabled until both paths are set); on click: `self.session = AppSession(config_path, output_dir)` → `journey_panel.advance(1)`.
- On `AppSession` construction error: `QMessageBox.critical(...)`, paths cleared, user can retry.

---

### Node 2 — Load Data (`load_data_view.py`)

- `QPushButton` → `QFileDialog.getExistingDirectory` → show selected path.
- "Load Data" button (disabled until dir selected) → calls `orchestrator.set_file_path(data_dir)`, starts `LoadWorker`.
- On `result_ready`: `journey_panel.advance(2)`.
- On `error_occurred`: `QMessageBox.critical(...)`.

---

### Node 3 — Preprocessing (`preprocessing_view.py`)

Inner `QStackedWidget` with 4 pages:

| Page | What's shown | What happens |
|---|---|---|
| 0 | Indeterminate `QProgressBar` + status label | `PreprocessingStep1Worker` calls `run_step1_prepare_ica()` |
| 1 | ICA review grid (4 columns, N rows) | User reviews and toggles components |
| 2 | Indeterminate `QProgressBar` + status label | `PreprocessingStep2Worker` calls `run_step2_finish_pipeline(excluded)` |
| 3 | Stats: epochs retained, components removed | `journey_panel.advance(3)` called |

**`ICAComponentCard(QWidget)` — Page 1:**
- `FigureCanvas` (matplotlib) with two axes: left = `mne.viz.plot_topomap(ica.get_components()[:, i], ica.info)`, right = ICA time series.
- Color badge: "SUGGESTED REJECT" (amber) or "KEEP" (gray), set from `suggested_components`.
- Keep / Reject toggle button (`QPushButton`, checkable).
- "Confirm" button below the grid reads all toggle states and transitions to Page 2.

**⚠ Backend data gap — resolve before implementing Step 7:**
The time series axis requires `ica.get_sources(raw).get_data()[i, :500]`, but `raw` is private to the orchestrator and not returned by any current API call. **The frontend must not access the orchestrator's internals or import backend classes to work around this.** Instead, open a separate backend plan to decide how to expose the data — for example, `run_step1_prepare_ica()` could return a third value (a pre-computed `(n_components, n_samples)` source array), or the orchestrator could gain a `get_ica_sources_preview() -> np.ndarray` method. Until that plan is written and the orchestrator updated, implement Step 7 with the topomap axis only and leave the time series axis as a placeholder.

---

### Node 4 — Evaluation (`evaluation_view.py`)

Inner `QStackedWidget` with 2 pages:

**Page 0 — Running:**
`EvaluationWorker` calls `orchestrator.run_evaluation()`. On completion → Page 1.

**Page 1 — Results (`QTabWidget`):**

*Summary tab:*
- `AUCChart`: one colored `matplotlib` line per decoder (`tasks[name]["diagonal_auc"]` vs `times`). Vertical dashed line at `suggested_timepoint`.
- `mpl_connect("button_press_event")` maps click x-coordinate to nearest time in `times` array → emits `timepoint_selected(float)`.
- Stats panel (right side): selected time display, per-decoder AUC at that time, average AUC.
- "Confirm Timepoint" button (disabled until timepoint selected) → stores timepoint in `Phase1Screen._selected_timepoint` → calls `journey_panel.advance(4)`.

*Per-decoder tabs (one per task name):*
- `AUCChart` (single decoder, same click behavior).
- `TGMChart`: `imshow(tasks[name]["tgm_matrix"])` with ms axis labels and colorbar.
- No topomaps at this stage (spatial patterns require training).

---

### Node 5 — Train & Save (`train_view.py`)

Inner `QStackedWidget` with 2 pages:

**Page 0 — Running:**
`TrainingWorker` calls `orchestrator.run_training(selected_timepoint)`. On completion → Page 1.

**Page 1 — Complete:**
- Read-only `QLineEdit` showing `model_filepath`.
- Grid of `TopomapWidget` instances, one per task: `mne.viz.plot_topomap(spatial_patterns[task], mne_info)`.
- Summary: number of decoders trained, selected timepoint in ms.

---

## Styling

`styles/theme.py` constants (verified against React mockup):

```python
PRIMARY_BLUE        = "#0078D4"   # primary CTAs, active nodes
SUCCESS_GREEN       = "#228B22"   # complete state
BG_LIGHT            = "#F3F3F3"  # outer window background
CARD_WHITE          = "#FFFFFF"  # workspace and journey panels
TEXT_PRIMARY        = "#1F2937"  # gray-800
TEXT_MUTED          = "#6B7280"  # gray-500
ALERT_RED           = "#C41E3A"  # error / reject
AMBER               = "#B45309"  # suggested reject badge
PRIMARY_BLUE_HOVER  = "#006CBE"
BORDER_GRAY         = "#E5E7EB"  # gray-200 panel borders
```

A global QSS string is applied via `QApplication.setStyleSheet(...)`:
- Main window background: `BG_LIGHT`.
- Workspace and journey panels: `CARD_WHITE` + `1px solid BORDER_GRAY`; journey panel adds `border-left` only.
- Card widgets: `CARD_WHITE` + `border-radius: 2px` (mockup uses `rounded-sm`, not 8 px).
- Primary buttons: `PRIMARY_BLUE` fill, white text, hover `PRIMARY_BLUE_HOVER`.
- Secondary buttons: white fill, `BORDER_GRAY` border, hover `#F3F4F6`.
- Disabled buttons: `#D1D5DB` fill, `TEXT_MUTED` text.

---

## Design Considerations

These are open questions that affect the robustness of the app but are not blocking for the initial implementation. Each should be resolved before the app is used in a real experiment session.

### AppSession as sole backend interface

`AppSession` (`src/backend/session.py`) is the **only** backend class the frontend may import. All UI code reaches the orchestrator via `session.offline`. No widget, view, or worker may import or instantiate `OfflineOrchestrator`, `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, or `SettingsManager` directly.

**If data the UI needs is not exposed by the orchestrator's current API, stop and write a separate backend plan.** Do not work around the gap by importing backend internals, accessing private attributes, or adding temporary shims in the frontend. The fix belongs in the backend, documented and reviewed before the frontend step that needs it is implemented. An example of this pattern is the ICA time series data needed by `ICAComponentCard` (see Step 7 above).

### Error handling

Workers catch all exceptions and emit `error_occurred(str)`. The UI shows `QMessageBox.critical` and re-enables the node's action button so the user can retry the step. This is safe because the orchestrator is designed to raise *before* updating internal state — a failed call leaves the orchestrator in its previous valid state. **To verify before implementation:** confirm that `run_step2_finish_pipeline` is atomic (no partial state left on mid-execution failure).

### App state and crash recovery

All EEG objects (raw data, ICA, epochs, evaluation results) live only in the orchestrator's memory. A crash loses the session. This is acceptable for a controlled lab setting. One improvement worth adding: save evaluation results (AUC arrays, TGM matrices) to disk as `.npz` after `run_evaluation()`. This prevents re-running the expensive cross-validation if the user needs to re-select the timepoint.

### Backward navigation

Not implemented in the initial version — the pipeline is strict linear forward-only. The most useful backward step would be returning to ICA review without re-running preprocessing step 1 (the ICA object is still in memory). Full backward navigation (e.g., re-running evaluation after training) involves resetting orchestrator state and should be designed carefully. Both are deferred as a separate milestone.

### File I/O ownership

The orchestrator owns all file writes. The UI never writes files directly — it only provides `output_dir` at startup and displays paths returned by the orchestrator (e.g., `model_filepath` from `run_training()`). Open question to resolve: who is responsible for creating the subject subfolder structure inside `output_dir` — the orchestrator on `__init__`, or on the first write?

---

## Verification Checklist

When implementation begins, verify end-to-end:

1. `python -m frontend.main` — window opens immediately (no dialogs).
2. **Node 1**: pick config YAML + output dir → settings sections populate → "Continue" becomes enabled → click → trail animates to Node 2.
3. **Node 2**: pick data dir → "Load Data" → indeterminate bar runs → trail animates to Node 3.
4. **Node 3 step 1**: "Run Preprocessing" → progress → ICA grid renders with real MNE topomaps.
5. **Node 3 step 2**: toggle rejections → "Confirm" → progress → complete view with epoch stats → trail to Node 4.
6. **Node 4 step 1**: "Run Evaluation" → progress → AUC chart renders with clickable timepoints.
7. **Node 4 step 2**: click timepoint → stats update → "Confirm Timepoint" → trail to Node 5.
8. **Node 5**: "Run Training" → progress → spatial topomaps render per decoder.
9. `decoder_pipeline.joblib` exists in the output directory.
10. No UI freeze during any backend call (main thread always responsive).
