# Phase 1 UI — Implementation Plan

Back to [Backend Architecture](backend_architecture.md) or [Docs Index](README.md).

---

## Status

This is the **active implementation contract** for the Phase 1 PyQt6 frontend.

**Progress:** Step 5 complete (Node 2: Load Data, backend wired). The shared `LoadingOverlay` is now the canonical pattern for every blocking backend call in Phase 1 — Steps 6+ reuse it directly instead of introducing embedded progress pages. Next: combined Step 6 — Node 3 Step 1 (overlay) + ICA review.

| Step | Description | Status |
|---|---|---|
| 1 | App shell + theme | ✅ Done |
| 2 | Journey panel (static) + Phase1Screen layout | ✅ Done |
| 3 | Stub workspace views + left–right wiring | ✅ Done |
| 4 | Node 1: Pipeline Settings (backend wired) | ✅ Done |
| 5 | Node 2: Load Data (backend wired) | ✅ Done |
| 6 | Node 3: Step 1 (overlay) + ICA review | — |
| 7 | Node 3: Preprocessing Step 2 + complete | — |
| 8 | Chart widgets (isolated) | — |
| 9 | Node 4: Evaluation (overlay + results view) | — |
| 10 | Topomap widget (isolated) | — |
| 11 | Node 5: Train & Save (overlay + complete) | — |
| 12 | (Optional) Embedded progress pages for Nodes 3 & 4 | — |

Design reference: React mockup at [`knowledge_base/02_reference/ui_demo/Phase1Screen.jsx`](../../../knowledge_base/02_reference/ui_demo/Phase1Screen.jsx) (originally from `https://github.com/roiguri/decoder_gui`).

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
│   ├── loading_overlay.py         # Semi-transparent overlay (message + indeterminate bar), parented to workspace card
│   ├── shared.py                  # FilePicker, ReadOnlyField, SectionCard
│   ├── ica_component_card.py      # Card: matplotlib topomap + keep/reject toggle (richer diagnostics deferred)
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
│   ├── config_loader_worker.py    # Wraps AppSession(config_path) construction
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
  - "Load Config File" `FilePicker` → `QFileDialog.getOpenFileName` filter `"Config (*.yaml *.yml)"` → stores `_config_path`; shows a checkmark label on success
  - "Select Output Directory" `FilePicker` → `QFileDialog.getExistingDirectory` → stores `_output_dir`; shows selected path
  - Read-only `SectionCard`s: **Preprocessing** (bandpass, resample, ICA, epochs) and **Model Evaluation** (model, CV folds, decoders) populated from `SettingsManager` after config load

**Wire:** Node 1 has no in-view "Continue" button — the **journey-panel** action button is the sole Continue affordance. It is gated by the [Ready Protocol](#ready-protocol): `SettingsView` emits `ready_changed(bool)` whenever both paths are set (or cleared), and `Phase1Screen` forwards it to `journey_panel.set_node_ready(0, ready)`. The panel button is wired to `SettingsView.trigger_continue`, which builds the `AppSession` and emits `session_ready`.

On `AppSession` construction error: `QMessageBox.critical(...)`, paths cleared, user can retry.

**Test:** pick the real `experiment_config.yaml` + a writable output dir → settings sections populate with correct values → the journey-panel "Continue" button (Node 1) becomes enabled → click → trail animates to Node 2.

---

### Step 5 — Node 2: Load Data (backend wired)

`workers/__init__.py` + `workers/base_worker.py` already exist from the Step 4 follow-up — reuse `BaseWorker`. The reusable `LoadingOverlay` on `Phase1Screen` is also already wired: views opt in by declaring `loading_requested(str)` and `loading_done()` signals; `Phase1Screen` connects them to `show_loading` / `hide_loading`.

**Create:**
- `workers/load_worker.py` — calls `orchestrator.load_raw_data()`
- `views/load_data_view.py` (replace stub) — dir picker `QPushButton` + path label + "Load Data" button (disabled until dir selected). Use the shared `LoadingOverlay` via the `loading_requested` / `loading_done` signals instead of an in-view `QProgressBar`.

**Wire:** "Load Data" → `orchestrator.set_file_path(data_dir)` → emit `loading_requested("Loading data…")` → start `LoadWorker` on `QThread` → `result_ready` → emit `loading_done()` → `journey_panel.advance(2)`.

**Test with a real `.vhdr` file:** pick dir → click "Load Data" → progress bar shows → trail animates to Node 3. Window stays responsive during load.

---

### Step 6 — Node 3: Step 1 (overlay) + ICA review

Combines the original Step 6 (preprocessing-step1 worker) and Step 7 (ICA component review). Step 1 progress is shown via the shared `LoadingOverlay` rather than an embedded progress page, so the inner stack drops from 4 pages to 2.

**Create:**
- `workers/preprocessing_worker.py` — `PreprocessingStep1Worker` calls `orchestrator.run_step1_prepare_ica()`; plus a no-op `PreprocessingStep2Worker` placeholder (immediately emits `result_ready({})`) so the Page 0 → Page 1 transition can be exercised end-to-end.
- `widgets/ica_component_card.py` — `ICAComponentCard(QWidget)`: embedded `FigureCanvas` with a single axis showing `mne.viz.plot_topomap(ica.get_components()[:, i], ica.info)`; amber/muted color badge; keep/reject `QPushButton` (checkable). Per-component time series, PSD, and ICLabel labels are deferred — see "Deferred per-component enhancements" under Node 3 detail.
- `views/preprocessing_view.py` (replace stub) — inner `QStackedWidget` with **3 pages**:
  - Page 0 = "Ready to Preprocess" landing (play-circle icon, title, description, in-view "Start Preprocessing" button).
  - Page 1 = ICA review grid (`QScrollArea` → 4-column grid of `ICAComponentCard` built from `(ica_obj, suggested_components)`); "Confirm" button below grid.
  - Page 2 = stub for the complete view (Step 7 next will replace).

**Wire:**
- In-view "Start Preprocessing" (Page 0) → `PreprocessingView.trigger_start` → emit `loading_requested("Running preprocessing…")` → start `PreprocessingStep1Worker` → on `result_ready((ica, suggested))`: hide overlay, populate grid, advance inner stack to Page 1, enable in-view "Confirm" button.
- Confirm (Page 1) → collect rejected indices → emit `loading_requested("Finishing preprocessing pipeline…")` → start placeholder `PreprocessingStep2Worker` → on `result_ready({})`: hide overlay, advance inner stack to Page 2 stub.
- The journey-panel Node 3 action is still wired to `trigger_start` for parity (so the Ready Protocol contract holds), and `ready_changed` follows the same gate as the in-view Start button: True while the user is on the Ready page with data loaded, False once preprocessing starts or completes. Clicking either button is equivalent.

**Card scope:** topomap + suggested/keep badge + Keep/Reject toggle only. The richer per-component signals (time series, PSD, ICLabel category) are explicitly deferred — see "Deferred per-component enhancements" under Node 3 detail for the rationale and recommended ordering.

**Test:** after Node 2, the workspace shows the Ready page → click in-view "Start Preprocessing" → overlay covers the workspace card during real preprocessing → grid renders on Page 1 with real MNE topomaps; toggling keep/reject changes the card style; Confirm → overlay shows briefly → Page 2 stub.

---

### Step 7 — Node 3: Preprocessing Step 2 + complete

**Create:**
- Replace the placeholder `PreprocessingStep2Worker` in `workers/preprocessing_worker.py` with a real implementation that calls `orchestrator.run_step2_finish_pipeline(excluded_components)`.
- Update `views/preprocessing_view.py` Page 1 — stats labels (epochs retained, components removed); calls `journey_panel.advance(3)` on display.

**Wire:** Confirm (from Step 6) → real `PreprocessingStep2Worker` (via the shared `LoadingOverlay`) → stats Page 1 → `journey_panel.advance(3)`.

**Test:** full Node 3 flow: preprocessing → ICA grid → Confirm → overlay → epoch stats → trail to Node 4.

---

### Step 8 — Chart widgets (isolated)

**Create:**
- `widgets/charts/__init__.py`
- `widgets/charts/auc_chart.py` — `AUCChart(FigureCanvas)`: plots one colored line per decoder (`diagonal_auc` vs `times`); vertical dashed line at `suggested_timepoint`; `mpl_connect("button_press_event")` snaps to nearest timepoint → emits `timepoint_selected(float)`
- `widgets/charts/tgm_chart.py` — `TGMChart(FigureCanvas)`: `imshow(tgm_matrix)` with ms axis labels and colorbar

**Test in isolation:** write `scripts/test_charts.py` — create fake numpy arrays matching the evaluation result shape, instantiate `AUCChart` and `TGMChart` in a minimal `QApplication`, verify `timepoint_selected` fires on click. Delete after confirming.

---

### Step 9 — Node 4: Evaluation (overlay + results view)

**Create:**
- `workers/evaluation_worker.py` — calls `orchestrator.run_evaluation()`
- Update `views/evaluation_view.py`: single `QTabWidget` results view.
  - Summary tab: `AUCChart` (all decoders) + stats panel (selected time, per-decoder AUC) + "Confirm Timepoint" button (disabled until timepoint selected)
  - Per-decoder tabs: individual `AUCChart` + `TGMChart`

**Wire:** Node 4 button → emit `loading_requested("Running evaluation…")` → start `EvaluationWorker` → on `result_ready`: hide overlay, build charts. Timepoint click → update stats panel. "Confirm Timepoint" → `Phase1Screen._selected_timepoint = t` → `journey_panel.advance(4)`.

**Test:** after Node 3, click "Run Evaluation" → overlay → AUC chart with all decoder lines; click different timepoints → stats panel updates; confirm → trail to Node 5.

---

### Step 10 — Topomap widget (isolated)

**Create:**
- `widgets/charts/topomap_widget.py` — `TopomapWidget(FigureCanvas)`: calls `mne.viz.plot_topomap(pattern, info, axes=ax, show=False)`

**Test in isolation:** write `scripts/test_topomap.py` — load a saved `.fif` or fabricate `mne.Info`, instantiate `TopomapWidget` in a minimal `QApplication`, confirm it renders without error. Delete after confirming.

---

### Step 11 — Node 5: Train & Save (overlay + complete)

**Create:**
- `workers/training_worker.py` — calls `orchestrator.run_training(selected_timepoint)`
- Update `views/train_view.py`: single complete view = read-only `QLineEdit` showing `model_filepath`; grid of `TopomapWidget` (one per decoder task using `spatial_patterns[task]` + `mne_info`).

**Wire:** Node 5 button → emit `loading_requested("Training decoders…")` → start `TrainingWorker(orchestrator, Phase1Screen._selected_timepoint)` → on `result_ready`: hide overlay, build the topomap grid.

**Test (full pipeline):** run all steps end-to-end: settings → load → preprocess → ICA review → evaluate → confirm timepoint → train → topomaps render; confirm `decoder_pipeline.joblib` exists in output dir; confirm no UI freeze at any stage.

---

### Step 12 — (Optional) Embedded progress pages for Nodes 3 & 4

Purely a UX polish, may be skipped. Replace the shared `LoadingOverlay` for the preprocessing-step1, preprocessing-step2, and evaluation calls with embedded indeterminate `QProgressBar` pages inside the inner stack, so that progress lives on the workspace card itself rather than as a transient block. If this step is taken, the "Loading Overlay" section above must be updated to allow the carve-out, and the Node 3 / Node 4 detail sections must add the progress pages back.

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
    finished       = Signal()        # emit in subclass run()'s finally — drives thread cleanup

    def run(self) -> None:
        ...  # override in each subclass
```

PyQt6 exports `pyqtSignal`, not `Signal`. All frontend code uses the alias `from PyQt6.QtCore import pyqtSignal as Signal` so the plan's `Signal(...)` notation matches the implementation directly.

**Standard UI lifecycle when a node button is clicked:**
1. Disable the button; emit `loading_requested("…")` so `Phase1Screen` shows the shared `LoadingOverlay`.
2. Instantiate `Worker(orchestrator, ...)` and `QThread()`.
3. Move worker to thread; connect `thread.started → worker.run`.
4. Connect `worker.result_ready` → handler (emits `loading_done()`, then advances sub-step or calls `journey_panel.advance()`).
5. Connect `worker.error_occurred` → handler (emits `loading_done()`, then `QMessageBox.critical(...)`).
6. Connect cleanup: `worker.finished → thread.quit`, `thread.finished → worker.deleteLater`, `thread.finished → thread.deleteLater`.
7. Keep `self._worker` and `self._thread` references on the view so they aren't GC'd while the thread is running. **Only null them in a slot connected to `thread.finished`** — nulling from `result_ready` is too early and will crash with `QThread: Destroyed while thread is still running` because `thread.quit()` is itself queued.
8. `thread.start()`.

---

## Loading Overlay

A reusable `LoadingOverlay` (`widgets/loading_overlay.py`) is mounted on the workspace card by `Phase1Screen`. It shows a centered message + indeterminate `QProgressBar` over a semi-transparent white background, and resizes with its host via an installed event filter. The journey panel sits outside the overlay's parent tree and stays visible/interactive at all times.

**View protocol:** any view that needs to block its workspace during a backend call declares two signals:

```python
loading_requested = Signal(str)
loading_done = Signal()
```

`Phase1Screen` connects them once in `__init__` to its own `show_loading(message)` / `hide_loading()`. Views never reference `Phase1Screen` directly — this matches the existing `session_ready` pattern and keeps view ↔ screen coupling one-way.

Every blocking backend call in Phase 1 routes through this overlay via `loading_requested` / `loading_done`. Views should not introduce their own in-view `QProgressBar` for transient backend calls; use the shared overlay instead. Embedded indeterminate progress pages inside inner stacked sub-pages are deferred — see the optional final step in the Serial Build Plan.

---

## Ready Protocol

Each node's primary "advance" action lives on the **journey-panel** action button — there is no duplicate Continue button inside the view. The button is gated by the view itself via a reusable signal/slot contract:

```python
# In a view that gates its node's action button:
ready_changed = Signal(bool)        # emit whenever prerequisites flip

def trigger_<action>(self) -> None:  # public slot the panel button calls
    ...
```

**Wiring (done once in `Phase1Screen.__init__`):**

```python
self._journey_panel.set_node_action(node_idx, view.trigger_<action>)
self._journey_panel.set_node_ready(node_idx, False)   # gate by default
view.ready_changed.connect(
    lambda ready, n=node_idx: self._journey_panel.set_node_ready(n, ready)
)
```

The view owns "what makes me ready" (file paths picked, timepoint selected, ICA confirmed, …) and never references the journey panel directly. `JourneyPanel.set_node_ready(node_index, ready)` toggles the action button's enabled state via `JourneyNode.set_action_enabled`.

Nodes 2–5 adopt the same protocol as they are implemented (e.g. `LoadDataView.ready_changed` flips when the data directory is picked; the panel button becomes `trigger_load`).

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

- "Load Config File" `FilePicker` → `QFileDialog.getOpenFileName` filter `"Config (*.yaml *.yml)"` → displays a "✓ Config loaded" label on success; shows read-only parsed settings in two `SectionCard`s:
  - **Preprocessing**: bandpass, resample rate, ICA components/method, epoch tmin/tmax
  - **Model Evaluation**: model type, CV folds, decoder task list
- "Select Output Directory" `FilePicker` → `QFileDialog.getExistingDirectory` → shows selected path.
- **No in-view Continue button.** The journey-panel Node 1 action button is the sole Continue affordance — gated via the [Ready Protocol](#ready-protocol). `SettingsView.trigger_continue()` is invoked when the panel button is clicked: it builds `AppSession`, emits `session_ready`, and `Phase1Screen` calls `journey_panel.advance(1)`.
- On `AppSession` construction error: `QMessageBox.critical(...)`, paths cleared, user can retry.

---

### Node 2 — Load Data (`load_data_view.py`)

- `QPushButton` → `QFileDialog.getExistingDirectory` → show selected path.
- "Load Data" button (disabled until dir selected) → calls `orchestrator.set_file_path(data_dir)`, starts `LoadWorker`.
- On `result_ready`: `journey_panel.advance(2)`.
- On `error_occurred`: `QMessageBox.critical(...)`.

---

### Node 3 — Preprocessing (`preprocessing_view.py`)

Inner `QStackedWidget` with 3 pages. Both blocking calls (`run_step1_prepare_ica`, `run_step2_finish_pipeline`) are wrapped by the shared `LoadingOverlay`; the view does not render its own progress page.

| Page | What's shown | What happens |
|---|---|---|
| 0 | "Ready to Preprocess" landing: big play-circle icon, title, descriptive text, in-view "Start Preprocessing" primary button | User clicks Start → `PreprocessingStep1Worker` runs (overlay covers workspace) |
| 1 | ICA review grid (4 columns, N rows) + in-view "Confirm" button | User reviews and toggles components; Confirm triggers `PreprocessingStep2Worker` (overlay covers workspace) |
| 2 | Stats: epochs retained, components removed | `journey_panel.advance(3)` called |

Both intra-node transitions (Page 0 → 1 and 1 → 2) are driven by **in-view buttons**, not the journey-panel button — Node 3's mockup explicitly hands the trigger to the workspace. The journey-panel Node 3 button is reserved for the final "Continue to Evaluation" action wired in the Step 7 follow-up once Page 2 renders.

**`ICAComponentCard(QWidget)` — Page 1:**
- `FigureCanvas` (matplotlib) with a single axis showing `mne.viz.plot_topomap(ica.get_components()[:, i], ica.info)`.
- Color badge: "SUGGESTED REJECT" (amber) or "KEEP" (gray), set from `suggested_components`.
- Keep / Reject toggle button (`QPushButton`, checkable).
- "Confirm" button below the grid reads all toggle states and transitions to Page 2.

#### Deferred per-component enhancements

The card intentionally shows only the topomap plus the suggested/keep badge in the current implementation. Three richer per-component signals were considered and are deferred — they should be designed together rather than landed piecemeal, because they all hang off the same backend exposure point (the dict returned by `run_step1_prepare_ica`). The in-code anchor is the TODO block at the top of `widgets/ica_component_card.py`.

1. **Per-component time series** — `ica.get_sources(raw).get_data()[i, :N]`. Visually distinguishes blinks (sharp transients) / ECG (rhythmic) / muscle (broadband). Requires backend exposure: `raw` is private to the orchestrator and the frontend must not reach for it directly. Likely shape: extend the diagnostics dict on each component with a `sources_short` field (first ~2 s for a card glance) and optionally `sources_long` for a future inspect dialog.

2. **Per-component PSD** — frequency-domain plot. Single most informative signal after the topomap for triage: line-noise spikes at 50/60 Hz, eye blinks concentrate <4 Hz, muscle is broadband >30 Hz, brain components show alpha (~10 Hz). Cheap to compute backend-side (FFT on the source). Card real estate (260×240) won't fit a readable PSD inline alongside the topomap — likely belongs in an `ICAComponentInspectDialog` opened by an "Inspect" affordance on each card.

3. **ICLabel category + confidence** — `mne-icalabel` classifies each component as `brain / muscle / eye / heart / line_noise / channel_noise / other` with a 7-way probability vector. Surfaces as a coloured class badge **alongside** the existing amber "SUGGESTED REJECT" badge (not replacing it — the two signals are complementary). **Caveats before adopting:** ICLabel was trained on a specific data preparation (1–100 Hz bandpass, extended-infomax ICA). Our pipeline currently uses a narrower bandpass and `fastica`, so ICLabel runs off-distribution and its confidences are miscalibrated. Two known fixes:
   - One-line config switch: ICA method `fastica` → `picard` with `fit_params={'ortho': False, 'extended': True}` (equivalent to extended infomax, faster, MNE-recommended).
   - Pipeline reorder so the production low-pass is applied **after** ICA fit, freeing ICA to see the wider band ICLabel expects. This is a deeper change to the preprocessor stage ordering.

   Until those land, ICLabel outputs would be hints, not authoritative. The product question: should the UI surface raw probabilities (requires the fixes above) or coarse buckets ("likely eye / likely brain / uncertain") that don't make false-precision claims?

**Recommended order when picked up:** (1) make the backend decision and extend the diagnostics dict returned by `run_step1_prepare_ica` with whichever per-component fields the UI will consume; (2) build the inspect dialog scaffolding for the heavier visuals (PSD, longer time series); (3) add the overview-card badges (ICLabel category) last, once the backend can produce calibrated outputs.

**Frontend rule (unchanged):** Do **not** access `raw` via `session.offline._raw` or any other orchestrator-internal field. The fix for every gap above belongs in the backend, documented and reviewed before the frontend step that needs it is implemented.

---

### Node 4 — Evaluation (`evaluation_view.py`)

Single results page (no inner stack — evaluation runs through the shared `LoadingOverlay`, then results render directly).

**Results (`QTabWidget`):**

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

Single complete page (no inner stack — training runs through the shared `LoadingOverlay`, then the complete view renders directly).

**Complete:**
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

**If data the UI needs is not exposed by the orchestrator's current API, stop and write a separate backend plan.** Do not work around the gap by importing backend internals, accessing private attributes, or adding temporary shims in the frontend. The fix belongs in the backend, documented and reviewed before the frontend step that needs it is implemented. The ICA time-series / PSD / ICLabel diagnostics needed by `ICAComponentCard` are the canonical example — see "Deferred per-component enhancements" under Node 3 detail.

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
3. **Node 2**: pick data dir → "Load Data" → shared overlay shows → trail animates to Node 3.
4. **Node 3 step 1**: "Run Preprocessing" → overlay shows → ICA grid renders with real MNE topomaps when overlay clears.
5. **Node 3 step 2**: toggle rejections → "Confirm" → overlay shows → complete view with epoch stats → trail to Node 4.
6. **Node 4**: "Run Evaluation" → overlay shows → AUC chart renders with clickable timepoints; click timepoint → stats update → "Confirm Timepoint" → trail to Node 5.
7. **Node 5**: "Run Training" → overlay shows → spatial topomaps render per decoder.
8. `decoder_pipeline.joblib` exists in the output directory.
9. No UI freeze during any backend call (main thread always responsive).
