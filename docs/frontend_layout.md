# Frontend Layout Reference

Back to [Docs Index](README.md) or [Phase 1 UI Plan](Phase1_UI_Plan.md).

---

## Overview

The frontend is a single `QMainWindow` that holds one "screen" at a time in a `QStackedWidget`. Currently only Phase 1 is wired up.

```
QApplication
└── MainWindow (QMainWindow 1280×800, min 960×600)
    └── QStackedWidget  (central widget — one screen at a time)
        └── Phase1Screen
```

---

## Phase 1 Screen — Two-Panel Shell

`screens/phase1_screen.py`

```
Phase1Screen (QWidget)
└── root (QHBoxLayout, 0 margins, 0 spacing)
    ├── outer (QWidget, #F3F3F3 bg, 24px padding all sides)  ← stretches to fill
    │   └── card (QFrame, white, 1px #E5E7EB border, 6px radius, drop shadow)
    │       ├── header_bar (QWidget, 48px tall, #FAFAFA bg, bottom border)
    │       │   └── _header_title (QLabel — updated per active node)
    │       └── _workspace (QStackedWidget, white bg)  ← one view at a time
    │           ├── [0] SettingsView       Node 1 — implemented
    │           ├── [1] LoadDataView       Node 2 — stub
    │           ├── [2] PreprocessingView  Node 3 — stub
    │           ├── [3] EvaluationView     Node 4 — stub
    │           └── [4] TrainView          Node 5 — stub
    └── _journey_panel (JourneyPanel, 320px fixed)          ← right sidebar
```

**Header title** is updated in `_on_node_changed()` using `_NODE_TITLES` (index-matched to workspace stack):

```python
_NODE_TITLES = [
    "Pipeline Settings", "Data Ingestion", "Preprocessing",
    "Model Evaluation", "Train & Save"
]
```

---

## Journey Panel

`widgets/journey_panel.py`

```
JourneyPanel (QWidget, 320px fixed)
└── QVBoxLayout (16px H / 20px V padding)
    ├── "TRAINING PIPELINE" header label
    ├── JourneyNode 1 — Settings
    ├── JourneyNode 2 — Load Data
    ├── JourneyNode 3 — Preprocessing
    ├── JourneyNode 4 — Evaluation
    ├── JourneyNode 5 — Train & Save
    └── stretch
```

Each `JourneyNode` has three visual states (inactive / active / complete) drawn in `paintEvent`. The trail line between nodes is also painted in `JourneyPanel.paintEvent`, not via child widgets.

`advance(n)` — marks node `n` complete, animates the trail to node `n+1`, activates `n+1`, emits `node_changed(n)`.

Node 1's action button is **overridden** via `set_node_action(0, handler)` — instead of auto-advancing, it opens the config file dialog.

---

## Workspace Views

### Node 1 — SettingsView (`views/settings_view.py`) — Implemented

```
SettingsView (QWidget)
└── QScrollArea (no frame)
    └── container (QWidget, 32px H / 24px V padding)
        └── inner (QWidget, max-width 720px, centered)
            ├── Setup section
            │   ├── FilePicker — "Load Config File"  + "✓ Config loaded" label
            │   └── FilePicker — "Select Output Directory"
            ├── Separator (1px #E5E7EB)
            ├── Preprocessing section (QLabel header + indented body)
            │   ├── Bandpass row    — ReadOnlyField × 2 + dim labels
            │   ├── Resample row    — ReadOnlyField × 1 + dim label
            │   ├── ICA row         — ReadOnlyField × 1 + dim labels
            │   ├── Epoch Size row  — ReadOnlyField × 2 + dim label
            │   └── Annotations     — rebuilt table widget on each config load
            ├── Model Evaluation section (QLabel header + indented body)
            │   ├── Model row       — 3 × QLabel badges (active one goes blue)
            │   ├── CV Folds row    — ReadOnlyField × 1
            │   └── Decoders        — rebuilt decoder cards on each config load
            ├── Continue button (right-aligned, disabled until both paths set)
            └── stretch
```

### Nodes 2–5 — Stubs

All currently placeholder `QLabel` centered in a `QVBoxLayout`. Each will be replaced with a full implementation per the [Phase 1 UI Plan](Phase1_UI_Plan.md).

---

## Shared Widgets

`widgets/shared.py` — reusable across all views.

### `FilePicker(QWidget)`
```
[  Button  ]   /path/to/selected/file-or-dir
```
- Secondary-style button opens `QFileDialog` (file or dir mode)
- `path_selected(str)` signal emitted on selection
- `path` property → `str | None`; `clear()` resets to unselected

### `ReadOnlyField(QWidget)`
```
┌──────────┐
│   value  │  unit
└──────────┘
```
- Bordered monospace `QLineEdit` (read-only, `#F9FAFB` bg)
- `set_value(v)` — `None` shows `—` placeholder
- `field_width` param controls the input box width

---

## Signal Flow

```
FilePicker.path_selected
    → SettingsView._on_config_selected(path)   loads AppSession, populates fields
    → SettingsView._on_output_dir_selected(path)

SettingsView.session_ready(AppSession)
    → Phase1Screen._on_session_ready()
        stores self.session
        → JourneyPanel.advance(1)

JourneyPanel.node_changed(completed_node: int)
    → Phase1Screen._on_node_changed()
        → _workspace.setCurrentIndex(next_idx)
        → _header_title.setText(_NODE_TITLES[next_idx])
```

---

## Styling Reference

Constants in `styles/theme.py`:

| Constant | Value | Used for |
|---|---|---|
| `BG_LIGHT` | `#F3F3F3` | Outer surround, window background |
| `CARD_WHITE` | `#FFFFFF` | Workspace card, journey panel |
| `BORDER_GRAY` | `#E5E7EB` | Card border, section separators |
| `PRIMARY_BLUE` | `#0078D4` | Active nodes, primary buttons |
| `SUCCESS_GREEN` | `#228B22` | Complete nodes, config loaded label |
| `TEXT_PRIMARY` | `#1F2937` | Main text |
| `TEXT_MUTED` | `#6B7280` | Labels, placeholders, dim text |

Global QSS (`styles/theme.py → GLOBAL_QSS`) defines only button styles (`class="primary"` / `class="secondary"`).
