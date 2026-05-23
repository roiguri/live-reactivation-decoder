# Phase 2 UI — Live Inference Implementation Plan

Back to [Docs Index](README.md) or [Phase 2 Backend Plan](Phase2_Implementation_Plan.md).

---

## Status

This is the **active implementation contract** for the Phase 2 PyQt6 live-inference frontend.

**Progress:** Not started. Phase 2 backend is fully implemented (`LiveStreamSession`, `StreamWorker`, `OnlinePreprocessor`, `LiveInferenceEngine`, `LSLReceiver`, `PredictionLogger`) and exposed via `AppSession.build_live_stream_session(...)`.

Each step below is one commit, self-contained, and independently verifiable. Tick the box when the commit is merged.

| # | Commit subject | Status |
|---|---|---|
| 1 | `refactor(frontend): generalize MainWindow to host multiple screens` | ☐ |
| 2 | `feat(phase2-ui): add Phase 2 screen shell and Go-Live handoff` | ☐ |
| 3 | `feat(phase2-ui): add debug-mode quick-jump to Phase 2 screen` | ☐ |
| 4 | `feat(phase2-ui): add LiveProbabilityChart widget (isolated)` | ☐ |
| 5 | `feat(phase2-ui): embed LiveProbabilityChart in Phase 2 screen` | ☐ |
| 6 | `feat(phase2-ui): wire LiveStreamSession start/halt to Phase 2 header` | ☐ |
| 7 | `feat(phase2-ui): connect prediction_ready to LiveProbabilityChart` | ☐ |
| 8 | `feat(phase2-ui): add rolling latency display and buffer-health indicator` | ☐ |
| 9 | `docs(phase2-ui): mark Phase 2 UI M1 complete + soak results` | ☐ |

Design reference: React mockup at [`knowledge_base/02_reference/ui_demo/Phase2Screen.jsx`](../../knowledge_base/02_reference/ui_demo/Phase2Screen.jsx) (in sync with [github.com/roiguri/decoder_gui](https://github.com/roiguri/decoder_gui) HEAD `64f08de` as of 2026-05-23).

---

## Scope (M1)

This plan covers **Milestone 1**: a POC of live decoding — handoff from Phase 1 Node 5 ("Go Live"), discover the LSL stream, run `LiveStreamSession`, plot rolling probability curves smoothly.

**In scope for M1:**
- Phase 2 screen registered in the main window.
- "Go Live" handoff from Phase 1 Node 5 reusing the in-memory `AppSession`.
- Debug-panel quick-jump that skips Phase 1.
- Header with LIVE / HALTED indicator, target hardware label, rolling latency (p50/p95), buffer health.
- Live probability graph: rolling 10 s window, one line per decoder, chance line (0.5), threshold line (read-only from config).
- Start / Halt inference button.
- Clean lifecycle: idempotent start/stop, screen-close stops the stream.

**Out of scope for M1 (queued for M2):**
Decision history strip · trigger log (terminal-style) · decoder visibility toggles + color picker · decision settings write-back (threshold, sustained activation, conflict resolution) · exit confirmation modal · full 3-pane mockup layout · frozen probability graph + event navigation · modular graph layouts · xdf replay UI · subject-folder-aware log paths.

---

## Why pyqtgraph for the live graph

Phase 1 uses matplotlib for static charts (AUC curves, TGMs, topomaps). Phase 2 cannot.

- `StreamWorker.prediction_ready` fires at ~25 Hz at the default 40-sample batch on a 1000 Hz stream.
- Each emission carries ~4 prediction rows per task at the 100 Hz target rate → ~100 points/sec/decoder.
- A 10 s rolling window with 3 decoders is ~3000 points repainted at 30 Hz.

Matplotlib via `FigureCanvasQTAgg` does not sustain this without visible stutter even with blitting. `pyqtgraph` (built directly on QPainter, designed for streaming data) handles it comfortably. **One new dep, scoped to Phase 2.**

---

## File Structure

```
online_decoder/src/frontend/
├── screens/
│   ├── phase1_screen.py
│   └── phase2_screen.py           # NEW
├── widgets/
│   ├── ...                        # existing Phase 1 widgets
│   └── live_probability_chart.py  # NEW — pyqtgraph chart, ring-buffered
└── debug/
    ├── phase1_screen_debug.py
    └── phase2_screen_debug.py     # NEW — quick-jump entry
```

Plus:
- `pyqtgraph>=0.13` added to `requirements.txt`.
- `Phase2Screen` registered in `MainWindow`'s `QStackedWidget`.
- Phase 1 Node 5 "Go Live" wired to switch the stack.

---

## Backend API Used by the UI

The frontend imports only `AppSession` (`src/backend/session.py`). The live screen uses:

| UI action | Call | Returns |
|---|---|---|
| Build live pipeline | `session.build_live_stream_session(decoder_pipeline_path, log_path=None)` | `LiveStreamSession` |
| Start inference | `live.start()` | `None` |
| Halt inference | `live.stop()` | `None` |
| Prediction stream | `live.prediction_ready` signal | `(predictions: dict[str, np.ndarray], out_ts: np.ndarray, markers: list[tuple[float, int]])` |
| Runtime errors | `live.error_occurred` signal | `(message: str)` |
| Latency diagnostics | `live.latency_ready` signal | `dict` with `total_ms`, `pending_samples`, etc. (see `stream_worker.py`) |

**Boundary rule:** the live screen must not import `StreamWorker`, `LSLReceiver`, `OnlinePreprocessor`, or `LiveInferenceEngine` directly. Everything goes through `LiveStreamSession`.

---

## Performance Budget

| Signal | Source rate | UI consumes at | Mechanism |
|---|---|---|---|
| `prediction_ready` | ~25 Hz | every emit (cheap append) | numpy ring buffer per task |
| Chart repaint | n/a | 30 Hz | `QTimer` calling `setData` on `pg.PlotDataItem`s |
| `latency_ready` | ~25 Hz | aggregated, displayed at 5 Hz | `collections.deque(maxlen=100)` + separate `QTimer` |
| `error_occurred` | rare | on emit | `QMessageBox.critical` |

**Key decisions:**
1. Repaints are **decoupled** from the signal rate. The signal only writes to ring buffers; a 30 Hz `QTimer` drives `setData` calls.
2. Latency display updates at 5 Hz, never on every batch.
3. All cross-thread signals use `Qt.ConnectionType.QueuedConnection` so the worker thread is never blocked by UI work.
4. Numpy ring buffers (preallocated), never Python lists with `.append`.

---

## Commit-by-Commit Build Plan

Each step below is **one commit** on a feature branch off `online-decoder`. Commits land in order: every commit assumes its predecessors are merged. Each step's **Acceptance** is the contract that proves the commit can land — if you can't reproduce it, the commit isn't done.

You may open one PR per commit or bundle multiple commits per PR; the boundary that matters is the commit itself.

---

### Commit 1 — Generalize `MainWindow` to host multiple screens

**Commit subject:** `refactor(frontend): generalize MainWindow to host multiple screens`

**Changes:**
- `online_decoder/src/frontend/main_window.py` — replace `add_screen` with `show_screen(widget: QWidget)`: registers the widget in the central `QStackedWidget` if not already present, then switches to it. Update the two in-repo callers (`frontend/main.py`, `frontend/debug/main.py`) to use the new name in the same commit.

**Acceptance:**
- `python -m frontend.main` opens Phase 1 unchanged — no behavior change.
- `pytest tests/` passes (no new tests; no regressions).
- Walk through Phase 1 via `python -m frontend.debug.main` end-to-end — no visual or behavioral change.

**Out of scope:** no Phase 2 screen yet, no Go-Live wiring.

---

### Commit 2 — Phase 2 screen shell + Go-Live handoff from Node 5

**Commit subject:** `feat(phase2-ui): add Phase 2 screen shell and Go-Live handoff`

**Changes:**
- `online_decoder/src/frontend/screens/phase2_screen.py` (new) — `Phase2Screen(QWidget)` with a header bar only: status label `INFERENCE HALTED`, target hardware label. Constructor takes `(session: AppSession, decoder_pipeline_path: Path)` and stores them. No Back button yet — back-flow semantics (Node 5 vs. journey reset; what to do with a live stream) are unresolved and tracked by a TODO in the screen file; Phase 2 is one-way for now (restart the app to leave).
- `online_decoder/src/frontend/screens/phase1_screen.py` — in `_on_train_results_displayed`, re-wire the Node 5 action button so it constructs `Phase2Screen(session=self.session, decoder_pipeline_path=...)` and registers it via `MainWindow.show_screen(...)`.
- `online_decoder/src/frontend/views/train_view.py` — flip `page1_ready` so the journey-panel Node 5 button is enabled after training completes (it was hardcoded to `False` with a "Phase 2 will wire this" TODO).

**Acceptance:**
- Run Phase 1 end-to-end (real or via the debug walkthrough). After Node 5 displays results, the "Go Live" button is enabled. Clicking it shows a blank Phase 2 screen with header.
- No background threads remain on screen entry/exit (verify with a short `threading.enumerate()` log).

**Out of scope:** no Back button (deferred until back-flow semantics are decided), no chart, no Start/Halt button, no `LiveStreamSession` construction.

---

### Commit 3 — Debug-mode quick-jump to Phase 2

**Commit subject:** `feat(phase2-ui): add debug-mode quick-jump to Phase 2 screen`

**Changes:**
- `online_decoder/src/frontend/debug/phase2_screen_debug.py` (new) — modeled on `phase1_screen_debug.py`. Loads `experiment_config.yaml` + a known `decoder_pipeline.joblib` path, builds `AppSession`, and shows `Phase2Screen` via `MainWindow.show_screen(...)`.
- `online_decoder/src/frontend/debug/main.py` — accept a `--phase2` CLI flag (or equivalent entry) that opens the Phase 2 debug screen directly.

**Acceptance:**
- `python -m frontend.debug.main --phase2` opens the Phase 2 screen in < 5 s with a valid session.
- `python -m frontend.debug.main` (no flag) still opens the Phase 1 debug walkthrough — unchanged.
- The debug session is real, not a stub: `phase2.session.settings` resolves to the loaded config (`preprocessing`, `decoders`, `event_mapping` sections).

**Out of scope:** still no chart, no live data.

---

### Commit 4 — `LiveProbabilityChart` widget (isolated, no integration)

**Commit subject:** `feat(phase2-ui): add LiveProbabilityChart widget (isolated)`

**Changes:**
- `online_decoder/requirements.txt` — add `pyqtgraph>=0.13`.
- `online_decoder/src/frontend/widgets/live_probability_chart.py` (new) — `LiveProbabilityChart(QWidget)`:
  - Constructor: `(task_names: list[str], window_seconds: float = 10.0, target_sfreq: float = 100.0, threshold: float = 0.85)`.
  - Internals: numpy ring buffer per task, shared timestamp ring, one `pg.PlotDataItem` per task, chance line at 0.5, threshold line at `threshold`. Y axis fixed to `[0, 1.05]`.
  - Public API: `append_predictions(predictions: dict[str, np.ndarray], timestamps: np.ndarray)`, `clear()`.
  - Internal `QTimer` (30 Hz) drives `_refresh()` → `setData` on each curve.
- `online_decoder/scripts/test_live_chart.py` (new) — feeds synthetic predictions at 25 Hz for 30 s then exits.

**Acceptance:**
- `pip install -r online_decoder/requirements.txt` succeeds.
- `python online_decoder/scripts/test_live_chart.py` shows three smoothly scrolling curves for 30 s with threshold and chance lines visible. Window closes cleanly; no exceptions on exit.

**Out of scope:** widget not yet embedded in `Phase2Screen`.

---

### Commit 5 — Embed `LiveProbabilityChart` in Phase 2 screen

**Commit subject:** `feat(phase2-ui): embed LiveProbabilityChart in Phase 2 screen`

**Changes:**
- `online_decoder/src/frontend/screens/phase2_screen.py` — read `task_names` from `self.session.settings["decoders"]["tasks"]` and instantiate `LiveProbabilityChart(...)`. Embed below the header with a `"Decoder Probabilities"` title.

**Acceptance:**
- Open Phase 2 (via Go Live or `--phase2`). Chart renders with empty curves and the correct number of legend entries (one per configured task).
- Threshold and chance lines visible at expected y positions.
- Status stays `INFERENCE HALTED`; no live data yet.

**Out of scope:** no Start/Halt button, no LSL.

---

### Commit 6 — Wire `LiveStreamSession` start/halt to the header

**Commit subject:** `feat(phase2-ui): wire LiveStreamSession start/halt to Phase 2 header`

**Changes:**
- `online_decoder/src/frontend/screens/phase2_screen.py`:
  - In `__init__`: `self._live = session.build_live_stream_session(decoder_pipeline_path)`.
  - Add a "Start Inference" / "Halt Inference" button in the header (right-aligned for M1; no separate right panel yet).
  - Start → `self._live.start()`, flip status to LIVE, swap button label.
  - Halt → `self._live.stop()`, flip status back to HALTED, swap button label.
  - Connect `self._live.error_occurred` → `QMessageBox.critical`, then auto-halt.
  - Back button calls `self._live.stop()` **before** switching screens.

**Acceptance:**
- With replay LSL (or live `LSLProxy.exe` if available): click Start → status flips to LIVE; console shows LSL pulls. Click Halt → flips back; LSL stops.
- Rapid start/stop ×5 leaves no orphan threads (verify with `threading.enumerate()` log).
- Back during streaming halts cleanly with no exception.

**Out of scope:** chart still doesn't receive data — that's Commit 7.

---

### Commit 7 — Connect `prediction_ready` to the chart

**Commit subject:** `feat(phase2-ui): connect prediction_ready to LiveProbabilityChart`

**Changes:**
- `online_decoder/src/frontend/screens/phase2_screen.py`:
  - `self._live.prediction_ready.connect(self._on_predictions, Qt.ConnectionType.QueuedConnection)`.
  - `_on_predictions(predictions, out_ts, markers)` calls `chart.append_predictions(predictions, out_ts)`. `markers` is ignored for M1.
  - On construction, read the threshold from `session.settings["decoders"]` if a `threshold` key exists; default 0.85; pass to the chart.

**Acceptance:**
- With replay LSL: click Start → graph scrolls with real decoder probabilities for ≥ 60 s.
- On a known high-probability segment, the curve crosses the threshold line visibly.
- Halt → curves freeze (last data retained); no exception in console.

**Out of scope:** no latency display, no buffer-health pill.

---

### Commit 8 — Rolling latency display + buffer-health pill

**Commit subject:** `feat(phase2-ui): add rolling latency display and buffer-health indicator`

**Changes:**
- `online_decoder/src/frontend/screens/phase2_screen.py`:
  - `collections.deque(maxlen=100)` ingests `latency_ready` `total_ms` values.
  - A 5 Hz `QTimer` reads the deque, computes `p50` / `p95`, updates a header label like `Latency: 22 / 38 ms`.
  - Buffer-health pill in the header: green when latest `pending_samples < batch_size_samples * 2`, amber otherwise.

**Acceptance:**
- During live streaming, latency label updates ~5×/sec with non-zero p50/p95.
- 5-minute soak with replay LSL: scroll stays smooth; memory delta `< 50 MB`; p95 `total_ms ≤ 40 ms`; no unhandled exceptions; buffer-health pill stays green.

**Out of scope:** no decision settings UI, no decoder toggles, no terminal trigger log — all M2.

---

### Commit 9 — Docs update + M1 sign-off

**Commit subject:** `docs(phase2-ui): mark Phase 2 UI M1 complete + soak results`

**Changes:**
- `online_decoder/docs/Phase2_UI_Plan.md` — flip every `☐` to `☑` in the Status table; append a `## M1 Soak Results` section with the actual numbers from Commit 8's 5-minute run (p50, p95, memory delta, exit notes).
- `online_decoder/docs/README.md` — add the Phase 2 UI plan to the doc index.
- `online_decoder/CLAUDE.md` — note the `Phase2Screen` surface and the pyqtgraph dependency for live plots.

**Acceptance:**
- Verification Checklist (below) walked end-to-end and every item ticks.
- `git log --oneline online-decoder..HEAD` lists exactly the 9 commit subjects above, in order.

**Out of scope:** M2 work — see "Out of scope (M1)" at the top of this plan.

---

## Design Considerations

### AppSession as sole backend interface

Same rule as Phase 1: the only backend class the live screen imports is `AppSession`. All live access goes through `LiveStreamSession` (returned by `session.build_live_stream_session(...)`). No imports of `StreamWorker`, `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine`, `PredictionLogger`, or `DecoderPipelineArtifact` from the frontend.

If the UI needs data not exposed by `LiveStreamSession`, **stop and write a separate backend plan**. Do not reach for `live._worker` or any other private field.

### Error handling

`LiveStreamSession.error_occurred` is the single error channel. The slot shows `QMessageBox.critical(...)`, calls `live.stop()` defensively, and resets the header to HALTED. Construction errors from `session.build_live_stream_session(...)` (e.g., artifact envelope mismatch, missing file) are caught in the Phase 2 screen constructor and shown the same way; the user is bounced back to Phase 1 so they can re-train or pick a different artifact.

### Memory and ring buffers

Every per-decoder ring buffer is preallocated. No `np.append` (which reallocates). The write index is a single integer; reads use modular indexing or a contiguous slice produced once per repaint. This keeps the per-batch hot path allocation-free.

### Screen lifecycle

Phase 2 screen is constructed fresh each time the user clicks "Go Live" (with the current in-memory artifact) and destroyed on back. The `LiveStreamSession` is created in the constructor and explicitly stopped in `closeEvent` / back-button path. We do **not** keep a Phase 2 screen alive across navigations — fresh state every time is simpler and safe.

### File I/O ownership

For M1, `log_path=None` is passed to `build_live_stream_session(...)`. CSV logging via `PredictionLogger` lands in M2 once subject-folder-aware paths are wired (PRD §5 directory structure).

---

## Verification Checklist

When implementation completes, verify end-to-end:

1. `python -m frontend.main` → window opens immediately (no dialogs).
2. Full Phase 1 walkthrough completes successfully (no regression).
3. Node 5 "Go Live" → Phase 2 screen appears with chart visible, status HALTED.
4. Click "Start Inference" → status flips to LIVE, latency numbers populate, chart begins scrolling with decoder probabilities.
5. Halt → status flips back to HALTED, chart stops scrolling (curves remain visible).
6. Restart in the same screen → resumes correctly.
7. Back button mid-stream → live stream halts cleanly, returns to Phase 1 Node 5.
8. `python -m frontend.debug.main --phase2` → Phase 2 opens directly with a session ready.
9. 5-minute soak: p95 latency ≤ 40 ms, memory delta < 50 MB, no exceptions.
10. Receiver error simulated → critical dialog shown, screen auto-halts.

---

## Open Questions

1. **Threshold source.** `session.settings["decoders"]` does not yet have a `threshold` field. Either add it to the config schema (`config_models.py`) or hardcode 0.85 for M1 and surface it in M2 along with the decision-settings panel.
2. **Stream-source UI.** M1 assumes `LSLReceiver` discovers the live stream via its defaults. xdf-replay-as-LSL is M2. If the operator needs to choose between sources before clicking Start, an M2 stream-source picker lands in the header.
3. **Subject-folder-aware log path.** `PredictionLogger` exists but `log_path=None` for M1. Once Phase 2 sessions need to write per-subject CSVs, the directory layout in PRD §5 must be wired (`phase2_live/live_stream_logs.csv` under the subject folder).
4. **Back-flow semantics.** Phase 2 has no Back button in Commit 2; the TODO sits in `phase2_screen.py`. Unresolved: should Back land on Node 5's results page (preserving the journey trail) or restart the journey from Node 1? What happens to a live stream that's running — auto-halt with a confirm? Commit 6's spec assumes Back calls `live.stop()` before switching; resolve this before Commit 6 lands.
