# Phase 2 UI — Live Inference Implementation Plan

Back to [Docs Index](README.md) or [Phase 2 Backend Plan](Phase2_Implementation_Plan.md).

---

## Status

This is the **active implementation contract** for the Phase 2 PyQt6 live-inference frontend.

**Progress:** Not started. Phase 2 backend is fully implemented (`LiveStreamSession`, `StreamWorker`, `OnlinePreprocessor`, `LiveInferenceEngine`, `LSLReceiver`, `PredictionLogger`) and exposed via `AppSession.build_live_stream_session(...)`.

| Step | Description | Status |
|---|---|---|
| 1 | Phase 2 screen shell + entry from Node 5 "Go Live" | — |
| 2 | Phase 2 debug entry (quick-jump for dev iteration) | — |
| 3 | pyqtgraph dep + `LiveProbabilityChart` widget (isolated) | — |
| 4 | `Phase2Screen` layout scaffold + chart embedded | — |
| 5 | `LiveStreamSession` lifecycle wired (start/stop) | — |
| 6 | `prediction_ready` → `LiveProbabilityChart` | — |
| 7 | Latency display + performance hardening (5-min soak) | — |
| 8 | Verification checklist + docs updates | — |

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

## Serial Build Plan

Each step is **independently runnable and visibly testable** — no step requires any future step to work.

---

### Step 1 — Phase 2 screen shell + entry from Phase 1

**Create:**
- `screens/phase2_screen.py` — empty `Phase2Screen(QWidget)` with a header bar only: "Back" button (returns to Phase 1 screen), status label "INFERENCE HALTED", target hardware label. Mirrors the mockup's Stream Header structure but with no live state yet.

**Update:**
- `MainWindow` — extend to host multiple screens cleanly. Add a `show_screen(widget)` helper that swaps the central `QStackedWidget` index.
- `Phase1Screen._on_train_results_displayed` — re-wire the Node 5 action button so "Go Live" constructs `Phase2Screen(session=self.session, decoder_pipeline_path=...)`, registers it on `MainWindow`, and switches the stack.
- Back button calls `MainWindow.show_screen(phase1_screen)`.

`Phase2Screen.__init__` accepts `session` (the live `AppSession`) and `decoder_pipeline_path` and stores them. No backend wiring yet.

**Test:** run end-to-end through Phase 1 (or via the debug walkthrough) → click "Go Live" → blank Phase 2 screen with header + back. Back → returns to Node 5.

**Success metrics:** navigation in both directions works; header text correct; no resource leaks on repeated switching.

---

### Step 2 — Phase 2 debug entry

**Create:** `frontend/debug/phase2_screen_debug.py` modeled on `phase1_screen_debug.py`. It loads `experiment_config.yaml` + a known `decoder_pipeline.joblib` snapshot path, builds `AppSession`, and shows `Phase2Screen` directly.

**Update:** `frontend/debug/main.py` (or add a new entry) to accept a `--phase2` flag (or equivalent) that opens Phase 2 immediately.

**Test:** `python -m frontend.debug.main --phase2` opens Phase 2 with a session ready in < 5 s.

**Success metric:** dev iteration loop from code change to seeing the live screen is < 5 s. This step exists because every subsequent step is faster to validate with this shortcut.

---

### Step 3 — pyqtgraph dependency + `LiveProbabilityChart` widget (isolated)

**Add:** `pyqtgraph>=0.13` to `requirements.txt`. Document in `online_decoder/CLAUDE.md` that Phase 2 live plots use pyqtgraph (vs Phase 1 matplotlib).

**Create:** `widgets/live_probability_chart.py` — `LiveProbabilityChart(QWidget)`:

- **Constructor:** `(task_names: list[str], window_seconds: float = 10.0, target_sfreq: float = 100.0, threshold: float = 0.85)`.
- **Internals:**
  - One numpy ring buffer per task, preallocated to `int(window_seconds * target_sfreq) + safety_margin` rows.
  - Shared timestamp ring buffer (same size).
  - One `pg.PlotDataItem` per task with a deterministic color from a fixed palette.
  - Horizontal `pg.InfiniteLine` at 0.5 (chance, dashed gray).
  - Horizontal `pg.InfiniteLine` at `threshold` (solid blue, semi-transparent).
- **Public API:**
  - `append_predictions(predictions: dict[str, np.ndarray], timestamps: np.ndarray)` — pure data ingestion, no repaint. Writes into the ring buffer and advances the write index.
  - `clear()` — resets buffers and write index.
- **Repaint:** a `QTimer` at 30 Hz drives `_refresh()`, which calls `setData` on each curve with the current rolling window slice.
- **Axes:** X axis rolls so the right edge is "now"; Y axis fixed to [0, 1.05].

**Test in isolation:** `scripts/test_live_chart.py` — feeds the widget synthetic predictions at 25 Hz for 30 s. Watch for smooth scroll, no flicker, CPU < 5% on a typical dev machine. Delete the script after confirming.

**Success metrics:** scrolls at the 30 fps target with 3 curves; no growing memory; no event-loop stalls; threshold and chance lines render correctly.

---

### Step 4 — `Phase2Screen` layout scaffold + chart embedded

**Update:** `screens/phase2_screen.py` to add a body region under the header. For M1 the simplest possible layout: the probability graph centered with a small title "Decoder Probabilities" above it. The 3-pane structure (decoder controls L, decision settings R) is M2 work — do **not** scaffold empty panes here.

**Wire:** instantiate `LiveProbabilityChart(task_names=...)` reading task names from `self.session.settings["decoders"]["tasks"]`. Embed it in the screen.

**Test:** open Phase 2 → chart renders with empty curves and correct legend/title. No live data yet.

**Success metric:** chart paints at 30 Hz idle (verify via logging) without rising memory.

---

### Step 5 — `LiveStreamSession` lifecycle wired (start / stop) — no chart data yet

**Update:** `Phase2Screen`:
- On `__init__`, lazily call `self._live = session.build_live_stream_session(decoder_pipeline_path)`. Pipeline construction is fast — no overlay needed.
- Add a "Start Inference" / "Halt Inference" button (right-aligned in header for M1 — no separate right panel yet).
- Start → `self._live.start()`, button label flips to "Halt Inference", status dot turns green-pulsing, label flips to "LIVE INFERENCE".
- Halt → `self._live.stop()`, button flips back, dot turns gray, label flips to "INFERENCE HALTED".
- Connect `self._live.error_occurred` → `QMessageBox.critical`, then auto-halt and reset header.
- Back button calls `self._live.stop()` **before** switching screens.

**Test (with replay LSL stream from `scripts/smoke_stream_worker.py` infrastructure or `LSLProxy.exe`):**
- Start → header reads LIVE, console logs show LSL pulls.
- Halt → header reads HALTED, LSL stops.
- Rapid start/stop 5 times leaves no orphan threads.

**Success metrics:** lifecycle is clean; no zombie threads; header reflects state correctly; back-during-streaming halts cleanly.

---

### Step 6 — Connect `prediction_ready` to `LiveProbabilityChart`

**Wire:** `self._live.prediction_ready.connect(self._on_predictions, Qt.ConnectionType.QueuedConnection)`. Slot reshapes the payload and calls `chart.append_predictions(...)`.

The `prediction_ready` payload is `(predictions: dict, out_ts: np.ndarray, markers: list)`. **Ignore `markers`** for M1 — the terminal trigger log is M2 work.

**Threshold value:** read from `session.settings["decoders"]` if a `threshold` key exists; otherwise default to 0.85. Pass it to the chart on construction (Step 4) so the dashed line sits at the configured value.

**Test (end-to-end with real or replayed LSL stream):**
- Graph scrolls smoothly with decoder probabilities for ≥ 60 s.
- On a high-probability segment of replay data, the curve climbs above the threshold line.
- Latency counts from `latency_ready` (sampled manually) stay healthy.

**Success metrics:** per-decoder curves visible and labeled; scrolling smooth at 30 fps; no dropped batches; no UI freeze.

---

### Step 7 — Latency display + performance hardening

**Add to `Phase2Screen` header:**
- `collections.deque(maxlen=100)` ingests `latency_ready` `total_ms` values.
- A separate `QTimer` (5 Hz) reads the deque, computes p50 / p95, updates the latency label.
- Buffer-health indicator: green if `pending_samples` from the most recent latency dict < `batch_size_samples * 2`, amber otherwise.

**5-minute soak test** with replay LSL. Confirm:
- Scrolling stays smooth.
- Memory plateaus (no leak from ring buffer or pyqtgraph).
- p95 `total_ms` stays below 40 ms (one batch interval at 1000 Hz / 40 samples).
- Closing the screen mid-stream is clean (no segfault, no leaked thread, no exception in console).

**Success metrics:**
- p95 ≤ 40 ms.
- Memory delta < 50 MB over 5 min.
- Zero unhandled exceptions in console.
- Buffer-health indicator stays green throughout.

---

### Step 8 — Verification checklist + docs

**Update:**
- `online_decoder/docs/README.md` — add the Phase 2 UI plan to the index.
- `online_decoder/CLAUDE.md` — mention `Phase2Screen` surface and the pyqtgraph dependency for live plots.
- This plan — update the status table as steps complete.

**End-to-end verification:**
1. `python -m frontend.main` → Phase 1 walkthrough → "Go Live" → graph runs.
2. `python -m frontend.debug.main --phase2` → graph runs directly.
3. Halt → restart in the same session → works.
4. Receiver error simulated (kill replay subprocess mid-stream) → `QMessageBox.critical`, screen auto-halts.
5. Back to Phase 1 → live stream cleanly stopped, no orphan thread.

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
