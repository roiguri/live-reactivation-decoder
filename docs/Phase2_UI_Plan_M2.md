# Phase 2 UI — M2 Implementation Plan

Back to [Docs Index](README.md) | Previous: [M1 Plan](Phase2_UI_Plan_M1.md)

---

## Current State

M1 shipped the Phase 2 live-inference POC on branch `feat/phase2-live-ui` (13 commits). The operator can transition from Phase 1 training to a live Phase 2 screen, start/halt an LSL stream, and see rolling decoder probabilities on a pyqtgraph chart.

**What works:**
- `MainWindow.show_screen()` hosts multiple screens.
- Phase 1 Node 5 "Go Live" constructs `Phase2Screen(session, decoder_pipeline_path)`.
- `python -m frontend.debug.main --phase2` skips Phase 1 entirely.
- Header with LIVE / HALTED status and target label.
- Sidebar with decoder checkboxes (visibility toggles) and a Decision Settings placeholder.
- `LiveProbabilityChart`: 10 s rolling window, 30 Hz repaint, ring-buffered, chance + threshold + NOW reference lines.
- Start/Halt button with three visual states. One-shot session rebuild on restart.
- `prediction_ready` signal forwarded to the chart via `QueuedConnection`.
- 14 headless lifecycle tests covering button states, error paths, close cleanup, prediction forwarding.

**What does not work or is unverified:**
- The XDF recording used for testing (`scripts/recordings/eeg_recording_with_trigger.xdf`) contains no stimulus events (trigger codes 31-110, but no 11/12/13 for red/green/yellow). Predictions on this data are noise — we have not yet seen the online pipeline produce meaningful class-correlated predictions.
- `OnlinePreprocessor.lsl_to_si_scale=1e-6` converts LSL microvolts to SI volts. Verified by z-score analysis against the fitted scaler, but not yet validated end-to-end against the offline pipeline on data with known labels.
- `markers` from `prediction_ready` are emitted but ignored by the UI.
- `latency_ready` signal is emitted but not consumed by the UI.
- No back button, no exit confirmation, no decision settings, no logging.

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| LSL unit scaling removed | Open — lab validation needed | The `lsl_to_si_scale=1e-6` factor was removed: VHDR replay via `PlayerLSL` streams in SI volts (MNE converts on load), so the scaling was incorrect for replay. Whether NeurOne's LSL proxy outputs µV or V is unverified — test in the lab and re-add scaling if needed. |
| No pipeline validation | Open | Online predictions have never been verified against offline on the same labeled data. |
| XDF test recording has no stimulus events | Open | Current replay file is from a non-task block. Need to replay actual training data (.vhdr) to validate. |

---

## Goals

| # | Goal | Status |
|---|------|--------|
| 1 | Pipeline Validation | Not started |
| 2 | Event Markers on Probability Graph | Not started |
| 3 | LSL Stream Picker | Not started |
| 4 | Trigger Log | Not started |
| 5 | Decision History Strip | Not started |
| 6 | Latency Display + Buffer Health | Not started |
| 7 | Subject-Folder Log Paths | Not started |
| 8 | Decision Settings UI | Not started |
| 9 | Frozen Event Graph | Not started |
| 10 | Back Button + Exit Flow | Not started |
| 11 | Past Events Dropdown | Not started |
| 12 | Unified Source Picker (XDF Replay) | Not started |
| 13 | Modular Graph Layouts | Not started |
| 14 | Per-Decoder Colour Picker | Not started |

---

## Goal 1 — Pipeline Validation

Prove that the online pipeline produces meaningful, class-correlated predictions on data with known labels. This is the foundation — all subsequent UI work assumes the predictions are trustworthy.

### Step 1: VHDR-to-LSL replay script

Create `scripts/replay_vhdr_to_lsl.py` that loads a BrainVision `.vhdr` file via MNE and streams it as LSL using `mne_lsl.player.PlayerLSL`. Must preserve the trigger channel so markers flow through the existing `LSLReceiver.split_eeg_and_markers` path.

- [ ] Script loads `.vhdr` and streams as LSL with correct channel count (64 EEG + 1 trigger)
- [ ] Trigger codes (11=red, 12=green, 13=yellow) are preserved in the stream
- [ ] `--stream-name`, `--repeat` flags match `replay_xdf_to_lsl.py` interface
- [ ] Verified: `LSLReceiver` pulls data and `StreamWorker` reports decoded markers in logs

### Step 2: End-to-end validation on training data

Run the full app from Phase 1 (offline evaluation, timepoint selection, training) through to Phase 2 live inference — all on the same subject data.

- [ ] Run Phase 1 offline pipeline on `subject_102_quarter` data: preprocess, evaluate, select timepoint, train
- [ ] Export `decoder_pipeline.joblib`
- [ ] Transition to Phase 2 via Go Live
- [ ] Start VHDR replay of the same subject data
- [ ] Observe: do decoder probabilities correlate with stimulus events? (red decoder rises after red stimuli, yellow after yellow, both low after green)
- [ ] Document findings (screenshots or logged predictions vs markers)

### Step 3: Quantitative offline-vs-online comparison

Compare online predictions against offline predictions on the same data segments.

- [ ] Script loads saved epochs, extracts features at training timepoint, runs `model.predict_proba()` — these are the offline ground-truth predictions
- [ ] Script replays the same raw data through `OnlinePreprocessor` + `LiveInferenceEngine` (headless, no UI)
- [ ] Compare: correlation between offline and online predictions per task
- [ ] Document expected gap (causal vs zero-phase filters) and whether it affects classification
- [ ] Decide: is the gap acceptable, or does the online preprocessor need adjustment?

### Step 4: Preprocessor numerical comparison (optional)

If Step 3 shows a significant gap, drill into the preprocessor.

- [ ] Feed identical raw segment through `OfflinePreprocessor` and `OnlinePreprocessor`
- [ ] Compare output at matching timepoints: correlation, scale, spatial pattern
- [ ] Identify which stage (filter, ICA, decimation) contributes most to the difference

---

## Goal 2 — Event Markers on Probability Graph

Display trigger events as vertical lines on the live probability chart. The `markers` tuple is already emitted by `prediction_ready` but currently ignored in `Phase2Screen._on_predictions`.

- [ ] `LiveProbabilityChart` gains an `append_markers(markers)` method
- [ ] Markers render as vertical lines at the correct timestamp, color-coded by trigger code
- [ ] Markers scroll with the data (same coordinate system as the probability curves)
- [ ] Marker legend or tooltip shows the trigger code / event name
- [ ] Verified: with VHDR replay, vertical lines appear at stimulus onset times

---

## Goal 3 — LSL Stream Picker

Replace the hardcoded `NeuroneStream` default with a discoverable stream selector in the header.

- [ ] Header gains a stream-name dropdown or picker widget
- [ ] On screen open (or on a "Discover" action), `LSLReceiver.discover_streams()` populates available streams
- [ ] Operator selects a stream before clicking Start
- [ ] `LSLReceiver.set_stream(name)` is called before `live.start()`
- [ ] Verified: with multiple LSL sources active, operator can pick the correct one

---

## Goal 4 — Trigger Log

Terminal-style scrolling log below the chart showing trigger events and system messages.

- [ ] New widget: `TriggerLog` (text-based, append-only, auto-scroll)
- [ ] Receives markers from `prediction_ready` and formats them as timestamped log lines
- [ ] Also logs lifecycle events: stream started, stream halted, errors
- [ ] Placed in the centre panel below the chart card
- [ ] Verified: during replay, trigger events appear in real time with correct codes

---

## Goal 5 — Decision History Strip

Row of recent decoder decisions displayed above the chart in the centre panel.

- [ ] New widget: `DecisionHistoryStrip`
- [ ] Consumes predictions and applies threshold logic to determine "decisions"
- [ ] Displays recent decisions as compact indicators (color-coded by decoder, with timestamp)
- [ ] Scrolls or truncates to a fixed visible count
- [ ] Verified: during replay, decisions appear when predictions cross the threshold

---

## Goal 6 — Latency Display + Buffer Health

Rolling latency readout and buffer-health indicator in the header. Deferred from M1 Commit 8.

- [ ] `Phase2Header` gains latency label (`Latency: p50 / p95 ms`) and buffer-health pill
- [ ] `Phase2Screen` subscribes to `latency_ready`, buffers in `deque(maxlen=100)`
- [ ] 5 Hz `QTimer` computes rolling percentiles and buffer-health state
- [ ] Green pill when `pending_samples < batch_size * 2`, amber otherwise
- [ ] Diagnostics clear on Halt
- [ ] Verified: during replay, latency numbers update ~5x/sec with non-zero values

---

## Goal 7 — Subject-Folder Log Paths

Wire `PredictionLogger` with subject-folder-aware output paths.

- [ ] Determine directory layout for Phase 2 logs (per PRD directory structure)
- [ ] `Phase2Screen` or `AppSession` resolves the log path from the current subject/session
- [ ] `build_live_stream_session(log_path=...)` passes a real path instead of `None`
- [ ] CSV logging starts on Start, file closes on Halt
- [ ] Verified: after a live session, a CSV exists at the expected path with correct columns

---

## Goal 8 — Decision Settings UI

Wire the sidebar's Decision Settings section with functional controls.

- [ ] Threshold slider — connected to `LiveProbabilityChart`'s threshold line
- [ ] Sustained-activation input — number of consecutive above-threshold predictions required
- [ ] Conflict-resolution select — behavior when multiple decoders cross threshold simultaneously
- [ ] Settings are read from config if available, otherwise use defaults
- [ ] Verified: adjusting the threshold slider moves the red dashed line on the chart in real time

---

## Goal 9 — Frozen Event Graph

On marker detection, display a separate chart showing the prediction window around that event.

- [ ] New widget: `FrozenEventChart` — fixed-window chart (e.g. -0.2s to +1.0s around the event)
- [ ] Triggered when a marker is detected in the prediction stream
- [ ] Shows all decoder probabilities for that window with the event onset marked
- [ ] Placed in the centre panel (exact position TBD — depends on Goal 13)
- [ ] Verified: during replay, an event triggers a frozen snapshot that stays visible

---

## Goal 10 — Back Button + Exit Flow

Resolve back-flow semantics and implement exit confirmation.

- [ ] Decide: Back lands on Node 5 results (preserve journey) or resets to Node 1
- [ ] Decide: running stream on Back — auto-halt with confirmation dialog, or silent halt
- [ ] Implement Back button in the header or sidebar footer
- [ ] Exit confirmation modal when a stream is running
- [ ] `closeEvent` shows the same confirmation if stream is active
- [ ] Verified: Back during LIVE shows confirmation; after confirm, returns to correct Phase 1 state

---

## Goal 11 — Past Events Dropdown

Browse earlier event snapshots in the Frozen Event Graph.

- [ ] Dropdown or navigation control above/beside the `FrozenEventChart`
- [ ] Stores a rolling history of event snapshots (capped to a reasonable count)
- [ ] Selecting a past event replaces the frozen chart content
- [ ] Current (most recent) event is the default view
- [ ] Verified: after multiple events, operator can navigate back to earlier ones

---

## Goal 12 — Unified Source Picker (XDF Replay)

Extend Goal 3's LSL stream picker to support file-based replay as an alternative source.

- [ ] Source picker gains a "Replay File" option alongside discovered LSL streams
- [ ] File browser for selecting `.xdf` or `.vhdr` files
- [ ] Replay runs in-process (or as a managed subprocess) and feeds data to the existing pipeline
- [ ] Operator can stop/restart replay like a live stream
- [ ] Verified: operator can switch between live NeurOne and file replay without restarting the app

---

## Goal 13 — Modular Graph Layouts

Allow the operator to show, hide, and resize chart panels.

- [ ] Probability chart and frozen event graph are in resizable/collapsible containers
- [ ] Operator can hide either panel to give the other full space
- [ ] Layout state persists within the session (not across restarts for now)
- [ ] Verified: hiding the frozen graph gives the probability chart full height; restoring it splits the space

---

## Goal 14 — Per-Decoder Colour Picker

Make sidebar decoder swatches interactive.

- [ ] Clicking a swatch opens a color picker dialog
- [ ] Selected color updates the swatch, the chart curve, and the decision history indicator
- [ ] Default colors remain as assigned by `chart_line_color()`
- [ ] Verified: changing a decoder's color is immediately reflected on the chart
