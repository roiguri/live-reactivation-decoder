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
- ~~`markers` from `prediction_ready` are emitted but ignored by the UI.~~ Now rendered as cue lines on the chart (Goal 2) — pending manual replay verification.
- `latency_ready` signal is emitted but not consumed by the UI.
- No back button, no exit confirmation, no decision settings, no logging.

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| LSL unit scaling removed | Open — lab validation needed | The `lsl_to_si_scale=1e-6` factor was removed: VHDR replay via `PlayerLSL` streams in SI volts (MNE converts on load), so the scaling was incorrect for replay. Whether NeurOne's LSL proxy outputs µV or V is unverified — test in the lab and re-add scaling if needed. |
| No pipeline validation | Open | Online predictions have never been verified against offline on the same labeled data. |
| XDF test recording has no stimulus events | Open | Current replay file is from a non-task block. Need to replay actual training data (.vhdr) to validate. |
| Proxy auto-launch during discovery | Open — [#3] | `discover_streams` / `start_stream_source` always launch `LSLProxy.exe`; redundant and risks a `NeuroneStream` name collision when an external stream is already publishing. |
| Stream-source locking model | Open — [#1] | `AppSession`'s coarse lock around the stream source held across the blocking proxy launch — revisit. |

---

## Update — stream-selection branch (2026-05-29)

Work from `feat/phase2-stream-selection` lands several M2 items early and changes Goal 12's direction.

**Landed:**
- **StreamSource refactor.** Proxy management moved out of `LSLReceiver` (now a pure *consumer*) into `LslProxySource` (`src/backend/online_phase/stream_source.py`). `AppSession` owns the active source via `start_stream_source()` / `stop_stream_source()` and exposes `discover_streams()`. `build_live_stream_session(..., stream_name=...)` no longer hardcodes the stream.
- **Goal 3 (LSL Stream Picker) — done** (header-launched dialog; see that section).
- **Goal 1 Step 1 (replay script) — done**, with deviations (see that section).

**Direction change — Goal 12 (in-app file replay) descoped.** An in-app `ReplaySource` (subprocess) + "Recording folder" dialog option was prototyped and reverted. **Decision: the app only consumes existing LSL streams; recording replay is out-of-process** via `scripts/replay_vhdr_to_lsl.py`, which publishes a NeurOne-like (`type=EEG`) stream the app discovers like hardware. This avoids in-app subprocess lifecycle and frozen-build (PyInstaller) entry-point complexity.

---

## Goals

| # | Goal | Status |
|---|------|--------|
| 1 | Pipeline Validation | Step 1 (replay script) done; validation pending |
| 2 | Event Markers on Probability Graph | In progress — rendering done; manual replay verification pending |
| 3 | LSL Stream Picker | ✅ Done (stream-selection branch) |
| 4 | Trigger Log | Not started |
| 5 | Decision History Strip | Not started |
| 6 | Latency Display + Buffer Health | Not started |
| 7 | Subject-Folder Log Paths | Not started |
| 8 | Decision Settings UI | Not started |
| 9 | Frozen Event Graph | Not started |
| 10 | Back Button + Exit Flow | Not started |
| 11 | Past Events Dropdown | Not started |
| 12 | Unified Source Picker (XDF Replay) | Revised — replay via external script; in-app picker descoped |
| 13 | Modular Graph Layouts | Not started |
| 14 | Per-Decoder Colour Picker | Not started |
| 15 | Probability Graph Window Length | Not started |
| 16 | Probability Graph History View | Not started — needs UX discussion |

---

## Goal 1 — Pipeline Validation

Prove that the online pipeline produces meaningful, class-correlated predictions on data with known labels. This is the foundation — all subsequent UI work assumes the predictions are trustworthy.

### Step 1: VHDR-to-LSL replay script — ✅ Done

`scripts/replay_vhdr_to_lsl.py` loads a BrainVision recording **directory** via MNE and publishes it as a live LSL stream. Two deviations from the original plan:

- **Raw `pylsl` outlet, not `mne_lsl.PlayerLSL`.** PlayerLSL derives the LSL `type` from MNE's channel kind (lowercase `eeg`, or `""` for mixed EEG+stim channels), which the app's `type == "EEG"` discovery filter skips. A manual outlet sets `type="EEG"` (name `NeuroneStream`, 65 ch @ the recording's rate) so the app discovers it exactly like NeurOne hardware.
- **Flags:** `--stream-name`, `--stream-type`, `--chunk-ms`, `--no-repeat` (loops by default); the `replay_xdf_to_lsl.py` interface was not mirrored.

- [x] Loads recording dir and streams with correct channel count (64 EEG + 1 trigger)
- [x] Trigger codes preserved (packed `code << 8`) and decoded by `LSLReceiver`
- [x] Stream advertises `type=EEG` so `AppSession.discover_streams()` lists it
- [x] Verified: consumer pulls data @ 1000 Hz and decodes markers

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

- [x] `LiveProbabilityChart` gains an `append_markers(markers)` method (data-only; lines materialise in `_refresh`)
- [x] Each trigger code is resolved against the configured event map; codes **not** in the map are filtered out
- [x] Markers render as bold black vertical lines (width 2) at the correct timestamp (colour is decoupled from the event name)
- [x] Markers scroll with the data (rebased onto the same `x = ts - latest_ts` axis as the curves; pruned when they leave the window)
- [x] Marker label shows the configured event name via the line's `InfLineLabel` — bold black text in a solid white, black-bordered box for contrast
- [ ] Verified: with VHDR replay, vertical lines appear at stimulus onset times (manual UI verification on Windows)

**Implementation notes:**
- `Phase2Screen` builds the `{code: name}` map by inverting `settings["event_mapping"]` (which `SettingsManager.get_settings()` exposes as `{name: id}`, flattened from `markers_mapping.events`) — tolerant of a missing map — and passes it to the chart constructor; `_on_predictions` now calls `append_markers(markers)` after `append_predictions`.
- The receiver emits **every** non-zero trigger edge, not just configured events. `append_markers` drops any code absent from the event map, so only declared events are drawn.
- Colour is intentionally **not** derived from the event name (names are arbitrary config labels, not colour hints). All markers are black; the label is the only per-event distinction.
- **Label placement.** A headroom band above the curves (`_Y_RANGE` top `1.1`, Y ticks capped at `1.0` so the band reads as blank margin) lets labels sit clear of the data. Labels are pinned to one fixed side via identical `InfLineLabel` `anchors` — this disables pyqtgraph's default view-edge flip, which otherwise makes two markers straddling the view centre point their labels inward and collide. Trade-off: a label on a marker near the right (NOW) edge can clip; flip the anchor side or centre it if that reads poorly in the lab.
- All marker scene mutation happens on the 30 Hz repaint tick (`_refresh_markers`), keeping `append_markers` off the scene like the prediction hot path. `_MAX_MARKERS` (128) backstops unbounded growth if the stream stalls.

---

## Goal 3 — LSL Stream Picker — ✅ Done (stream-selection branch)

Implemented as a **header-launched dialog** rather than an inline header dropdown.

- [x] Header "Choose target…" button opens `TargetSelectionDialog`
- [x] Manual **Refresh** runs `AppSession.discover_streams()` on a `StreamDiscoveryWorker` thread and populates a combo
- [x] Operator selects a stream before Start; Start is guarded against no target
- [x] Chosen `stream_name` is passed to `build_live_stream_session(stream_name=...)` (no `set_stream` call)
- [ ] Caveat: discovery currently always launches `LSLProxy.exe` — see [#3]

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

## Goal 12 — Unified Source Picker (XDF Replay) — Revised: descoped

Original intent was an in-app "Replay File" source alongside discovered streams. This was prototyped (`ReplaySource` subprocess + a "Recording folder" dialog option) and **reverted**.

**Decision: the app only consumes existing LSL streams; recording replay is out-of-process** via `scripts/replay_vhdr_to_lsl.py`, which publishes a NeurOne-like (`type=EEG`) stream the app discovers like hardware. This keeps the app a pure consumer and avoids in-app subprocess lifecycle + frozen-build (PyInstaller) entry-point complexity. If an in-app picker is ever wanted, the prototype is recoverable from git history.

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

---

## Goal 15 — Probability Graph Window Length

Today `LiveProbabilityChart` shows a **fixed 10 s** rolling window: the x-axis is locked to `[-window_seconds, NOW_GAP]`, timestamps are rebased so the latest sample sits at 0, and the curve scrolls right-to-left. The ring buffer's capacity equals the window. Give the operator control over **how much recent history is visible**, while still following the latest sample (no scrolling back — that's Goal 16).

- [ ] A window-length control (e.g. segmented buttons / dropdown — 5 / 10 / 30 / 60 s) in the chart panel header
- [ ] Selecting a length updates the x-range and the ring-buffer capacity together
- [ ] Chart keeps following the latest sample; incoming data is never dropped during the change
- [ ] Y-axis stays fixed at `[0, 1]` (probabilities are comparable across decoders)
- [ ] Decide whether the choice persists within the session (across restarts: no, for now)
- [ ] Verified: switching window length immediately re-scales the live chart

---

## Goal 16 — Probability Graph History View

**Needs UX discussion before spec.** Goal 15 only changes how much of the *live tail* is visible. This goal is about **reviewing earlier predictions** — looking back at history beyond the current window while (or after) the stream runs. "In some way" is deliberately open: the mechanism is undecided.

Approaches to weigh:
- **Scroll/pan the live chart.** A scrollbar or drag to move back in time, with a **live/paused** mode and a "Jump to now" control to resume following. Needs a backing buffer larger than the visible window (a capped scrollback history) instead of the current window-sized ring buffer.
- **Separate review/history view.** A distinct chart or mode for browsing past windows, leaving the live chart untouched.
- **Lean on existing goals.** Goals 9 (Frozen Event Graph) and 11 (Past Events Dropdown) already snapshot and browse *per-event* windows; history review may be partly covered there, or should be unified with them rather than built twice.

Open questions:
- Continuous scrollback over the whole session, or only around detected events (overlap with Goals 9/11)?
- How much history to retain (memory cap / max scrollback duration)?
- Does entering history pause the live follow, and how does the operator return to live?
- Persist position/zoom within the session?

- [ ] Decide the mechanism (scroll/pan vs separate view vs fold into Goals 9/11) and the retention cap
- [ ] Implement the chosen history buffer + live/paused state (if applicable)
- [ ] Verified: operator can review earlier predictions and return to live without losing incoming data
