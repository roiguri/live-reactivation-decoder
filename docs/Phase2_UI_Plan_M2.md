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
- **LSL Stream Picker** (Goal 3) — header-launched `TargetSelectionDialog` with manual refresh.
- **Event markers on the probability graph** (Goal 2) — trigger codes resolved against the configured event map and rendered as labelled vertical lines.
- 14 headless lifecycle tests covering button states, error paths, close cleanup, prediction forwarding.

**What does not work or is unverified:**
- **Pipeline fidelity is the open blocker.** Online predictions have never been validated against the offline pipeline on labeled data. A known ~60 ms pipeline group delay plus a causal-vs-zero-phase filter mismatch (see `online-inference-fidelity-bug`, diagnosed 2026-05-30) makes live inference *look* dead even when it is recoverable. **Goal 18 (Group Delay Deep Dive) characterizes this before any further UI work assumes the predictions are trustworthy.**
- The XDF recording used for early testing (`scripts/recordings/eeg_recording_with_trigger.xdf`) contains no stimulus events (trigger codes 31-110, but no 11/12/13 for red/green/yellow). Predictions on that data are noise. Validation must replay actual training data (`.vhdr`) via `scripts/replay_vhdr_to_lsl.py`.
- `latency_ready` is emitted by `StreamWorker` but consumed by nobody.
- `PredictionLogger` exists but is never wired: `build_live_stream_session(log_path=None)`.
- No back button, no exit confirmation, no decision settings.

---

## Priority Sequence (revised 2026-06-05)

M2 is re-sequenced around one principle: **nothing downstream is meaningful until live predictions are proven trustworthy.** The first four items below build the diagnostic scaffolding, diagnose the fidelity bug, capture the evidence, and close the validation gate. The frozen event graph is the first "real" feature that rides on validated predictions. After that, the backlog resumes as originally planned.

| Seq | Goal | What it unlocks |
|-----|------|-----------------|
| **1** | Goal 17 — Debug Profiles | Repeatable runs with different settings + debug data; prerequisite for every diagnostic below |
| **2** | Goal 18 — Group Delay Deep Dive | Characterizes the ~60 ms delay + filter mismatch; produces a delay constant the next steps consume |
| **3** | Goal 7 — System Logging (prediction + latency timepoints) | Persists the evidence validation needs |
| **4** | Goal 1 — Pipeline Validation (Steps 2-4) | Closes the trust gate using profiles (1), the delay model (2), and logs (3) |
| **5** | Goal 9 — Frozen Event Graph | First UI feature on validated predictions |
| backlog | Goals 4, 5, 6, 8, 10, 11, 13, 14, 15, 16 | Resume original plan |

> **Phase 1 cross-cutting note — per-decoder training timepoint selection.** Each decoder is trained at a **single** timepoint, and the backend **already supports a *different* timepoint per decoder**: `ModelTrainer.run_training()` accepts `dict[str, float]` (task → one timepoint), the trained feature stays `n_channels` wide, and `DecoderPipelineMetadata.decoding_timepoints` carries the per-task values. This is *not* multi-timepoint/multi-slice training — there is no `n_channels × k` feature and **no online impact**: `LiveInferenceEngine.predict()` runs every decoder on the same sliding `feature_width` window, and each fires when its pattern slides through.
>
> The **only** missing piece is the **Phase 1 (offline) UI** to let the operator *choose/override* each decoder's timepoint. **✅ Now done — see [Goal 19](#goal-19--per-decoder-timepoint-selection-phase-1-offline-ui----done).** The operator picks/confirms each decoder's timepoint in the Evaluation roster and the selected dict is passed to `run_training(...)`. (The old auto-derivation `_derive_per_task_timepoints` and the legacy single-float path were removed in the process.)

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| Online/offline pipeline fidelity (group delay + filter mismatch) | Open — Goal 18 | ~60 ms group delay + causal-vs-zero-phase mismatch makes live inference look dead. Recoverable with delay compensation. See `online-inference-fidelity-bug`. |
| LSL unit scaling removed | Open — lab validation needed | The `lsl_to_si_scale=1e-6` factor was removed: VHDR replay via `PlayerLSL` streams in SI volts (MNE converts on load), so the scaling was incorrect for replay. Whether NeurOne's LSL proxy outputs µV or V is unverified — test in the lab and re-add scaling if needed. |
| XDF test recording has no stimulus events | Open | Current replay file is from a non-task block. Validation replays actual training data (.vhdr). |
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

| # | Goal | Status | Seq |
|---|------|--------|-----|
| 17 | Debug Profiles | ✅ Done (1 profile seeded; validation profile pending) | **1** |
| 18 | Group Delay Deep Dive | Not started | **2** |
| 7 | System Logging (prediction + latency timepoints) | Not started | **3** |
| 1 | Pipeline Validation | Step 1 (replay script) done; Steps 2-4 pending | **4** |
| 9 | Frozen Event Graph | Not started | **5** |
| 2 | Event Markers on Probability Graph | ✅ Done | — |
| 3 | LSL Stream Picker | ✅ Done (stream-selection branch) | — |
| 4 | Trigger Log | Not started | backlog |
| 5 | Decision History Strip | Not started | backlog |
| 6 | Latency Display + Buffer Health | Not started | backlog |
| 8 | Decision Settings UI | Not started | backlog |
| 10 | Back Button + Exit Flow | Not started | backlog |
| 11 | Past Events Dropdown | Not started | backlog |
| 12 | Unified Source Picker (XDF Replay) | Revised — replay via external script; in-app picker descoped | — |
| 13 | Modular Graph Layouts | Not started | backlog |
| 14 | Per-Decoder Colour Picker | Not started | backlog |
| 15 | Probability Graph Window Length | Not started | backlog |
| 16 | Probability Graph History View | Not started — needs UX discussion | backlog |
| 19 | Per-Decoder Timepoint Selection (Phase 1 offline UI) | ✅ Done (merged, PR #7) | Phase 1 |

---

# Critical Path

## Goal 17 — Debug Profiles (Seq 1) — ✅ Done

`python -m frontend.debug.main --phase2` was hardcoded to one config and one pipeline. Diagnosing the fidelity bug and validating the pipeline both need to swap *settings*, *trained artifact*, and *replay recording* together, repeatably. A **debug profile** bundles those so a single flag selects a known scenario.

Implemented as **self-describing directories** (not a central registry): each `debug_snapshots/<name>/` carries a minimal `manifest.yaml` (`name`, copied-in `config`, `raw_data_dir` path-only) plus the snapshots, `models/`, and `epochs/` it produces. Discovery lists subdirs with a manifest. Self-contained in `src/frontend/debug/`; production `frontend.main` stays byte-for-byte unaffected. Full design + usage: [docs/features/debug_profiles.md](features/debug_profiles.md) and `src/frontend/debug/README.md`.

- [x] `DebugProfile` (`src/frontend/debug/profiles.py`) — resolved paths from a 3-field manifest. *Trimmed from the original sketch: no central registry, no `stream_name`/`notes`/explicit `pipeline` fields (snapshot/pipeline/epochs paths are conventions); add back if needed.*
- [x] `frontend.debug.main` gains `--profile <name>`, `--list-profiles`, and `--config` / `--data` overrides (pipeline path is a convention, so no `--pipeline`)
- [x] `build_debug_phase2(profile)` resolves config + pipeline from the profile; **also** `DebugPhase1Screen(profile)` (the Phase 1 walkthrough is now profile-driven too — beyond the original Phase-2-only scope)
- [x] Each profile records the recording (`raw_data_dir`) the operator replays via `scripts/replay_vhdr_to_lsl.py` (app stays a pure consumer; replay out-of-process per Goal 12)
- [ ] **Pending:** a second labeled-training-data profile for validation (Goal 1). One profile (`default`) is seeded so far.
- [x] `scripts/demo_seed_debug_snapshots.py` is profile-aware (bootstrap copies the config in + writes the manifest; re-seed reuses it) and writes the snapshots inside the profile dir
- [x] Verified: `--list-profiles` prints the registry; the `default` artifact loads with all 3 decoders + online_state + metadata; 14 `test_debug_profiles` tests pass. *GUI launch against the profile left to the operator.*

---

## Goal 18 — Group Delay Deep Dive (Seq 2)

**The fidelity gate.** Before logging or validation, characterize *why* live inference looks dead. The known diagnosis (`online-inference-fidelity-bug`, 2026-05-30): a ~60 ms pipeline group delay plus a causal (online, `lfilter`-style) vs zero-phase (offline, `filtfilt`) filter mismatch. Offline features are computed from a centred, non-causal window; online features come from a causal sliding window, so the same neural event appears **later and differently shaped** online. The goal is a written characterization and a concrete delay/compensation model the downstream steps consume — not a UI change.

Existing diagnostic assets to build on: `scripts/preproc_parity_check.py`, `scripts/offline_inference_check.py`, `scripts/full_recording_live_inference_check.py`, `scripts/inspect_decoder_internals.py`, and `knowledge_base/01_timeline/03_online_stage_design/Decoder Pipeline Investigation.md`.

**Characterize:**
- [ ] Drive a known signal (impulse / labeled epoch) through `OnlinePreprocessor` and `OfflinePreprocessor` and measure the time lag of matching features (cross-correlation peak)
- [ ] Decompose the delay per stage: causal highpass, notch, lowpass, decimation/`final_resample` — quantify each contribution and the total
- [ ] Quantify shape distortion (not just lag): correlation and scale of the causal vs zero-phase feature at the aligned timepoint
- [ ] Confirm whether the offline decoders' training timepoints (`decoding_timepoints`) sit where the causal pipeline can actually reach them in real time

**Decide compensation strategy (pick and document):**
- [ ] (a) **Alignment-only** — treat the delay as a constant `group_delay_ms`; shift marker/prediction timestamps by it when logging/comparing (cheapest; no signal change)
- [ ] (b) **Filter-matched** — redesign online filters (e.g. shorter causal filters, or a min-phase design) to reduce the delay/distortion, trading fidelity for latency
- [ ] (c) **Accept + document** — if classification survives the delay, record the gap as a known characteristic

**Output (consumed by Goals 7 and 1):**
- [ ] A written characterization committed under `knowledge_base/` (extend the Decoder Pipeline Investigation note)
- [ ] A single `group_delay_ms` constant (and its provenance) that System Logging stamps and Pipeline Validation uses for offline/online alignment
- [ ] A go/no-go call on whether the online preprocessor needs adjustment before validation, or whether alignment compensation is sufficient

---

## Goal 7 — System Logging (prediction + latency timepoints) (Seq 3)

Wire persistent logging so the validation step has hard evidence and live sessions are auditable. `PredictionLogger` already writes `timestamp, marker_code, <per-task probs>` to CSV but is **never instantiated** (`build_live_stream_session(log_path=None)`). `StreamWorker.latency_ready` emits a full per-batch timing breakdown (pull / accumulation / preprocessing / inference / emit ms, plus `pending_samples`) that **no one consumes**. This goal connects both, and stamps the `group_delay_ms` from Goal 18 so logged prediction timepoints are interpretable against marker times.

**Prediction log (extends Goal 7's original "Subject-Folder Log Paths"):**
- [ ] Determine the per-subject/session directory layout for Phase 2 logs (per PRD directory structure)
- [ ] `Phase2Screen` / `AppSession` resolves the log path from the current subject/session
- [ ] `build_live_stream_session(log_path=...)` passes a real path instead of `None`
- [ ] CSV logging starts on Start, file closes on Halt
- [ ] Each prediction row carries the wall-clock prediction timepoint and the marker-aligned timepoint (apply `group_delay_ms` from Goal 18)

**Latency / timing log:**
- [ ] A second sink consumes `latency_ready` and persists per-batch timing (or rolling p50/p95) + backlog (`pending_samples`)
- [ ] Throttle/aggregate so the ~25 Hz `latency_ready` stream does not bloat the log (rolling summary, not every batch)

- [ ] Verified: after a live replay session, a prediction CSV and a timing log exist at the expected paths with correct columns, and prediction timepoints line up with replayed markers once the group delay is applied

> Goals 4 (Trigger Log) and 6 (Latency Display + Buffer Health) are the **UI views** of this same data and remain in the backlog; this goal is the persistence/backend half they will later surface.

---

## Goal 1 — Pipeline Validation (Seq 4)

Prove that the online pipeline produces meaningful, class-correlated predictions on data with known labels. With Goal 18's delay model and Goal 7's logs in hand, this becomes a measurement rather than a guess.

### Step 1: VHDR-to-LSL replay script — ✅ Done

`scripts/replay_vhdr_to_lsl.py` loads a BrainVision recording **directory** via MNE and publishes it as a live LSL stream. Two deviations from the original plan:

- **Raw `pylsl` outlet, not `mne_lsl.PlayerLSL`.** PlayerLSL derives the LSL `type` from MNE's channel kind (lowercase `eeg`, or `""` for mixed EEG+stim channels), which the app's `type == "EEG"` discovery filter skips. A manual outlet sets `type="EEG"` (name `NeuroneStream`, 65 ch @ the recording's rate) so the app discovers it exactly like NeurOne hardware.
- **Flags:** `--stream-name`, `--stream-type`, `--chunk-ms`, `--no-repeat` (loops by default); the `replay_xdf_to_lsl.py` interface was not mirrored.

- [x] Loads recording dir and streams with correct channel count (64 EEG + 1 trigger)
- [x] Trigger codes preserved (packed `code << 8`) and decoded by `LSLReceiver`
- [x] Stream advertises `type=EEG` so `AppSession.discover_streams()` lists it
- [x] Verified: consumer pulls data @ 1000 Hz and decodes markers

### Step 2: End-to-end validation on training data

Run the full app from Phase 1 (offline evaluation, timepoint selection, training) through to Phase 2 live inference — all on the same subject data, driven by a Goal 17 debug profile.

- [ ] Run Phase 1 offline pipeline on `subject_102_quarter` data: preprocess, evaluate, select timepoint, train
- [ ] Export `decoder_pipeline.joblib`
- [ ] Transition to Phase 2 via Go Live (or the matching debug profile)
- [ ] Start VHDR replay of the same subject data
- [ ] Observe: do decoder probabilities correlate with stimulus events, **after applying `group_delay_ms`**? (red decoder rises after red stimuli, yellow after yellow, both low after green)
- [ ] Document findings against the logged predictions vs markers (Goal 7 output)

### Step 3: Quantitative offline-vs-online comparison

Compare online predictions against offline predictions on the same data segments, aligned by the Goal 18 delay model.

- [ ] Script loads saved epochs, extracts features at each decoder's training timepoint, runs `model.predict_proba()` — these are the offline ground-truth predictions
- [ ] Script replays the same raw data through `OnlinePreprocessor` + `LiveInferenceEngine` (headless, no UI)
- [ ] Compare: correlation between offline and online predictions per task, after shifting by `group_delay_ms`
- [ ] Document the residual gap (causal vs zero-phase filters) and whether it affects classification
- [ ] Decide: is the gap acceptable, or does the online preprocessor need adjustment (feeds back into Goal 18 strategy (b))?

### Step 4: Preprocessor numerical comparison (optional)

If Step 3 shows a significant residual gap beyond the modeled delay, drill into the preprocessor.

- [ ] Feed identical raw segment through `OfflinePreprocessor` and `OnlinePreprocessor`
- [ ] Compare output at matching timepoints: correlation, scale, spatial pattern
- [ ] Identify which stage (filter, ICA, decimation) contributes most to the difference

---

## Goal 9 — Frozen Event Graph (Seq 5)

On marker detection, display a separate chart showing the prediction window around that event. The first "real" UI feature, now riding on validated predictions and rendered markers (Goal 2, done).

- [ ] New widget: `FrozenEventChart` — fixed-window chart (e.g. -0.2s to +1.0s around the event)
- [ ] Triggered when a marker is detected in the prediction stream
- [ ] Shows all decoder probabilities for that window with the event onset marked
- [ ] Placed in the centre panel (exact position TBD — depends on Goal 13)
- [ ] Verified: during replay, an event triggers a frozen snapshot that stays visible

---

# Backlog (resume original plan)

## Goal 4 — Trigger Log

Terminal-style scrolling log below the chart showing trigger events and system messages. (UI surface over the Goal 7 logging backend.)

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

Rolling latency readout and buffer-health indicator in the header. (UI surface over the Goal 7 timing log.) Deferred from M1 Commit 8.

- [ ] `Phase2Header` gains latency label (`Latency: p50 / p95 ms`) and buffer-health pill
- [ ] `Phase2Screen` subscribes to `latency_ready`, buffers in `deque(maxlen=100)`
- [ ] 5 Hz `QTimer` computes rolling percentiles and buffer-health state
- [ ] Green pill when `pending_samples < batch_size * 2`, amber otherwise
- [ ] Diagnostics clear on Halt
- [ ] Verified: during replay, latency numbers update ~5x/sec with non-zero values

---

## Goal 8 — Decision Settings UI

Wire the sidebar's Decision Settings section with functional controls.

- [ ] Threshold slider — connected to `LiveProbabilityChart`'s threshold line
- [ ] Sustained-activation input — number of consecutive above-threshold predictions required
- [ ] Conflict-resolution select — behavior when multiple decoders cross threshold simultaneously
- [ ] Settings are read from config if available, otherwise use defaults
- [ ] Verified: adjusting the threshold slider moves the red dashed line on the chart in real time

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

---

# Phase 1 Cross-Cutting Work

These items are **offline (Phase 1)** and have **no Phase-2 / online impact**. They are tracked here because they surfaced during M2 planning, but the work belongs to the Phase 1 surface — not on the Phase-2 critical path above. Move to a dedicated Phase 1 plan if/when one exists.

## Goal 19 — Per-Decoder Timepoint Selection (Phase 1 offline UI) — ✅ Done

Each decoder is now operator-selectable at its own timepoint, end-to-end. Shipped on `feat/per-decoder-timepoints` (7 commits); full design in [docs/feature_plans/per_decoder_timepoint_selection.md](feature_plans/per_decoder_timepoint_selection.md).

- [x] `EvaluationView` Summary tab is a **per-decoder roster** — each decoder has its own timepoint spinbox, AUC@t, read-only Peak column, and Confirm; pre-filled with its evaluator `peak_timepoint`
- [x] Per-decoder editing (roster spinbox, decoder-tab spinbox, or chart click) moves **only that decoder**, synced bidirectionally; "Approve && Continue" gates on **all** decoders confirmed
- [x] Selected timepoints flow as a `dict[str, float]` through `evaluation_complete → TrainView.set_timepoints → TrainingWorker → OfflineOrchestrator.run_training(dict)`
- [x] Out-of-bounds timepoints raise `ValueError` in `ModelTrainer` (surfaced via the worker error path)
- [x] Evaluator exposes per-task `peak_timepoint` (canonical suggestion); cross-task `suggested_timepoint` surfaced in the roster caption with an info badge ("peak of mean AUC", not a plain average)
- [x] Debug seeder + walkthrough pass per-decoder dicts; `debug_snapshots/default` re-seeded with 6 decoders
- [x] Verified: exported `decoding_timepoints` holds the **distinct** per-decoder values

**Deviations from the original sketch:** the legacy single-float `run_training` path and `_derive_per_task_timepoints` were **removed** (not kept as a fallback); the singular `decoding_timepoint` metadata field was **removed** entirely (`decoding_timepoints` is now the authoritative required field) rather than kept as a representative mean.

### Goal 19 — Deferred follow-ups (tracked here; feature otherwise complete)

- [ ] **Diagnostic scripts left stale.** Four single-timepoint diagnostics still read the removed `metadata["decoding_timepoint"]` — `scripts/preproc_parity_check.py` (subscript → **KeyError** on artifacts trained after the change), `offline_inference_check.py`, `inspect_decoder_internals.py`, `full_recording_live_inference_check.py` (`.get(...)` → `None`). They still work against *existing* artifacts but break on newly-trained ones. **Fix:** replace the read with the local mean of `decoding_timepoints`, e.g. `float(sum(d.values()) / len(d))`. (Left untouched on purpose to avoid conflicts with in-flight Goal-18 work.)
- [x] **sklearn `penalty` → `l1_ratio` migration.** ✅ Done — merged to `main` (PR #8, merge commit `f532011`; branch deleted). scikit-learn 1.8 deprecated `LogisticRegression(penalty=...)` (removed in 1.10); `penalty: "l1"` emitted a `FutureWarning` per fit. **Fix applied:** `_CLASSIFIER_DEFAULTS["Logistic"]` `penalty: "l1"` → `l1_ratio: 1` and `_VALID_PARAMS_BY_MODEL["Logistic"]` now lists `l1_ratio` (and **drops** `penalty` — Option A, so stale configs fail loudly). `build_classifier` forwards `**params`, no change. **Caveats discovered during the fix (the original sketch was incomplete):** (1) `solver: "liblinear"` **must stay** — `l1_ratio=1` with the default `lbfgs` solver raises `ValueError`; (2) `l1_ratio=1` produces **bit-identical coefficients** to `penalty="l1"` (no model drift); (3) `experiment_config.full.yaml` migrated `penalty: l2` → `l1_ratio: 0`; (4) `requirements.txt` floor bumped `scikit-learn>=1.4` → `>=1.8` (the liblinear `l1_ratio` spelling only exists in 1.8+; on older sklearn it is silently ignored → L2). LDA/SVM unaffected.

### Goal 19 — Remaining wrap-up
- [x] `docs/backend_architecture.md` `run_training` signature reflects `dict[str, float]` (light targeted touch; doc has broader pre-existing drift)
- [x] Merge `feat/per-decoder-timepoints` to `main` (PR #7, merge commit `e09817a`; branch deleted)

---

# Completed Goals

## Goal 2 — Event Markers on Probability Graph — ✅ Done

Trigger events render as vertical lines on the live probability chart. The `markers` tuple from `prediction_ready` is resolved against the configured event map and drawn by `LiveProbabilityChart`.

- [x] `LiveProbabilityChart` gains an `append_markers(markers)` method (data-only; lines materialise in `_refresh`)
- [x] Each trigger code is resolved against the configured event map; codes **not** in the map are filtered out
- [x] Markers render as bold black vertical lines (width 2) at the correct timestamp (colour is decoupled from the event name)
- [x] Markers scroll with the data (rebased onto the same `x = ts - latest_ts` axis as the curves; pruned when they leave the window)
- [x] Marker label shows the configured event name via the line's `InfLineLabel` — bold black text in a solid white, black-bordered box for contrast
- [x] Verified with VHDR replay: vertical lines appear at stimulus onset times

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
