# Plan: Decision rules for live decoding

**Branch:** `decisions`

**Goal:** Turn Phase 2's raw per-decoder probability stream into **per-decoder
on/off activation decisions** — then display, log, and live-tune them. Decoders are
**independent: multiple can be active at once.**

## Model

Per sample, per decoder, two composable criteria collapse to **one latched boolean**:

```
proba[decoder] ──► [threshold] ──► [sustain gate] ──► latched active[decoder]
                   (instantaneous)   (temporal, stateful, per decoder)
```

- **Threshold** — instantaneous, per-decoder (`proba ≥ threshold`).
- **Sustain gate** — per-decoder state machine: latch `on` after `sustain_timepoints` of
  continuous passing, `off` after `release_timepoints` of misses. Counted in **timepoints**
  (one prediction each) — no frequency needed, since predictions are already per-timepoint.

Cross-decoder arbitration ("pick one winner above the others by a margin") is **out of
scope** — that is a downstream *trigger* concern and is deferred (see *Trigger seam*).

## What we save

A decision is a deterministic function of `(probabilities, parameters)`. Probabilities are
already saved losslessly, so the only new information is **the parameter timeline** and **what
fired live**. New files live in the run dir beside `predictions.csv`, share the `lsl_timestamp`
join key, and are owned by `LiveSessionLogger` (crash-safe line-buffered append).

```
<run_dir>/
├── predictions.csv       (unchanged)
├── markers.csv           (unchanged)
├── decisions.csv         NEW  dense, one row per sample: lsl_timestamp, t_sec, <decoder1..N bool>, config_version
├── decision_config.jsonl NEW  one full config snapshot per version (append-only)
├── manifest.json         (extended) + decision_schema_version, decision_initial_config, n_decision_samples
└── predictions.npz       (unchanged — stays lean; no decision array)
```

**`decisions.csv`** — the faithful live record, dense like `predictions.csv`. Each decoder
column is `True`/`False` (the latched state that sample). `config_version` joins to the
timeline. Onset/offset *edges* are recovered offline by diffing the columns (`episodes_from_decisions`
helper), so we store the actual per-sample state — no reconstruction ambiguity around config
changes.

```
lsl_timestamp, t_sec, animate decoder, inanimate decoder, config_version
179034.10,     12.10, True,            False,             0
179034.14,     12.14, True,            True,              0
```

**`decision_config.jsonl`** — the parameter timeline, in human units (seconds, per-decoder
threshold map). Version 0 written at logger construction (`lsl_timestamp: null` = in effect from
the start); every `update_config` appends a version. Config at time T = latest version with
`lsl_timestamp ≤ T` (version 0 = −inf).

```json
{"config_version": 0, "lsl_timestamp": null,     "config": {"threshold": 0.85, "sustain_timepoints": 10, "release_timepoints": 1}}
{"config_version": 1, "lsl_timestamp": 179035.9, "config": {"threshold": 0.70, "sustain_timepoints": 10, "release_timepoints": 1}}
```

**`predictions.npz`** — unchanged. Decision booleans are lossless in the CSV already (no
precision concern), so the npz stays lean; nothing decision-related is added.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Dense per-sample vs. edges | **Dense `decisions.csv`** (bool per decoder) | Faithful actuals, trivial join to predictions, no offline reconstruction. Edges are one `diff` away. |
| Config provenance | **Separate `decision_config.jsonl`** (full snapshots) + `config_version` col | Decision rows stay criteria-invariant; the config (threshold + sustain/release) fits JSON, not CSV columns. |
| Threshold scope | **Single global threshold** (shared by all decoders) | One knob, one chart line. Decoders still latch independently; a per-decoder threshold isn't needed and adds UI/log clutter. |
| Latch on `update_config` | **Reset sustain counters; keep already-latched activations** | A count accrued under a different threshold is meaningless, but an on decoder shouldn't blink off on a knob change. |
| Sustain unit | **Timepoints** (one prediction each) in config/UI/log | Predictions/decisions are already per-timepoint; seconds+frequency was an unnecessary round-trip with rounding fuzz. Integer, exact, no `target_sfreq` in the engine. |
| npz decision array | **Omitted** | Booleans are lossless in the CSV; npz's full-precision reason doesn't apply. |
| Initial settings source | **Hardcoded module constants**; optional YAML override added last (Phase D) | Iterate on the numbers via the UI before freezing a schema; existing configs keep loading. |

## Incremental steps

Backend-first in pure layers, then Qt wiring, then UI — each step keeps `pytest tests/` green
and is verifiable before any UI exists. Decisions travel on a **parallel** `decision_ready`
channel; `prediction_ready` (raw probas → chart/logger) is untouched, so the layer is strictly
additive. **The whole feature starts on hardcoded default settings; the optional YAML override is
deferred to the very end (Phase D).**

### Phase A — Backend logic (pure Python, no Qt, unit-tested)

- **A1 — Primitives.** New `src/backend/online_phase/decision_engine.py`: `DecisionConfig`
  (frozen: global `threshold: float`, int `sustain_timepoints`, `release_timepoints`) seeded from
  **hardcoded module-level defaults** (`DEFAULT_THRESHOLD`, `DEFAULT_SUSTAIN_SECONDS`,
  `DEFAULT_RELEASE_SECONDS`); a YAML override is deferred to Phase D. `ThresholdCriterion`
  (instantaneous), `SustainGate` (per-decoder latch state machine).
  *Verify:* latch fires exactly `sustain_samples` after crossing; release timing; decoders independent.
- **A2 — Engine.** `DecisionEngine(decoder_names, config, target_sfreq)`:
  `process_batch(predictions, timestamps) -> DecisionResult` (iterates samples in order, threads
  `SustainGate` across batch boundaries, emits per-sample `active` arrays; **at the top, applies a
  staged config if present** — swap, reset counters, keep latches, bump version, stamp a
  `ConfigChange` with `out_ts[0]`, and attach it to the result). `set_pending_config(cfg)` — the
  thread-safe stash under a lock, the only cross-thread mutation entry point; `reset()`. Pure
  Python — no Qt; the `decision_ready` signal is added by the Phase B binding.
  *Verify:* batch-boundary continuity; a staged config applies at the next batch with the right
  version + timestamp.
- **A3 — Logging.** Extend `LiveSessionLogger` with `on_decisions(result)`: append the dense rows
  and, when `result.config_change` is set, append its snapshot to `decision_config.jsonl` — so the
  version bump and the first row under it come from the same result (no cross-file ordering risk).
  Version-0 snapshot at construction; manifest additions; `episodes_from_decisions(...)` reader.
  *Verify:* rows/ordering/`config_version` join, snapshot written on the applying batch, crash-safe
  flush, edge recovery incl. trailing open episode.

### Phase B — Backend integration (Qt wiring, headless-testable)

- **B1 — Wire (sibling consumer of `prediction_ready`, `StreamWorker` untouched).** The engine
  joins the pipeline the *same way the logger does* — as a consumer of `prediction_ready`, not by
  modifying the worker loop. A thin QObject binding wraps the pure Phase-A engine: an
  `on_predictions(predictions, out_ts, markers)` slot that passes only `(predictions, out_ts)` to
  `engine.process_batch(...)` (markers accepted to match the signal but unused — decisions are
  free-running) and emits `decision_ready = pyqtSignal(object)`.
  `AppSession.build_live_stream_session` constructs the engine (from the **hardcoded default
  `DecisionConfig`** — Phase D adds the YAML override) + binding, then connects (mirroring
  `session.py:253`, all DirectConnection on the worker thread):
  - `worker.prediction_ready → binding.on_predictions`
  - `binding.decision_ready → logger.on_decisions` (only when `log_dir` is set)

  `LiveStreamSession` owns the binding, `reset()`s the engine on `start()`, forwards
  `decision_ready`, and exposes `update_decision_config(cfg)` → `engine.set_pending_config(cfg)`
  (see [Live config changes](#live-config-changes-thread-safe-apply-gated)); the engine applies,
  versions, and (via the result) logs it on the worker thread. Decisions are computed and reach the
  UI even when logging is off (no logger). *Verify:* headless run produces a populated
  `decisions.csv` + `decision_config.jsonl`; a `log_dir=None` run still emits `decision_ready`; an
  applied config change bumps `config_version` at the right timestamp.

> **End of core backend.** Decisions are computed + logged every run on hardcoded defaults; the app
> looks identical to the user. Safe, shippable checkpoint (UI + YAML override still to come).

### Phase C — Frontend (UI only, headless tests)

- **C1 — Decision panel.** ✅ Done. `widgets/phase2/decision_panel.py`: a single row of per-decoder
  tiles, each highlighted in the decoder's colour while latched-active (name only — no probability
  or status text; multiple can be lit at once). `Phase2Screen._on_decision(result)`
  (QueuedConnection), reads `result` duck-typed (no backend import).
- **C2 — Chart threshold line.** Folded into C4. Make the single global threshold line live — a
  `set_threshold()` driven by the applied config (replacing the hardcoded `0.85`), moving on Apply.
  No activation shading — the decision panel already shows what's currently active.
- **C3 — History strip (deferred).** Piano-roll under the chart, one lane per decoder, sharing the
  chart's time axis; fed from `result.active`. Kept in the plan; not built yet.
- **C4 — Decision settings (apply-gated).** Threshold + sustain controls in `Phase2SettingsPanel`
  edit a local **draft** (seeded from the applied config); no backend call while editing.
  **Apply** commits the draft via `update_decision_config(draft)` — the one logged provenance
  event; **Reset** reverts the draft to the currently-applied config (pure UI, no log). Apply/Reset
  enabled only while `draft ≠ applied`. The chart threshold line follows the **applied** config
  (moves on Apply). *Verify:* editing doesn't call the backend; Apply commits once and logs one
  version; Reset restores controls without a backend call.

### Phase D — Config schema (optional YAML override; last)

Only now — after the engine, wiring, and UI are all proven on the hardcoded defaults — do we let
the YAML seed the initial settings.

- **D1 — Config.** Optional `decision_rules:` block + `DecisionRulesConfig` in `config_models.py`
  (global threshold, sustain/release timepoints); **absent → the hardcoded
  defaults** (existing configs keep loading unchanged). `AppSession` builds the initial
  `DecisionConfig` from the block when present, else the constants. *Verify:* valid block, absent
  block → defaults, bad values rejected.

### Phase E — Trigger seam (design only)

Not built here. Activation is the clean upstream layer a future trigger consumes: it will sit
downstream of `DecisionResult.active`, arbitrate a single winner among already-active decoders,
and log to its own `trigger_events.csv` — no engine change needed.

## Live config changes (thread-safe, apply-gated)

Decision settings change only on an explicit **Apply**, never on every slider move — so the
operator commits a *deliberate* config (70%→80% never passes through 75%) and the log records one
version bump per intent, not a sweep. Editing mutates a UI-local **draft** only; **Reset** reverts
it to the currently-applied config. Neither touches the backend until Apply.

Apply runs on the UI thread but the engine runs on the worker thread, so the commit is a
lock-guarded handoff applied at a batch boundary — the engine's decision state has exactly one
writer (the worker thread):

1. **UI thread** — `update_decision_config(cfg)` stores `cfg` in a `_pending_config` slot under a
   small lock and does nothing else (no counters/latches/version touched).
2. **Worker thread**, top of the next `process_batch` — under the lock: swap active config ←
   pending, **reset sustain counters, keep latched activations**, bump `config_version`, stamp a
   `ConfigChange` with this batch's first `out_ts`, clear the slot, and attach the `ConfigChange`
   to this batch's `DecisionResult` (the logger appends its jsonl snapshot when it sees one). The
   batch's rows are then decided under the new config.

Between batches the pending slot is **last-writer-wins** — only the most recent Apply survives,
which is fine since Apply is deliberate and rare. This guarantees no torn reads (config read once
per batch, swapped only at the boundary) and exact provenance: `decisions.csv`'s `config_version`
boundary and `decision_config.jsonl`'s `lsl_timestamp` for that version both derive from the same
`out_ts[0]`.

## Notes

- **No-internal-imports contract:** `Phase2Screen` imports only `AppSession`; `DecisionResult`
  crosses the signal boundary as a duck-typed `object`; tuning calls back through `AppSession`.
- **Engine placement:** the engine is a **sibling consumer** of `prediction_ready` (a thin
  QObject binding), exactly like the logger — `StreamWorker` is not modified, preserving its
  "injected-dependency micro-batch loop only" contract. The engine produces; logger (persist),
  UI (display), and any future trigger (act) are independent consumers of `decision_ready`.
- **Threading:** `decision_ready` → logger via DirectConnection (worker thread, single writer),
  → UI via QueuedConnection — identical to `prediction_ready`.
- **`target_sfreq`** is injected (artifact metadata), never inferred.
