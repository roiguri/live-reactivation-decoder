# Plan: Decision rules for live decoding (incremental)

**Branch:** `feat/decision-rules`
**Goal:** Turn the raw per-decoder probability stream of Phase 2 into **per-decoder
activation decisions** — a latched on/off state derived from two composable criteria
(probability **threshold** + activation **sustainability**) — then display, log, and live-tune
those decisions. Multiple decoders can be active at once.

**Explicitly out of scope for this feature (moved to the future trigger feature):** the
"above other decoders by a margin" rule. That is a *cross-decoder, single-winner arbitration*
concern, which conflicts with "multiple decoders active." It belongs **downstream** of
activation, in the trigger layer, and is deferred (see *Trigger seam* at the end). This plan
leaves the seam in place and builds nothing of it.

**Approach:** Build **backend-first, in pure layers, then integrate, then add UI** — so each
step compiles, `pytest tests/` stays green, and every step is independently verifiable
*before* any UI exists. The backend decision logic (Phase A) is pure Python with no Qt and no
I/O and is exhaustively unit-tested on synthetic probability sequences. Logging (Phase A3) is
unit-tested against `tmp_path`. Only once the backend emits and logs decisions do we wire the
frontend (Phase C), which consumes decisions through the **same no-internal-imports contract**
the Phase 2 screen already honours. Each step is one commit/PR.

## The two-layer model (why activation ≠ trigger)

```
ACTIVATION layer  (this feature)            TRIGGER layer  (future, seam only)
──────────────────────────────             ──────────────────────────────────
per-decoder, independent                   cross-decoder arbitration
threshold + sustain → latched boolean       among ACTIVE decoders, "above others
multiple can be active at once              by margin" picks the winner(s)
                                            + which decoders may emit a trigger
logged as edges (decision_events.csv)       + the output sink
                                            logged separately (trigger_events.csv)
```

Per sample, per decoder, the engine collapses the criteria to **one latched boolean**:

```
proba[decoder] ──► [Threshold] ──► instantaneous pass ──► [Sustain gate] ──► latched "active"
                   (instantaneous)    (per sample)          (temporal, stateful)   (per decoder)
```

An **edge** is the diff of that latched boolean over time (`off→on` / `on→off`). The edge log
is **criteria-invariant**: the criteria identity and values live entirely in the config
snapshot (`config_version` → `decision_config.jsonl`), never in the edge row.

## Core principle for what we save

> **A decision is a deterministic function of `(probabilities, parameters)`.** We already
> persist probabilities losslessly (`predictions.npz`/`.csv`). So the only genuinely new
> information is (1) the **parameter timeline** — what the rules were at each instant — and
> (2) **what actually fired live** — a faithful audit record (mandatory once a trigger acts on
> a subject). Per-sample decision booleans are pure derived bulk and are **not** persisted as
> source of truth.

This is what makes "a decision at 85% threshold ≠ a decision at 50%" unambiguous: every edge
carries the `config_version` in effect, and the timeline says which config was live at every
moment — while the saved probabilities let any alternative parameter set be recomputed offline.

## Decisions baked into this plan (defaults — correct me if wrong)

| Decision | Default chosen | Rationale |
|---|---|---|
| What-fired format | **Edges** (one row per on/off transition), like `markers.csv` | Crash-safe (onset flushed immediately), stateless sink, honours mid-episode config changes, consistent with existing marker logging. Episodes reconstructed offline. |
| `reason`/gating field on edges | **Omitted** | Derivable from probas+config; would couple the log to the criteria set. Add later only if live debugging needs it. |
| "Above other decoders" criterion | **Omitted from activation** → deferred to trigger feature | Single-winner arbitration belongs downstream, not in multi-active activation. |
| Parameter timeline storage | **Full snapshots in `decision_config.jsonl`** (append-only) | Self-contained per line, trivial "config at time T" lookup, crash-safe append; config changes are infrequent (human tuning) so size is a non-issue. |
| `.npz` per-sample decision arrays | **Omitted — npz stays lean** | Booleans are derived bulk; recompute from probas + timeline. Avoid the appearance of source-of-truth. |
| `update_config` latch behaviour | **Reset sustain counters; keep already-latched activations** until they release naturally | A sustain count accrued under a different threshold is meaningless; but a currently-on decoder shouldn't blink off just because a knob moved. |
| Sustain unit | **Seconds** in config/UI/log; converted to samples via `target_sfreq` inside the engine | Human-meaningful and `target_sfreq`-independent in the persisted record. |
| Onset edge timestamp | **Latch time** (after the sustain window), not the raw threshold crossing | The latch *is* the decision and is what a trigger would act on; the raw crossing stays visible in `predictions.npz`. |

---

## Data saved (full specification)

All new artifacts live in the existing per-run directory next to `predictions.csv` /
`markers.csv`, share the same **`lsl_timestamp`** join key and the same crash-safe,
line-buffered append model, and are owned by `LiveSessionLogger` (which already owns `run_dir`,
`lsl_t0`, the manifest, and the `close()` lifecycle — see *Logger ownership* below).

Run directory after this feature:

```
<run_dir>/
├── predictions.csv        (unchanged) lsl_timestamp, t_sec, <decoder1..N>
├── markers.csv            (unchanged) lsl_timestamp, t_sec, code, name
├── decision_events.csv    NEW  edges:   lsl_timestamp, t_sec, decoder, edge, proba, config_version
├── decision_config.jsonl  NEW  timeline: one JSON object per config version
├── manifest.json          (extended) + decision_schema_version, decision_initial_config, n_decision_events
└── predictions.npz        (unchanged — stays lean; decisions recomputed offline)
```

**`decision_events.csv`** — the faithful record of what fired live. One row per latch
transition, flushed immediately (crash-safe). `edge ∈ {on, off}`. `proba` is the decoder's
probability at the edge sample (a convenience annotation; the full cross-decoder context is in
`predictions.npz`). `config_version` joins to the timeline.

```
lsl_timestamp, t_sec,  decoder, edge, proba,   config_version
179034.120000, 12.10,  reach,   on,   0.91000, 0
179035.400000, 13.38,  reach,   off,  0.62000, 0
179036.010000, 13.99,  grasp,   on,   0.88000, 1
```

**`decision_config.jsonl`** — the parameter timeline. Append-only, one full snapshot per
version. Version 0 is written at logger construction (the YAML/UI defaults that were live from
the start); every `update_config` appends a new version. Config is stored in **human units**
(seconds, per-decoder threshold map) — not engine-internal samples.

```json
{"config_version": 0, "lsl_timestamp": null,         "t_sec": 0.0,  "config": {"thresholds": {"reach": 0.85, "grasp": 0.85}, "sustain_seconds": 0.3, "release_seconds": 0.0}}
{"config_version": 1, "lsl_timestamp": 179035.9,     "t_sec": 14.0, "config": {"thresholds": {"reach": 0.70, "grasp": 0.85}, "sustain_seconds": 0.3, "release_seconds": 0.0}}
```

`lsl_timestamp: null` on version 0 means "in effect from the beginning" (the floor). To find
the config active at any edge: take the latest version whose `lsl_timestamp ≤ edge_ts`,
treating version 0 as `-inf`.

**`manifest.json` additions** — `decision_schema_version` (int), `decision_initial_config`
(the version-0 snapshot, for at-a-glance reproducibility), `n_decision_events` (int, filled at
`close()`).

**`predictions.npz`** — unchanged. Decisions are *not* added as per-sample arrays; they are
recomputed offline from probas + the timeline (or read from the edge CSV). This keeps the npz
lean and avoids implying the booleans are ground truth.

**Recovery / `export_session_npz`** — both new files are independently crash-safe (line-buffered
append), so a session that dies before `close()` still has a complete, interpretable decision
record on disk. `export_session_npz` is unaffected (it rebuilds only the prediction/marker
bundle, which does not carry decisions).

**Reading helper (ships with the logger module).** Nobody hand-rolls edge-pairing:

```python
def episodes_from_events(events) -> list[Episode]:
    """Pair on/off edges per decoder into closed intervals. A trailing 'on'
    with no 'off' (run ended mid-activation) yields an open episode
    (offset_ts=None). Carries onset/offset proba + config_version; peak proba
    over the episode is a separate enrichment that needs predictions.npz."""
```

---

## Logger ownership (decision)

Decision logging **extends `LiveSessionLogger`** rather than introducing a separate
`DecisionLogger`, because decision events must align to the same `lsl_t0`/`lsl_timestamp`
clock the logger already owns, and the manifest has a single writer. The logger gains two new
sinks, mirroring `on_predictions`:

```python
def on_decisions(self, result) -> None:          # appends edges to decision_events.csv
def on_config_change(self, change) -> None:       # appends a snapshot to decision_config.jsonl
```

Both run on the worker thread via `DirectConnection` (single writer, no locking), exactly like
`on_predictions` today. `close()` finalizes `n_decision_events` into the manifest. The module
docstring ("two pieces, deliberately decoupled") is updated to describe the decision streams.

---

## Frontend/backend separation (the contract)

- **Phases A–B are backend-only.** No Qt in Phase A (pure logic + file I/O). Phase B adds the
  Qt signal wiring in `StreamWorker`/`LiveStreamSession`/`AppSession`. The UI is untouched and
  the app behaves identically (decisions are computed + logged, nothing is shown yet).
- **Phase C is frontend-only.** It adds widgets and wires them to the new `decision_ready`
  signal. It introduces **no backend logic**.
- **No-internal-imports contract (unchanged from CLAUDE.md):** `Phase2Screen` imports only
  `AppSession`. The `DecisionResult` crosses the signal boundary as a duck-typed `object` (the
  signal is `pyqtSignal(object)`), and the screen reads its attributes (`.active`, `.onsets`,
  `.offsets`, `.timestamps`) without importing the backend type — exactly as `_on_predictions`
  consumes the raw `dict` today. Live tuning calls back through `AppSession`/`LiveStreamSession`
  (a thin `update_decision_config(...)` forwarder), never a backend internal.
- `prediction_ready` is **left untouched** — raw probabilities keep flowing to the chart and
  logger. Decisions travel on a **parallel** `decision_ready` channel. This keeps every UI
  consumer of raw probas working unchanged and makes the decision layer purely additive.

---

## Incremental steps

### Phase A — Backend decision logic (pure; no Qt, unit-tested)

#### Step A1 — Config + criteria primitives
- New module `src/backend/online_phase/decision_engine.py` (or a `decision/` package if it
  grows). Add:
  - `DecisionConfig` (frozen dataclass): `thresholds: dict[str, float]`,
    `sustain_seconds: float`, `release_seconds: float = 0.0`. (No margin — deferred.)
  - `ThresholdCriterion` — instantaneous, per-decoder: `evaluate(probs) -> dict[str, bool]`.
  - `SustainGate` — temporal, **per-decoder independent state machine**: tracks consecutive
    passing samples; latches `on` after `sustain_samples`, latches `off` after `release_samples`
    of misses (`release_samples=0` → drop on first miss). `step(passed) -> dict[str, bool]`,
    `reset()`.
- Pure: no Qt, no I/O, no numpy-array emission yet (operates per-sample).
- **Verify:** `tests/online_phase/test_decision_engine.py` — feed synthetic per-sample
  sequences; assert latch occurs exactly `sustain_samples` after the crossing, release timing,
  and that decoders are independent (one latching doesn't affect another).

#### Step A2 — `DecisionEngine` + `DecisionResult`
- `DecisionResult` (frozen dataclass): `timestamps: np.ndarray (n,)`,
  `active: dict[str, np.ndarray]` (per-decoder bool `(n,)`), `onsets: list[ActivationEvent]`,
  `offsets: list[ActivationEvent]`. `ActivationEvent`: `decoder, timestamp, proba`.
- `DecisionEngine(decoder_names, config, target_sfreq)`:
  - `process_batch(predictions, timestamps) -> DecisionResult` — iterates the `n_out` samples
    **in temporal order**, threading `SustainGate` state across batch boundaries, collecting
    edge events.
  - `update_config(config) -> ConfigChange` — bumps `config_version`, resets sustain counters,
    **keeps latched activations**, returns the change record (full new snapshot + version + the
    timestamp basis) for the timeline.
  - `reset()` — called on session start.
- Converts `sustain_seconds`/`release_seconds` → samples via `target_sfreq` internally.
- **Verify:** unit tests for batch-boundary continuity (a sustain window spanning two
  `process_batch` calls latches correctly), edge emission (onset at latch time, not crossing),
  and a mid-stream `update_config` (counters reset, latched stays on, version bumps).

#### Step A3 — Decision logging (data-saving focus)
- Extend `LiveSessionLogger` with `on_decisions(result)` and `on_config_change(change)`, the
  two new files (`decision_events.csv`, `decision_config.jsonl`), version-0 snapshot at
  construction, and the manifest additions — all per the **Data saved** spec above.
- Add the `episodes_from_events(...)` reading helper to the same module.
- **Verify:** `tests/online_phase/test_session_logger_decisions.py` (or extend the existing
  logger test) — drive `on_decisions`/`on_config_change` into a `tmp_path` run dir; read the
  CSV/JSONL back and assert rows, ordering, `config_version` joins, crash-safe flush (file
  contents present before `close()`), and `episodes_from_events` pairing (incl. a trailing
  open episode).

### Phase B — Backend integration (Qt wiring; headless-testable)

#### Step B1 — Config schema + defaults
- Add an optional `decision_rules:` block to the experiment config with a Pydantic
  `DecisionRulesConfig` in `config_models.py` (per-decoder thresholds with a global fallback,
  `sustain_seconds`, `release_seconds`). Absent block → sensible hardcoded defaults so existing
  configs keep loading (`extra="forbid"` compliance: the block is optional, not required).
- `AppSession` builds a `DecisionConfig` from it (resolving the global→per-decoder fallback
  against the artifact's decoder names).
- **Verify:** `tests/core/` config-validation tests (valid block, absent block → defaults,
  bad values rejected); a test that `AppSession` produces the expected `DecisionConfig`.

#### Step B2 — Wire engine into the live pipeline
- `StreamWorker`: accept an injected `decision_engine`; after `inference_engine.predict(...)`,
  call `decision_engine.process_batch(predictions, out_ts)` and
  `self.decision_ready.emit(result)` (new `decision_ready = pyqtSignal(object)`).
  `prediction_ready` is unchanged.
- `LiveStreamSession`: own the engine; `reset()` it on `start()`; forward `decision_ready`;
  expose `update_decision_config(config)` that calls `engine.update_config(...)` and feeds the
  returned change to the logger.
- `AppSession.build_live_stream_session(...)`: construct the `DecisionEngine` from B1's config
  and the artifact decoder names; connect `worker.decision_ready` to
  `logger.on_decisions` via `DirectConnection` (mirroring the existing `prediction_ready`
  logger hookup at session.py:253), and the version-0 snapshot is written by the logger at
  construction.
- **Verify:** a headless backend integration test (fake receiver, à la
  `scripts/smoke_stream_worker.py` / existing Phase 2 lifecycle tests) asserting
  `decision_ready` fires and the run dir contains a populated `decision_events.csv` +
  `decision_config.jsonl`. Optionally extend `smoke_stream_worker.py` to print decisions.

> **End of backend.** Decisions are computed, emitted, and logged on every run. The app looks
> identical to a user — nothing is displayed yet. This is a safe, shippable checkpoint.

### Phase C — Frontend (UI only; headless UI tests)

#### Step C1 — Decision panel widget
- New `src/frontend/widgets/phase2/decision_panel.py`: one row per decoder — on/off indicator,
  live proba, and sustain progress (e.g. `3/5`). Multiple lit rows = multiple active.
- `Phase2Screen._on_decision(result)` slot (QueuedConnection, like `_on_predictions`) updates
  the panel. Reads `result` attributes duck-typed (no backend import).
- **Verify:** `tests/frontend/test_decision_panel.py` (headless) — feed a synthetic
  `DecisionResult`-shaped object; assert indicators/progress reflect it.

#### Step C2 — Chart threshold lines + activation shading
- `LiveProbabilityChart`: draw a per-decoder threshold line (replacing the hardcoded
  `threshold=0.85`), and shade/emphasize a trace while its decoder is latched-active.
- Driven by the decision stream + the active thresholds (passed in / updated on config change).
- **Verify:** headless chart test asserting threshold line position and that shading toggles
  with `active`.

#### Step C3 — Decision history strip
- New widget: a piano-roll/raster under the chart — one lane per decoder, filled bars over
  activation episodes, sharing the chart's scrolling time axis. Fed from `result.active`.
- **Verify:** headless test feeding a sequence of batches; assert bars appear/clear in step
  with activation, and scrolling matches the chart window.

#### Step C4 — Live tuning controls
- Add threshold (global + per-decoder) and sustain-seconds controls to `Phase2SettingsPanel`.
  Editing calls `session_live.update_decision_config(new_config)` — which is exactly what bumps
  the version and appends the `decision_config.jsonl` snapshot. So **the UI tweak *is* the
  logged provenance event**; there is no separate "log it" path.
- The chart's threshold line(s) and the panel update to the new config.
- **Verify:** headless test — adjust a control, assert `update_decision_config` is called and a
  new config version is logged with the new value.

### Phase D — Trigger seam (design only; built later, separate feature)

Not implemented here. This plan only ensures activation is the clean upstream layer the trigger
will consume:
- The trigger will sit downstream of `DecisionResult.active`, applying a `TriggerConfig`
  (`enabled_decoders`, `margin`) to arbitrate a single winner among **already-active** decoders
  ("above max of others by margin"), and emit trigger events through a `TriggerSink`
  (`on_activation(event)`) registered behind `LiveStreamSession`.
- It gets its **own** `trigger_events.csv` + its slice of the config timeline, keeping trigger
  provenance separate from activation.
- No `StreamWorker`/`DecisionEngine` change is needed to add it — the modularity payoff of
  keeping arbitration out of the activation layer.

---

## Notes / risks

- **Behavior parity for raw probabilities:** `prediction_ready` and the chart/logger paths that
  consume it are untouched in every step. The decision layer is strictly additive; if it were
  removed, Phase 2 would behave exactly as today.
- **Threading:** decisions reuse the existing model — `decision_ready` connects to the logger
  via `DirectConnection` (worker thread, single writer) and to the UI via `QueuedConnection`
  (marshalled to the UI thread), identical to `prediction_ready`.
- **Sustain across batch boundaries** is the one piece of cross-batch state; it lives entirely
  in the engine's `SustainGate` and is reset on `start()`/`reset()` and on `update_config`.
- **`target_sfreq` dependency:** the engine needs it to convert seconds→samples;
  `AppSession`/`LiveStreamSession` already know it (it's in the artifact metadata / used to size
  the chart), so it is injected, never inferred.
- **Config version floor:** readers must treat version 0 (`lsl_timestamp: null`) as `-inf` when
  resolving "config active at time T".
```