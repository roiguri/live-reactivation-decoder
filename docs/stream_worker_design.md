# StreamWorker + PredictionLogger — Backend-Only Plan

## Context

The online-phase backend has three committed components — `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` — but no live runtime that drives them. This PR adds the runtime (`StreamWorker`) and a persistence sink (`PredictionLogger`), plus a factory on `AppSession` so the frontend (partner's PR) can spin them up without knowing the internals.

**Work split:**
- **This PR (backend, you):** `StreamWorker`, `PredictionLogger`, `AppSession.online` factory, headless smoke test, updates to `backend_architecture.md` documenting the contract.
- **Partner's PR (frontend):** `ProbabilityBuffer`, `ProbabilityPlotWidget`, UI lifecycle, button-to-action wiring. They connect to the backend through the documented `AppSession.online` interface.

The deliverable from this PR is a **contract**: signals + types + threading guarantees that the frontend partner can build against in isolation.

## Scope

**In:**
- `StreamWorker` (QThread orchestrator)
- `PredictionLogger` (CSV sink)
- `AppSession.online.build_stream_session(...)` factory
- **Modifications to `LSLReceiver` + `split_eeg_and_markers`** to return per-marker timestamps (decision: exact timing — see Resolved Decisions §1)
- Headless smoke test (worker + logger driven by replayed/fake LSL → CSV)
- `docs/backend_architecture.md` updates: StreamWorker section rewrite + new "Frontend integration contract" subsection
- Updates to existing `LSLReceiver` tests for the new marker return type

**Out (for partner / future):**
- `ProbabilityBuffer`, `ProbabilityPlotWidget`, any UI code
- Trigger-aligned epoch viewer
- Network sinks (MQTT/UDP)
- Persistence formats other than CSV (parquet, HDF5)

## Architecture

```
┌──────────────────────────────────────────────┐
│              BACKEND (this PR)               │
│                                              │
│  LSLReceiver ─► StreamWorker ─► (signal) ────┼──► to frontend partner's
│                  │                           │    ProbabilityBuffer
│                  ├─► OnlinePreprocessor      │    (then ProbabilityPlotWidget)
│                  └─► LiveInferenceEngine     │
│                                              │
│                  prediction_ready signal     │
│                  also feeds:                 │
│                  └─► PredictionLogger ──► CSV│
│                                              │
│  Factory: AppSession.online.build_stream_session(...)
│  Returns: a handle exposing the worker + signal
└──────────────────────────────────────────────┘
                       │
              prediction_ready signal
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           FRONTEND (partner's PR)            │
│                                              │
│  worker.prediction_ready ─► ProbabilityBuffer│
│                              │               │
│                              ▼               │
│                       ProbabilityPlotWidget  │
└──────────────────────────────────────────────┘
```

`StreamWorker` is a pure orchestrator. It emits raw per-batch numeric data and knows nothing about plots, files, or windows. `PredictionLogger` is one consumer of that signal (CSV). The frontend partner attaches other consumers (buffer, plot) via the same signal.

## Class specs

### 1. `StreamWorker`

**File:** `online_decoder/src/backend/online_phase/stream_worker.py`

**Inherits:** `PyQt6.QtCore.QThread`

**Constructor:**
```python
def __init__(
    self,
    receiver: LSLReceiver,
    preprocessor: OnlinePreprocessor,
    inference_engine: LiveInferenceEngine,
    batch_size_samples: int = 40,        # raw-rate samples per batch
    poll_interval_sec: float = 0.01,
    parent: QObject | None = None,
) -> None:
```

**Signal:**
```python
prediction_ready = pyqtSignal(dict, np.ndarray, list)
# (predictions: dict[task_name -> (n_rows,) float ndarray of P(class=1)],
#  timestamps:  (n_rows,) float64 ndarray, LSL clock, monotonic non-decreasing,
#  markers:     list[(timestamp: float, code: int)] — LSL clock, per-sample-accurate)
```

**Public methods:**
- `run() -> None` — Qt calls this after `start()`.
- `stop() -> None` — sets `_stop_requested`; loop exits at next iteration; caller does `worker.wait()` to join.

**Internal state:**
- `_batch_ts: list[np.ndarray]`, `_batch_eeg: list[np.ndarray]` — accumulators.
- `_pending_markers: list[(ts, code)]` — markers detected but not yet emitted.
- `_stop_requested: bool`.

**Loop body (per iteration):**
1. `ts, eeg, markers = receiver.pull_new_data()` — `markers` is now `list[(ts, code)]` after the LSLReceiver change.
2. Append `(ts, eeg)` to internal accumulators; extend `_pending_markers` with `markers`.
3. While `accumulated_samples >= batch_size_samples`:
   1. Pop a `batch_size_samples`-wide chunk from accumulators (record its last raw timestamp as `batch_end_ts`).
   2. `out_eeg, out_ts = preprocessor.process_batch(batch_eeg, batch_ts)`.
   3. `predictions = inference_engine.predict(out_eeg)` — channels-are-features (Resolved §2).
   4. Take markers from `_pending_markers` whose timestamps are `<= batch_end_ts` (the rest stay for the next batch).
   5. `self.prediction_ready.emit(predictions, out_ts, batch_markers)`.
4. If no batch was ready: `time.sleep(poll_interval_sec)`.

**Threading guarantee:** `prediction_ready` is connected with the default `Qt.AutoConnection`. When the receiving slot lives on a different thread (the UI thread), Qt uses a queued connection — the slot runs in the receiver's thread, not the worker's. Numpy arrays passed in the signal payload are shared by reference; we accept that and document that slots must NOT mutate them.

**Does NOT know about:** plots, ring buffers, files, task display names, the UI thread.

---

### 2. `PredictionLogger`

**File:** `online_decoder/src/backend/online_phase/prediction_logger.py`

**Inherits:** `PyQt6.QtCore.QObject`

**Constructor:**
```python
def __init__(
    self,
    out_path: str | Path,
    task_names: list[str],
    parent: QObject | None = None,
) -> None:
```

On construction:
- Opens `out_path` for writing (text, line-buffered).
- Writes header: `timestamp,marker_code,<task_1>,<task_2>,...`.

**Slot:**
```python
@pyqtSlot(dict, np.ndarray, list)
def on_predictions(
    self,
    predictions: dict[str, np.ndarray],
    timestamps: np.ndarray,
    markers: list[tuple[float, int]],
) -> None:
```

Writes one row per timestamp. Marker codes are matched to rows by nearest-timestamp within `0.5 / target_sfreq` tolerance; unmatched rows have empty `marker_code`. Uses `csv.writer`; flushes after each batch.

**Public methods:**
- `close() -> None` — flushes and closes the file. Called by the factory's cleanup path.

**Does NOT know about:** plots, buffers, the worker class, the UI thread.

---

### 3. `AppSession.online.build_stream_session(...)` factory

**File:** `online_decoder/src/backend/session.py` (extend existing `AppSession`)

Mirror the existing `session.offline` shape. Add a new `session.online` namespace exposing factories for Phase 2.

**Proposed interface:**
```python
@dataclass
class StreamSessionHandle:
    """Bundle returned by build_stream_session. Frontend connects to .worker.prediction_ready."""
    worker: StreamWorker
    logger: PredictionLogger | None
    receiver: LSLReceiver
    preprocessor: OnlinePreprocessor
    inference_engine: LiveInferenceEngine

    def start(self) -> None:
        """Start receiver and worker. Idempotent."""

    def stop(self) -> None:
        """Stop worker, wait for join, close logger, stop receiver. Idempotent."""


class OnlinePhase:  # exposed as AppSession.online
    def build_stream_session(
        self,
        decoder_pipeline_path: Path,
        log_path: Path | None = None,
        batch_size_samples: int = 40,
    ) -> StreamSessionHandle:
        """Construct the full backend pipeline, wired but not started.

        decoder_pipeline_path: path to decoder_pipeline.joblib (Phase 1 export).
        log_path: if provided, a PredictionLogger is created and connected. If None, no logging.
        batch_size_samples: passed to StreamWorker.

        Returns a handle the caller (frontend) connects signals to and then calls .start().
        """
```

The factory:
1. Loads `DecoderPipelineArtifact` from `decoder_pipeline_path`.
2. Constructs `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` from artifact + settings.
3. Constructs `StreamWorker`.
4. If `log_path`: constructs `PredictionLogger` and `worker.prediction_ready.connect(logger.on_predictions)`.
5. Returns the handle. Does NOT call `.start()` — the caller does that after attaching its own consumers.

**Why a handle instead of just returning the worker:** the frontend partner needs `worker` (to `.connect` to its buffer), plus the logger needs to be `.close()`d on shutdown, plus the receiver needs to be `.stop()`ed. Bundling lifecycle into the handle keeps the frontend ignorant of which lower-level objects exist.

---

## Frontend integration contract (goes into `backend_architecture.md`)

This section is the exact thing the frontend partner builds against. Copy this verbatim into the doc.

> ### Backend → Frontend contract: live decoder output
>
> **Entry point:** `AppSession.online.build_stream_session(decoder_pipeline_path, log_path=None, batch_size_samples=40) -> StreamSessionHandle`
>
> **Lifecycle:**
> 1. Frontend calls `build_stream_session(...)` to get a `StreamSessionHandle`.
> 2. Frontend connects its UI-side slots to `handle.worker.prediction_ready`.
> 3. Frontend calls `handle.start()`.
> 4. On shutdown (e.g., window close), frontend calls `handle.stop()`.
>
> **Signal:** `StreamWorker.prediction_ready = pyqtSignal(dict, np.ndarray, list)`
>
> **Payload:**
> - `predictions: dict[str, np.ndarray]` — keys are task names (model identifiers from the loaded `DecoderPipelineArtifact`); values are `float32` arrays of shape `(n_rows,)`. Each value is `P(class=1)` for that task at that timestamp.
> - `timestamps: np.ndarray` — `float64`, shape `(n_rows,)`, LSL clock seconds. Monotonic non-decreasing across emissions.
> - `markers: list[tuple[float, int]]` — `(timestamp, trigger_code)`. Timestamps are per-sample-accurate on the LSL clock. Codes match the trigger codes documented in `experiment_config.yaml`.
>
> **Threading:**
> - The signal is emitted from the `StreamWorker` background thread.
> - Connect with default `Qt.AutoConnection`; Qt uses a queued connection automatically when the receiver lives on the main/UI thread, so the slot runs in the UI thread.
> - **Slots must not mutate the numpy arrays in the payload.** Copy first if you need to.
>
> **Cadence:**
> - Emissions occur once per accumulated batch of `batch_size_samples` raw-rate samples (default 40 samples @ 1000 Hz = ~25 Hz emission rate, with `n_rows` ≈ 4 at 100 Hz target).
> - Emission rate is governed by LSL pull cadence; if the upstream stalls, emissions stall.
>
> **What the frontend is expected to build:**
> - A consumer (e.g., `ProbabilityBuffer`) that maintains a rolling window of (timestamps, predictions, markers) and emits a plot-ready signal.
> - A renderer (e.g., `ProbabilityPlotWidget`) that consumes the buffer's signal and draws.
> - Wiring code that calls `build_stream_session(...)`, connects slots, and manages `start()/stop()`.
>
> **What the frontend should NOT do:**
> - Reach into `handle.worker.<private>` or any of the other handle members. Use the signal and `start()/stop()` only.
> - Block in slots — slots run on the UI thread.

---

## Modifications to existing code: `LSLReceiver` + `split_eeg_and_markers`

To deliver per-sample-accurate marker timestamps, two committed files change.

**`split_eeg_and_markers(samples, eeg_channel_count, trigger_channel_index, previous_trigger_code)`** in [lsl_receiver.py](online_decoder/src/backend/online_phase/lsl_receiver.py) (look for the helper called at line 278):
- Current return: `(eeg_chunk, chunk_markers: list[int], last_trigger_code: int)`.
- New return: `(eeg_chunk, chunk_markers: list[tuple[int, int]], last_trigger_code: int)` where each `chunk_markers` entry is `(sample_index_within_chunk, code)`.
- Why this shape: keeps the helper timestamp-agnostic. The receiver maps `sample_index → LSL timestamp` using the chunk's `timestamps_array` it already has.

**`LSLReceiver.pull_new_data()`** at [lsl_receiver.py:262-307](online_decoder/src/backend/online_phase/lsl_receiver.py#L262-L307):
- Current return: `tuple[np.ndarray, np.ndarray, list[int]]`.
- New return: `tuple[np.ndarray, np.ndarray, list[tuple[float, int]]]` — each marker is `(timestamp, code)`.
- Implementation: after `split_eeg_and_markers` returns `(sample_index, code)` pairs, the receiver does `marker_ts = float(timestamps_array[sample_index])` for each and extends a `list[(ts, code)]`.

**Test updates:**
- Existing tests under `online_decoder/tests/online_phase/test_lsl_receiver.py` (the `FakeInlet` / `FakePylslModule` tests) assert on the marker return type. Update assertions to expect `list[tuple[float, int]]` instead of `list[int]`.
- Add at least one new case asserting marker timestamps line up with the right sample (synthetic chunk where a trigger fires at a known sample index → assert the returned `ts` equals the corresponding `timestamps_array` entry).

**Compatibility:** Nothing else in the committed codebase consumes `pull_new_data()` (StreamWorker is the only planned consumer), so this is a contained break.

## `docs/backend_architecture.md` changes

1. **Replace lines 1136–1202** (the tentative StreamWorker sketch) with the final spec from §1 above (constructor, signal, loop body, threading guarantees).
2. **Add a new section** `## Backend → Frontend contract: live decoder output` placed after the StreamWorker section, with the contents shown in the "Frontend integration contract" block above.
3. **Update the Phase 2 component table** (if present elsewhere in the doc) to mark `StreamWorker` and `PredictionLogger` as committed.

## File layout (backend only)

```
online_decoder/src/backend/
├── online_phase/
│   ├── lsl_receiver.py            (exists)
│   ├── online_preprocessor.py     (exists)
│   ├── live_inference.py          (exists)
│   ├── stream_worker.py           (NEW)
│   └── prediction_logger.py       (NEW)
└── session.py                     (extend with .online namespace)

online_decoder/tests/online_phase/
├── test_stream_worker.py          (NEW)
├── test_prediction_logger.py      (NEW)
└── test_stream_session_smoke.py   (NEW, opt-in like the LSL integration test)

online_decoder/scripts/
└── smoke_stream_worker.py         (NEW, headless replay)
```

## Testing strategy (backend only)

| Component | Test type | Tooling |
|---|---|---|
| `StreamWorker` | Integration (Qt) | `pytest-qt`, stub LSLReceiver (use existing `FakeInlet` pattern), real preprocessor, deterministic fake inference. Assert via `qtbot.waitSignal(worker.prediction_ready)`. Cover: empty pulls, partial batches, marker timing, stop()/join. |
| `PredictionLogger` | Pure unit | `tmp_path`, direct slot calls with synthetic numpy arrays, parse output with `pandas.read_csv`. |
| `AppSession.online.build_stream_session` | Construction smoke | Patch `LSLReceiver.start`, assert the handle's members are wired and `.stop()` is idempotent. |
| End-to-end | Opt-in smoke | `scripts/smoke_stream_worker.py` plays back a recorded session against a fake LSL inlet; assert CSV row count matches expected after N seconds. |

Tests live alongside existing tests; smoke script follows the `scripts/smoke_test_lsl_receiver.py` convention.

## Verification

1. `pytest online_decoder/tests/online_phase/ -v` → unit and Qt-integration tests green.
2. `python online_decoder/scripts/smoke_stream_worker.py --duration 5 --log /tmp/smoke.csv` → runs without UI, produces a CSV. Manually inspect:
   - Header row matches loaded model task names.
   - Row count ≈ duration × target_sfreq.
   - Timestamps are monotonic.
   - At least one `marker_code` is populated when the replayed stream contains triggers.
3. Read-only check: `backend_architecture.md` "Backend → Frontend contract" section is present, accurate, and the partner can implement against it without reading any backend source.

## Implementation: per-commit execution checklist

Tick boxes as you go. Each commit must leave the tree green (`pytest online_decoder/tests/ -v`) before moving on.

**TODOs in code, not in this plan.** Any unresolved Open Decision that doesn't block this PR is implemented with the documented default *plus* a `# TODO(open):` comment at the call site referencing the relevant Open Decisions §. Examples:
- `# TODO(open): see docs/stream_worker_design.md Open §1 — make tolerance config-driven`
- `# TODO(open): see docs/stream_worker_design.md Open §2 — accept in-memory artifact`

Future decisions discovered mid-implementation that aren't worth blocking on go in the same pattern — `# TODO(open): <one-line problem statement>` — and get appended to Open Decisions in the next plan update.

---


### Commit 1 — `feat(lsl_receiver): return per-marker timestamps`

**Files:**
- `online_decoder/src/backend/online_phase/lsl_receiver.py`
- `online_decoder/tests/online_phase/test_lsl_receiver.py`

**Implementation:**
- [x] In `split_eeg_and_markers` (helper called at [lsl_receiver.py:278](online_decoder/src/backend/online_phase/lsl_receiver.py#L278)): change `chunk_markers` return from `list[int]` to `list[tuple[int, int]]` where each tuple is `(sample_index_within_chunk, code)`.
- [x] In `LSLReceiver.pull_new_data` ([lsl_receiver.py:262-307](online_decoder/src/backend/online_phase/lsl_receiver.py#L262-L307)):
  - [x] Update return annotation to `tuple[np.ndarray, np.ndarray, list[tuple[float, int]]]`.
  - [x] After `split_eeg_and_markers` returns, map each `(sample_idx, code)` → `(float(timestamps_array[sample_idx]), code)` and extend the markers list.
  - [x] Update the empty-return branch (lines 297-302) — still returns empty list.

**Tests:**
- [x] Update every existing `test_lsl_receiver.py` assertion on marker shape from `list[int]` to `list[tuple[float, int]]`.
- [x] Add a new test: synthesize a chunk with a trigger edge at a known sample index; assert returned `marker_ts == timestamps_array[that_index]`.
- [x] Add a new test: trigger held high across two `pull_new_data()` calls only emits once (existing behavior preserved with new tuple shape).

**TODOs in code:** none.

**Verify:**
- [x] `pytest online_decoder/tests/online_phase/test_lsl_receiver.py -v` green.
- [x] `pytest online_decoder/tests/ -v` everything else still green.

**Commit:**
- [x] `git commit -m "feat(lsl_receiver): return per-marker timestamps"`

---

### Commit 2 — `feat(online_phase): add PredictionLogger`

**Files:**
- `online_decoder/src/backend/online_phase/prediction_logger.py` (NEW)
- `online_decoder/tests/online_phase/test_prediction_logger.py` (NEW)

**Implementation:**
- [x] `PredictionLogger(QObject)`: ctor `(out_path, task_names, target_sfreq, parent=None)`.
- [x] On init: open `out_path` in `"w"` mode (line-buffered), write header `timestamp,marker_code,<task_1>,<task_2>,...` using `csv.writer`.
- [x] Slot `on_predictions(predictions, timestamps, markers)`:
  - [x] For each row `i`: build row `[timestamps[i], <code or "">, predictions[task][i] for task in task_names]`.
  - [x] Marker matching: tolerance = `0.5 / target_sfreq`; nearest marker within tolerance gets attached to the row.
  - [x] `csv.writer.writerow(row)`; flush after each batch.
- [x] `close()`: flush + close; idempotent.

**Tests:**
- [x] Construct with `tmp_path / "session.csv"`, feed one batch (3 rows, empty markers); assert header + 3 rows with empty `marker_code`.
- [x] Feed a batch where one marker's timestamp matches a row's timestamp; assert that row's `marker_code` is the trigger code.
- [x] Feed two batches sequentially; assert rows accumulate in input order.
- [x] Call `close()` twice; assert no exception.
- [x] Assert header column order matches `task_names` order.

**TODOs in code:**
- [x] `# TODO(open): see docs/stream_worker_design.md Open §1 — wire tolerance from SettingsManager` at the tolerance calculation site.

**Verify:**
- [x] `pytest online_decoder/tests/online_phase/test_prediction_logger.py -v` green.

**Commit:**
- [x] `git commit -m "feat(online_phase): add PredictionLogger"`

---

### Commit 3 — `feat(online_phase): add StreamWorker`

**Files:**
- `online_decoder/src/backend/online_phase/stream_worker.py` (NEW)
- `online_decoder/tests/online_phase/test_stream_worker.py` (NEW)
- `online_decoder/requirements-dev.txt` (add `pytest-qt` if missing)

**Implementation:**
- [x] Add `pytest-qt` to `requirements-dev.txt` if not already present.
- [x] `StreamWorker(QThread)`: ctor `(receiver, preprocessor, inference_engine, batch_size_samples=40, poll_interval_sec=0.01, parent=None)`.
- [x] Signal: `prediction_ready = pyqtSignal(dict, np.ndarray, list)`.
- [x] Internal state: `_batch_ts`, `_batch_eeg`, `_pending_markers`, `_stop_requested`.
- [x] `run()` loop body (see class spec §1 above for steps):
  - [x] pull → accumulate → extend `_pending_markers`
  - [x] while accumulated ≥ batch_size: pop batch, preprocess, predict, take ripe markers, emit
  - [x] short sleep if no batch was ready
  - [x] honor `_stop_requested`
- [x] `stop()`: set `_stop_requested = True`.

**Tests:**
- [x] Build a `FakeReceiver` exposing `pull_new_data()` returning a scripted sequence of chunks (some empty, some with markers).
- [x] Build a deterministic fake `LiveInferenceEngine` (returns input-dependent dict).
- [x] Use real `OnlinePreprocessor` with simple settings.
- [x] `qtbot.waitSignal(worker.prediction_ready)` — assert payload types and shapes match the contract.
- [x] Below-threshold accumulation does not emit until a batch is ready.
- [x] Marker with `ts <= batch_end_ts` is included in that emission; marker with `ts > batch_end_ts` is deferred to the next emission.
- [x] `worker.stop()` → `worker.wait(timeout=2000)` returns True; thread is no longer running.

**TODOs in code:** none.

**Verify:**
- [x] `pytest online_decoder/tests/online_phase/test_stream_worker.py -v` green.

**Commit:**
- [x] `git commit -m "feat(online_phase): add StreamWorker"`

---

### Commit 4 — `feat(session): add AppSession.online factory`

**Files:**
- `online_decoder/src/backend/session.py` (extend with `.online` namespace)
- `online_decoder/tests/test_session_online.py` (NEW, or extend existing `test_session.py` if present)

**Implementation:**
- [ ] `@dataclass class StreamSessionHandle` with members `worker, logger, receiver, preprocessor, inference_engine` and methods `start()`, `stop()`.
- [ ] `class OnlinePhase` exposing `build_stream_session(decoder_pipeline_path, log_path=None, batch_size_samples=40) -> StreamSessionHandle`.
- [ ] Add `AppSession.online` property returning an `OnlinePhase` instance (lazy or eager — match `session.offline` style).
- [ ] In `build_stream_session`:
  - [ ] Load `DecoderPipelineArtifact` from `decoder_pipeline_path`.
  - [ ] Construct `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` from artifact + `SettingsManager` settings.
  - [ ] Construct `StreamWorker`.
  - [ ] If `log_path`: construct `PredictionLogger`; `worker.prediction_ready.connect(logger.on_predictions)`.
  - [ ] Return handle. **Do not call `start()`.**
- [ ] `handle.start()`: `receiver.start()` then `worker.start()`. Idempotent.
- [ ] `handle.stop()`: `worker.stop(); worker.wait(); logger.close() if logger; receiver.stop()`. Idempotent.

**Tests:**
- [ ] Patch `LSLReceiver.start` so no real LSL connection is attempted; construct via factory and assert handle members exist.
- [ ] With `log_path=None`: assert `handle.logger is None` and only one slot is connected to `prediction_ready`.
- [ ] With `log_path` set: assert two slots are connected.
- [ ] `handle.start()` then `handle.start()` again — no exception.
- [ ] `handle.stop()` then `handle.stop()` again — no exception.

**TODOs in code:**
- [ ] `# TODO(open): see docs/stream_worker_design.md Open §2 — accept in-memory DecoderPipelineArtifact` at the artifact load site.
- [ ] `# TODO(open): see docs/stream_worker_design.md Open §3 — read LSL stream name / target_sfreq from SettingsManager once Phase 2 config schema is defined` at the receiver/preprocessor construction site.

**Verify:**
- [ ] `pytest online_decoder/tests/ -v` everything green.

**Commit:**
- [ ] `git commit -m "feat(session): add AppSession.online factory"`

---

### Commit 5 — `feat(scripts): add headless StreamWorker smoke script`

**Files:**
- `online_decoder/scripts/smoke_stream_worker.py` (NEW)

**Implementation:**
- [ ] CLI args: `--duration` (seconds, default 5), `--log` (CSV path, default `/tmp/smoke.csv`), `--pipeline` (path to `decoder_pipeline.joblib`).
- [ ] Construct `AppSession`, call `session.online.build_stream_session(pipeline_path, log_path)`.
- [ ] `handle.start()`; sleep for `duration`; `handle.stop()`.
- [ ] Print row count from log file and timestamp monotonicity check at the end.
- [ ] Follow the existing `scripts/smoke_test_lsl_receiver.py` conventions (logging, argparse).

**Tests:** none (smoke script).

**TODOs in code:** none.

**Verify:**
- [ ] `python online_decoder/scripts/smoke_stream_worker.py --duration 5 --log /tmp/smoke.csv --pipeline <path>` against a fake/replayed LSL stream → CSV produced, row count ≈ `duration × target_sfreq`, header matches task names, timestamps monotonic, ≥1 marker code present when triggers were replayed.

**Commit:**
- [ ] `git commit -m "feat(scripts): add headless StreamWorker smoke script"`

---

### Commit 6 — `docs(backend_architecture): document StreamWorker + frontend contract`

**Files:**
- `online_decoder/docs/backend_architecture.md` (update)
- `online_decoder/docs/Phase2_Implementation_Plan.md` (check off completed items)
- `online_decoder/CLAUDE.md` (update Current Backend Scope)

**Implementation:**
- [ ] Replace the tentative StreamWorker sketch in [backend_architecture.md:1136-1202](online_decoder/docs/backend_architecture.md#L1136-L1202) with the final spec from `stream_worker_design.md` §1 (constructor, signal, loop body summary, threading guarantees).
- [ ] Add a new section `## Backend → Frontend contract: live decoder output` after the StreamWorker section, copying the contract block from `stream_worker_design.md`.
- [ ] In `Phase2_Implementation_Plan.md`: tick off StreamWorker, add and tick PredictionLogger and `AppSession.online`.
- [ ] In `CLAUDE.md`: update "Current Backend Scope" — StreamWorker, PredictionLogger, AppSession.online are now committed; remove "next planned" note for StreamWorker.

**Tests:** none (documentation only).

**TODOs in code:** none.

**Verify:**
- [ ] Cold-read the new `backend_architecture.md` contract section as if you were the frontend partner — can you implement against it without opening any backend `src/` file? If yes, ship.

**Commit:**
- [ ] `git commit -m "docs(backend_architecture): document StreamWorker + frontend contract"`

---

### After commit 6

- [ ] Push branch, open PR for review.
- [ ] Notify partner that `AppSession.online.build_stream_session(...)` is ready and `docs/backend_architecture.md` describes the contract.

## Resolved decisions

1. **Marker timestamp resolution.** Chosen: modify `LSLReceiver` + `split_eeg_and_markers` to return per-sample-accurate `(timestamp, code)` tuples. See the "Modifications to existing code" section above for the exact changes. Tradeoff accepted: small break to a committed interface in exchange for clean per-sample timing in the worker output and a simpler contract for the frontend.
2. **Feature shape into `LiveInferenceEngine.predict()`.** Chosen: **channels-are-features.** The preprocessor's `(n_out, n_channels)` output is passed directly to `predict()` — `feature_width == n_channels`. No feature-extraction helper needed in the worker; `_build_features(out_eeg)` collapses to `features = out_eeg`. (The worker may skip the indirection entirely and call `inference_engine.predict(out_eeg)` directly.)

## Open decisions (please push back where you disagree)

1. **`PredictionLogger` marker-matching tolerance.** The logger matches markers to rows by nearest timestamp. Either pass `target_sfreq` as a constructor arg (tolerance = `0.5 / target_sfreq`), or hardcode a tolerance like `0.005` s (= 5 ms). Default in this plan: pass `target_sfreq` for correctness across configs.
2. **Should the factory accept an in-memory artifact instead of a path?** `AppSession` may already hold a loaded artifact in memory from a Phase 1 run. Two overloads, or always reload from disk? Default in this plan: always load from disk path for simplicity.
3. **Where does Phase 2 settings live?** The factory needs LSL stream name, target_sfreq, etc. Are these in `experiment_config.yaml` (read by `SettingsManager`) or passed as factory args? Default in this plan: factory reads from `SettingsManager` (which `AppSession` already owns).
