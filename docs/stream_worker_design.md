# StreamWorker + LiveStreamSession + PredictionLogger — Backend-Only Plan

## Context

The online-phase backend has three committed components — `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` — but no live runtime that drives them. This PR adds the runtime loop (`StreamWorker`), a live-session lifecycle wrapper (`LiveStreamSession`), and a persistence sink (`PredictionLogger`), plus a factory on `AppSession` so the frontend (partner's PR) can spin them up without knowing the internals.

**Work split:**
- **This PR (backend, you):** `StreamWorker`, `LiveStreamSession`, `PredictionLogger`, `AppSession.build_live_stream_session(...)` factory, headless smoke test, updates to `backend_architecture.md` documenting the contract.
- **Partner's PR (frontend):** `ProbabilityBuffer`, `ProbabilityPlotWidget`, UI lifecycle, button-to-action wiring. They connect to the backend through the documented `AppSession` interface.

The deliverable from this PR is a **contract**: signals + types + threading guarantees that the frontend partner can build against in isolation.

## Scope

**In:**
- `StreamWorker` (QThread orchestrator)
- `PredictionLogger` (CSV sink)
- `LiveStreamSession` (online lifecycle wrapper)
- `AppSession.build_live_stream_session(...)` factory
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
│  Factory: AppSession.build_live_stream_session(...)
│  Returns: LiveStreamSession exposing signal + start/stop
└──────────────────────────────────────────────┘
                       │
              prediction_ready signal
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           FRONTEND (partner's PR)            │
│                                              │
│  live.prediction_ready ─► ProbabilityBuffer  │
│                              │               │
│                              ▼               │
│                       ProbabilityPlotWidget  │
└──────────────────────────────────────────────┘
```

`StreamWorker` owns only the background micro-batch loop. It receives `LSLReceiver`, `OnlinePreprocessor`, and `LiveInferenceEngine` as constructor dependencies and keeps references to use inside `run()`, but it does not create them, start/stop the LSL connection, own logging, or close files. `LiveStreamSession` owns the lifecycle of the composed live run. `PredictionLogger` is one consumer of the worker signal (CSV); the frontend partner attaches other consumers (buffer, plot) via the same signal.

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

error_occurred = pyqtSignal(str)        # worker/runtime failures
latency_ready = pyqtSignal(dict)        # planned hardening: timing diagnostics
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

**Does NOT know about:** artifact loading, config loading, plots, ring buffers, files, task display names, the UI thread, or the start/stop lifecycle of the receiver/logger.

---

## Post-basic validation and hardening

The basic Phase 2 backend flow is considered implemented once replayed LSL data can run through `LiveStreamSession` and produce a prediction CSV. The following items are additions/hardening before experiment use, not prerequisites for the core architecture:

1. **Error surfacing from `StreamWorker`.**
   - Status: implemented in the working tree; commit pending.
   - `StreamWorker.error_occurred = pyqtSignal(str)` reports unrecoverable receiver, preprocessing, batch accumulation, and inference failures.
   - `LiveStreamSession.error_occurred` forwards the worker signal so callers do not reach into worker internals.
   - The worker stops the loop cleanly after emitting the error.
   - Tests: fake receiver/preprocessor/inference failures and malformed receiver payloads emit the error signal and stop the worker instead of silently killing the thread.

2. **Latency/runtime diagnostics.**
   - Track per-batch timings for pull, preprocessing, inference, total batch handling, and optionally signal/log writing.
   - Emit a diagnostics payload such as `latency_ready = pyqtSignal(dict)` or log periodically.
   - Tests: use deterministic fakes or monkeypatched timers to assert diagnostics fields exist and are non-negative.

3. **Real Phase 1 artifact compatibility.**
   - Add an explicit smoke/test path using a `decoder_pipeline.joblib` exported by `OfflineOrchestrator.run_training(...)`, not only a synthetic artifact.
   - Verify that the exported artifact loads through `AppSession.build_live_stream_session(...)`, feature width matches live preprocessor output, and predictions contain finite probabilities for every task.
   - This validates Phase 1 → Phase 2 artifact compatibility, not decoder scientific quality.

4. **Replay and lab validation.**
   - Replay validation: run `scripts/smoke_stream_worker.py` against `scripts/recordings/eeg_recording_with_trigger.xdf` and a compatible artifact; require nonzero rows, monotonic timestamps, expected task headers, and marker propagation.
   - Real lab validation: repeat the smoke against the NeurOne/LSLProxy live stream on the decoding machine.

5. **Decoder quality validation.**
   - Out of scope for this backend plan. The backend can validate finite probabilities, task keys, shape, timestamps, and class labels; scientific decoder quality needs separate model/performance criteria.

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

### 3. `LiveStreamSession` + `AppSession.build_live_stream_session(...)`

**File:** `online_decoder/src/backend/session.py` (extend existing `AppSession`)

`AppSession` is the app-level orchestrator of orchestrators. The frontend imports only `AppSession`; `AppSession` owns `SettingsManager`, creates `session.offline` for Phase 1, and exposes `build_live_stream_session(...)` directly for Phase 2. Do not introduce a separate `OnlinePhase` namespace/class.

`LiveStreamSession` is the object returned by the factory. It owns the lifecycle of one composed live run: start the receiver, start the worker, stop the worker, wait for the thread, close the optional logger, and stop the receiver. It exposes the worker's `prediction_ready` and `error_occurred` signals as properties, but the frontend should not need to reach into the worker for normal use.

**Proposed interface:**
```python
@dataclass
class LiveStreamSession:
    """One composed live decoding run. Frontend connects to .prediction_ready."""
    _receiver: LSLReceiver
    _worker: StreamWorker
    _logger: PredictionLogger | None

    @property
    def prediction_ready(self):
        """Forward the worker signal without exposing worker internals."""
        return self._worker.prediction_ready

    @property
    def error_occurred(self):
        """Forward worker runtime errors without exposing worker internals."""
        return self._worker.error_occurred

    @property
    def latency_ready(self):
        """Forward worker runtime diagnostics without exposing worker internals."""
        return self._worker.latency_ready

    def start(self) -> None:
        """Start receiver and worker. Idempotent."""

    def stop(self) -> None:
        """Stop worker, wait for join, close logger, stop receiver. Idempotent."""


class AppSession:
    offline: OfflineOrchestrator | None

    def build_live_stream_session(
        self,
        decoder_pipeline_path: Path,
        log_path: Path | None = None,
        batch_size_samples: int = 40,
    ) -> LiveStreamSession:
        """Construct the full backend pipeline, wired but not started.

        decoder_pipeline_path: path to decoder_pipeline.joblib (Phase 1 export).
        log_path: if provided, a PredictionLogger is created and connected. If None, no logging.
        batch_size_samples: passed to StreamWorker.

        Returns a live session the caller (frontend) connects signals to and then calls .start().
        """
```

The factory:
1. Loads `DecoderPipelineArtifact` from `decoder_pipeline_path`.
2. Constructs `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` from artifact + settings.
3. Constructs `StreamWorker`.
4. If `log_path`: constructs `PredictionLogger` and `worker.prediction_ready.connect(logger.on_predictions)`.
5. Returns `LiveStreamSession`. Does NOT call `.start()` — the caller does that after attaching its own consumers.

**Why `LiveStreamSession` instead of just returning the worker:** the frontend needs a signal plus `start()`/`stop()`, but shutdown also needs `worker.wait()`, optional `logger.close()`, and `receiver.stop()`. Keeping that lifecycle in `LiveStreamSession` avoids pushing backend internals into the frontend and avoids making `StreamWorker` responsible for artifact loading, logging, or external resource cleanup.

---

## Frontend integration contract (goes into `backend_architecture.md`)

This section is the exact thing the frontend partner builds against. Copy this verbatim into the doc.

> ### Backend → Frontend contract: live decoder output
>
> **Entry point:** `AppSession.build_live_stream_session(decoder_pipeline_path, log_path=None, batch_size_samples=40) -> LiveStreamSession`
>
> **Lifecycle:**
> 1. Frontend calls `build_live_stream_session(...)` to get a `LiveStreamSession`.
> 2. Frontend connects its UI-side slots to `live.prediction_ready`, `live.error_occurred`, and optionally `live.latency_ready`.
> 3. Frontend calls `live.start()`.
> 4. On shutdown (e.g., window close), frontend calls `live.stop()`.
>
> **Signals:**
> - `StreamWorker.prediction_ready = pyqtSignal(dict, np.ndarray, list)`, exposed as `live.prediction_ready`.
> - `StreamWorker.error_occurred = pyqtSignal(str)`, exposed as `live.error_occurred`.
> - `StreamWorker.latency_ready = pyqtSignal(dict)`, exposed as `live.latency_ready`.
>
> If `live.error_occurred` fires, the worker loop has exited but external resources are still owned by `LiveStreamSession`; caller code should still call `live.stop()` to close the logger and stop the receiver.
>
> **Payload:**
> - `predictions: dict[str, np.ndarray]` — keys are task names (model identifiers from the loaded `DecoderPipelineArtifact`); values are `float32` arrays of shape `(n_rows,)`. Each value is `P(class=1)` for that task at that timestamp.
> - `timestamps: np.ndarray` — `float64`, shape `(n_rows,)`, LSL clock seconds. Monotonic non-decreasing across emissions.
> - `markers: list[tuple[float, int]]` — `(timestamp, trigger_code)`. Timestamps are per-sample-accurate on the LSL clock. Codes match the trigger codes documented in `experiment_config.yaml`.
> - `latency_ready` payload: dictionary with millisecond timing keys `pull_ms`, `accumulation_ms`, `preprocessing_ms`, `inference_ms`, `emit_ms`, `total_ms`, plus `input_samples`, `emitted_rows`, `marker_count`, and `pending_samples`.
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
> - Wiring code that calls `build_live_stream_session(...)`, connects slots, and manages `start()/stop()`.
>
> **What the frontend should NOT do:**
> - Reach into `LiveStreamSession` private members or the underlying `StreamWorker`. Use `prediction_ready`, `error_occurred`, `latency_ready`, and `start()/stop()` only.
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

**Compatibility:** Nothing else in the committed codebase consumes `pull_new_data()` directly (`StreamWorker` is the intended consumer), so this is a contained break.

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
└── session.py                     (extend AppSession with build_live_stream_session and LiveStreamSession)

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
| `LiveStreamSession` | Lifecycle unit | Fakes for receiver/worker/logger; assert start/stop order and idempotence. |
| `AppSession.build_live_stream_session` | Construction smoke | Patch runtime dependencies, assert returned session exposes `prediction_ready` and `.stop()` is idempotent. |
| End-to-end | Opt-in smoke | `scripts/smoke_stream_worker.py` plays back a recorded session against a fake LSL inlet; assert CSV row count matches expected after N seconds. |
| Error handling | Unit/Qt integration | Fake receiver/preprocessor/inference exceptions and malformed receiver payloads; assert `error_occurred` emits and worker stops cleanly. Implemented in working tree; commit pending. |
| Latency diagnostics | Unit/Qt integration | Deterministic timers/fakes; assert latency payload fields exist and values are non-negative. |
| Phase 1 artifact compatibility | Smoke/integration | Use a real `decoder_pipeline.joblib` exported by `OfflineOrchestrator`; assert finite predictions for every task. |

Tests live alongside existing tests; smoke script follows the `scripts/smoke_test_lsl_receiver.py` convention.

## Verification

1. `pytest online_decoder/tests/online_phase/ -v` → unit and Qt-integration tests green.
2. `python online_decoder/scripts/smoke_stream_worker.py --duration 5 --log /tmp/smoke.csv` → runs without UI, produces a CSV. Manually inspect:
   - Header row matches loaded model task names.
   - Row count ≈ duration × target_sfreq.
   - Timestamps are monotonic.
   - At least one `marker_code` is populated when the replayed stream contains triggers.
3. Synthetic-artifact replay smoke is acceptable for plumbing validation only; real Phase 1 artifact smoke is still required before experiment use.
4. Read-only check: `backend_architecture.md` "Backend → Frontend contract" section is present, accurate, and the partner can implement against it without reading any backend source.

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

### Commit 4 — `feat(session): add live stream session factory`

**Files:**
- `online_decoder/src/backend/session.py` (extend `AppSession` directly)
- `online_decoder/tests/test_session_live_stream.py`

**Implementation:**
- [x] `@dataclass class LiveStreamSession` with private members `_receiver`, `_worker`, optional `_logger`, `prediction_ready` property, and idempotent `start()`, `stop()`.
- [x] Add `AppSession.build_live_stream_session(decoder_pipeline_path, log_path=None, batch_size_samples=40) -> LiveStreamSession`.
- [x] Do not add `OnlinePhase`; do not expose `session.online`.
- [x] In `AppSession.build_live_stream_session`:
  - [x] Load `DecoderPipelineArtifact` from `decoder_pipeline_path`.
  - [x] Construct `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine` from artifact + `SettingsManager` settings.
  - [x] Construct `StreamWorker` with injected receiver/preprocessor/inference dependencies.
  - [x] If `log_path`: construct `PredictionLogger`; `worker.prediction_ready.connect(logger.on_predictions)`.
  - [x] Return `LiveStreamSession`. **Do not call `start()`.**
- [x] `live.start()`: `receiver.start()` then `worker.start()`. Idempotent.
- [x] `live.stop()`: `worker.stop(); worker.wait(); logger.close() if logger; receiver.stop()`. Idempotent.

**Tests:**
- [x] Patch `LSLReceiver.start` so no real LSL connection is attempted; construct via `session.build_live_stream_session(...)` and assert `prediction_ready` is exposed.
- [x] With `log_path=None`: assert no logger slot is connected to `prediction_ready`.
- [x] With `log_path` set: assert the logger slot is connected and receives emitted predictions.
- [x] `live.start()` then `live.start()` again — no duplicate start calls.
- [x] `live.stop()` then `live.stop()` again — no duplicate stop/close calls.

**TODOs in code:**
- [x] `# TODO(open): Avoid unnecessary disk reload when Phase 1 already has an in-memory DecoderPipelineArtifact...` at the artifact load site.
- [x] `# TODO(open): Stop hardcoding default LSLReceiver settings once Phase 2 config defines stream name/type...` at the receiver construction site.

**Verify:**
- [x] `pytest tests/online_phase tests/test_session_live_stream.py -q` green.

**Commit:**
- [x] `git commit -m "feat(session): add live stream session factory"`

---

### Commit 5 — `feat(scripts): add headless StreamWorker smoke script`

**Files:**
- `online_decoder/scripts/smoke_stream_worker.py` (NEW)

**Implementation:**
- [x] CLI args: `--duration` (seconds, default 5), `--log` (CSV path, default `/tmp/smoke.csv`), `--pipeline` (path to `decoder_pipeline.joblib`), plus config/stream/replay options needed for headless use.
- [x] Construct `AppSession`, call `session.build_live_stream_session(pipeline_path, log_path)`.
- [x] `live.start()`; sleep for `duration`; `live.stop()`.
- [x] Print row count from log file and timestamp monotonicity check at the end.
- [x] Follow the existing `scripts/smoke_test_lsl_receiver.py` conventions (argparse, optional replay subprocess, explicit summary).

**Tests:** none (smoke script).

**TODOs in code:** none.

**Verify:**
- [x] `python scripts/smoke_stream_worker.py --help` validates imports and CLI parsing.
- [ ] `python online_decoder/scripts/smoke_stream_worker.py --duration 5 --log /tmp/smoke.csv --pipeline <path>` against a fake/replayed LSL stream → CSV produced, row count ≈ `duration × target_sfreq`, header matches task names, timestamps monotonic, ≥1 marker code present when triggers were replayed.

**Commit:**
- [x] `git commit -m "feat(scripts): add headless StreamWorker smoke script"`

---

### Commit 6 — `docs(backend_architecture): document StreamWorker + frontend contract`

**Files:**
- `online_decoder/docs/backend_architecture.md` (update)
- `online_decoder/docs/Phase2_Implementation_Plan.md` (check off completed items)
- `online_decoder/CLAUDE.md` (update Current Backend Scope)

**Implementation:**
- [x] Replace the tentative StreamWorker sketch in [backend_architecture.md:1136-1202](online_decoder/docs/backend_architecture.md#L1136-L1202) with the final spec from `stream_worker_design.md` §1 (constructor, signal, loop body summary, threading guarantees).
- [x] Add a new section `## Backend → Frontend contract: live decoder output` after the StreamWorker section, copying the contract block from `stream_worker_design.md`.
- [x] In `Phase2_Implementation_Plan.md`: tick off StreamWorker, add and tick PredictionLogger and `AppSession.build_live_stream_session(...)`.
- [x] In `CLAUDE.md`: update "Current Backend Scope" — StreamWorker, LiveStreamSession, PredictionLogger, and `AppSession.build_live_stream_session(...)` are the current backend scope; remove "next planned" note for StreamWorker.

**Tests:** none (documentation only).

**TODOs in code:** none.

**Verify:**
- [ ] Cold-read the new `backend_architecture.md` contract section as if you were the frontend partner — can you implement against it without opening any backend `src/` file? If yes, ship.

**Commit:**
- [ ] `git commit -m "docs(backend_architecture): document StreamWorker + frontend contract"`

---

### Commit 7 — `feat(online_phase): surface StreamWorker runtime errors`

**Files:**
- `online_decoder/src/backend/online_phase/stream_worker.py`
- `online_decoder/src/backend/session.py`
- `online_decoder/tests/online_phase/test_stream_worker.py`
- `online_decoder/tests/test_session_live_stream.py`
- `online_decoder/docs/backend_architecture.md` (contract/status update)

**Implementation:**
- [x] Add `error_occurred = pyqtSignal(str)` to `StreamWorker`.
- [x] Forward `error_occurred` through `LiveStreamSession`.
- [x] Wrap receiver pull, preprocessing, batch accumulation, and inference in runtime error handling.
- [x] Emit a concise error message that identifies the failing stage.
- [x] Stop the loop cleanly after emitting the error.
- [x] Preserve current `prediction_ready` payload/API.

**Tests:**
- [x] Fake receiver raises; assert `error_occurred` emits and worker exits.
- [x] Malformed receiver payload raises during batch accumulation; assert `error_occurred` emits and worker exits.
- [x] Fake preprocessor raises; assert `error_occurred` emits and worker exits.
- [x] Fake inference engine raises; assert `error_occurred` emits and worker exits.
- [x] Session tests assert `LiveStreamSession.error_occurred` forwards the worker signal.

**Commit:**
- [ ] `git commit -m "feat: surface StreamWorker runtime errors"`

---

### Commit 8 — `feat: add StreamWorker latency diagnostics`

**Files:**
- `online_decoder/src/backend/online_phase/stream_worker.py`
- `online_decoder/tests/online_phase/test_stream_worker.py`
- `online_decoder/docs/backend_architecture.md` (contract/status update)

**Implementation:**
- [x] Add diagnostics signal or logging hook, e.g. `latency_ready = pyqtSignal(dict)`.
- [x] Track per-batch timing for preprocessing, inference, and total batch handling.
- [x] Include batch size and emitted row count in the diagnostics payload.
- [x] Keep diagnostics optional for consumers; do not alter `prediction_ready`.

**Tests:**
- [x] Assert diagnostics emit for processed batches.
- [x] Assert payload includes stable keys and non-negative timing values.

**Commit:**
- [ ] `git commit -m "feat: add StreamWorker latency diagnostics"`

---

### Commit 9 — `test(phase2): validate real Phase 1 artifact compatibility`

**Files:**
- `online_decoder/tests/` (exact location TBD)
- Optional fixture/helper under `online_decoder/tests/data/` if a small committed artifact is acceptable
- `online_decoder/docs/Phase2_Implementation_Plan.md` / this plan (status update)

**Implementation:**
- [ ] Produce or locate a real `decoder_pipeline.joblib` exported by `OfflineOrchestrator.run_training(...)`.
- [ ] Run it through `AppSession.build_live_stream_session(...)` with replayed LSL data or a deterministic fake receiver.
- [ ] Assert finite predictions for every model task.
- [ ] Assert feature width compatibility between online preprocessor output and loaded decoder models.

**Tests/Verify:**
- [ ] Automated compatibility test if a small artifact can be generated cheaply.
- [ ] Otherwise, documented smoke command with artifact path and expected CSV checks.

**Commit:**
- [ ] `git commit -m "test(phase2): validate Phase 1 artifact compatibility"`

---

### After commit 9

- [ ] Push branch, open PR for review.
- [ ] Notify partner that `AppSession.build_live_stream_session(...)` is ready and `docs/backend_architecture.md` describes the contract.

## Resolved decisions

1. **Marker timestamp resolution.** Chosen: modify `LSLReceiver` + `split_eeg_and_markers` to return per-sample-accurate `(timestamp, code)` tuples. See the "Modifications to existing code" section above for the exact changes. Tradeoff accepted: small break to a committed interface in exchange for clean per-sample timing in the worker output and a simpler contract for the frontend.
2. **Feature shape into `LiveInferenceEngine.predict()`.** Chosen: **channels-are-features.** The preprocessor's `(n_out, n_channels)` output is passed directly to `predict()` — `feature_width == n_channels`. No feature-extraction helper needed in the worker; `_build_features(out_eeg)` collapses to `features = out_eeg`. (The worker may skip the indirection entirely and call `inference_engine.predict(out_eeg)` directly.)

## Open decisions (please push back where you disagree)

1. **`PredictionLogger` marker-matching tolerance.** The logger matches markers to rows by nearest timestamp. Either pass `target_sfreq` as a constructor arg (tolerance = `0.5 / target_sfreq`), or hardcode a tolerance like `0.005` s (= 5 ms). Default in this plan: pass `target_sfreq` for correctness across configs.
2. **Should the factory accept an in-memory artifact instead of a path?** `AppSession` may already hold a loaded artifact in memory from a Phase 1 run. Two overloads, or always reload from disk? Default in this plan: always load from disk path for simplicity.
3. **Where does Phase 2 settings live?** The factory needs LSL stream name, target_sfreq, etc. Are these in `experiment_config.yaml` (read by `SettingsManager`) or passed as factory args? Default in this plan: factory reads from `SettingsManager` (which `AppSession` already owns).
