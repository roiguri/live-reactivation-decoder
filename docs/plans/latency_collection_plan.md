# Live-pipeline latency collection for the project report

## Goal

Produce the end-to-end (E2E) latency numbers the report's results chapter needs:
across all recordings for all three subjects, report the pipeline latency from
"last raw sample acquired" to "decision produced", with a stage breakdown
(buffering floor + causal preprocessing + inference).

## What already exists

The online path already computes latency per micro-batch and emits it on
`StreamWorker.latency_ready` (~25×/s), re-exposed as
`LiveStreamSession.latency_ready` (`src/backend/online_phase/stream_worker.py`):

- **`sample_to_decision_ms`** — the true E2E latency:
  `local_clock() - (batch_end_ts + time_correction)`, i.e. last raw sample in
  the batch → decision, corrected for the LSL clock offset. **This is the
  headline number.**
- **`total_ms`** — compute-only pipeline time (`pull + accumulation + preprocess
  + inference + emit`), plus the per-stage fields `pull_ms`, `accumulation_ms`,
  `preprocessing_ms`, `inference_ms`, `emit_ms`.

Today this data is **not persisted** — it only feeds the header's rolling label
and a DEBUG log summary of `total_ms`. So there is nothing on disk to aggregate.

E2E latency is a **real-time property**: `sample_to_decision_ms` is only
meaningful when the recording is replayed through the actual online path at real
time (LSLReceiver → StreamWorker → preprocessor → inference). A fast offline
batch loop cannot produce it.

## Approach (scripts-only; no `src/` change)

Reuses the existing real-time replay (`scripts/replay_vhdr_to_lsl.py`) and the
headless `LiveStreamSession` construction pattern from
`scripts/smoke_stream_worker.py`. Latency is captured by subscribing to the
already-emitted `latency_ready` signal with a `DirectConnection` (the same
cross-thread pattern the `LiveSessionLogger` prediction sink uses), so no
production code changes are needed.

### 1. Capture — `scripts/collect_latency.py`

Per recording (auto-discovered as every directory containing a `.vhdr` under a
subject folder — robust to the `functinal_localizer`/`functional_localizer`/
`task`/`phase2` naming differences):

1. Spawn `replay_vhdr_to_lsl.py <dir> --no-repeat` as a child process
   (real-time NeurOne-like LSL publish that ends with the recording).
2. Build `AppSession → build_live_stream_session(...)` against that subject's
   `models/decoder_pipeline.joblib` and its own `experiment_config.yaml`.
3. Attach a `latency_ready` consumer that appends every payload as a row to a
   per-recording `<recording>_latency.csv`.
4. Run until the replay child exits (or `--max-seconds` for a smoke run), drain
   briefly, then `live.stop()`.

CSV columns (full payload, nothing dropped):
`wall_time, sample_to_decision_ms, total_ms, pull_ms, accumulation_ms,
preprocessing_ms, inference_ms, emit_ms, pending_samples, marker_count`.

Output layout: `<out-root>/<subject>/<recording>_latency.csv`
(default out-root `docs/project_docs/latency`).

### 2. Aggregate — `scripts/summarize_latency.py`

Reads every `*_latency.csv`, groups by subject, and reports for
`sample_to_decision_ms` (headline) and `total_ms` + stage breakdown:

- **Per subject**: n_batches, minutes replayed, mean, median, p95, max.
- **Pooled (all subjects)**: same stats → the report's headline sentence.
- **Breakdown**: mean buffering floor (batch_size / sfreq ≈ 40 ms), mean
  `preprocessing_ms`, mean `inference_ms`, mean queue/`emit_ms`.

Writes `<out-root>/latency_summary.csv` + prints tables to stdout.

### 3. Report text

A short paragraph + table in the results chapter, framed honestly (see caveats).

## Caveats to state in the report

- Replay reproduces the **software** path (buffering + causal preprocessing +
  inference + queueing) but **not** amplifier acquisition/transport latency —
  replayed timestamps are generated at push time. It is pipeline-E2E, not
  electrode-to-decision.
- There is a hard **~40 ms floor**: a decision waits for a full 40-sample batch
  at 1000 Hz. This is the buffering term in the breakdown.
- Latency is steady-state / content-independent, so numbers are stable across a
  recording.

## Verification

1. **Smoke** (fast): run the harness on one subject with `--max-seconds 20` and
   confirm rows are written and `sample_to_decision_ms` is populated (non-empty).
2. **Full sweep** (~2 h wall-clock, real-time; must run on **Windows**): all FL +
   task recordings, all 3 subjects. Command provided after smoke passes.
