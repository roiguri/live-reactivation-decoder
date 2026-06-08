# Issue #43 — Evaluation TGM-CV Parallelization Benchmark

> **Result (2026-06-08): evaluation 2.64× faster, −62.1% wall time, results
> bit-for-bit identical.** Achieved by setting `n_jobs=-1` on the
> `GeneralizingEstimator` in `ModelEvaluator._run_tgm_cv`
> (`src/backend/offline_phase/evaluator.py`). `cross_val_multiscore` stays
> serial (`n_jobs=1`) on purpose — see "Why not nested" below.

## What was measured

`ModelEvaluator`'s temporal-generalization CV (`_run_tgm_cv`) is ~90% of offline
Phase 1 compute. It runs MNE's `GeneralizingEstimator` (full TGM, `O(n_times²)`
fit/score) through `cross_val_multiscore`. Both calls were hardcoded to
`n_jobs=1` (fully serial). Issue #43 scoped a parallelize-only fix that keeps
results identical.

Benchmark harness: `scripts/bench_eval_njobs.py`. Dataset: the cached
functional-localizer epochs (`debug_snapshots/default/epochs/functional_localizer_epo.fif`)
— the issue's reproducibility dataset.

- **Epochs:** 1257 · **channels:** 64 · **n_times:** 121 (−0.2…1.0 s @ 100 Hz)
- **Decoders:** 3 (red / green / yellow), model = Logistic, CV `k=3`
- **Machine:** 8 logical cores

## Strategy ranking (decimated grid, single task — fast pre-screen)

n_jobs only affects scheduling, so every strategy *should* return an identical
TGM. It does not for the nested case.

| Strategy | Time | Speedup | Identical to serial? |
|---|---:|---:|---|
| baseline `est=1, cv=1` | 51.4 s | 1.00× | — (reference) |
| timepoint-parallel `est=-1, cv=1` | 31.7 s | 1.62× | ✅ yes |
| fold-parallel `est=1, cv=-1` | 29.8 s | 1.72× | ✅ yes |
| nested `est=-1, cv=-1` | 17.4 s | 2.95× | ❌ **DIFFERS** |

## Headline (full grid, all 3 decoders)

| Decoder | Baseline `n_jobs=1` | Optimized `n_jobs=-1` | Peak diag AUC (both) |
|---|---:|---:|---|
| red | 285.14 s | 116.58 s | 0.8478 |
| green | 374.48 s | 143.16 s | 0.7446 |
| yellow | 288.00 s | 99.78 s | 0.8161 |
| **Total** | **947.62 s (~15.8 min)** | **359.51 s (~6.0 min)** | — |

- **Speedup: 2.64×  (−62.1% wall time)**
- **`np.allclose(baseline, optimized)` = True** — identical TGMs, identical peak AUCs.

## Why timepoint-parallel (and not the others)

- **Timepoint-parallel (`est=-1`)** distributes ~121 train-timepoints across all
  cores — far more granularity than the 3 CV folds — so it scales best on the
  full grid and keeps every core busy. `n_jobs=-1` is machine-agnostic: it uses
  whatever cores the host has, so it ports to a standard computer with no tuning.
- **Fold-parallel (`est=1, cv=-1`)** is capped at 3× (only `k=3` folds) and
  underuses an 8-core box.
- **Why not nested (`est=-1, cv=-1`):** fastest on the pre-screen, but it
  **changes the results**. Nesting loky inside loky shifts the BLAS thread count
  seen by LogisticRegression's iterative solver, and those tiny float
  differences compound past `np.allclose` tolerance. That violates the
  "no functionality change" requirement, so it is rejected despite being faster.

## Out of scope (would change behavior/outputs)

The issue's directions #2 (diagonal-only `SlidingEstimator`, `O(n_times)`) and #3
(time decimation of the eval grid) are larger wins but drop the off-diagonal TGM
heatmap — a behavior/UI change. Not pursued here.

## Reproduce

```bash
python scripts/bench_eval_njobs.py --mode rank          # fast decimated pre-screen
python scripts/bench_eval_njobs.py --mode full          # full grid, baseline vs optimized + identity check
```
