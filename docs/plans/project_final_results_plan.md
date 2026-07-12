# Report analysis: 3-step decoding validation (offline CV, online-pipeline validation, cross-domain generalization)

## Context

The project report's "Analysis of results" / "Conclusions" chapters are unwritten. Live/grounded inspection of saved outputs (`data/sub_001/live_summary.joblib`, `debug_snapshots/animacy_fl/held_out_{encoding,retrieval}_summary.joblib`) showed:

- The FL functional-localizer decoder has a genuine, honest cross-validated AUC (~0.73–0.76 for sub_001) that clears the work plan's own bar (chance+20pp) — this is a real result, not overfitting.
- The only "generalization" evidence currently produced (encoding/retrieval `diag_at_tp` / mean P(t) trajectories) comes from replaying the *already-frozen, FL-trained* decoder live on encoding/retrieval recordings — there is no proper held-out **AUC** for that question, and no test at all of whether the **online (causal) pipeline itself** degrades performance relative to offline processing.
- `save_run_summary`'s `peak_auc`/`diagonal_auc` fields are copied from the FL-only CV (`plots.cv_auc(ctx)`) into every source's summary regardless of `SOURCE` — i.e. encoding/retrieval summaries currently carry FL's numbers, not their own.

This plan builds three independent, complementary analyses to replace that gap, each answering a distinct question the report needs:

1. **Step 1 — Pure offline**: is the category information decodable at all, under best-case (offline) processing? (Already-existing machinery — collect/report it properly.)
2. **Step 2 — Online pipeline validation**: does switching to the real causal/streaming pipeline lose any of that decodability? (Goal defined below; implementation deliberately left open.)
3. **Step 3 — Cross-domain generalization**: does a decoder trained purely on FL generalize to encoding/retrieval — and at what latency, via a cross-domain TGM? (New code — the actual scientific question: reactivation detection.)

Scope: build all three analysis capabilities and validate them end-to-end (smoke-tested on `debug_snapshots/animacy_fl` and run for real on `data/sub_001`). Running the *existing* fl/encoding/retrieval notebook flow to fill sub_002/sub_003's missing summary files is explicitly **out of scope** — steps 1–3 read directly from each subject's `epochs/`, `models/`, and raw recording directories, so they don't depend on those older saved summaries and can run on any of the 3 subjects directly once built.

## Step 1 — Pure offline (no new analysis code)

Reuse `ModelEvaluator.run_evaluation()` (`src/backend/offline_phase/evaluator.py`) via the existing `plots.cv_auc(ctx)` wrapper (`tests/notebooks/analysis/analysis_lib/plots.py:80`) per subject, per decoder task:

- `diagonal_auc`, `peak_auc`, `peak_timepoint`, full `tgm_matrix` (train-time × test-time, from `GeneralizingEstimator` + `cross_val_multiscore`, 5-fold `StratifiedKFold`).
- Spatial-pattern topomap: reuse `ModelTrainer._calculate_spatial_patterns` output (`src/backend/offline_phase/trainer.py:100`) — already computed during Phase 1 training and already rendered by the GUI; surface it in the notebook rather than recomputing.

No new production code. Just a notebook section that runs this per subject and collects the numbers into one place for the report.

## Step 2 — Online pipeline validation

**Goal**: determine whether the real-time (causal, streaming) signal-processing pipeline loses any of the decodability that step 1 established under offline (best-case) processing. This is the work plan's own requirement — "the accuracy (AUC) of the online decoder during simulation with pre-recorded data should not degrade by more than 10% compared to the purely offline performance on the same data partition" — turned into an actual measurement instead of an assumption.

Concretely: for data step 1 already showed to be decodable, is that decodability preserved when features are computed the way the live system actually computes them (causal filters, micro-batching, frozen-ICA replay) instead of the way step 1's offline evaluation computes them? The result is a degradation figure — how much AUC is lost, if any — attributable specifically to the online signal-processing pipeline, not to the classifier or to which trials were tested.

**Methodology (implemented).** Deployment-faithful, paired: on the same FL trials and the same CV folds, fit the decoder on **offline** features, then score the held-out trials twice — once offline-processed (best-case reference + internal sanity anchor), once online-processed (as-deployed). Only the test-time processing differs, so the AUC gap is attributable purely to the pipeline. The frozen ICA/interp/bad-channel matrices come from each subject's Phase-1 artifact (`models/decoder_pipeline.joblib` → `online_state`) and are identical on both sides, so ICA is never a confound.

**Interval classes reproduced online.** A decoder's negative class can include `intervals:` classes (rest windows) that exist only as offline synthetic epochs — no trigger marker. Rather than exclude them (which would drop to the harder image-vs-image contrast), every offline trial is given an online counterpart so the CV partition matches step 1's exactly (standard `StratifiedKFold`, all classes in train and test): stimulus trials pair to their online trigger marker by nearest onset; marker-less interval trials are epoched from the continuous online feature stream at their own offline onset (legitimately online-pipeline-processed features for those baseline windows). Because rest separates easily from stimulus, the AUC matches step 1's rest-inflated number — intentional, since the work-plan bar is degradation on the *same data partition*. Read the result as "the causal pipeline preserves decodability on step 1's partition," not as a category-discrimination claim; the headline is the offline-vs-online *delta* on identical trials.

**Pairing** is by marker class + onset in seconds (never sample index — offline is 100 Hz decimated, online is native ~1000 Hz, `first_samp` differs), nearest-onset within a class.

**Outputs (per subject):** the deployment operating-point degradation at the decoder's trained timepoint (paired AUC offline vs online, absolute/relative drop, bootstrap CI, pass/fail vs the 10% bar), an offline-vs-online diagonal-AUC-over-time overlay (exposes latency shift), and the cross-pipeline TGM (train-offline-time × test-online-time).

**Files (implemented):**
- `tests/notebooks/analysis/analysis_lib/streaming.py` — added `make_epocher_multichannel` (multi-channel sibling of `make_epocher`).
- `tests/notebooks/analysis/analysis_lib/online_validation.py` (new) — `build_paired_features` (all trials paired, incl. interval classes epoched online at their onset), `cross_pipeline_tgm`, `operating_point_degradation` (standard `StratifiedKFold`).
- `tests/notebooks/analysis/report_online_validation.ipynb` (new) — per-subject run + tables/figures, uses `context.load_context`.
- `tests/notebooks/analysis/test_analysis_lib.py` — synthetic unit tests for the multi-channel epocher and the cross-pipeline metrics.

## Step 3 — Cross-domain generalization (new code)

**New module**: `tests/notebooks/analysis/analysis_lib/cross_domain.py`, plus additions to `sources.py`.

**Feature extraction for encoding/retrieval**: `sources.py`'s existing `build_encoding_epochs`/`build_retrieval_epochs` epoch the *decoder's output probability* stream, not raw channel features — not usable for fitting/testing a fresh `GeneralizingEstimator`. Add `build_encoding_features`/`build_retrieval_features` (same file, reusing `_extract_task_markers`, `task_labels.encoding_trials`/`retrieval_trials`) that epoch the **multi-channel** online-replayed feature stream instead of `preds`, producing `X_test (n_trials, n_channels, n_grid)` + `y_test` per category. This needs a new epocher — generalize `streaming.make_epocher`/`_interp_epoch` (currently interpolates a 1-D probability array) to interpolate an `(n_channels, n_time)` array per trial, e.g. `streaming.make_epocher_multichannel`.

**Cross-domain TGM**: no CV needed (retrieval/encoding trials were never seen during FL training, so there's no leakage to guard against by holding out FL data):

1. `X_fl, y_fl = get_task_data(fl_epochs, task_cfg)` — all of FL.
2. `ge = GeneralizingEstimator(build_classifier(settings), scoring="roc_auc", n_jobs=-1); ge.fit(X_fl, y_fl)` — one fit, all FL timepoints.
3. `scores = ge.score(X_test, y_test)` → `(n_fl_times, n_test_times)` matrix: FL train-timepoint × encoding/retrieval test-timepoint AUC. This is the cross-domain equivalent of step 1's TGM — extends the same King & Dehaene TGM framing the report's theoretical background already uses, and lets a later/smeared retrieval-phase effect show up even if it doesn't align with FL's own peak latency.
4. Run this separately for `source="encoding"` and `source="retrieval"`.

**Significance**: permutation test on the test-side labels only (no refitting — reuse `ge`'s already-fitted per-timepoint estimators' predicted probabilities, computed once, then shuffle `y_test` and recompute `roc_auc_score` from the cached probabilities many times) → null distribution per (train_t, test_t) cell or just at the operating point (FL peak tp × best test-time). Report observed score + permutation p-value + bootstrap CI (resample test trials) at the headline cell, alongside the full matrix for the figure.

## Cross-subject aggregation (steps 1 and 3; step 2 is per-subject only)

With n=3 subjects, group-level parametric stats are not meaningful — structure results per the earlier discussion:

- **Per-subject, inferential**: report each subject's AUC + permutation p-value individually for steps 1 and 3 (trial counts per subject are large enough — dozens to ~60 per class — for this to be well-powered).
- **Across-subject, descriptive**: grand-average curves/TGMs (thin per-subject lines + bold mean), and a table of the 3 peak-AUC/p-value pairs — framed explicitly as "consistency across subjects," not a powered group test.
- **Across-subject, combined significance (step 3 only)**: Stouffer's method — combine the 3 subjects' per-subject z-scores (from their permutation p-values) into one meta z-score/p-value. Standard small-N technique for "do independent results converge" without pretending a powered group design exists.
- **Step 2**: purely an engineering-validation number, per subject — not a scientific claim needing cross-subject inference. Exact reporting shape depends on the implementation, once designed.

## Files

Covers steps 1 and 3 only — step 2 has no implementation yet.

- New: `tests/notebooks/analysis/analysis_lib/cross_domain.py` (step 3)
- Modified: `tests/notebooks/analysis/analysis_lib/streaming.py` — add multi-channel epoching (`make_epocher_multichannel` or a `channels=` option on the existing epocher)
- Modified: `tests/notebooks/analysis/analysis_lib/sources.py` — add `build_encoding_features`/`build_retrieval_features`
- New notebook: `tests/notebooks/analysis/report_validation.ipynb` — runs steps 1 and 3 per subject (`data/sub_001`, `sub_002`, `sub_003`) and produces the aggregated tables/figures for the report
- New tests: extend `tests/notebooks/analysis/test_analysis_lib.py` (or a sibling `test_cross_domain.py`) with small-synthetic-array unit tests for the new pure-logic pieces (multi-channel epocher shape/interpolation correctness, permutation-test null shape/p-value bounds), matching the existing style (e.g. `test_make_epocher_grid_and_shape`, `test_perm_band_shapes`)

## Verification

Covers steps 1 and 3 only — step 2 has no implementation yet.

1. **Unit tests**: `pytest tests/notebooks/analysis/` — new synthetic-data tests for the multi-channel epocher and permutation test pass alongside existing `analysis_lib` tests.
2. **Smoke test**: run the new notebook against `debug_snapshots/animacy_fl` (small, fast, already seeded) end-to-end — confirm step 1's CV-AUC matches the previously-seen ~0.73 figure, and step 3 produces a correctly-shaped `(n_fl_times, n_test_times)` matrix with p-values in [0, 1].
3. **Real run**: run the notebook against `data/sub_001`, `data/sub_002`, `data/sub_003` to produce the actual per-subject and aggregated numbers/figures for the report's results chapter.
