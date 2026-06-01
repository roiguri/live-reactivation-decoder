# Per-Class Overlay Diagnostic + Semester-A Reproduction (Hypothetical)

## Context

The current offline-vs-online diagnostic in
[scripts/offline_inference_check.py](../scripts/offline_inference_check.py)
produces 4 composite comparison PNGs (offline | online side-by-side) plus 2
CV graphs. Quantitatively the streaming-path comparison on
BindingDecoding001 shows shape correlations 0.98–0.99 between the offline
and online mean curves, with diagonal-dominance 6/6 on both paths. The one
notable amplitude divergence is the living_room decoder dropping from
0.721 (offline) to 0.562 (online) — a 0.16 drop the correlation metric
doesn't penalize.

But the **side-by-side composite layout makes it hard to eyeball
differences**. The eye has to jump between two panels per decoder per
plot type. An overlay-on-same-axes design with one plot per stimulus
class is much more direct: the offline and online mean curves either
sit on top of each other or they don't.

Separately, we want a hypothetical plan for **cross-subject validation
using semester-A's data** (assuming we obtain the raw VHDRs), so we have
a published reference number to compare our offline preprocessing against.

This plan has two phases plus a doc-publishing step:

0. **Publish the plan to `docs/`** as a tracked planning document.
1. **Improve the existing diagnostic** with the new per-class overlay
   design. Doable now, on the data we already have.
2. **Semester-A reproduction with hypothetical raw VHDRs** to validate
   offline preprocessing cross-subject + retest the streaming path on
   a different subject. Blocked on data we don't have today.

---

# Phase 0 — Publish plan to docs/ and commit

1. Save this plan verbatim (content from `# Per-Class Overlay…` through end
   of file) to
   [docs/Per_Class_Overlay_And_Semester_A_Plan.md](./Per_Class_Overlay_And_Semester_A_Plan.md)
   so the team has a tracked planning doc covering both phases. No edits to
   content — just write the same text to that file.
2. Commit just that file before any code changes, so the plan is anchored
   in git history before the implementation lands.
   - Suggested message:
     `docs: add per-class overlay diagnostic + semester-A reproduction plan`
   - Single file: `docs/Per_Class_Overlay_And_Semester_A_Plan.md`
   - Direct commit on `main` (no branch) is fine since it's a docs-only
     change with no production impact.

---

# Phase 1 — Per-Class Overlay Diagnostic (executable now)

## What changes

**Replace** the 4 composite comparison PNGs with **6 separate per-class PNGs**.
**Keep** the 2 CV evaluation PNGs (`cv_auc_curves.png`, `cv_tgm_heatmaps.png`)
unchanged.

### New output

```
debug_snapshots/plots/offline_sanity_check/
  comparison_red.png             ← red decoder on red trials, offline vs online
  comparison_green.png
  comparison_yellow.png
  comparison_living_room.png
  comparison_bathroom.png
  comparison_kitchen.png
  cv_auc_curves.png              ← unchanged
  cv_tgm_heatmaps.png            ← unchanged
```

### What each per-class PNG shows

For class `X` (e.g. `red`):
- **X axis**: time from marker (s), tmin to tmax (default -0.2 to 1.0)
- **Y axis**: P(positive), fixed range [0, 1]
- **Offline mean curve** (e.g. navy blue), bold line, averaged across all
  N trials where marker X was presented, from the offline-preprocessed
  epochs through the X decoder
- **Offline ±SEM band**, light navy, `std / sqrt(N)` per timepoint
- **Online mean curve** (e.g. crimson), bold line, same computation but
  trajectories built by streaming through `OnlinePreprocessor`
- **Online ±SEM band**, light crimson
- **Vertical line at t=0** (marker onset), black dotted
- **Vertical dashed line at the X decoder's trained timepoint**, black dashed
- **Horizontal line at 0.5** (chance), gray
- **Title**: `"{class} decoder on {class} trials  (n={N_trials} trials | trained tp {tp:.2f}s)"`
- **Legend** top-right: "offline" / "online"
- **No per-trial faint lines** — only means + bands

### Removed from the script

- `render_comparison_individual_trials` (replaced by new design)
- `render_comparison_marker_overlay`
- `render_comparison_decoder_overlay`
- `render_comparison_decoder_overlay_zscore`
- Side-by-side composite builders: `_make_two_column_composite`,
  `_annotate_composite`, `_hide_unused_panels`
- Panel drawers no longer used: `_draw_individual_epochs_panel`,
  `_draw_marker_overlay_panel`, `_draw_decoder_overlay_panel`
- Z-score machinery: `compute_decoder_baselines`, `zscore_epoched`,
  `compute_offline_baselines_from_prestim`, `_compute_shared_zscore_ylim`
- `--skip-online` offline-only fallback (`render_offline_only_grid`) — keep
  the flag but in fallback mode just render the 6 per-class PNGs with the
  online side omitted (no second curve, no SEM band for online).

### Kept unchanged

- `build_offline_trajectories`, `build_online_trajectories` (trajectory
  builders — the data is fine, only the rendering changes)
- `render_cv_auc_curves`, `render_cv_tgm_heatmaps`
- CLI: `--mode` not needed; existing `--skip-online`, `--skip-cv`, `--raw`,
  `--tmin`, `--tmax`, `--out-dir` all preserved

### New function

`render_per_class_overlay(class_name, decoder_task_name, offline_trials,
online_trials, t_grid, trained_tp, out_path)` — ~40 lines. Single matplotlib
figure, two `plot` + two `fill_between` calls, decorations, save.

Called 6 times in main, once per class in `markers_of_interest`. The
decoder→class mapping comes from `task_to_marker` (already built).

### File-level impact

[scripts/offline_inference_check.py](../scripts/offline_inference_check.py)
goes from 1031 lines to ~500 lines (we delete more than we add).

## Verification (Phase 1)

1. `python scripts/offline_inference_check.py` from repo root produces 8 PNGs
   (6 comparison_*.png + 2 cv_*.png). Old 4 composite files are NOT in the
   output directory. Stale files from prior runs can be cleaned manually.
2. `comparison_red.png` shows: offline mean rising sharply to ~0.81 near
   0.17s, online mean rising to ~0.76 near the same time, both bands tight
   and overlapping. Visual takeaway in one second.
3. `comparison_living_room.png` shows the documented amplitude gap:
   offline mean reaching ~0.72 near 0.31s, online mean only reaching ~0.56.
   This is the case where mean-curve correlation hid an amplitude drop;
   the new design makes the gap obvious.
4. `cv_auc_curves.png` and `cv_tgm_heatmaps.png` are byte-identical to
   prior runs (they're driven by `eval_done.joblib`, no changes to
   their renderers).
5. `--skip-online --skip-cv` produces only the 6 `comparison_*.png`
   with offline curve + band visible, online curve absent. Used for
   fast iteration on plot styling.

## Critical files (Phase 1)

- **MODIFIED** [scripts/offline_inference_check.py](../scripts/offline_inference_check.py)
  — strip 5 renderers + composite helpers + z-score machinery; add
  `render_per_class_overlay`; update `main` to loop over 6 classes
- Reused: `build_offline_trajectories`, `build_online_trajectories` (no change)

---

# Phase 2 — Semester-A Reproduction (hypothetical; blocked on raw data)

## Why

We've validated the streaming path on a single recording. We want
**cross-subject evidence**, and an **external published reference** for the
offline preprocessing quality. Semester A's published result for the
`object_vs_other` decoder is **grand-average peak AUC = 0.582 at 490 ms**
across 7 subjects (LDA lsqr + shrinkage=auto, per-subject 3-fold
StratifiedKFold, `GeneralizingEstimator`) — saved at
[reactivation-decoder/results/group_analyses/LDA_lsqr_shrinkauto_Group_object_LDA_lsqr_shrinkauto_N7_20251218-164020/](file:///home/itaipap/projects/university/reactivation-decoder/results/group_analyses/LDA_lsqr_shrinkauto_Group_object_LDA_lsqr_shrinkauto_N7_20251218-164020/),
including per-subject `.pkl` files with the full diagonal AUC curves and
TGMs that we can load as a reference overlay.

**This phase assumes we obtain the raw VHDRs for those 7 subjects.** We
don't have them today — only preprocessed FIFs, which would not test our
offline preprocessing.

The output is **graphs, not scores**.

## Step 2A — Cross-subject offline-preprocessing sanity

Run OUR offline preprocessor on each subject's raw VHDR, then reproduce
semester-A's per-subject 3-fold CV with our `ModelEvaluator`, then plot.

CV scheme MUST be per-subject 3-fold StratifiedKFold averaged across
subjects (matching semester-A) — NOT pooled. Override `cv.k = 3` in the
new config to match semester-A's `cv_folds=3`.

### Step 2A graphs (saved to `debug_snapshots/plots/semester_a_reproduction/`)

1. **`auc_per_subject_overlay.png`** (headline graph):
   - X axis: time (s), -0.6 to 0.99 (matches semester-A epoch window)
   - Y axis: AUC, 0.40–0.70
   - 7 thin lines (one per subject): our pipeline, slate-blue
   - 1 bold line: our grand-average curve
   - 1 thick dashed line: semester-A grand-average curve (loaded from .pkl files)
   - Vertical marker at 490 ms (semester-A peak time)
   - Horizontal dotted line at chance (0.5)
   - Title: "object-vs-other CV (k=3) — our pipeline vs semester-A reference"

2. **`auc_grand_avg_with_band.png`** (clean version):
   - Bold our grand-average + shaded ±SEM band across 7 subjects
   - Reference dashed line for semester-A
   - For writeups; (1) is for diagnostic eyeballing

3. **`tgm_grand_average.png`**:
   - Side-by-side: our grand-average TGM | semester-A grand-average TGM
   - Same `imshow / RdBu_r / vmin=0.3 / vmax=0.7 / crosshair` pattern as
     `cv_tgm_heatmaps.png`

4. **`tgm_per_subject_grid.png`**:
   - 7 panels, one TGM per subject — diagnostic for which subjects
     succeeded vs which struggled

## Step 2B — Per-subject offline-vs-online streaming test

Pick the best Step-2A CV performer. Train a `object_vs_other` decoder
at that subject's CV-peak timepoint. Run the (now overlay-style) Phase-1
diagnostic on that subject's raw VHDR.

Output: 1 `comparison_object.png` (overlay-style, the new Phase-1 design)
+ `cv_auc_curves.png` (single panel, 1 decoder) + `cv_tgm_heatmaps.png`
(single panel). Saved to `debug_snapshots/plots/semester_a_subject_<n>_comparison/`.

This re-confirms streaming-path quality on a different recording with a
freshly-trained decoder, ruling out that the BindingDecoding001 result
was a single-subject artifact.

## Files to create (Phase 2)

- **NEW** `tests/notebooks/semester_a_cv_reproduction.ipynb` — Step-2A as a
  notebook (matches the existing `validate_live_inference_epoched.ipynb` /
  `validate_preproc_parity.ipynb` pattern). Cells:
  1. Imports + paths (raw-dir, reference-results, config, out-dir)
  2. Load semester-A reference: stack per-subject `.pkl` files →
     reference AUC array + grand-average curve
  3. Loop over subjects: drive `OfflinePreprocessor` per raw VHDR →
     `mne.Epochs`. Cell saves per-subject epochs as intermediate so
     reruns of CV don't redo preprocessing.
  4. Loop over subjects: run `ModelEvaluator.run_evaluation` on
     `object_vs_other` → per-subject `diagonal_auc` + `tgm_matrix`.
  5. Stack across subjects → `ours_per_subject_auc` + `ours_grand_avg`.
  6. Plot `auc_per_subject_overlay` (inline).
  7. Plot `auc_grand_avg_with_band` (inline).
  8. Plot `tgm_grand_average` side-by-side (inline).
  9. Plot `tgm_per_subject_grid` (inline).
  10. Final cell: save all 4 figures to
      `debug_snapshots/plots/semester_a_reproduction/` as PNGs for sharing.
- **NEW** `experiment_config.semester_a.yaml` — tracked config matching
  semester-A protocol: same `preprocessing` block as
  `experiment_config.full.yaml`, single `object_vs_other` decoder task,
  `markers_mapping.events` reflecting semester-A codes
  (31=object, 32=feature, 33=scene, 41=retrieval, 51=binding, 61=baseline).
- **MODIFIED (small)** [scripts/offline_inference_check.py](../scripts/offline_inference_check.py)
  — add `--markers` CLI override so we can run with semester-A markers
  instead of FL stimuli (~10 lines on top of Phase-1 changes). Used by
  Step 2B via subprocess from a follow-up notebook cell (or directly from
  the terminal — same script either way).

## Verification (Phase 2)

1. `auc_per_subject_overlay.png` displays. Bold (ours) and dashed
   (semester-A) lines within ±0.03 across the epoch, peaks within ±50 ms
   near 490 ms. Bottom line: **does the bold line track the dashed line?**
2. Fail mode: bold line consistently 0.05+ below dashed line → our
   preprocessing is degrading signal. Inspect thin per-subject lines to
   find which subjects diverge most.
3. Step 2B: `comparison_object.png` shows offline and online mean curves
   that overlay closely, mirroring the BindingDecoding001 result.
4. No regression on Phase 1: re-running `offline_inference_check.py` on
   BindingDecoding001 after the `--markers` CLI addition still produces
   the same 6 `comparison_*.png` + 2 CV PNGs.

## Assumptions

- We obtain raw VHDRs for the 7 semester-A subjects. None of these
  directories I checked has them; only preprocessed FIFs exist today.
- Semester-A's preprocessing recipe may differ from ours in non-obvious
  ways (different `l_freq`, ICA settings, epoch window). Worth verifying
  their config matches ours before claiming apples-to-apples; if it
  differs, the AUC delta isn't purely about streaming-path quality.

## Critical files (Phase 2)

- **NEW** [tests/notebooks/semester_a_cv_reproduction.ipynb](../tests/notebooks/semester_a_cv_reproduction.ipynb)
- **NEW** [experiment_config.semester_a.yaml](../experiment_config.semester_a.yaml)
- **MODIFIED** [scripts/offline_inference_check.py](../scripts/offline_inference_check.py)
  — `--markers` flag (after Phase-1 rewrite); Step 2B calls it as a
  subprocess
- Reference: [reactivation-decoder/results/group_analyses/LDA_lsqr_shrinkauto_Group_object_LDA_lsqr_shrinkauto_N7_20251218-164020/data/](file:///home/itaipap/projects/university/reactivation-decoder/results/group_analyses/LDA_lsqr_shrinkauto_Group_object_LDA_lsqr_shrinkauto_N7_20251218-164020/data/)
- Reused: [src/backend/offline_phase/preprocessor.py](../src/backend/offline_phase/preprocessor.py),
  [src/backend/offline_phase/evaluator.py](../src/backend/offline_phase/evaluator.py)
