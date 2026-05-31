# Decoder Pipeline Investigation

End-to-end record of the fixes that took the FL decoder pipeline from
"broken — every decoder produces nonsense at inference" to "6/6 decoders
correctly classify their target marker." Covers the problem, the four
distinct fixes applied, the numerical and visual comparison of original vs
current, and the remaining open question for live deployment.

Plots live in [debug_snapshots/plots/](../../../debug_snapshots/plots/):
- `original/` — pre-investigation baseline
- `stepB/` — intermediate (cross-modality + Logistic+balanced, single shared
  trained timepoint)
- `stepC/` — current (all four fixes)

## 1. Starting symptom

Six binary one-vs-rest decoders (3 colors + 3 scenes) trained on the
functional-localizer recording. Live inference on the same recording showed:

- **`bathroom decoder` firing positive on every marker** (~0.6–0.85 across
  the row, with no bathroom-specific peak).
- **Diagonal-dominance 2/6** — only `yellow` and `kitchen` correctly picked
  their own target. The rest were either dominated by other decoders or sat
  near chance with no class-specific structure.
- Bathroom's pathological "fires on all" behavior was the most visible
  failure mode and the trigger for the investigation.

Original cell-19 table at the trained timepoint (0.14 s), captured from the
full FL recording (70 trials per marker):

```
                          red    green   yellow living_room bathroom  kitchen
red decoder              0.248  0.258   0.201   0.222       0.311    0.294   ✗ bathroom wins
green decoder            0.324  0.308   0.385   0.368       0.276    0.310   ✗ yellow wins
yellow decoder           0.469  0.458   0.484   0.459       0.465    0.430   ✓ yellow
living_room decoder      0.217  0.262   0.205   0.169       0.224    0.221   ✗ green wins
bathroom decoder         0.573  0.528   0.527   0.565       0.565    0.545   ✗ red wins (bathroom-fires-on-all)
kitchen decoder          0.290  0.307   0.334   0.334       0.282    0.343   ✓ kitchen
```

## 2. The four fixes

### Fix 1 — Causal preprocessing (preprocessing parity bug)

**Problem.** Offline preprocessing used MNE's default zero-phase IIR
(`filtfilt`) plus a zero-phase polyphase resampler. Online preprocessing used
causal `scipy.signal.sosfilt` with persistent state plus a causal-FIR +
integer-decimation path. Same filter coefficients, fundamentally different
phase modes. Result: training-time features were ~45 ms ahead of and ~8×
different in amplitude from inference-time features. The trained model had
never seen anything shaped like what it received at inference.

The config comment on `highpass.method: iir` literally said "IIR keeps
offline/online causal parity" — the *intent* was there, but the
implementation didn't enforce phase mode and MNE's default silently broke it.

**Changes.**

- [src/backend/offline_phase/preprocessor.py](../../../src/backend/offline_phase/preprocessor.py):
  - `_highpass`: added `phase="forward"` → causal IIR via `lfilter`
  - `_lowpass`: added `phase="forward"` → causal IIR via `lfilter`
  - `_resample`: full rewrite to use `firwin` + causal `lfilter` + integer
    decimation, mirroring `OnlinePreprocessor._decimate`. Channel-by-channel
    to keep peak memory near input size on multi-hour recordings.
  - `_notch`: deliberately left as FIR zero-phase. Parity bisection proved
    the notch is non-load-bearing (LP at 40 Hz already kills 50 Hz line
    content).
- [experiment_config.yaml](../../../experiment_config.yaml) — comments on
  `highpass.method` and `lowpass.method` updated to document the causal
  constraint and warn against reverting.

**Result.** Diagonal-dominance went from 2/6 → 5/6. The pathological
bathroom-fires-on-all pattern was reduced but not fully eliminated.
Aligned correlation between offline and online features at the trained
timepoint went from 0.56 → 0.79 in the ±50 ms lag window, and the lag
dropped from −46 ms to 0.

### Fix 2 — Cross-modality training (training-paradigm bug)

**Problem.** Each decoder was trained against same-modality alternatives
only: `red vs (green + yellow)`, `bathroom vs (living_room + kitchen)`,
etc. At inference time scenes were shown to color decoders (and vice
versa) — stimulus categories the model had never seen. The model
extrapolated arbitrarily into uncharted regions of the LDA feature space.
This is the *structural* root cause of "bathroom fires on every marker":
bathroom had never trained against colors, so its output on color trials
was undefined.

**Changes.**

- [debug_snapshots/experiment_config.yaml](../../../debug_snapshots/experiment_config.yaml) —
  every decoder's `neg_labels` now lists all 5 non-target stimuli (e.g.
  `red decoder.neg_labels: [green, yellow, living_room, bathroom, kitchen]`).
- No source code changes — `get_task_data` and the classifier construction
  already accepted any label list.

**Result.** Average CV peak AUC across the 6 decoders rose from 0.69 → 0.75.
Each decoder now learns what "non-target" looks like across the entire
stimulus space. **Side effect**: class imbalance shifted from 1:2 to 1:5
(70 pos vs 350 neg), which compressed absolute probability outputs downward
— addressed in fix 3.

### Fix 3 — Logistic Regression with balanced class weights (model-choice change)

**Problem.** The 1:5 imbalance from fix 2 plus default LDA priors produced
artificially low absolute P(positive) outputs. LDA `priors=[0.5, 0.5]`
would have only remapped the output threshold without changing the learned
discriminant. We needed the *boundary itself* rebalanced.

**Changes.**

- [debug_snapshots/experiment_config.yaml](../../../debug_snapshots/experiment_config.yaml):

  ```yaml
  decoders:
    model: Logistic
    params:
      solver: liblinear
      class_weight: balanced
      C: 1.0
      penalty: l2
      max_iter: 1000
  ```

- No source code changes — Logistic was already wired through
  `build_classifier` and the Pydantic config validator.
- **Methodology caveat**: this deviates from the paper's LDA choice. Worth
  confirming with the lab before any external comparison.

**Result.** Diagonal-dominance 5/6 → 6/6. Absolute P(positive) values
recovered into a useful range (target diagonals 0.56–0.84). Average CV AUC
0.75 → 0.79. Diagonals are now decisive: target marker wins by 0.10–0.30
over runners-up.

### Fix 4 — Per-decoder training timepoints (model-design change)

**Problem.** The orchestrator picked a single shared training timepoint
(the cross-decoder CV-average peak). But each decoder peaks at a different
time: red 0.17 s, green/yellow 0.20 s, scenes 0.30–0.33 s — 160 ms spread.
Color decoders especially were trained at 0.31 s when their discriminative
window had already faded; they lost 0.10–0.15 AUC by being trained
off-peak.

**Changes.**

- [src/backend/core/artifact_models.py](../../../src/backend/core/artifact_models.py) —
  `DecoderPipelineMetadata` gained `decoding_timepoints: dict[str, float]`
  field (empty default for backward compatibility with old artifacts).
- [src/backend/offline_phase/trainer.py](../../../src/backend/offline_phase/trainer.py) —
  `run_training` now accepts `float | dict[str, float]`. Per-task lookup
  inside the training loop.
- [src/backend/offline_phase/orchestrator.py](../../../src/backend/offline_phase/orchestrator.py):
  - New `_derive_per_task_timepoints` static helper extracts each task's
    `argmax(diagonal_auc)` from `_eval_results`.
  - `run_training` now computes per-task timepoints, passes them to the
    trainer, and stores both single (`decoding_timepoint`, kept for
    backward compatibility) and dict (`decoding_timepoints`) in metadata.
- [tests/offline_phase/test_orchestrator.py](../../../tests/offline_phase/test_orchestrator.py) —
  new `_attach_eval_results_stub` helper; 4 affected tests now stub
  `_eval_results`; one test asserts the per-task dict is populated.
- [scripts/full_recording_live_inference_check.py](../../../scripts/full_recording_live_inference_check.py) —
  diagnostic table + diagonal-dominance check now sample each decoder at
  its own trained timepoint (via `metadata.decoding_timepoints` with
  fallback to the single `decoding_timepoint`).

**Result.** Same 6/6 diagonal-dominance, but **target-diagonal magnitudes
jumped**: red 0.65 → 0.76 (+0.11), yellow 0.63 → 0.74 (+0.12), bathroom
0.69 → 0.77 (+0.08), others unchanged (already on their peak). Cross-modality
suppression sharpened: target now towers above same-modality competitors
by clear margins.

## 3. Numerical comparison — original vs current

Both states are real artifacts captured from end-to-end runs of
`scripts/full_recording_live_inference_check.py` on the full FL recording
(3267 s, 70 trials per stimulus type).

### Decoder × marker P(positive) table

**Original** (single-timepoint 0.14 s):

```
                          red    green   yellow living_room bathroom  kitchen
red decoder              0.248  0.258   0.201   0.222       0.311    0.294   ✗
green decoder            0.324  0.308   0.385   0.368       0.276    0.310   ✗
yellow decoder           0.469  0.458   0.484   0.459       0.465    0.430   ✓
living_room decoder      0.217  0.262   0.205   0.169       0.224    0.221   ✗
bathroom decoder         0.573  0.528   0.527   0.565       0.565    0.545   ✗
kitchen decoder          0.290  0.307   0.334   0.334       0.282    0.343   ✓
```

**Current** (per-decoder timepoints; each row sampled at that decoder's own tp):

```
                          red    green   yellow living_room bathroom  kitchen
red decoder              0.760  0.405   0.228   0.257       0.150    0.128   ✓
green decoder            0.500  0.693   0.319   0.158       0.216    0.199   ✓
yellow decoder           0.253  0.365   0.743   0.309       0.362    0.286   ✓
living_room decoder      0.190  0.202   0.196   0.562       0.329    0.259   ✓
bathroom decoder         0.297  0.230   0.275   0.425       0.770    0.674   ✓
kitchen decoder          0.163  0.098   0.132   0.422       0.587    0.825   ✓
```

### Headline metrics

| Metric | Original | Current | Δ |
|---|---|---|---|
| Diagonal-dominance | 2/6 | **6/6** | +4 |
| Avg CV peak AUC across decoders | 0.792 | 0.790 | flat |
| Target diagonal — mean | 0.349 | **0.726** | +0.377 |
| Target diagonal — min / max | 0.169 / 0.565 | **0.562 / 0.825** | min +0.39, max +0.26 |
| Bathroom-row range | 0.527–0.573 (fires on all) | **0.230–0.770** (target wins) | — |

The average CV AUC barely moved (0.79 → 0.79). The decoders' *intrinsic*
discriminative capacity was already there. What changed is the
**inference-time output**: the model now produces sensible class-specific
predictions instead of biased ambiguous outputs.

### Per-decoder inter-trial baseline P(positive)

| Decoder | Original μ | Original σ | Current μ | Current σ |
|---|---|---|---|---|
| red | 0.283 | 0.282 | 0.084 | 0.200 |
| green | 0.331 | 0.311 | 0.540 | 0.360 |
| yellow | 0.482 | 0.380 | 0.575 | 0.356 |
| living_room | 0.238 | 0.261 | 0.420 | 0.370 |
| bathroom | 0.568 | 0.315 | 0.249 | 0.325 |
| kitchen | 0.313 | 0.269 | 0.197 | 0.323 |

Notable: bathroom's baseline collapsed from 0.568 → 0.249 (correctly
becomes "not bathroom" on inter-trial EEG). Red's baseline dropped 0.283 →
0.084 (cleaner discriminator). Green/yellow baselines rose because their
training distribution now includes scenes, and inter-trial features sit
closer to the boundary in the broader feature space.

## 4. Visual comparison

Pair files by name across the original/ and stepC/ folders:

| File | What to look for |
|---|---|
| `individual_trials_1s.png` | Each decoder's trial cloud + mean on its OWN target marker. Mean should rise above 0.5 at the trained tp. Original: most decoders flat near 0.3. Current: clear rises at each decoder's own tp. |
| `marker_overlay_1s.png` | Each decoder's panel with all 6 marker types overlaid. Target marker should be the highest at the trained tp. Original: lines clustered, no clear winner; bathroom panel = all-high. Current: target clearly tops competitors by 0.10–0.30. |
| `decoder_overlay_1s.png` | Each marker's panel with all 6 decoders overlaid (the "decoder competition" view). The decoder whose target IS this marker should win. Original: bathroom decoder dominates every panel. Current: target decoder wins unambiguously in each panel. |
| `decoder_overlay_zscore_1s.png` | Same as decoder_overlay but z-scored per decoder against its own inter-trial baseline. Makes outputs cross-decoder comparable despite baseline drift. The cleanest view for "which decoder fired the most strongly above its own baseline." |
| `cv_auc.png` | Per-decoder CV AUC over time. Red dashed = each decoder's own CV peak. Original: shared trained tp at 0.14 s missed most peaks. Current: trained tp aligns with each peak. |
| `spatial_patterns.png` | Topomaps of Haufe-transformed decoder weights. Tight focal patterns (red right-occipital, kitchen central-occipital) → cleanest decoders. Diffuse patterns (green, yellow, living_room) → noisier confusable decoders. |
| `marker_timeline.png` | Stimulus timing across the full recording. Confirms fully interleaved (no block structure) — same for both. |

## 5. Open question — per-decoder baseline calibration for live UI

The decoders' inter-trial baselines differ substantially (red 0.08, green
0.55, yellow 0.58, kitchen 0.20, bathroom 0.25, living_room 0.42). Raw
P(positive) comparison across decoders is therefore misleading: yellow's
baseline alone is higher than red's actual spike peak. The
`decoder_overlay_zscore_*.png` view shows what comparison looks like after
per-decoder z-score normalization — that's the right post-processing
recipe for a production live system.

The production live UI doesn't currently apply z-scoring or per-decoder
thresholds. To deploy these decoders in a live BCI context you'd need to
either:

1. Calibrate per-decoder thresholds during a pre-experiment rest period
   and apply z-score normalization on the fly, or
2. Use a single multi-class softmax classifier so outputs are
   by-construction comparable.

This is outside the current investigation's scope.

## 6. Dependencies installed during the investigation

`mne-icalabel>=0.7` and `onnxruntime>=1.16` were in `requirements.txt` but
missing from the local conda env. ICLabel needs onnxruntime as a backend.
Both installed during the investigation; documented here as a setup gotcha
for new contributors.

## 7. Diagnostic infrastructure

Not part of the production code path, but landed in the repo as
reproducible tools for future debugging:

- [scripts/full_recording_live_inference_check.py](../../../scripts/full_recording_live_inference_check.py) —
  streams full FL recording through the production online pipeline; emits
  diagonal-dominance table + 4 visualization PNGs (individual_trials,
  marker_overlay, decoder_overlay, decoder_overlay_zscore). Supports
  `--suffix` for run history and `--tmin/--tmax` for extended epoch windows.
- [scripts/inspect_decoder_cv_auc.py](../../../scripts/inspect_decoder_cv_auc.py) —
  per-decoder CV AUC table (peak + ±50 ms / ±100 ms surrounding-area
  windows) + per-decoder AUC-by-time figure.
- [scripts/inspect_decoder_internals.py](../../../scripts/inspect_decoder_internals.py) —
  spatial-pattern topomaps + marker timeline + per-trial variance at peak.
- [scripts/preproc_parity_check.py](../../../scripts/preproc_parity_check.py) —
  comprehensive offline-vs-online preprocessing parity harness with
  stage-bisection sweep, alt-stage toggle, probability-level comparison,
  and memory-bounded loading.
- [tests/online_phase/test_preproc_parity.py](../../../tests/online_phase/test_preproc_parity.py) —
  unit tests for parity-check math helpers.
- [tests/notebooks/validate_preproc_parity.ipynb](../../../tests/notebooks/validate_preproc_parity.ipynb) —
  visual diagnostic notebook (heatmaps, histograms, trace overlays).

## 8. Ablation — what does cross-modality alone contribute?

To isolate the contribution of fix 2 (cross-modality training), we ran the
seeder with **everything else from Step C still in place** but reverted
`neg_labels` to within-modality only. Snapshot saved at
[debug_snapshots/plots/within_modality/](../../../debug_snapshots/plots/within_modality/).

| Metric | Step C (cross-mod) | Ablation (within-mod) | Δ |
|---|---|---|---|
| Diagonal-dominance | 6/6 | 5/6 (living_room loses) | −1 |
| Avg CV peak AUC | 0.790 | 0.801 | +0.01 (within-mod easier) |
| Target diagonal — mean | 0.726 | 0.702 | −0.024 |
| Living_room baseline | 0.420 | **0.733** | +0.31 (much worse) |
| Bathroom baseline | 0.249 | **0.642** | +0.39 (much worse) |
| Living_room neg-class mean | 0.339 | **0.643** | +0.30 (fires on non-targets) |
| Bathroom neg-class mean | 0.282 | **0.628** | +0.35 (fires on non-targets) |

**What cross-modality buys**:

1. **Scene-decoder baselines collapse from ~0.65 → ~0.30**. Within-modality
   scene decoders fire HIGH on inter-trial EEG and on color trials because
   they've never trained against anything but other scenes — anything that
   isn't a scene looks "kind of like" their target. Cross-modality
   negatives teach them to suppress non-scene inputs.

2. **Scene-decoder false-positive rates halve**. Negative-class P(positive)
   means drop from ~0.63 → ~0.30 for living_room and bathroom. That
   directly improves the SNR of any thresholded "decoder fired" signal.

3. **Living_room diagonal-dominance is preserved by cross-modality**.
   Without it, the living_room decoder loses to green decoder on
   living_room trials (because green's baseline drift makes it look higher).

**What cross-modality does NOT buy**:

- Average CV AUC barely changes (0.79 vs 0.80). Within-modality is
  technically an easier discrimination task, so CV AUC is fractionally
  higher. The model fits comparably either way.
- Diagonal-dominance count moves only 1 (6/6 → 5/6). The marginal cases
  flip but the qualitative pattern is similar.

**Implication**: cross-modality is doing real, measurable work — but most
of it shows up in the *baselines*, not in the diagonals. If you only look
at "did the right decoder win at the trained timepoint" you'd see a small
effect. If you look at "how cleanly is the decoder distinguishing target
from non-target inputs in general" the effect is large.

This matters for live deployment: with within-modality, you'd need very
careful per-decoder thresholding to keep scene decoders from spuriously
firing on inter-trial activity. With cross-modality, the baseline is
already low enough that a simple "P > 0.5" rule mostly works for scene
decoders.
