# Decoder Pipeline — Problem, Fixes, and Results

Single source of truth for the offline→online decoder parity investigation:
why trained decoders produced nonsense at live inference, the four fixes, and
the before/after numbers. Consolidates the former
`Decoder Pipeline Investigation.md` (knowledge base).

## 1. The problem

Six binary one-vs-rest decoders (3 colors + 3 scenes) were trained on the
functional-localizer (FL) recording. Offline they looked healthy —
**CV AUC ≈ 0.79** — but live inference on the *same* recording was broken:

- The **`bathroom` decoder fired positive on every marker** (~0.53–0.57
  across the row, no bathroom-specific peak).
- **Diagonal-dominance 2/6** — only `yellow` and `kitchen` picked their own
  target; the rest were dominated by other decoders or sat at chance.

Decoder × marker P(positive) at the shared trained timepoint (0.14 s):

```
                     red    green  yellow living_rm bathroom kitchen
red decoder         0.248  0.258  0.201   0.222     0.311   0.294   ✗ bathroom wins
green decoder       0.324  0.308  0.385   0.368     0.276   0.310   ✗ yellow wins
yellow decoder      0.469  0.458  0.484   0.459     0.465   0.430   ✓
living_room decoder 0.217  0.262  0.205   0.169     0.224   0.221   ✗ green wins
bathroom decoder    0.573  0.528  0.527   0.565     0.565   0.545   ✗ fires-on-all
kitchen decoder     0.290  0.307  0.334   0.334     0.282   0.343   ✓
```

The key insight: CV AUC was fine, so the decoders *could* discriminate. The
failure was a **train-time vs. inference-time mismatch** — four stacked bugs.

## 2. The four fixes

### Fix 1 — Causal preprocessing parity (the code fix)
Offline preprocessing used MNE's default **zero-phase** filtering (`filtfilt`)
plus a zero-phase polyphase resampler; online used **causal** `sosfilt` +
causal-FIR decimation. Same coefficients, opposite phase mode → training
features were ~45 ms time-shifted and ~8× different in amplitude from
inference features. The model had never seen anything shaped like its input.

Changes in `src/backend/offline_phase/preprocessor.py`:
- `_highpass` / `_lowpass`: `phase="forward"` → causal IIR via `lfilter`.
- `_resample`: rewritten to `firwin` + causal `lfilter` + integer decimation,
  mirroring `OnlinePreprocessor._decimate` (channel-by-channel to bound memory
  on multi-hour recordings).
- `_notch`: left as FIR zero-phase — parity bisection proved it
  non-load-bearing (the 40 Hz low-pass already removes 50 Hz line content).

→ **Diagonal-dominance 2/6 → 5/6**; aligned offline/online feature
correlation at the trained tp rose 0.56 → 0.79, lag −46 ms → 0.

### Fix 2 — Cross-modality training (config)
Each decoder trained only against same-modality alternatives
(`red vs green+yellow`). At inference, scenes were shown to color decoders —
a category they'd **never trained against** → undefined extrapolation. This is
the structural root of "bathroom fires on everything."
→ `neg_labels` now lists all 5 other stimuli per decoder.
→ **Avg CV AUC 0.69 → 0.75** (but created a 1:5 class imbalance).

### Fix 3 — Logistic + balanced class weights (config)
The 1:5 imbalance plus default LDA priors squashed probability outputs.
Switched to Logistic regression with `class_weight: balanced` (rebalances the
decision boundary itself, not just the threshold).
→ **Diagonal-dominance 5/6 → 6/6**; target diagonals recovered to 0.56–0.84.
⚠️ Deviates from the paper's LDA — confirm with the lab before external comparison.

### Fix 4 — Per-decoder training timepoints (code)
The orchestrator trained every decoder at one shared CV-peak time, but
decoders peak at different latencies (red 0.17 s, scenes 0.30–0.33 s — a
160 ms spread); color decoders were trained off-peak.
- `artifact_models.py`: `DecoderPipelineMetadata.decoding_timepoints: dict[str,float]`.
- `trainer.py`: `run_training` accepts `float | dict[str,float]`.
- `orchestrator.py`: `_derive_per_task_timepoints` extracts each task's
  `argmax(diagonal_auc)`; stores both single and per-task metadata.
→ Same 6/6, but target magnitudes jumped (red 0.65→0.76, yellow 0.63→0.74).

## 3. Before / after

Decoder × marker P(positive), current state (each row at its own trained tp):

```
                     red    green  yellow living_rm bathroom kitchen
red decoder         0.760  0.405  0.228   0.257     0.150   0.128   ✓
green decoder       0.500  0.693  0.319   0.158     0.216   0.199   ✓
yellow decoder      0.253  0.365  0.743   0.309     0.362   0.286   ✓
living_room decoder 0.190  0.202  0.196   0.562     0.329   0.259   ✓
bathroom decoder    0.297  0.230  0.275   0.425     0.770   0.674   ✓
kitchen decoder     0.163  0.098  0.132   0.422     0.587   0.825   ✓
```

| Metric | Before | After |
|---|---|---|
| Diagonal-dominance | 2/6 | **6/6** |
| Target-diagonal mean | 0.349 | **0.726** |
| Target-diagonal min / max | 0.169 / 0.565 | **0.562 / 0.825** |
| Avg CV AUC | 0.792 | 0.790 (flat) |

The flat CV AUC is the whole point: **the decoders were always capable; the
bugs corrupted inference-time output, not model capacity.**

## 4. Ablation — what cross-modality (fix 2) buys

Reverting only `neg_labels` to within-modality (everything else kept):

| Metric | Cross-mod | Within-mod | Δ |
|---|---|---|---|
| Diagonal-dominance | 6/6 | 5/6 | −1 |
| Avg CV AUC | 0.790 | 0.801 | +0.01 (within-mod easier) |
| living_room baseline | 0.420 | **0.733** | +0.31 |
| bathroom baseline | 0.249 | **0.642** | +0.39 |

Most of cross-modality's benefit is in the **baselines**, not the diagonals:
it teaches scene decoders to suppress non-scene inputs (their resting
P(positive) collapses ~0.65 → ~0.30), halving false positives. Look only at
"did the right decoder win" and you'd see a small effect; look at "how cleanly
target is separated from non-target in general" and the effect is large.

## 5. Open question — per-decoder baseline calibration

Inter-trial baselines differ a lot (red 0.08, yellow 0.58, living_room 0.42).
Raw cross-decoder P(positive) comparison is therefore misleading. A production
live UI needs either (1) per-decoder z-scoring against a rest-period baseline,
or (2) a single multi-class softmax so outputs are comparable by construction.
Not yet implemented.

## 6. Reproducible tooling

| Tool | Purpose |
|---|---|
| `scripts/offline_inference_check.py` | Per-class offline-vs-online overlay diagnostic (offline+online trajectories + CV AUC/TGM plots) → `debug_snapshots/plots/offline_sanity_check/` |
| `tests/notebooks/validate_live_inference_epoched.ipynb` | Runs the real online inference path on a recording and epochs the probability stream by marker (the "does inference make sense?" view that surfaced the bugs) |

**Setup gotcha:** `mne-icalabel` needs a backend (`onnxruntime`, or `torch` for
newer versions); install CPU-only torch on machines without CUDA.

## 7. Possible future work — cross-subject reproduction

A natural follow-up (not started) is to run our offline pipeline on a second
cohort to validate the offline preprocessing cross-subject and compare against
an external published reference (e.g. semester-A's `object_vs_other`
grand-average peak AUC ≈ 0.582 @ 490 ms). This was scoped but deliberately
shelved; revisit only if cross-subject validation becomes a priority.
