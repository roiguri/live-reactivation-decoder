"""Hardcoded preprocessing recipe constants.

The preprocessing pipeline is a fixed, paper-aligned reproduction that is not
varied between runs or subjects. Rather than carry it as configuration, the
recipe lives here as named constants imported directly by both the offline
(:mod:`backend.offline_phase.preprocessor`) and online
(:mod:`backend.online_phase.online_preprocessor`) preprocessors. A single
source of truth guarantees the two phases cannot drift apart.

These were migrated out of ``experiment_config.yaml`` one block at a time;
see ``docs/plans/minimize_settings_plan.md`` for the sequence. The pipeline
runs a single fixed ordering (LP + decimate before the spatial transforms);
the former ``resample_filter_stage`` early/late toggle has been removed.
"""

from __future__ import annotations

# ── Low-pass filter ───────────────────────────────────────────────────────────
# Paper-aligned LP for the 100 Hz target sfreq. IIR keeps offline/online causal
# parity (the streaming side uses scipy.signal.sosfilt, which cannot run a
# zero-phase FIR).
LOWPASS_H_FREQ: float = 40.0
LOWPASS_METHOD: str = "iir"

# ── ICA + ICLabel (offline only) ────────────────────────────────────────────────
# ``ICA_N_COMPONENTS = None`` lets MNE/infomax decide (rank = n_electrodes - 1
# after average reference). ``ICA_FIT_L_FREQ`` is the high-pass cutoff for the
# ICA-fit copy of the epochs only. ICLabel pre-selects a component for exclusion
# only when its predicted class is a key in ``ICLABEL_REJECT_THRESHOLDS`` *and*
# the model's confidence in that class is >= the per-class threshold (a fraction
# in [0, 1]). A threshold of ``0.0`` rejects the class at any confidence; classes
# absent from the dict — notably "brain" and "other" (ICLabel's low-confidence
# catch-all) — are never auto-suggested, so the operator decides on those
# manually.
ICA_METHOD: str = "infomax"
ICA_EXTENDED: bool = True
ICA_N_COMPONENTS: int | None = None
ICA_FIT_L_FREQ: float = 1.0
ICLABEL_ENABLED: bool = True
ICLABEL_REJECT_THRESHOLDS: dict[str, float] = {
    "muscle artifact": 0.85,
    "eye blink": 0.85,
    "heart beat": 0.0,
    "line noise": 0.80,
    "channel noise": 0.90,
}

# ── Channel hygiene (offline only) ──────────────────────────────────────────────
# Dataset-specific channel fixups applied before filtering: drop the EMG channel,
# rename HEGOC → HEOG, set the hardware montage, and fix the AFz/Afz case mismatch
# in MNE's standard montage. The boolean flags keep their guards so a dev can
# disable a step by flipping the constant.
CHANNEL_DROP_EMG: bool = True
CHANNEL_RENAME_HEGOC_TO_HEOG: bool = True
CHANNEL_MONTAGE_NAME: str = "easycap-M1"
CHANNEL_AFZ_CASE_FIX: bool = True

# ── High-pass filter ────────────────────────────────────────────────────────────
# Paper-aligned HP. IIR keeps offline/online causal parity (matched by the
# streaming side's scipy.signal.sosfilt).
HIGHPASS_L_FREQ: float = 0.1
HIGHPASS_METHOD: str = "iir"

# ── Notch filter ──────────────────────────────────────────────────────────────
# Line-noise notch at the regional mains frequency. ``None`` disables the notch
# entirely (the preprocessors keep a guard for that case).
NOTCH_FREQ: float | None = 50.0

# ── Final resample ──────────────────────────────────────────────────────────────
# Paper-aligned training/inference sample rate. The online decimation requires
# the LSL input rate to be an integer multiple of this.
FINAL_RESAMPLE_RATE: int = 100

# ── Epoching (offline only) ─────────────────────────────────────────────────────
# Epoch window around each stimulus. ``EPOCH_BASELINE = None`` is paper-aligned —
# baseline correction is omitted (set to e.g. (None, 0.0) to re-enable pre-stim
# mean subtraction). Passed directly to ``mne.Epochs``.
EPOCH_TMIN: float = -0.2
EPOCH_TMAX: float = 1.0
EPOCH_BASELINE: tuple[float | None, float | None] | None = None

assert EPOCH_TMIN < EPOCH_TMAX, "EPOCH_TMIN must be less than EPOCH_TMAX"
