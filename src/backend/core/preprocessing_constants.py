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
