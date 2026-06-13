"""Hardcoded preprocessing recipe constants.

The preprocessing pipeline is a fixed, paper-aligned reproduction that is not
varied between runs or subjects. Rather than carry it as configuration, the
recipe lives here as named constants imported directly by both the offline
(:mod:`backend.offline_phase.preprocessor`) and online
(:mod:`backend.online_phase.online_preprocessor`) preprocessors. A single
source of truth guarantees the two phases cannot drift apart.

These were migrated out of ``experiment_config.yaml`` one block at a time;
see ``docs/plans/minimize_settings_plan.md`` for the sequence. Blocks not yet
migrated (e.g. ``resample_filter_stage``, which still toggles the early/late
pipeline variant) remain in the YAML schema for now.
"""

from __future__ import annotations

# ── Low-pass filter ───────────────────────────────────────────────────────────
# Paper-aligned LP for the 100 Hz target sfreq. IIR keeps offline/online causal
# parity (the streaming side uses scipy.signal.sosfilt, which cannot run a
# zero-phase FIR).
LOWPASS_H_FREQ: float = 40.0
LOWPASS_METHOD: str = "iir"

# ── Final resample ──────────────────────────────────────────────────────────────
# Paper-aligned training/inference sample rate. The online decimation requires
# the LSL input rate to be an integer multiple of this.
FINAL_RESAMPLE_RATE: int = 100
