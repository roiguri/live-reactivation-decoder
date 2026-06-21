"""Pin the hardcoded preprocessing recipe constants.

These values were previously carried in ``experiment_config.yaml`` and are now
fixed in ``backend.core.preprocessing_constants``. The asserts here guard
against an accidental value change during the block-by-block migration (see
``docs/plans/minimize_settings_plan.md``): the constants must keep matching the
recipe that produced the shipped decoder artifact.
"""

from __future__ import annotations

from backend.core import preprocessing_constants as pc


class TestLowpass:
    def test_h_freq(self):
        assert pc.LOWPASS_H_FREQ == 40.0

    def test_method(self):
        assert pc.LOWPASS_METHOD == "iir"


class TestHighpass:
    def test_l_freq(self):
        assert pc.HIGHPASS_L_FREQ == 0.1

    def test_method(self):
        assert pc.HIGHPASS_METHOD == "iir"


class TestNotch:
    def test_freq(self):
        assert pc.NOTCH_FREQ == 50.0


class TestFinalResample:
    def test_rate(self):
        assert pc.FINAL_RESAMPLE_RATE == 100


class TestEpochs:
    def test_tmin(self):
        assert pc.EPOCH_TMIN == -0.2

    def test_tmax(self):
        assert pc.EPOCH_TMAX == 1.0

    def test_baseline_is_none(self):
        assert pc.EPOCH_BASELINE is None

    def test_tmin_below_tmax(self):
        assert pc.EPOCH_TMIN < pc.EPOCH_TMAX
