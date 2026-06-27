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


class TestChannelHygiene:
    def test_drop_emg(self):
        assert pc.CHANNEL_DROP_EMG is True

    def test_rename_hegoc_to_heog(self):
        assert pc.CHANNEL_RENAME_HEGOC_TO_HEOG is True

    def test_montage_name(self):
        assert pc.CHANNEL_MONTAGE_NAME == "easycap-M1"

    def test_afz_case_fix(self):
        assert pc.CHANNEL_AFZ_CASE_FIX is True


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


class TestIca:
    def test_method(self):
        assert pc.ICA_METHOD == "infomax"

    def test_extended(self):
        assert pc.ICA_EXTENDED is True

    def test_n_components_auto(self):
        assert pc.ICA_N_COMPONENTS is None

    def test_fit_l_freq(self):
        assert pc.ICA_FIT_L_FREQ == 1.0


class TestIclabel:
    def test_enabled(self):
        assert pc.ICLABEL_ENABLED is True

    def test_reject_thresholds(self):
        assert pc.ICLABEL_REJECT_THRESHOLDS == {
            "muscle artifact": 0.85,
            "eye blink": 0.85,
            "heart beat": 0.0,
            "line noise": 0.80,
            "channel noise": 0.90,
        }

    def test_reject_thresholds_in_unit_range(self):
        assert all(0.0 <= t <= 1.0 for t in pc.ICLABEL_REJECT_THRESHOLDS.values())


class TestEpochs:
    def test_tmin(self):
        assert pc.EPOCH_TMIN == -0.2

    def test_tmax(self):
        assert pc.EPOCH_TMAX == 1.0

    def test_baseline_is_none(self):
        assert pc.EPOCH_BASELINE is None

    def test_tmin_below_tmax(self):
        assert pc.EPOCH_TMIN < pc.EPOCH_TMAX
