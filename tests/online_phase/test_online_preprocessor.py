"""
Unit tests for OnlinePreprocessor.

Tests are organized by commit stage. Each stage builds on the previous.
Private methods are tested directly where possible to isolate failures.
"""

from __future__ import annotations

import copy

import mne
import numpy as np
import pytest

from backend.online_phase.online_preprocessor import OnlinePreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

EEG_CH_NAMES = [
    "Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4",
    "P3", "Pz", "P4", "O1", "O2", "F7", "F8", "T7",
    "T8", "P7", "P8", "Oz",
]
N_CHANNELS = len(EEG_CH_NAMES)
INPUT_SFREQ = 1000.0
TARGET_SFREQ = 256


def _make_online_state(
    ch_names: list[str] = EEG_CH_NAMES,
    bad_channels: list[str] | None = None,
    interp_weights: np.ndarray | None = None,
    n_components: int = 4,
    sfreq_offline: float = float(TARGET_SFREQ),
) -> dict:
    """Build a minimal but valid online_state dict."""
    if bad_channels is None:
        bad_channels = []

    n_ch = len(ch_names)
    rng = np.random.default_rng(0)

    pca_components = rng.standard_normal((n_components, n_ch))
    unmixing = rng.standard_normal((n_components, n_components))
    mixing = np.linalg.pinv(unmixing)

    return {
        "bad_channels": bad_channels,
        "interp_weights": interp_weights,
        "ch_names": ch_names,
        "ica_unmixing": unmixing,
        "ica_mixing": mixing,
        "ica_pca_components": pca_components,
        "ica_pca_mean": np.zeros(n_ch),
        "ica_exclude": [],
        "sfreq_offline": sfreq_offline,
    }


def _make_settings(target_rate: int = TARGET_SFREQ) -> dict:
    """Build a minimal valid preprocessing_settings dict."""
    return {
        "bandpass": {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "method": "iir",
            "notch": 50.0,
        },
        "resample": {"target_rate": target_rate},
    }


@pytest.fixture
def valid_settings() -> dict:
    return _make_settings()


@pytest.fixture
def valid_online_state() -> dict:
    return _make_online_state()


@pytest.fixture
def preprocessor(valid_settings, valid_online_state) -> OnlinePreprocessor:
    return OnlinePreprocessor(valid_settings, valid_online_state, input_sfreq=INPUT_SFREQ)


# ── Commit 2: Constructor validation ─────────────────────────────────────────

class TestConstructorValidation:
    def test_valid_construction_succeeds(self, valid_settings, valid_online_state):
        p = OnlinePreprocessor(valid_settings, valid_online_state, input_sfreq=INPUT_SFREQ)
        assert p is not None

    def test_raises_if_sfreq_offline_mismatches_target_rate(self, valid_settings):
        state = _make_online_state(sfreq_offline=512.0)
        with pytest.raises(ValueError, match="sfreq_offline"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_if_ch_names_count_inconsistent_with_ica_pca(self, valid_settings):
        state = _make_online_state()
        state["ica_pca_components"] = np.zeros((4, N_CHANNELS + 5))
        with pytest.raises(ValueError, match="ch_names"):
            OnlinePreprocessor(valid_settings, state)


# ── Commit 2: Properties ──────────────────────────────────────────────────────

class TestProperties:
    def test_n_channels(self, preprocessor):
        assert preprocessor.n_channels == N_CHANNELS

    def test_target_sfreq(self, preprocessor):
        assert preprocessor.target_sfreq == float(TARGET_SFREQ)

    def test_input_sfreq(self, preprocessor):
        assert preprocessor.input_sfreq == INPUT_SFREQ


# ── Commit 2: process_batch stub ──────────────────────────────────────────────

class TestProcessBatchStub:
    def test_raises_not_implemented(self, preprocessor):
        batch = np.zeros((40, N_CHANNELS))
        timestamps = np.linspace(0, 0.04, 40)
        with pytest.raises(NotImplementedError):
            preprocessor.process_batch(batch, timestamps)


# ── Commit 2: reset_state ────────────────────────────────────────────────────

class TestResetState:
    def test_reset_clears_bandpass_zi(self, preprocessor):
        preprocessor._bandpass_zi = np.ones((4, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._bandpass_zi is None

    def test_reset_clears_notch_zi(self, preprocessor):
        preprocessor._notch_zi = np.ones((4, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._notch_zi is None

    def test_reset_clears_decimate_zi(self, preprocessor):
        preprocessor._decimate_zi = np.ones((10, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._decimate_zi is None

    def test_reset_zeros_decimate_phase(self, preprocessor):
        preprocessor._decimate_phase = 99
        preprocessor.reset_state()
        assert preprocessor._decimate_phase == 0


# ── Commit 3: causal bandpass + notch filter ──────────────────────────────────

def _make_sinusoid(freq_hz: float, n_samples: int, n_channels: int, sfreq: float) -> np.ndarray:
    t = np.arange(n_samples) / sfreq
    return np.tile(np.sin(2 * np.pi * freq_hz * t)[:, np.newaxis], (1, n_channels))


class TestApplyFilter:
    def test_equal_chunks_match_single_pass(self):
        """Filtering in N equal chunks must equal filtering the whole array at once."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((1000, N_CHANNELS)) * 1e-5

        p1 = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        result_single = p1._apply_filter(data.copy())

        p2 = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        chunks = [p2._apply_filter(data[i * 100:(i + 1) * 100].copy()) for i in range(10)]
        result_chunked = np.concatenate(chunks)

        np.testing.assert_allclose(result_single, result_chunked, atol=1e-10)

    def test_irregular_chunks_match_single_pass(self):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((1000, N_CHANNELS)) * 1e-5

        p1 = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        result_single = p1._apply_filter(data.copy())

        p2 = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        sizes = [37, 51, 29, 83, 100, 200, 500]  # sums to 1000
        chunks, idx = [], 0
        for s in sizes:
            chunks.append(p2._apply_filter(data[idx:idx + s].copy()))
            idx += s
        result_chunked = np.concatenate(chunks)

        np.testing.assert_allclose(result_single, result_chunked, atol=1e-10)

    def test_highfreq_attenuated(self):
        """Sinusoid well above h_freq=40 Hz must be strongly attenuated."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(100.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        p = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        # Take second half to skip filter transient.
        # Causal single-pass IIR has half the effective order of offline filtfilt,
        # so -30 dB (not -40 dB) is the meaningful guarantee at 2.5× the cutoff.
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / data[half:].std() + 1e-30)
        assert ratio_db < -30, f"Expected >30 dB attenuation, got {ratio_db:.1f} dB"

    def test_lowfreq_attenuated(self):
        """Sinusoid well below l_freq=1 Hz must be strongly attenuated."""
        n = int(INPUT_SFREQ * 30)  # need longer signal to see sub-1 Hz
        data = _make_sinusoid(0.1, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        p = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / data[half:].std() + 1e-30)
        assert ratio_db < -40, f"Expected >40 dB attenuation, got {ratio_db:.1f} dB"

    def test_notch_attenuates_50hz(self):
        """50 Hz sinusoid must be strongly attenuated when notch=50."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(50.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        settings = _make_settings()
        settings["bandpass"]["notch"] = 50.0
        p = OnlinePreprocessor(settings, _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db < -20, f"Expected notch attenuation, got {ratio_db:.1f} dB"

    def test_no_notch_leaves_50hz_intact(self):
        """50 Hz sinusoid must NOT be attenuated when notch=None."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(50.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        settings = _make_settings()
        settings["bandpass"]["notch"] = None
        settings["bandpass"]["h_freq"] = 80.0  # widen bandpass so 50 Hz passes
        p = OnlinePreprocessor(settings, _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db > -6, f"50 Hz should pass through, got {ratio_db:.1f} dB"

    def test_bandpass_zi_set_after_first_call(self, preprocessor):
        assert preprocessor._bandpass_zi is None
        preprocessor._apply_filter(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        assert preprocessor._bandpass_zi is not None

    def test_reset_clears_filter_zi(self, preprocessor):
        preprocessor._apply_filter(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        preprocessor.reset_state()
        assert preprocessor._bandpass_zi is None
        assert preprocessor._notch_zi is None


# ── Commit 4: stateful decimation ─────────────────────────────────────────────

class TestDecimate:
    def _make_p(self) -> OnlinePreprocessor:
        return OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)

    def _make_data(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n, dtype=float) / INPUT_SFREQ
        return data, timestamps

    def test_40_samples_give_10_outputs(self):
        """First batch of 40 samples at 1000 Hz → 10 outputs at 256 Hz."""
        data, ts = self._make_data(40)
        p = self._make_p()
        out, out_ts = p._decimate(data, ts)
        assert out.shape[0] == 10
        assert out_ts.shape[0] == 10

    def test_sample_count_large_equals_chunked(self):
        """Total output samples must be the same whether input arrives as one batch or many."""
        n_total = 1000
        data, ts = self._make_data(n_total)

        p1 = self._make_p()
        _, out_ts_single = p1._decimate(data, ts)
        n_single = len(out_ts_single)

        p2 = self._make_p()
        n_chunked = 0
        for i in range(25):  # 25 × 40 = 1000
            _, o = p2._decimate(data[i*40:(i+1)*40], ts[i*40:(i+1)*40])
            n_chunked += len(o)

        assert n_single == n_chunked

    def test_output_timestamps_are_subset_of_input(self):
        """Every output timestamp must correspond to an actual input timestamp."""
        data, ts = self._make_data(200)
        p = self._make_p()
        _, out_ts = p._decimate(data, ts)
        for t in out_ts:
            assert np.any(np.isclose(ts, t)), f"Output timestamp {t} not in input timestamps"

    def test_37_samples_at_phase_3_give_9_outputs(self):
        """37 input samples starting at phase=3 → 9 outputs (verified analytically)."""
        # Advance phase to 3 by processing 3 samples first
        p = self._make_p()
        seed_data, seed_ts = self._make_data(3)
        p._decimate(seed_data, seed_ts)
        assert p._decimate_phase == 3 * 32 % 125  # phase after 3 samples: 96 % 125 = 96

        # Manually set phase to 3 for the documented test case
        p._decimate_phase = 3
        data, ts = self._make_data(37)
        out, _ = p._decimate(data, ts)
        assert out.shape[0] == 9

    def test_empty_input_returns_empty(self):
        p = self._make_p()
        data = np.empty((0, N_CHANNELS))
        timestamps = np.empty((0,))
        out, out_ts = p._decimate(data, timestamps)
        assert out.shape == (0, N_CHANNELS)
        assert out_ts.shape == (0,)

    def test_output_is_2d(self):
        data, ts = self._make_data(40)
        p = self._make_p()
        out, _ = p._decimate(data, ts)
        assert out.ndim == 2
        assert out.shape[1] == N_CHANNELS

    def test_reset_clears_decimate_state(self, preprocessor):
        data, ts = self._make_data(40)
        preprocessor._decimate(data, ts)
        preprocessor.reset_state()
        assert preprocessor._decimate_zi is None
        assert preprocessor._decimate_phase == 0

    def test_reset_gives_identical_output(self):
        """After reset, reprocessing the same data from scratch gives identical output."""
        data, ts = self._make_data(120)
        p = self._make_p()

        chunks = [(data[i*40:(i+1)*40], ts[i*40:(i+1)*40]) for i in range(3)]
        outputs_first = [p._decimate(d, t) for d, t in chunks]

        p.reset_state()
        outputs_second = [p._decimate(d, t) for d, t in chunks]

        for (o1, t1), (o2, t2) in zip(outputs_first, outputs_second):
            np.testing.assert_array_equal(o1, o2)
            np.testing.assert_array_equal(t1, t2)
