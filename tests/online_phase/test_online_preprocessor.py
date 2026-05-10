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
        # Default to ones so synthetic-matrix tests get an identity rescaling.
        # The real-ICA fixture overrides this with ica.pre_whitener_.
        "pre_whitener": np.ones((n_ch, 1)),
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


def _make_offline_settings(target_rate: int = TARGET_SFREQ) -> dict:
    """Build preprocessing settings with the fields OfflinePreprocessor needs."""
    return {
        "random_state": 42,
        "bandpass": {"l_freq": 1.0, "h_freq": 40.0, "method": "iir", "notch": 50.0},
        "resample": {"target_rate": target_rate},
        "reject_criteria": {
            "hard_amplitude": 1e-3,
            "flat_threshold": 0.5e-6,
            "noisy_z_score": 3.0,
        },
        "ica": {"n_components": 4, "method": "fastica", "fit_l_freq": 1.0},
        "epochs": {"tmin": -0.1, "tmax": 0.5, "baseline": [None, 0]},
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


# ── Commit 5: spatial transforms ──────────────────────────────────────────────

def _make_preprocessor_with_bad_channel() -> OnlinePreprocessor:
    """Preprocessor with Fp1 declared as bad and interp_weights set."""
    import mne
    ch_names = list(EEG_CH_NAMES)
    bad_channels = ["Fp1"]
    bad_idx = ch_names.index("Fp1")
    good_indices = [i for i in range(len(ch_names)) if i != bad_idx]

    # Compute real interp weights via MNE identity-basis trick
    n_eeg = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    raw = mne.io.RawArray(np.eye(n_eeg), info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=False, verbose=False)
    interp_data = raw.get_data()
    bad_local = [bad_idx]
    good_local = good_indices
    weights = interp_data[np.ix_(bad_local, good_local)].T  # (n_good, 1)

    state = _make_online_state(
        ch_names=ch_names,
        bad_channels=bad_channels,
        interp_weights=weights,
    )
    return OnlinePreprocessor(_make_settings(), state, INPUT_SFREQ)


class TestApplyBadChannelInterpolation:
    def test_no_bad_channels_data_unchanged(self, preprocessor):
        data = np.random.default_rng(0).standard_normal((50, N_CHANNELS))
        original = data.copy()
        preprocessor._apply_bad_channel_interpolation(data)
        np.testing.assert_array_equal(data, original)

    def test_interpolated_values_match_mne(self):
        """Bad channel values after interpolation must match MNE's interpolate_bads output."""
        import mne
        p = _make_preprocessor_with_bad_channel()
        bad_idx = EEG_CH_NAMES.index("Fp1")
        good_indices = [i for i in range(N_CHANNELS) if i != bad_idx]

        rng = np.random.default_rng(3)
        data = rng.standard_normal((30, N_CHANNELS)) * 1e-5

        # Compute MNE reference: interpolate bad channel from good channels
        weights = p._interp_weights  # (n_good, 1)
        expected_bad = data[:, good_indices] @ weights  # (30, 1)

        result = data.copy()
        p._apply_bad_channel_interpolation(result)

        np.testing.assert_allclose(
            result[:, bad_idx:bad_idx+1], expected_bad, atol=1e-10
        )

    def test_good_channels_unchanged(self):
        p = _make_preprocessor_with_bad_channel()
        bad_idx = EEG_CH_NAMES.index("Fp1")
        good_indices = [i for i in range(N_CHANNELS) if i != bad_idx]

        data = np.random.default_rng(4).standard_normal((30, N_CHANNELS)) * 1e-5
        good_before = data[:, good_indices].copy()
        p._apply_bad_channel_interpolation(data)
        np.testing.assert_array_equal(data[:, good_indices], good_before)


class TestApplyAverageReference:
    def test_mean_across_all_channels_is_zero(self, preprocessor):
        data = np.random.default_rng(5).standard_normal((50, N_CHANNELS))
        preprocessor._apply_average_reference(data)
        np.testing.assert_allclose(data.mean(axis=1), 0.0, atol=1e-12)

    def test_idempotent(self, preprocessor):
        data = np.random.default_rng(6).standard_normal((50, N_CHANNELS))
        preprocessor._apply_average_reference(data)
        after_first = data.copy()
        preprocessor._apply_average_reference(data)
        # Second application changes nothing (already zero-mean), but floating-point
        # arithmetic introduces sub-epsilon differences — use allclose, not array_equal.
        np.testing.assert_allclose(data, after_first, atol=1e-14)


class TestApplyICA:
    def _make_raw_for_ica(self, sfreq: float = float(TARGET_SFREQ)) -> mne.io.RawArray:
        rng = np.random.default_rng(42)
        n_comp = 4
        n_times = int(sfreq * 10)
        t = np.arange(n_times) / sfreq

        sources = np.vstack([
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 25 * t),
            np.sign(np.sin(2 * np.pi * 7 * t)),
            rng.standard_normal(n_times) + rng.laplace(size=n_times),
        ])
        mixing_true = rng.standard_normal((N_CHANNELS, n_comp))
        data = (mixing_true @ sources) * 1e-6

        info = mne.create_info(ch_names=EEG_CH_NAMES, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
        return raw

    def _build_preprocessor_with_real_ica(self) -> tuple[OnlinePreprocessor, object]:
        """Fit a real MNE ICA on synthetic data and return both the online preprocessor
        and the fitted ICA object so tests can compare against mne.ICA.apply().

        Uses non-Gaussian sources (sinusoids + Laplace noise) mixed into N_CHANNELS
        channels, then band-pass filtered so FastICA converges reliably.
        """
        import mne
        rng = np.random.default_rng(42)
        n_comp = 4
        n_times = int(1000.0 * 10)
        t = np.arange(n_times) / 1000.0

        # Independent, non-Gaussian sources — FastICA requires non-Gaussianity
        sources = np.vstack([
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 25 * t),
            np.sign(np.sin(2 * np.pi * 7 * t)),   # square wave — strongly non-Gaussian
            rng.standard_normal(n_times) + rng.laplace(size=n_times),
        ])  # (4, n_times)
        mixing_true = rng.standard_normal((N_CHANNELS, n_comp))
        data = (mixing_true @ sources) * 1e-6  # (20, n_times)

        info = mne.create_info(ch_names=EEG_CH_NAMES, sfreq=1000.0, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.filter(1.0, 40.0, verbose=False)  # MNE recommends high-pass before ICA

        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica",
                                    random_state=42, max_iter=2000)
        ica.fit(raw, verbose=False)
        ica.exclude = [0]

        n_comp_fit = ica.n_components_
        state = _make_online_state(n_components=n_comp_fit)
        state["ica_unmixing"] = ica.unmixing_matrix_.copy()
        state["ica_mixing"] = ica.mixing_matrix_.copy()
        state["ica_pca_components"] = ica.pca_components_[:n_comp_fit].copy()
        state["ica_pca_mean"] = ica.pca_mean_.copy() if ica.pca_mean_ is not None else None
        state["ica_exclude"] = list(ica.exclude)
        state["pre_whitener"] = ica.pre_whitener_.copy()

        p = OnlinePreprocessor(_make_settings(), state, INPUT_SFREQ)
        return p, ica, raw

    def test_ica_matches_mne_apply(self):
        """_apply_ica() must reproduce mne.preprocessing.ICA.apply() to within 1e-8."""
        import mne
        p, ica, raw = self._build_preprocessor_with_real_ica()

        raw_copy = raw.copy()
        ica.apply(raw_copy, verbose=False)
        expected = raw_copy.get_data().T  # (n_times, n_ch)

        data = raw.get_data().T.copy()
        p._apply_ica(data)

        np.testing.assert_allclose(data, expected, atol=1e-8)

    def test_exported_offline_state_matches_mne_apply(self, tmp_path):
        """Offline export_online_state() must contain enough ICA state for online parity."""
        from backend.offline_phase.preprocessor import OfflinePreprocessor

        raw = self._make_raw_for_ica()
        offline = OfflinePreprocessor(
            data_dir=tmp_path / "Sub_001",
            preprocessing_settings=_make_offline_settings(),
        )
        offline.raw = raw.copy()
        offline._fit_ica()
        offline.ica.exclude = [0, 2]

        state = offline.export_online_state()
        online = OnlinePreprocessor(
            preprocessing_settings=_make_settings(),
            online_state=state,
            input_sfreq=INPUT_SFREQ,
        )

        raw_expected = offline.raw.copy()
        offline.ica.apply(raw_expected, verbose=False)
        expected = raw_expected.get_data().T

        data = offline.raw.get_data().T.copy()
        online._apply_ica(data)

        np.testing.assert_allclose(data, expected, atol=1e-8)

    def test_empty_exclude_leaves_data_unchanged(self):
        """With no excluded components, ICA apply is identity (within float precision)."""
        import mne
        p, ica, raw = self._build_preprocessor_with_real_ica()
        p._ica_exclude = []  # override exclusions

        data = raw.get_data().T.copy()
        original = data.copy()
        p._apply_ica(data)
        np.testing.assert_allclose(data, original, atol=1e-8)

    def test_pca_mean_none_does_not_crash(self):
        state = _make_online_state()
        state["ica_pca_mean"] = None
        p = OnlinePreprocessor(_make_settings(), state, INPUT_SFREQ)
        data = np.random.default_rng(9).standard_normal((20, N_CHANNELS))
        p._apply_ica(data)  # must not raise

    def test_ica_chunked_matches_single_pass(self):
        """_apply_ica is a per-sample transform (no temporal state) — chunked == single pass."""
        p, ica, raw = self._build_preprocessor_with_real_ica()
        data = raw.get_data().T.copy()  # (n_times, n_ch)

        data_single = data.copy()
        p._apply_ica(data_single)

        # Apply in 10 equal chunks in-place on the same array
        data_chunked = data.copy()
        chunk_size = len(data_chunked) // 10
        for i in range(10):
            chunk = data_chunked[i * chunk_size:(i + 1) * chunk_size]
            p._apply_ica(chunk)

        np.testing.assert_allclose(data_single, data_chunked, atol=1e-12)

    def test_ica_pca_mean_none_matches_mne(self):
        """Output must match MNE's ICA.apply() even when pca_mean is None."""
        p, ica, raw = self._build_preprocessor_with_real_ica()

        # Force pca_mean to None in both MNE and our preprocessor
        ica.pca_mean_ = None
        p._ica_pca_mean = None

        raw_copy = raw.copy()
        ica.apply(raw_copy, verbose=False)
        expected = raw_copy.get_data().T

        data = raw.get_data().T.copy()
        p._apply_ica(data)

        np.testing.assert_allclose(data, expected, atol=1e-8)


# ── Commit 6: process_batch() ─────────────────────────────────────────────────

class TestProcessBatch:
    def _make_p(self) -> OnlinePreprocessor:
        return OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)

    def _make_batch(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(11)
        data = rng.standard_normal((n, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n, dtype=float) / INPUT_SFREQ
        return data, timestamps

    def test_wrong_channel_count_raises(self):
        p = self._make_p()
        bad_batch = np.zeros((40, N_CHANNELS + 1))
        timestamps = np.zeros(40)
        with pytest.raises(ValueError):
            p.process_batch(bad_batch, timestamps)

    def test_timestamp_length_mismatch_raises(self):
        p = self._make_p()
        batch = np.zeros((40, N_CHANNELS))
        timestamps = np.zeros(39)
        with pytest.raises(ValueError):
            p.process_batch(batch, timestamps)

    def test_empty_batch_returns_empty_without_state_change(self):
        p = self._make_p()
        assert p._bandpass_zi is None
        out, out_ts = p.process_batch(np.empty((0, N_CHANNELS)), np.empty(0))
        assert out.shape == (0, N_CHANNELS)
        assert out_ts.shape == (0,)
        assert p._bandpass_zi is None  # state not touched

    def test_output_shape(self):
        p = self._make_p()
        batch, timestamps = self._make_batch(40)
        out, out_ts = p.process_batch(batch, timestamps)
        assert out.ndim == 2
        assert out.shape[1] == N_CHANNELS
        assert out_ts.shape == (out.shape[0],)

    def test_chunked_matches_single_pass_values(self):
        """Processing 400 samples in 40-sample chunks must give values identical
        to processing all 400 at once (validates full pipeline state continuity)."""
        rng = np.random.default_rng(12)
        n_total = 400
        data = rng.standard_normal((n_total, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n_total, dtype=float) / INPUT_SFREQ

        p1 = self._make_p()
        out_single, ts_single = p1.process_batch(data, timestamps)

        p2 = self._make_p()
        out_chunks, ts_chunks = [], []
        for i in range(10):
            o, t = p2.process_batch(data[i*40:(i+1)*40], timestamps[i*40:(i+1)*40])
            out_chunks.append(o)
            ts_chunks.append(t)
        out_chunked = np.concatenate(out_chunks)
        ts_chunked = np.concatenate(ts_chunks)

        np.testing.assert_allclose(out_single, out_chunked, atol=1e-10)
        np.testing.assert_allclose(ts_single, ts_chunked, atol=1e-12)

    def test_reset_then_reprocess_gives_identical_output(self):
        """After reset, reprocessing the same data produces identical output to the first run."""
        p = self._make_p()
        batch, timestamps = self._make_batch(400)

        out1, ts1 = p.process_batch(batch, timestamps)
        p.reset_state()
        out2, ts2 = p.process_batch(batch, timestamps)

        np.testing.assert_allclose(out1, out2, atol=1e-10)
        np.testing.assert_allclose(ts1, ts2, atol=1e-12)

    def test_does_not_mutate_input(self):
        p = self._make_p()
        batch, timestamps = self._make_batch(40)
        original = batch.copy()
        p.process_batch(batch, timestamps)
        np.testing.assert_array_equal(batch, original)
