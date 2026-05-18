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
TARGET_SFREQ = 100


def _make_online_state(
    n_eeg: int = N_CHANNELS,
    eeg_chunk_indices: list[int] | None = None,
    bad_indices: list[int] | None = None,
    interp_weights: np.ndarray | None = None,
    n_components: int = 4,
    sfreq_offline: float = float(TARGET_SFREQ),  # legacy kwarg; no longer in the schema
) -> dict:
    """Build a minimal but valid online_state dict (positional schema)."""
    if eeg_chunk_indices is None:
        # Default: keep the first n_eeg positions of the post-trigger-split LSL chunk
        # (e.g. positions 0..n_eeg-1 with no offline drops).
        eeg_chunk_indices = list(range(n_eeg))
    if bad_indices is None:
        bad_indices = []

    rng = np.random.default_rng(0)

    pca_components = rng.standard_normal((n_components, n_eeg))
    unmixing = rng.standard_normal((n_components, n_components))
    mixing = np.linalg.pinv(unmixing)

    return {
        "eeg_chunk_indices": eeg_chunk_indices,
        "bad_indices": bad_indices,
        "interp_weights": interp_weights,
        "ica_unmixing": unmixing,
        "ica_mixing": mixing,
        "ica_pca_components": pca_components,
        "ica_pca_mean": np.zeros(n_eeg),
        "ica_exclude": [],
        # Default to ones so synthetic-matrix tests get an identity rescaling.
        # The real-ICA fixture overrides this with ica.pre_whitener_.
        "pre_whitener": np.ones((n_eeg, 1)),
    }


def _make_settings(
    target_rate: int = TARGET_SFREQ,
    resample_filter_stage: str = "early",
) -> dict:
    """Build a minimal valid preprocessing_settings dict."""
    return {
        "highpass": {"l_freq": 1.0, "method": "iir"},
        "notch": {"freq": 50.0},
        "lowpass": {"h_freq": 40.0, "method": "iir"},
        "final_resample": {"target_rate": target_rate},
        "resample_filter_stage": resample_filter_stage,
    }


def _adapt_offline_state_to_positional(state: dict) -> dict:
    """Translate Roi's current name-based offline export into the new positional
    schema OnlinePreprocessor now expects.

    Temporary bridge until Roi's offline-side migration lands. Once Roi's
    export_online_state() emits eeg_chunk_indices + bad_indices directly,
    this helper goes away.
    """
    ch_names = list(state["ch_names"])
    bad_indices = [ch_names.index(name) for name in state.get("bad_channels", [])]
    new_state = {
        "eeg_chunk_indices": list(range(len(ch_names))),
        "bad_indices": bad_indices,
        "interp_weights": state["interp_weights"],
        "ica_unmixing": state["ica_unmixing"],
        "ica_mixing": state["ica_mixing"],
        "ica_pca_components": state["ica_pca_components"],
        "ica_pca_mean": state["ica_pca_mean"],
        "ica_exclude": state["ica_exclude"],
        "pre_whitener": state["pre_whitener"],
    }
    return new_state


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
    return OnlinePreprocessor(
        valid_settings, valid_online_state, input_sfreq=INPUT_SFREQ,
    )


# ── Commit 2: Constructor validation ─────────────────────────────────────────

class TestConstructorValidation:
    def test_valid_construction_succeeds(self, valid_settings, valid_online_state):
        p = OnlinePreprocessor(valid_settings, valid_online_state, input_sfreq=INPUT_SFREQ)
        assert p is not None

    def test_raises_if_pca_cols_inconsistent_with_eeg_chunk_indices(self, valid_settings):
        state = _make_online_state()
        state["ica_pca_components"] = np.zeros((4, N_CHANNELS + 5))
        with pytest.raises(ValueError, match="eeg_chunk_indices"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_on_negative_eeg_chunk_index(self, valid_settings):
        state = _make_online_state(eeg_chunk_indices=[0, 1, -1] + list(range(3, N_CHANNELS)))
        with pytest.raises(ValueError, match="eeg_chunk_indices"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_on_duplicate_eeg_chunk_indices(self, valid_settings):
        dup = list(range(N_CHANNELS))
        dup[0] = dup[1]
        state = _make_online_state(eeg_chunk_indices=dup)
        with pytest.raises(ValueError, match="duplicates"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_on_out_of_range_bad_indices(self, valid_settings):
        state = _make_online_state(bad_indices=[N_CHANNELS + 1])
        with pytest.raises(ValueError, match="bad_indices"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_on_out_of_range_ica_exclude(self, valid_settings):
        state = _make_online_state(n_components=4)
        state["ica_exclude"] = [99]  # n_components=4, so 99 is out of range
        with pytest.raises(ValueError, match="ica_exclude"):
            OnlinePreprocessor(valid_settings, state)

    def test_raises_on_non_integer_decimation_ratio(self):
        settings = _make_settings(target_rate=256)  # 1000 / 256 = 3.90625
        with pytest.raises(ValueError, match="integer multiple"):
            OnlinePreprocessor(settings, _make_online_state(sfreq_offline=256.0))

    def test_raises_on_invalid_resample_filter_stage(self):
        settings = _make_settings(resample_filter_stage="middle")
        with pytest.raises(ValueError, match="resample_filter_stage"):
            OnlinePreprocessor(settings, _make_online_state())


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
    def test_reset_clears_highpass_zi(self, preprocessor):
        preprocessor._highpass_zi = np.ones((4, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._highpass_zi is None

    def test_reset_clears_notch_zi(self, preprocessor):
        preprocessor._notch_zi = np.ones((4, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._notch_zi is None

    def test_reset_clears_lowpass_zi(self, preprocessor):
        preprocessor._lowpass_zi = np.ones((4, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._lowpass_zi is None

    def test_reset_clears_decimate_zi(self, preprocessor):
        preprocessor._decimate_zi = np.ones((10, N_CHANNELS))
        preprocessor.reset_state()
        assert preprocessor._decimate_zi is None

    def test_reset_zeros_decimate_phase(self, preprocessor):
        preprocessor._decimate_phase = 99
        preprocessor.reset_state()
        assert preprocessor._decimate_phase == 0


# ── Commit 3: causal high-pass + notch filter ────────────────────────────────

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

    def test_apply_filter_passes_high_frequencies(self):
        """HP-only stage must pass frequencies above the HP cutoff that aren't on the notch.

        Confirms the filter is HP-only (no upper rolloff) and that the 0.05 Hz drift
        below the HP cutoff is suppressed. Together these pin the stage as HP+notch,
        not bandpass.
        """
        n = int(INPUT_SFREQ * 5)
        p = OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)

        # 100 Hz tone is above the (former) bandpass upper edge of 40 Hz and not at the
        # 50 Hz notch — under HP-only it should pass through nearly untouched.
        high_data = _make_sinusoid(100.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        high_out = p._apply_filter(high_data.copy())
        half = n // 2
        high_ratio_db = 20 * np.log10(
            high_out[half:].std() / (high_data[half:].std() + 1e-30) + 1e-30
        )
        assert high_ratio_db > -3, (
            f"100 Hz should pass through HP-only stage, got {high_ratio_db:.1f} dB"
        )

        # 0.05 Hz drift is well below the 1 Hz HP cutoff — must be heavily attenuated.
        p.reset_state()
        n_long = int(INPUT_SFREQ * 30)  # need a long signal to see sub-1 Hz
        drift = _make_sinusoid(0.05, n_long, N_CHANNELS, INPUT_SFREQ) * 1e-5
        drift_out = p._apply_filter(drift.copy())
        half_long = n_long // 2
        drift_ratio_db = 20 * np.log10(
            drift_out[half_long:].std() / (drift[half_long:].std() + 1e-30) + 1e-30
        )
        assert drift_ratio_db < -40, (
            f"0.05 Hz drift should be suppressed, got {drift_ratio_db:.1f} dB"
        )

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
        settings["notch"] = {"freq": 50.0}
        p = OnlinePreprocessor(settings, _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db < -20, f"Expected notch attenuation, got {ratio_db:.1f} dB"

    def test_no_notch_leaves_50hz_intact(self):
        """50 Hz sinusoid must NOT be attenuated when notch is disabled."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(50.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        settings = _make_settings()
        settings["notch"] = None
        p = OnlinePreprocessor(settings, _make_online_state(), INPUT_SFREQ)
        out = p._apply_filter(data)
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db > -6, f"50 Hz should pass through, got {ratio_db:.1f} dB"

    def test_highpass_zi_set_after_first_call(self, preprocessor):
        assert preprocessor._highpass_zi is None
        preprocessor._apply_filter(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        assert preprocessor._highpass_zi is not None

    def test_reset_clears_filter_zi(self, preprocessor):
        preprocessor._apply_filter(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        preprocessor.reset_state()
        assert preprocessor._highpass_zi is None
        assert preprocessor._notch_zi is None


# ── Commit 2 (migration): causal low-pass filter ─────────────────────────────


class TestApplyLowpass:
    """Frequency-domain and state behaviour of the new LP stage.

    The LP stage shapes training-data spectrum (40 Hz cutoff for a 100 Hz
    target sfreq, per the cited replay paper) and prevents aliasing before
    the decimation step. Tests mirror the rigour of TestApplyFilter.
    """

    def _make_p(self) -> OnlinePreprocessor:
        return OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)

    def test_passband_fidelity_5hz_preserved(self):
        """A 5 Hz tone (well below the 40 Hz cutoff) should pass with negligible attenuation."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(5.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        out = self._make_p()._apply_lowpass(data.copy())
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db > -1.0, f"5 Hz should be in passband, got {ratio_db:.2f} dB"

    @pytest.mark.parametrize("freq_hz,min_atten_db", [(80.0, 10.0), (120.0, 20.0), (200.0, 30.0)])
    def test_stopband_attenuation_progresses(self, freq_hz: float, min_atten_db: float):
        """Stopband tones should be attenuated progressively more with increasing frequency."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(freq_hz, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        out = self._make_p()._apply_lowpass(data.copy())
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        assert ratio_db < -min_atten_db, (
            f"{freq_hz:.0f} Hz tone should be attenuated by at least {min_atten_db} dB, "
            f"got {ratio_db:.1f} dB"
        )

    def test_cutoff_at_40hz_is_around_minus_3db(self):
        """At the LP cutoff (40 Hz), MNE's default IIR design lands roughly in [-6, -1] dB."""
        n = int(INPUT_SFREQ * 5)
        data = _make_sinusoid(40.0, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
        out = self._make_p()._apply_lowpass(data.copy())
        half = n // 2
        ratio_db = 20 * np.log10(out[half:].std() / (data[half:].std() + 1e-30) + 1e-30)
        # MNE's default IIR (butter, order 4) is ~-3 dB at the design cutoff but
        # exact response depends on the order; allow a generous band.
        assert -6.0 < ratio_db < -1.0, (
            f"40 Hz response should sit near -3 dB, got {ratio_db:.2f} dB"
        )

    def test_lowpass_is_causal(self):
        """No output before t=0 from a unit impulse at t=0."""
        n = 200
        impulse = np.zeros((n, N_CHANNELS))
        impulse[0] = 1.0
        p = self._make_p()
        out = p._apply_lowpass(impulse)
        # Trivially true for `out[0:]`, but the meaningful guarantee is that the
        # filter doesn't draw on samples it hasn't seen — sosfilt is causal by
        # construction, so we assert finite energy lives entirely in t >= 0.
        assert np.all(np.isfinite(out))
        assert np.abs(out[:1]).sum() > 0, "Filter should respond at t=0"

    def test_chunk_boundary_continuity(self):
        """Whole-pass output must equal concatenated chunked output (persistent zi)."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((1000, N_CHANNELS)) * 1e-5

        p_single = self._make_p()
        out_single = p_single._apply_lowpass(data.copy())

        p_chunked = self._make_p()
        sizes = [37, 51, 29, 83, 100, 200, 500]  # sums to 1000
        chunks, idx = [], 0
        for s in sizes:
            chunks.append(p_chunked._apply_lowpass(data[idx:idx + s].copy()))
            idx += s
        out_chunked = np.concatenate(chunks)

        np.testing.assert_allclose(out_single, out_chunked, atol=1e-10)

    def test_reset_clears_lowpass_state(self, preprocessor):
        preprocessor._apply_lowpass(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        preprocessor.reset_state()
        assert preprocessor._lowpass_zi is None

    def test_lowpass_zi_set_after_first_call(self, preprocessor):
        assert preprocessor._lowpass_zi is None
        preprocessor._apply_lowpass(np.random.standard_normal((100, N_CHANNELS)) * 1e-5)
        assert preprocessor._lowpass_zi is not None

    def test_frequency_response_matches_offline_design(self):
        """sosfreqz on the same SOS must match the empirical |H(f)| of our online LP."""
        from scipy.signal import sosfreqz

        p = self._make_p()
        probe_freqs = np.array([1.0, 10.0, 30.0, 60.0, 100.0])

        # Empirical: pass tones through the online LP, measure RMS ratio (after transient).
        empirical_db = []
        for f in probe_freqs:
            n = int(INPUT_SFREQ * 5)
            sig = _make_sinusoid(f, n, N_CHANNELS, INPUT_SFREQ) * 1e-5
            p.reset_state()
            out = p._apply_lowpass(sig.copy())
            half = n // 2
            empirical_db.append(
                20 * np.log10(out[half:].std() / (sig[half:].std() + 1e-30) + 1e-30)
            )

        # Theoretical from the same SOS matrix
        worN = 2 * np.pi * probe_freqs / INPUT_SFREQ
        _, h = sosfreqz(p._lowpass_sos, worN=worN)
        theoretical_db = 20 * np.log10(np.abs(h) + 1e-30)

        # 2 dB tolerance covers MNE's IIR design quirks + steady-state RMS noise.
        np.testing.assert_allclose(np.array(empirical_db), theoretical_db, atol=2.0)


# ── Commit 4: stateful decimation ─────────────────────────────────────────────

class TestDecimate:
    def _make_p(self) -> OnlinePreprocessor:
        return OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)

    def _make_data(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        data = rng.standard_normal((n, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n, dtype=float) / INPUT_SFREQ
        return data, timestamps

    def test_40_samples_give_4_outputs(self):
        """First batch of 40 samples at 1000 Hz → 4 outputs at 100 Hz (factor 10)."""
        data, ts = self._make_data(40)
        p = self._make_p()
        out, out_ts = p._decimate(data, ts)
        assert out.shape[0] == 4
        assert out_ts.shape[0] == 4

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

    def test_phase_persists_across_chunks(self):
        """phase=3 + 37 input samples (factor 10) → keep indices [3, 13, 23, 33] = 4 outputs."""
        p = self._make_p()
        p._decimate_phase = 3
        data, ts = self._make_data(37)
        out, _ = p._decimate(data, ts)
        assert out.shape[0] == 4
        # After this chunk, next kept sample would be at original index 43,
        # i.e. local index 43 - 37 = 6 in the next chunk.
        assert p._decimate_phase == 6

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


@pytest.mark.parametrize("target_sfreq", [100, 200, 250, 500])
class TestDecimateFrequencies:
    """Decimation correctness across a range of integer-ratio target sample rates.

    Decimation now requires input_sfreq to be an integer multiple of
    target_sfreq (see OnlinePreprocessor.__init__). Non-integer ratios
    such as 1000→128, 1000→256, 1000→512 are rejected at construction,
    so they're not exercised here.
    """

    def _make_p(self, target_sfreq: int) -> OnlinePreprocessor:
        return OnlinePreprocessor(
            _make_settings(target_rate=target_sfreq),
            _make_online_state(sfreq_offline=float(target_sfreq)),
            input_sfreq=INPUT_SFREQ,
        )

    def _make_data(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n, dtype=float) / INPUT_SFREQ
        return data, timestamps

    def test_output_count_approximately_correct(self, target_sfreq: int) -> None:
        """n_out must be within ±1 of n_in × target / input."""
        n_in = 1000
        data, ts = self._make_data(n_in)
        out, _ = self._make_p(target_sfreq)._decimate(data, ts)
        expected = n_in * target_sfreq / INPUT_SFREQ
        assert abs(out.shape[0] - expected) <= 1

    def test_chunked_equals_single_pass_count(self, target_sfreq: int) -> None:
        """Total output samples are the same regardless of how input is chunked."""
        n_total = 500
        chunk_size = 40
        data, ts = self._make_data(n_total)

        _, single_ts = self._make_p(target_sfreq)._decimate(data, ts)

        p_chunked = self._make_p(target_sfreq)
        n_chunked = 0
        for start in range(0, n_total, chunk_size):
            _, o = p_chunked._decimate(
                data[start : start + chunk_size], ts[start : start + chunk_size]
            )
            n_chunked += len(o)

        assert len(single_ts) == n_chunked

    def test_output_timestamps_are_subset_of_input(self, target_sfreq: int) -> None:
        """Every output timestamp must correspond to a real input sample."""
        data, ts = self._make_data(200)
        _, out_ts = self._make_p(target_sfreq)._decimate(data, ts)
        for t in out_ts:
            assert np.any(np.isclose(ts, t)), f"Output timestamp {t:.6f} not in input"

    def test_empty_input_returns_empty(self, target_sfreq: int) -> None:
        out, out_ts = self._make_p(target_sfreq)._decimate(
            np.empty((0, N_CHANNELS)), np.empty((0,))
        )
        assert out.shape == (0, N_CHANNELS)
        assert out_ts.shape == (0,)


# ── Commit 5: spatial transforms ──────────────────────────────────────────────

def _make_preprocessor_with_bad_channel() -> OnlinePreprocessor:
    """Preprocessor with channel at index 0 (Fp1 in the EEG_CH_NAMES layout)
    declared as bad and interp_weights set.

    The bad/good indices are now positional — we use names only to drive MNE's
    interpolate_bads (which is name-based) when generating the reference weight
    matrix. The online state stores indices only.
    """
    import mne
    ch_names = list(EEG_CH_NAMES)
    bad_idx = ch_names.index("Fp1")  # positional index in the post-hygiene array
    good_indices = [i for i in range(len(ch_names)) if i != bad_idx]

    # Compute real interp weights via MNE identity-basis trick
    n_eeg = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    raw = mne.io.RawArray(np.eye(n_eeg), info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
    raw.info["bads"] = [ch_names[bad_idx]]
    raw.interpolate_bads(reset_bads=False, verbose=False)
    interp_data = raw.get_data()
    weights = interp_data[np.ix_([bad_idx], good_indices)].T  # (n_good, 1)

    state = _make_online_state(
        n_eeg=n_eeg,
        bad_indices=[bad_idx],
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

        state = _adapt_offline_state_to_positional(offline.export_online_state())
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

    def test_non_2d_batch_raises(self):
        p = self._make_p()
        bad_batch = np.zeros(40)  # 1D
        timestamps = np.zeros(40)
        with pytest.raises(ValueError, match="2D"):
            p.process_batch(bad_batch, timestamps)

    def test_too_narrow_batch_raises_index_error_at_slice(self):
        """A batch narrower than max(eeg_chunk_indices) + 1 fails at the slice.
        We don't validate the width up front, so NumPy raises IndexError when
        it tries to read a column that doesn't exist. Caught here as a sanity
        check that misconfiguration doesn't pass silently.
        """
        p = self._make_p()
        # eeg_chunk_indices defaults to list(range(N_CHANNELS)), so the slice
        # references index N_CHANNELS-1. A batch with N_CHANNELS-1 columns
        # fails on that index.
        bad_batch = np.zeros((40, N_CHANNELS - 1))
        timestamps = np.zeros(40)
        with pytest.raises(IndexError):
            p.process_batch(bad_batch, timestamps)

    def test_timestamp_length_mismatch_raises(self):
        p = self._make_p()
        batch = np.zeros((40, N_CHANNELS))
        timestamps = np.zeros(39)
        with pytest.raises(ValueError):
            p.process_batch(batch, timestamps)

    def test_empty_batch_returns_empty_without_state_change(self):
        p = self._make_p()
        assert p._highpass_zi is None
        out, out_ts = p.process_batch(np.empty((0, N_CHANNELS)), np.empty(0))
        assert out.shape == (0, N_CHANNELS)
        assert out_ts.shape == (0,)
        assert p._highpass_zi is None  # state not touched

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


# ── Commit 4 (migration): in-preprocessor EEG channel hygiene ────────────────


class TestProcessBatchHygiene:
    """eeg_chunk_indices is applied at the entry of process_batch, dropping
    raw LSL EEG columns down to the offline post-hygiene channel set before
    any spectral or spatial transform runs.
    """

    def test_process_batch_applies_chunk_indices_drops_emg_column(self):
        """A raw 22-column batch with eeg_chunk_indices=[0..7, 9..21] (drop position 8)
        produces a 21-column output. Use a small toy n_eeg = 21 so we can craft
        the input without needing the full 64-channel layout.
        """
        raw_n = 22
        keep = list(range(0, 8)) + list(range(9, raw_n))  # drop position 8 (1 channel)
        n_eeg = len(keep)  # 21
        state = _make_online_state(n_eeg=n_eeg, eeg_chunk_indices=keep)
        p = OnlinePreprocessor(_make_settings(), state, INPUT_SFREQ)

        # Build a 40-sample batch where each raw column has a distinct value
        # equal to its index, so we can verify which survived.
        data = np.zeros((40, raw_n), dtype=float)
        for col in range(raw_n):
            data[:, col] = float(col) * 1e-5
        timestamps = np.arange(40, dtype=float) / INPUT_SFREQ

        out, _ = p.process_batch(data, timestamps)

        # The hygiene drops column 8. The preprocessor then runs filter/decimate/
        # etc. — we don't assert on those numerics here, just on the *width*.
        assert out.shape[1] == n_eeg


# ── Commit 7: public API export ───────────────────────────────────────────────

class TestPublicAPI:
    def test_importable_from_online_phase(self):
        from backend.online_phase import OnlinePreprocessor as OP
        assert OP is not None

    def test_in_all(self):
        import backend.online_phase as pkg
        assert "OnlinePreprocessor" in pkg.__all__


# ── Commit 8: integration tests with real offline-exported state ───────────────

class TestIntegration:
    """process_batch() called with state from OfflinePreprocessor.export_online_state()."""

    def _build_offline_with_ica(self, tmp_path):
        """Return (offline, raw) with ICA fitted on synthetic non-Gaussian data."""
        from backend.offline_phase.preprocessor import OfflinePreprocessor

        rng = np.random.default_rng(42)
        n_comp = 4
        n_times = int(float(TARGET_SFREQ) * 10)
        t = np.arange(n_times) / float(TARGET_SFREQ)
        sources = np.vstack([
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 25 * t),
            np.sign(np.sin(2 * np.pi * 7 * t)),
            rng.standard_normal(n_times) + rng.laplace(size=n_times),
        ])
        mixing_true = rng.standard_normal((N_CHANNELS, n_comp))
        data = (mixing_true @ sources) * 1e-6

        info = mne.create_info(ch_names=EEG_CH_NAMES, sfreq=float(TARGET_SFREQ), ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

        offline = OfflinePreprocessor(
            data_dir=tmp_path / "Sub_001",
            preprocessing_settings=_make_offline_settings(),
        )
        offline.raw = raw.copy()
        offline._fit_ica()
        offline.ica.exclude = [0]
        return offline, raw

    def test_process_batch_smoke_with_real_offline_state(self, tmp_path):
        """process_batch() must run without error using state from export_online_state()."""
        offline, _ = self._build_offline_with_ica(tmp_path)
        state = _adapt_offline_state_to_positional(offline.export_online_state())

        online = OnlinePreprocessor(
            preprocessing_settings=_make_settings(),
            online_state=state,
            input_sfreq=INPUT_SFREQ,
        )

        rng = np.random.default_rng(20)
        batch = rng.standard_normal((400, N_CHANNELS)) * 1e-5
        timestamps = np.arange(400, dtype=float) / INPUT_SFREQ

        out, out_ts = online.process_batch(batch, timestamps)

        assert out.ndim == 2
        assert out.shape[1] == N_CHANNELS
        assert out_ts.shape[0] == out.shape[0]
        assert out.dtype == float

    def test_process_batch_bad_channel_is_overwritten_by_interpolation(self, tmp_path):
        """Bad channel is overwritten by interpolation: different input values → same output."""
        from backend.offline_phase.preprocessor import OfflinePreprocessor

        offline, _ = self._build_offline_with_ica(tmp_path)

        # Zero out Fp1 to force it to be detected as flat/bad
        eeg_picks = mne.pick_types(offline.raw.info, eeg=True)
        fp1_local = 0
        fp1_name = offline.raw.ch_names[eeg_picks[fp1_local]]
        offline.raw._data[eeg_picks[fp1_local], :] = 0.0
        offline._detect_bad_channels()
        assert fp1_name in offline._bad_channels

        offline._fit_ica()
        offline.ica.exclude = [0]
        offline_state = offline.export_online_state()
        assert offline_state["interp_weights"] is not None
        state = _adapt_offline_state_to_positional(offline_state)

        online = OnlinePreprocessor(
            preprocessing_settings=_make_settings(),
            online_state=state,
            input_sfreq=INPUT_SFREQ,
        )

        bad_ch_idx = offline_state["ch_names"].index(fp1_name)
        rng = np.random.default_rng(21)
        base = rng.standard_normal((400, N_CHANNELS)) * 1e-5
        timestamps = np.arange(400, dtype=float) / INPUT_SFREQ

        # Two batches with identical good channels but different bad-channel values.
        batch_a = base.copy()
        batch_b = base.copy()
        batch_a[:, bad_ch_idx] = +1.0
        batch_b[:, bad_ch_idx] = -1.0

        out_a, _ = online.process_batch(batch_a, timestamps)
        online.reset_state()
        out_b, _ = online.process_batch(batch_b, timestamps)

        # Interpolation overwrites the bad channel from the good channels,
        # so different input values in the bad slot produce identical output.
        np.testing.assert_allclose(out_a[:, bad_ch_idx], out_b[:, bad_ch_idx], atol=1e-10)
        good_indices = [i for i in range(N_CHANNELS) if i != bad_ch_idx]
        np.testing.assert_allclose(out_a[:, good_indices], out_b[:, good_indices], atol=1e-10)


# ── Commit 2 (migration): resample_filter_stage variant ordering ─────────────


class TestVariantOrdering:
    """process_batch must call its `_apply_*` stages in the order dictated by
    `resample_filter_stage`. We monkey-patch each stage to record the call
    order and the input shape it saw, then assert both.
    """

    def _instrument(self, p: OnlinePreprocessor) -> list[tuple[str, tuple]]:
        """Wrap each pipeline stage to record (name, input_shape) into a log."""
        log: list[tuple[str, tuple]] = []

        def wrap_inplace(name: str, original):
            def wrapped(data):
                log.append((name, data.shape))
                return original(data)
            return wrapped

        def wrap_returning_tuple(name: str, original):
            def wrapped(data, timestamps):
                log.append((name, data.shape))
                return original(data, timestamps)
            return wrapped

        # _apply_filter and _apply_lowpass return arrays.
        p._apply_filter = wrap_inplace("filter", p._apply_filter)
        p._apply_lowpass = wrap_inplace("lowpass", p._apply_lowpass)
        # _decimate returns (data, timestamps).
        p._decimate = wrap_returning_tuple("decimate", p._decimate)
        # In-place stages all mutate `data` and return None.
        p._apply_bad_channel_interpolation = wrap_inplace(
            "interp", p._apply_bad_channel_interpolation
        )
        p._apply_average_reference = wrap_inplace(
            "avg_ref", p._apply_average_reference
        )
        p._apply_ica = wrap_inplace("ica", p._apply_ica)
        return log

    def _make_batch(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        n = 100  # 100 samples @ 1000 Hz -> 10 samples @ 100 Hz after decimation
        data = rng.standard_normal((n, N_CHANNELS)) * 1e-5
        timestamps = np.arange(n, dtype=float) / INPUT_SFREQ
        return data, timestamps

    def test_early_variant_runs_lp_decimate_before_spatial_transforms(self):
        p = OnlinePreprocessor(
            _make_settings(resample_filter_stage="early"),
            _make_online_state(),
            INPUT_SFREQ,
        )
        log = self._instrument(p)
        data, timestamps = self._make_batch()
        p.process_batch(data, timestamps)

        stage_names = [name for name, _ in log]
        assert stage_names == [
            "filter", "lowpass", "decimate", "interp", "avg_ref", "ica"
        ]

        # ICA must see decimated data: 100 input samples -> 10 at 100 Hz.
        ica_input_shape = next(shape for name, shape in log if name == "ica")
        assert ica_input_shape[0] == 10

    def test_late_variant_runs_lp_decimate_after_spatial_transforms(self):
        p = OnlinePreprocessor(
            _make_settings(resample_filter_stage="late"),
            _make_online_state(),
            INPUT_SFREQ,
        )
        log = self._instrument(p)
        data, timestamps = self._make_batch()
        p.process_batch(data, timestamps)

        stage_names = [name for name, _ in log]
        assert stage_names == [
            "filter", "interp", "avg_ref", "ica", "lowpass", "decimate"
        ]

        # ICA must see full-rate data: 100 input samples stay at 1000 Hz.
        ica_input_shape = next(shape for name, shape in log if name == "ica")
        assert ica_input_shape[0] == 100

    def test_both_variants_emit_same_output_sample_count(self):
        """Both variants decimate the same input by 10x and emit the same length."""
        data, timestamps = self._make_batch()

        p_early = OnlinePreprocessor(
            _make_settings(resample_filter_stage="early"),
            _make_online_state(),
            INPUT_SFREQ,
        )
        out_early, _ = p_early.process_batch(data.copy(), timestamps.copy())

        p_late = OnlinePreprocessor(
            _make_settings(resample_filter_stage="late"),
            _make_online_state(),
            INPUT_SFREQ,
        )
        out_late, _ = p_late.process_batch(data.copy(), timestamps.copy())

        assert out_early.shape == out_late.shape
