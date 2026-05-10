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
