"""
Staged unit tests for OfflinePreprocessor.

Each TestStage class validates one processing stage independently.
Private methods are called directly (after injecting synthetic self.raw)
so failures pinpoint the broken stage without MNE's full pipeline overhead.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from backend.offline_phase.preprocessor import OfflinePreprocessor


# ── Stage 1: File discovery ────────────────────────────────────────────────

class TestStage1FindVhdr:
    def test_finds_single_vhdr(self, tmp_path):
        data_dir = tmp_path / "Sub_001"
        data_dir.mkdir()
        vhdr = data_dir / "test.vhdr"
        vhdr.touch()

        p = OfflinePreprocessor(data_dir, {})
        assert p.vhdr == vhdr

    def test_returns_none_when_missing(self, tmp_path):
        data_dir = tmp_path / "Sub_001"
        data_dir.mkdir()
        p = OfflinePreprocessor(data_dir, {})
        assert p.vhdr is None

    def test_uses_first_when_multiple(self, tmp_path, caplog):
        data_dir = tmp_path / "Sub_001"
        data_dir.mkdir()
        (data_dir / "a.vhdr").touch()
        (data_dir / "b.vhdr").touch()

        import logging
        with caplog.at_level(logging.WARNING):
            p = OfflinePreprocessor(data_dir, {})
        assert p.vhdr is not None
        assert "multiple" in caplog.text.lower()

    def test_raises_when_no_vhdr_on_step1(self, make_preprocessor):
        # vhdr is None → run_step1 must raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            make_preprocessor.run_step1_prepare_ica()

    def test_subject_id_is_folder_name(self, tmp_path):
        data_dir = tmp_path / "Sub_042"
        data_dir.mkdir()
        p = OfflinePreprocessor(data_dir, {})
        assert p.subject_id == "Sub_042"


# ── Stage 2: Signal cleaning chain ────────────────────────────────────────

class TestStage2Filter:
    def test_data_is_changed_after_filter(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        original = p.raw.get_data().copy()
        p.settings = preprocessing_settings
        p._filter()
        assert not np.allclose(p.raw.get_data(), original)

    def test_sfreq_unchanged_by_filter(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        sfreq_before = p.raw.info["sfreq"]
        p._filter()
        assert p.raw.info["sfreq"] == sfreq_before


class TestStage2Resample:
    def test_downsamples_to_target_rate(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._resample()
        assert p.raw.info["sfreq"] == preprocessing_settings["resample"]["target_rate"]

    def test_skips_resample_if_already_at_target(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        # Start at target rate already
        target = preprocessing_settings["resample"]["target_rate"]
        low_rate_raw = synthetic_raw.copy().resample(target, verbose=False)
        p.raw = low_rate_raw
        p.settings = preprocessing_settings
        n_times_before = p.raw.n_times
        p._resample()
        assert p.raw.n_times == n_times_before


class TestStage2BadChannels:
    def test_flat_channel_detected_and_interpolated(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        raw = synthetic_raw.copy()
        # Zero out first EEG channel → flat
        picks = mne.pick_types(raw.info, eeg=True)
        raw._data[picks[0], :] = 0.0
        flat_ch = raw.ch_names[picks[0]]

        p.raw = raw
        p.settings = preprocessing_settings
        p._detect_bad_channels()

        assert flat_ch in p._bad_channels

    def test_noisy_channel_detected(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        raw = synthetic_raw.copy()
        # Make one channel 100× noisier
        picks = mne.pick_types(raw.info, eeg=True)
        raw._data[picks[1], :] *= 100
        noisy_ch = raw.ch_names[picks[1]]

        p.raw = raw
        p.settings = preprocessing_settings
        p._detect_bad_channels()

        assert noisy_ch in p._bad_channels

    def test_no_bads_when_data_is_clean(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._detect_bad_channels()
        assert p._bad_channels == []


class TestStage2Reference:
    def test_average_reference_applied(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._reference()
        # After average reference, mean across EEG channels at each time ≈ 0
        eeg_data = p.raw.get_data(picks="eeg")
        assert np.abs(eeg_data.mean(axis=0)).max() < 1e-15


# ── Stage 3: ICA fit ────────────────────────────────────────────────────────

class TestStage3ICAFit:
    def test_returns_list_of_ints(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        suggested = p._fit_ica()

        assert isinstance(suggested, list)
        assert all(isinstance(i, int) for i in suggested)

    def test_ica_object_stored(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._fit_ica()
        assert p.ica is not None
        assert isinstance(p.ica, mne.preprocessing.ICA)

    def test_no_components_excluded_yet(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        # _fit_ica only suggests — it must NOT set ica.exclude
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._fit_ica()
        assert p.ica.exclude == []

    def test_step1_raises_without_vhdr(self, make_preprocessor):
        assert make_preprocessor.vhdr is None
        with pytest.raises(FileNotFoundError):
            make_preprocessor.run_step1_prepare_ica()

    def test_step2_raises_before_step1(self, make_preprocessor):
        with pytest.raises(RuntimeError, match="run_step1"):
            make_preprocessor.run_step2_finish_pipeline([], {}, Path("/tmp"))


# ── Stage 4: Epoching ──────────────────────────────────────────────────────

class TestStage4Epoch:
    def test_epochs_shape(self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        event_mapping = {"red": 1, "green": 2}

        # Manually convert annotations to MNE format
        mapping = {"red": 1, "green": 2}
        p.epochs = p._epoch(event_mapping)

        assert isinstance(p.epochs, mne.Epochs)
        assert p.epochs.get_data().ndim == 3  # (n_epochs, n_channels, n_times)

    def test_epoch_event_ids_match_mapping(self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        event_mapping = {"red": 1, "green": 2}
        p.epochs = p._epoch(event_mapping)

        assert "red" in p.epochs.event_id
        assert "green" in p.epochs.event_id

    def test_epoch_time_window(self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})

        ep = preprocessing_settings["epochs"]
        assert np.isclose(p.epochs.tmin, ep["tmin"], atol=1 / p.raw.info["sfreq"])
        assert np.isclose(p.epochs.tmax, ep["tmax"], atol=1 / p.raw.info["sfreq"])

    def test_unrecognised_events_ignored(self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        # Mapping that doesn't match any annotation
        p.epochs = p._epoch({"blue": 99})
        # Falls back to all found events — should still produce epochs
        assert len(p.epochs) > 0


# ── Stage 5: AutoReject + save + export_online_state ──────────────────────

class TestStage5AutoReject:
    def test_autoreject_called_and_epochs_updated(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings, tmp_path
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        n_before = len(p.epochs)

        mock_log = MagicMock()
        mock_log.bad_epochs = np.zeros(n_before, dtype=bool)
        mock_ar = MagicMock()
        mock_ar.fit_transform.return_value = (p.epochs, mock_log)

        with patch("backend.offline_phase.preprocessor.AutoReject", return_value=mock_ar):
            p._autoreject()

        mock_ar.fit_transform.assert_called_once()


class TestStage5Save:
    def test_fif_file_created(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings, tmp_path
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})

        output_dir = tmp_path / "output"
        p._save(output_dir)

        saved = list(output_dir.glob("*.fif"))
        assert len(saved) == 1

    def test_saved_filename_contains_subject_id(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings, tmp_path
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})

        output_dir = tmp_path / "output"
        p._save(output_dir)

        saved = list(output_dir.glob("*.fif"))[0]
        assert p.subject_id in saved.name


class TestStage5ExportOnlineState:
    def _build_preprocessor_with_ica(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._fit_ica()
        p.ica.exclude = [0]
        p.ica.apply(p.raw, verbose=False)
        return p

    def test_raises_before_pipeline(self, make_preprocessor):
        with pytest.raises(RuntimeError):
            make_preprocessor.export_online_state()

    def test_returns_required_keys(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()

        required = {
            "bad_channels", "ica_unmixing", "ica_mixing",
            "ica_pca_components", "ica_pca_mean", "ica_exclude",
            "ch_names", "sfreq_offline", "interp_weights",
        }
        assert required <= state.keys()

    def test_ica_matrices_are_numpy(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()

        assert isinstance(state["ica_unmixing"], np.ndarray)
        assert isinstance(state["ica_mixing"], np.ndarray)
        assert isinstance(state["ica_pca_components"], np.ndarray)

    def test_exclude_list_matches(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()
        assert state["ica_exclude"] == [0]

    def test_sfreq_is_float(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()
        assert isinstance(state["sfreq_offline"], float)

    def test_ch_names_is_list_of_strings(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()
        assert isinstance(state["ch_names"], list)
        assert all(isinstance(n, str) for n in state["ch_names"])

    # ── interp_weights ────────────────────────────────────────────────────────

    def _build_preprocessor_with_bad_channel(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        """Preprocessor with Fp1 detected as flat and interpolated, ICA applied."""
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        eeg_picks = mne.pick_types(p.raw.info, eeg=True)
        p.raw._data[eeg_picks[0], :] = 0.0  # Fp1 → flat
        p._detect_bad_channels()
        p._fit_ica()
        p.ica.exclude = [0]
        p.ica.apply(p.raw, verbose=False)
        return p

    def test_interp_weights_key_present(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()
        assert "interp_weights" in state

    def test_interp_weights_none_when_no_bad_channels(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_ica(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()
        assert state["interp_weights"] is None

    def test_interp_weights_shape_with_bad_channels(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = self._build_preprocessor_with_bad_channel(make_preprocessor, synthetic_raw, preprocessing_settings)
        state = p.export_online_state()

        eeg_picks = mne.pick_types(p.raw.info, eeg=True)
        n_eeg = len(eeg_picks)
        n_bad = len(p._bad_channels)
        n_good = n_eeg - n_bad

        assert state["interp_weights"] is not None
        assert state["interp_weights"].shape == (n_good, n_bad)

    def test_interp_weights_correctness(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings

        eeg_picks = mne.pick_types(p.raw.info, eeg=True)
        eeg_ch_names = [p.raw.ch_names[i] for i in eeg_picks]
        fp1_local_idx = eeg_ch_names.index("Fp1")
        good_local_indices = [i for i in range(len(eeg_ch_names)) if i != fp1_local_idx]

        # Save good-channel data before any modification
        p.raw._data[eeg_picks[fp1_local_idx], :] = 0.0
        good_data_before = p.raw.get_data(picks="eeg")[good_local_indices, :].copy()

        # _detect_bad_channels() interpolates Fp1 and stores _interp_weights
        p._detect_bad_channels()

        # MNE's interpolated Fp1 is now in p.raw
        mne_interpolated = p.raw.get_data(picks="eeg")[fp1_local_idx, :]
        weights = p._interp_weights  # (n_good, 1)

        # Apply weights: data_bad = W.T @ data_good  (channels × time format)
        predicted = weights.T @ good_data_before  # (1, n_times)

        np.testing.assert_allclose(predicted[0], mne_interpolated, atol=1e-10)
