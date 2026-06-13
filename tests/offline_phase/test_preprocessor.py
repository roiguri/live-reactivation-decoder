"""
Staged unit tests for OfflinePreprocessor (new reference pipeline).

Each TestStage class validates one processing stage independently. Private
methods are called directly (after injecting synthetic self.raw / self.epochs)
so failures pinpoint the broken stage without MNE's full pipeline overhead.
"""

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest

from backend.core.preprocessing_constants import FINAL_RESAMPLE_RATE
from backend.offline_phase.preprocessor import OfflinePreprocessor


def _raw_with_extra_channels(synthetic_raw, extra_types: dict) -> mne.io.RawArray:
    """Append extra named channels (e.g. {'EMG': 'eeg'}) to a synthetic raw."""
    n_times = synthetic_raw.n_times
    rng = np.random.default_rng(1)
    extra_data = rng.standard_normal((len(extra_types), n_times)) * 10e-6
    extra_info = mne.create_info(
        list(extra_types.keys()), synthetic_raw.info["sfreq"],
        ch_types=list(extra_types.values()),
    )
    extra_raw = mne.io.RawArray(extra_data, extra_info, verbose=False)
    return synthetic_raw.copy().add_channels([extra_raw], force_update_info=True)


# ── Stage 1: Init ─────────────────────────────────────────────────────────

class TestStage1Init:
    def test_subject_id_is_folder_name(self, tmp_path):
        data_dir = tmp_path / "Sub_042"
        data_dir.mkdir()
        p = OfflinePreprocessor(data_dir, {})
        assert p.subject_id == "Sub_042"

    def test_raw_none_by_default(self, tmp_path):
        data_dir = tmp_path / "Sub_001"
        data_dir.mkdir()
        p = OfflinePreprocessor(data_dir, {})
        assert p.raw is None

    def test_raw_set_when_passed(self, tmp_path, synthetic_raw):
        data_dir = tmp_path / "Sub_001"
        data_dir.mkdir()
        p = OfflinePreprocessor(data_dir, {}, raw=synthetic_raw)
        assert p.raw is synthetic_raw

    def test_step1a_raises_without_raw(self, make_preprocessor):
        assert make_preprocessor.raw is None
        with pytest.raises(RuntimeError, match="raw must be set"):
            make_preprocessor.run_step1a_filter()

    def test_step1b_raises_before_step1a(self, make_preprocessor):
        with pytest.raises(RuntimeError, match="run_step1a_filter"):
            make_preprocessor.run_step1b_fit_ica({})

    def test_step2_raises_before_step1b(self, make_preprocessor):
        with pytest.raises(RuntimeError, match="run_step1b_fit_ica"):
            make_preprocessor.run_step2_apply_and_save([], Path("/tmp"))


# ── Stage 2: Channel hygiene ──────────────────────────────────────────────

class TestStage2ChannelHygiene:
    def test_emg_dropped(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = _raw_with_extra_channels(synthetic_raw, {"EMG": "eeg"})
        p.settings = preprocessing_settings
        assert "EMG" in p.raw.ch_names
        p._channel_hygiene()
        assert "EMG" not in p.raw.ch_names
        assert "EMG" in p._dropped_channels

    def test_hegoc_renamed_to_heog(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = _raw_with_extra_channels(synthetic_raw, {"HEGOC": "eeg"})
        p.settings = preprocessing_settings
        p._channel_hygiene()
        assert "HEGOC" not in p.raw.ch_names
        assert "HEOG" in p.raw.ch_names

    def test_montage_set_and_eeg_names_captured(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p._channel_hygiene()
        assert p.raw.get_montage() is not None
        assert len(p._post_hygiene_eeg_names) == len(
            mne.pick_types(p.raw.info, eeg=True)
        )

    def test_hygiene_skipped_when_disabled(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = _raw_with_extra_channels(synthetic_raw, {"EMG": "eeg"})
        preprocessing_settings["channel_hygiene"]["drop_emg"] = False
        p.settings = preprocessing_settings
        p._channel_hygiene()
        assert "EMG" in p.raw.ch_names


# ── Stage 3: Filtering / resampling ───────────────────────────────────────

class TestStage3Filter:
    def test_highpass_changes_data(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        original = p.raw.get_data().copy()
        p._highpass()
        assert not np.allclose(p.raw.get_data(), original)

    def test_highpass_keeps_sfreq(self, make_preprocessor, synthetic_raw, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        sfreq_before = p.raw.info["sfreq"]
        p._highpass()
        assert p.raw.info["sfreq"] == sfreq_before

    def test_early_stage_resamples_raw_to_target(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings  # stage == "early"
        p.run_step1a_filter()
        assert p.raw.info["sfreq"] == FINAL_RESAMPLE_RATE

    def test_late_stage_keeps_raw_full_rate(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        preprocessing_settings["resample_filter_stage"] = "late"
        p.settings = preprocessing_settings
        p.run_step1a_filter()
        assert p.raw.info["sfreq"] == synthetic_raw.info["sfreq"]

    def test_resample_epochs_returns_consistent_times(
        self, make_preprocessor, preprocessing_settings
    ):
        """_resample on Epochs must return epochs whose time vector matches the
        decimated data. Regression: the late path previously left a stale
        full-rate `times` (data 121 samples but len(times) 1201)."""
        p = make_preprocessor
        p.settings = preprocessing_settings  # FINAL_RESAMPLE_RATE == 100

        # Full-rate (1000 Hz) epochs: 4 trials, 8 ch, 600 samples (-0.1..0.5 s).
        sfreq, n_times, n_trials = 1000.0, 600, 4
        ch_names = [f"EEG{i:03d}" for i in range(8)]
        data = np.random.default_rng(0).standard_normal(
            (n_trials, len(ch_names), n_times)
        ) * 1e-5
        info = mne.create_info(ch_names, sfreq, "eeg")
        events = np.column_stack(
            [np.arange(n_trials), np.zeros(n_trials, int), np.ones(n_trials, int)]
        )
        epochs = mne.EpochsArray(
            data, info, events=events, tmin=-0.1, event_id={"x": 1}, verbose=False
        )

        out = p._resample(epochs)

        assert out.info["sfreq"] == 100.0
        assert len(out.times) == out.get_data().shape[-1]  # the bug
        assert out.get_data().shape[0] == n_trials         # trials preserved
        assert out.times[0] == pytest.approx(-0.1)         # tmin preserved

    def test_resample_raw_returns_consistent_times(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        """_resample on Raw returns a decimated Raw with consistent times."""
        p = make_preprocessor
        p.settings = preprocessing_settings
        out = p._resample(synthetic_raw.copy())
        assert out.info["sfreq"] == 100.0
        assert len(out.times) == out.get_data().shape[-1]


# ── Stage 4: Bad channels ─────────────────────────────────────────────────

class TestStage4BadChannels:
    def test_set_bad_channels_stores_list(self, make_preprocessor):
        make_preprocessor.set_bad_channels(["Fp1", "Oz"])
        assert make_preprocessor._bad_channels == ["Fp1", "Oz"]

    def test_no_bads_yields_none_weights(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings
        p.set_bad_channels([])
        p._interpolate_bads()
        assert p._interp_weights is None

    def test_interp_weights_correctness(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw.copy()
        p.settings = preprocessing_settings

        eeg_picks = mne.pick_types(p.raw.info, eeg=True)
        eeg_ch_names = [p.raw.ch_names[i] for i in eeg_picks]
        fp1_local_idx = eeg_ch_names.index("Fp1")
        good_local_indices = [i for i in range(len(eeg_ch_names)) if i != fp1_local_idx]

        p.raw._data[eeg_picks[fp1_local_idx], :] = 0.0
        good_data_before = p.raw.get_data(picks="eeg")[good_local_indices, :].copy()

        p.set_bad_channels(["Fp1"])
        p._interpolate_bads()

        mne_interpolated = p.raw.get_data(picks="eeg")[fp1_local_idx, :]
        weights = p._interp_weights  # (n_good, 1)
        predicted = weights.T @ good_data_before  # (1, n_times)

        np.testing.assert_allclose(predicted[0], mne_interpolated, atol=1e-10)


# ── Stage 5: Epoching ─────────────────────────────────────────────────────

class TestStage5Epoch:
    def test_epochs_shape(self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        assert isinstance(p.epochs, mne.Epochs)
        assert p.epochs.get_data().ndim == 3

    def test_epoch_event_ids_match_mapping(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        assert "red" in p.epochs.event_id
        assert "green" in p.epochs.event_id

    def test_unrecognised_events_fall_back(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"blue": 99})
        assert len(p.epochs) > 0

    def test_baseline_none_supported(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = synthetic_raw_with_events.copy()
        preprocessing_settings["epochs"]["baseline"] = None
        p.settings = preprocessing_settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        assert p.epochs.baseline is None


# ── Stage 6: Reference + ICA ──────────────────────────────────────────────

class TestStage6ReferenceAndICA:
    def _epoch_and_ref(self, p, raw, settings):
        p.raw = raw.copy()
        p.settings = settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        p._reference()
        return p

    def test_average_reference_applied(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._epoch_and_ref(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        eeg = p.epochs.get_data(picks="eeg")
        assert np.abs(eeg.mean(axis=1)).max() < 1e-12

    def test_fit_ica_returns_list_and_stores_ica(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._epoch_and_ref(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        suggested = p._fit_ica()
        assert isinstance(suggested, list)
        assert isinstance(p.ica, mne.preprocessing.ICA)
        assert p.ica.exclude == []

    def test_iclabel_suggestion_uses_drop_labels(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        preprocessing_settings["ica"]["iclabel"] = {
            "enabled": True, "drop_labels": ["eye", "muscle"],
        }
        p = self._epoch_and_ref(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        fake = {
            "labels": ["brain", "eye", "muscle", "brain"],
            "y_pred_proba": np.array([0.91, 0.99, 0.85, 0.77]),
        }
        with patch("mne_icalabel.label_components", return_value=fake):
            suggested = p._fit_ica()
        assert suggested == [1, 2]

    def test_component_labels_populated_aligned_by_index(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        preprocessing_settings["ica"]["iclabel"] = {
            "enabled": True, "drop_labels": ["eye"],
        }
        p = self._epoch_and_ref(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        fake = {
            "labels": ["brain", "eye", "muscle", "brain"],
            "y_pred_proba": np.array([0.91, 0.99, 0.85, 0.77]),
        }
        with patch("mne_icalabel.label_components", return_value=fake):
            p._fit_ica()
        assert p.component_labels == [
            ("brain", pytest.approx(0.91)),
            ("eye", pytest.approx(0.99)),
            ("muscle", pytest.approx(0.85)),
            ("brain", pytest.approx(0.77)),
        ]

    def test_iclabel_disabled_returns_empty(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._epoch_and_ref(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        # fixture default has iclabel disabled
        assert p._fit_ica() == []
        assert p.component_labels is None


# ── Stage 7: Apply + save ─────────────────────────────────────────────────

class TestStage7ApplyAndSave:
    def _prepare(self, p, raw, settings):
        p.raw = raw.copy()
        p.settings = settings
        p.epochs = p._epoch({"red": 1, "green": 2})
        p._reference()
        p._fit_ica()
        return p

    def test_fif_saved_with_subject_id(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings, tmp_path
    ):
        p = self._prepare(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        out = tmp_path / "epochs"
        result = p.run_step2_apply_and_save([0], out)
        saved = list(out.glob("*.fif"))
        assert len(saved) == 1
        assert p.subject_id in saved[0].name
        assert result["n_excluded"] == 1

    def test_late_stage_resamples_epochs(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings, tmp_path
    ):
        preprocessing_settings["resample_filter_stage"] = "late"
        p = self._prepare(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        assert p.epochs.info["sfreq"] == synthetic_raw_with_events.info["sfreq"]
        p.run_step2_apply_and_save([], tmp_path / "epochs")
        assert p.epochs.info["sfreq"] == FINAL_RESAMPLE_RATE


# ── Stage 8: export_online_state ──────────────────────────────────────────

class TestStage8ExportOnlineState:
    def _full(self, p, raw, settings):
        p.raw = raw.copy()
        p.settings = settings
        p._original_ch_names = list(p.raw.ch_names)
        p._post_hygiene_eeg_names = [
            p.raw.ch_names[i] for i in mne.pick_types(p.raw.info, eeg=True)
        ]
        p.epochs = p._epoch({"red": 1, "green": 2})
        p._reference()
        p._fit_ica()
        p.ica.exclude = [0]
        p.ica.apply(p.epochs, verbose=False)
        return p

    def test_raises_before_pipeline(self, make_preprocessor):
        with pytest.raises(RuntimeError):
            make_preprocessor.export_online_state()

    def test_returns_positional_keys_only(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._full(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        state = p.export_online_state()
        required = {
            "eeg_chunk_indices", "bad_indices", "interp_weights",
            "ica_unmixing", "ica_mixing", "ica_pca_components",
            "ica_pca_mean", "ica_exclude", "pre_whitener",
        }
        assert required <= state.keys()
        assert "ch_names" not in state
        assert "sfreq_offline" not in state
        assert "bad_channels" not in state

    def test_ica_matrices_are_numpy_and_exclude_matches(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._full(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        state = p.export_online_state()
        assert isinstance(state["ica_unmixing"], np.ndarray)
        assert isinstance(state["ica_pca_components"], np.ndarray)
        assert state["ica_exclude"] == [0]

    def test_pre_whitener_shape(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._full(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        state = p.export_online_state()
        n_eeg = len(p._post_hygiene_eeg_names)
        assert state["pre_whitener"].shape == (n_eeg, 1)

    def test_eeg_chunk_indices_drop_emg_positionally(
        self, make_preprocessor, synthetic_raw, preprocessing_settings
    ):
        p = make_preprocessor
        p.raw = _raw_with_extra_channels(synthetic_raw, {"EMG": "eeg"})
        p.settings = preprocessing_settings
        emg_pos = p.raw.ch_names.index("EMG")
        n_before = len(p.raw.ch_names)
        p.run_step1a_filter()
        idx = p._compute_eeg_chunk_indices()
        assert emg_pos not in idx
        assert len(idx) == n_before - 1

    def test_bad_indices_are_post_hygiene_positions(
        self, make_preprocessor, synthetic_raw_with_events, preprocessing_settings
    ):
        p = self._full(make_preprocessor, synthetic_raw_with_events, preprocessing_settings)
        p._bad_channels = [p._post_hygiene_eeg_names[3]]
        state = p.export_online_state()
        assert state["bad_indices"] == [3]
