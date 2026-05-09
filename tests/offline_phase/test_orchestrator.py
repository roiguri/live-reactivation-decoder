from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from backend.offline_phase.orchestrator import OfflineOrchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_orchestrator(
    tmp_path: Path,
    settings_manager: Any = None,
) -> OfflineOrchestrator:
    if settings_manager is None:
        settings_manager = MagicMock()
        settings_manager.get_preprocessing_params.return_value = {}
        settings_manager.get_decoder_settings.return_value = {}
        settings_manager.get_event_mapping.return_value = {}
    return OfflineOrchestrator(settings_manager, tmp_path)


def _attach_preprocessor_stub(
    orchestrator: OfflineOrchestrator,
    *,
    raw: Any = "stub",
    ica: Any = "stub",
    epochs: Any = None,
) -> MagicMock:
    """Injects a mock OfflinePreprocessor and sets orchestrator raw to a sentinel."""
    stub = MagicMock()
    stub.raw = raw
    stub.ica = ica
    stub.epochs = epochs
    stub.export_online_state.return_value = {
        "bad_channels": [],
        "ica_unmixing": np.eye(3),
        "ica_mixing": np.eye(3),
        "ica_pca_components": np.eye(3),
        "ica_pca_mean": None,
        "ica_exclude": [],
        "ch_names": ["Fz", "Cz", "Pz"],
        "sfreq_offline": 256.0,
    }
    orchestrator._preprocessor = stub
    orchestrator._raw = raw  # simulate load_raw_data() having run
    return stub


# ── TestSetFilePath ───────────────────────────────────────────────────────────


class TestSetFilePath:
    def test_stores_path(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(tmp_path)
        assert orc._data_dir == tmp_path

    def test_accepts_string(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(str(tmp_path))
        assert orc._data_dir == tmp_path


# ── TestLoadRawData ───────────────────────────────────────────────────────────


class TestLoadRawData:
    def test_raises_without_file_path(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(ValueError, match="set_file_path"):
            orc.load_raw_data()

    def test_raises_when_no_vhdr(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(tmp_path)
        with pytest.raises(FileNotFoundError):
            orc.load_raw_data()

    def test_stores_raw_and_does_not_create_preprocessor(
        self, tmp_path: Path, synthetic_raw: mne.io.RawArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(tmp_path)

        with (
            patch.object(orc, "_find_vhdr", return_value=tmp_path / "test.vhdr"),
            patch.object(orc, "_load_eeg_raw", return_value=synthetic_raw),
        ):
            orc.load_raw_data()

        assert orc._raw is synthetic_raw
        assert orc._preprocessor is None  # preprocessor not created yet


# ── TestRunStep1PrepareIca ────────────────────────────────────────────────────


class TestRunStep1PrepareIca:
    def test_raises_if_raw_not_loaded(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="load_raw_data"):
            orc.run_step1_prepare_ica()

    def test_creates_preprocessor_with_raw_in_constructor(
        self, tmp_path: Path, synthetic_raw: mne.io.RawArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(tmp_path)
        orc._raw = synthetic_raw

        fake_ica = MagicMock(spec=mne.preprocessing.ICA)

        with patch(
            "backend.offline_phase.orchestrator.OfflinePreprocessor"
        ) as MockPrep:
            instance = MockPrep.return_value
            instance.ica = fake_ica
            instance.run_step1_prepare_ica.return_value = [0, 2]

            ica_obj, suggested = orc.run_step1_prepare_ica()

        # raw must be passed as constructor keyword argument
        _, kwargs = MockPrep.call_args
        assert kwargs.get("raw") is synthetic_raw
        instance.run_step1_prepare_ica.assert_called_once()
        assert ica_obj is fake_ica
        assert suggested == [0, 2]


# ── TestRunStep2FinishPipeline ────────────────────────────────────────────────


class TestRunStep2FinishPipeline:
    def test_raises_if_no_preprocessor(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step1_prepare_ica"):
            orc.run_step2_finish_pipeline([0])

    def test_raises_if_no_ica(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        _attach_preprocessor_stub(orc, ica=None)
        with pytest.raises(RuntimeError, match="run_step1_prepare_ica"):
            orc.run_step2_finish_pipeline([0])

    def test_calls_preprocessor_and_returns_stats(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        stub = _attach_preprocessor_stub(orc)

        def _side_effect(**_kwargs):
            stub.epochs = synthetic_epochs

        stub.run_step2_finish_pipeline.side_effect = _side_effect

        result = orc.run_step2_finish_pipeline([0, 1])

        stub.run_step2_finish_pipeline.assert_called_once()
        assert orc._epochs is synthetic_epochs
        assert result == {"n_epochs": len(synthetic_epochs)}

    def test_passes_excluded_components_to_preprocessor(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        stub = _attach_preprocessor_stub(orc)

        def _side_effect(exclude_components, **_kwargs):
            stub.epochs = synthetic_epochs

        stub.run_step2_finish_pipeline.side_effect = _side_effect

        orc.run_step2_finish_pipeline([3, 5])

        call_kwargs = stub.run_step2_finish_pipeline.call_args
        assert call_kwargs.kwargs["exclude_components"] == [3, 5]


# ── TestRunEvaluation ─────────────────────────────────────────────────────────


class TestRunEvaluation:
    def test_raises_if_no_epochs(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step2_finish_pipeline"):
            orc.run_evaluation()

    def test_returns_evaluator_result(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs

        result = orc.run_evaluation()

        assert "suggested_timepoint" in result
        assert "times" in result
        assert "tasks" in result
        assert orc._eval_results is result


# ── TestRunTraining ───────────────────────────────────────────────────────────


class TestRunTraining:
    def test_raises_if_no_epochs(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_evaluation"):
            orc.run_training(0.350)

    def test_saves_joblib_to_disk(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs

        timepoint = float(synthetic_epochs.times[10])
        result = orc.run_training(timepoint)

        expected_path = tmp_path / "models" / "decoder_pipeline.joblib"
        assert expected_path.exists()
        assert result["model_filepath"] == expected_path

    def test_assembles_online_state(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs

        timepoint = float(synthetic_epochs.times[10])
        orc.run_training(timepoint)

        state = orc.online_state
        assert "models" in state
        assert "spatial_patterns" in state
        assert "mne_info" in state
        assert "decoding_timepoint" in state
        assert state["decoding_timepoint"] == timepoint
        assert "ica_unmixing" in state
        assert "ch_names" in state

    def test_returns_spatial_patterns_and_info(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs

        timepoint = float(synthetic_epochs.times[10])
        result = orc.run_training(timepoint)

        assert "spatial_patterns" in result
        assert "mne_info" in result


# ── TestGetOnlineState ────────────────────────────────────────────────────────


class TestGetOnlineState:
    def test_raises_if_training_not_done(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_training"):
            orc.get_online_state_for_live_phase()

    def test_returns_same_dict_as_online_state(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs

        timepoint = float(synthetic_epochs.times[10])
        orc.run_training(timepoint)

        state = orc.get_online_state_for_live_phase()
        assert state is orc.online_state


# ── TestStateOrdering ─────────────────────────────────────────────────────────


class TestStateOrdering:
    """Confirm that out-of-order calls raise descriptive RuntimeErrors."""

    def test_prepare_ica_before_load(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_step1_prepare_ica()

    def test_pipeline_before_prepare_ica(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_step2_finish_pipeline([])

    def test_evaluation_before_pipeline(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_evaluation()

    def test_training_before_evaluation(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_training(0.350)
