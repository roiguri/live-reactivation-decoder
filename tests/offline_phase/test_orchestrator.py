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
        "eeg_chunk_indices": [0, 1, 2],
        "bad_indices": [],
        "interp_weights": None,
        "ica_unmixing": np.eye(3),
        "ica_mixing": np.eye(3),
        "ica_pca_components": np.eye(3),
        "ica_pca_mean": None,
        "ica_exclude": [],
        "pre_whitener": np.ones((3, 1)),
    }
    orchestrator._preprocessor = stub
    orchestrator._raw = raw  # simulate load_raw_data() having run
    return stub


def _attach_eval_results_stub(
    orchestrator: OfflineOrchestrator,
    epochs: mne.BaseEpochs,
    task_names: list[str],
) -> dict[str, Any]:
    """Inject a minimal ``_eval_results`` so ``run_training`` can derive
    per-task timepoints. Required since Step C — orchestrator pulls each task's
    CV-peak time from ``_eval_results['tasks'][name]['diagonal_auc']``.
    """
    times = np.asarray(epochs.times, dtype=float)
    eval_results = {
        "times": times,
        "tasks": {
            name: {"diagonal_auc": np.linspace(0.5, 0.9, times.size)}
            for name in task_names
        },
    }
    orchestrator._eval_results = eval_results
    return eval_results


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
        assert orc._preprocessor is None

    def test_load_eeg_raw_preserves_vmrk_annotations(
        self, tmp_path: Path, synthetic_raw_with_events: mne.io.RawArray
    ) -> None:
        """The revert: events come from the .vmrk (loaded by
        read_raw_brainvision); _load_eeg_raw must not overwrite or decode them."""
        orc = _make_orchestrator(tmp_path)
        src = synthetic_raw_with_events.copy()  # carries 'red'/'green' annotations
        expected = set(src.annotations.description)

        with patch(
            "mne.io.read_raw_brainvision", return_value=src
        ) as mock_read:
            result = orc._load_eeg_raw(tmp_path / "rec.vhdr")

        mock_read.assert_called_once()
        # Annotations survived untouched (no decoder, no set_annotations).
        assert set(result.annotations.description) == expected
        assert len(result.annotations) == len(synthetic_raw_with_events.annotations)
        # All synthetic channels are EEG → none dropped by the IO boundary.
        assert result.ch_names == synthetic_raw_with_events.ch_names


# ── TestRunStep1aFilter ───────────────────────────────────────────────────────


class TestRunStep1aFilter:
    def test_raises_if_raw_not_loaded(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="load_raw_data"):
            orc.run_step1a_filter()

    def test_creates_preprocessor_with_raw_and_returns_raw(
        self, tmp_path: Path, synthetic_raw: mne.io.RawArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        orc.set_file_path(tmp_path)
        orc._raw = synthetic_raw

        with patch(
            "backend.offline_phase.orchestrator.OfflinePreprocessor"
        ) as MockPrep:
            instance = MockPrep.return_value
            instance.run_step1a_filter.return_value = "filtered_raw"

            result = orc.run_step1a_filter()

        _, kwargs = MockPrep.call_args
        assert kwargs.get("raw") is synthetic_raw
        instance.run_step1a_filter.assert_called_once()
        assert result == "filtered_raw"


# ── TestSetBadChannels ────────────────────────────────────────────────────────


class TestSetBadChannels:
    def test_raises_if_no_preprocessor(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step1a_filter"):
            orc.set_bad_channels(["Fp1"])

    def test_forwards_to_preprocessor(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        stub = _attach_preprocessor_stub(orc)
        orc.set_bad_channels(["Fp1", "Oz"])
        stub.set_bad_channels.assert_called_once_with(["Fp1", "Oz"])


# ── TestRunStep1bFitIca ───────────────────────────────────────────────────────


class TestRunStep1bFitIca:
    def test_raises_if_no_preprocessor(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step1a_filter"):
            orc.run_step1b_fit_ica()

    def test_returns_ica_epochs_suggested_and_passes_event_mapping(
        self, tmp_path: Path
    ) -> None:
        sm = MagicMock()
        sm.get_event_mapping.return_value = {"red": 1}
        orc = _make_orchestrator(tmp_path, sm)
        stub = _attach_preprocessor_stub(orc)
        stub.run_step1b_fit_ica.return_value = ("ica", "epochs", [0, 2])

        ica, epochs, suggested = orc.run_step1b_fit_ica()

        stub.run_step1b_fit_ica.assert_called_once_with({"red": 1})
        assert (ica, epochs, suggested) == ("ica", "epochs", [0, 2])
        assert orc._epochs == "epochs"


# ── TestRunStep2ApplyAndSave ──────────────────────────────────────────────────


class TestRunStep2ApplyAndSave:
    def test_raises_if_no_preprocessor(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step1b_fit_ica"):
            orc.run_step2_apply_and_save([0])

    def test_raises_if_no_ica(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        _attach_preprocessor_stub(orc, ica=None)
        with pytest.raises(RuntimeError, match="run_step1b_fit_ica"):
            orc.run_step2_apply_and_save([0])

    def test_calls_preprocessor_and_returns_stats(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        stub = _attach_preprocessor_stub(orc)
        stub.run_step2_apply_and_save.return_value = {"n_epochs": 90, "n_excluded": 2}
        stub.epochs = synthetic_epochs

        result = orc.run_step2_apply_and_save([0, 1])

        stub.run_step2_apply_and_save.assert_called_once()
        assert orc._epochs is synthetic_epochs
        assert result == {"n_epochs": 90, "n_excluded": 2}

    def test_passes_excluded_components_to_preprocessor(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray
    ) -> None:
        orc = _make_orchestrator(tmp_path)
        stub = _attach_preprocessor_stub(orc)
        stub.run_step2_apply_and_save.return_value = {"n_epochs": 1, "n_excluded": 2}
        stub.epochs = synthetic_epochs

        orc.run_step2_apply_and_save([3, 5])

        call = stub.run_step2_apply_and_save.call_args
        assert call.kwargs["exclude_components"] == [3, 5]


# ── TestRunEvaluation ─────────────────────────────────────────────────────────


class TestRunEvaluation:
    def test_raises_if_no_epochs(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_step2_apply_and_save"):
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
            orc.run_training({"red decoder": 0.350})

    def test_saves_joblib_to_disk(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs
        _attach_eval_results_stub(
            orc, synthetic_epochs, [t["name"] for t in evaluator_settings["tasks"]]
        )

        timepoints = {
            t["name"]: float(synthetic_epochs.times[10])
            for t in evaluator_settings["tasks"]
        }
        result = orc.run_training(timepoints)

        expected_path = tmp_path / "models" / "decoder_pipeline.joblib"
        assert expected_path.exists()
        assert result["model_filepath"] == expected_path

    def test_assembles_live_artifact_spec_and_ui_state(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs
        task_names = [t["name"] for t in evaluator_settings["tasks"]]
        _attach_eval_results_stub(orc, synthetic_epochs, task_names)

        t10 = float(synthetic_epochs.times[10])
        timepoints = {name: t10 for name in task_names}
        orc.run_training(timepoints)

        spec = orc._live_artifact_spec
        assert spec is not None
        assert spec.models
        assert "ica_unmixing" in spec.online_state
        assert "eeg_chunk_indices" in spec.online_state
        assert "bad_indices" in spec.online_state
        assert "ch_names" not in spec.online_state
        # Per-task timepoints stored verbatim (the authoritative field).
        assert spec.metadata.decoding_timepoints == pytest.approx(timepoints)
        for name in task_names:
            assert isinstance(spec.metadata.decoding_timepoints[name], float)

        ui = orc._ui_state
        assert ui is not None
        assert "spatial_patterns" in ui
        assert "mne_info" in ui

    def test_returns_spatial_patterns_and_info(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs
        _attach_eval_results_stub(
            orc, synthetic_epochs, [t["name"] for t in evaluator_settings["tasks"]]
        )

        timepoints = {
            t["name"]: float(synthetic_epochs.times[10])
            for t in evaluator_settings["tasks"]
        }
        result = orc.run_training(timepoints)

        assert "spatial_patterns" in result
        assert "mne_info" in result

    def test_dict_timepoints_used_verbatim(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        """An explicit per-task dict is stored verbatim (no auto-derivation)."""
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs
        task_names = [t["name"] for t in evaluator_settings["tasks"]]
        _attach_eval_results_stub(orc, synthetic_epochs, task_names)

        # Distinct per-decoder timepoints (the whole point of the feature).
        chosen = {
            name: float(synthetic_epochs.times[5 + 3 * i])
            for i, name in enumerate(task_names)
        }
        orc.run_training(chosen)

        spec = orc._live_artifact_spec
        assert spec is not None
        assert spec.metadata.decoding_timepoints == pytest.approx(chosen)


# ── TestGetLiveArtifactSpec ───────────────────────────────────────────────────


class TestGetLiveArtifactSpec:
    def test_raises_if_training_not_done(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="run_training"):
            orc.get_live_artifact_spec()

    def test_returns_live_artifact_spec(
        self, tmp_path: Path, synthetic_epochs: mne.EpochsArray, evaluator_settings: dict
    ) -> None:
        sm = MagicMock()
        sm.get_decoder_settings.return_value = evaluator_settings
        orc = _make_orchestrator(tmp_path, sm)
        _attach_preprocessor_stub(orc)
        orc._epochs = synthetic_epochs
        _attach_eval_results_stub(
            orc, synthetic_epochs, [t["name"] for t in evaluator_settings["tasks"]]
        )

        timepoints = {
            t["name"]: float(synthetic_epochs.times[10])
            for t in evaluator_settings["tasks"]
        }
        orc.run_training(timepoints)

        spec = orc.get_live_artifact_spec()
        assert spec is orc._live_artifact_spec


# ── TestStateOrdering ─────────────────────────────────────────────────────────


class TestStateOrdering:
    """Confirm that out-of-order calls raise descriptive RuntimeErrors."""

    def test_step1a_before_load(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_step1a_filter()

    def test_step1b_before_step1a(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_step1b_fit_ica()

    def test_step2_before_step1b(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_step2_apply_and_save([])

    def test_evaluation_before_pipeline(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_evaluation()

    def test_training_before_evaluation(self, tmp_path: Path) -> None:
        orc = _make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError):
            orc.run_training({"red decoder": 0.350})
