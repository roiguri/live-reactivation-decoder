"""
Unit tests for ModelEvaluator.

Each test class covers one logical concern. Private methods are called directly
where they are the simplest target; run_evaluation() is used for integration-level
checks that require the full pipeline.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from backend.offline_phase.evaluator import ModelEvaluator


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_times_match_epochs(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        np.testing.assert_array_equal(ev.times, synthetic_epochs.times)

    def test_settings_stored(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        assert ev.settings is evaluator_settings


# ── _get_task_data ────────────────────────────────────────────────────────────

class TestGetTaskData:
    def test_X_shape(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]
        X, _ = ev._get_task_data(task)
        assert X.ndim == 3
        assert X.shape[1] == synthetic_epochs.info["nchan"]
        assert X.shape[2] == len(synthetic_epochs.times)

    def test_y_shape_matches_X(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]
        X, y = ev._get_task_data(task)
        assert y.ndim == 1
        assert len(y) == X.shape[0]

    def test_binary_labels(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        for task in evaluator_settings["tasks"]:
            _, y = ev._get_task_data(task)
            assert set(y).issubset({0, 1})

    def test_pos_trials_are_labeled_one(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]  # pos: red (30 trials)
        _, y = ev._get_task_data(task)
        assert np.sum(y == 1) == len(synthetic_epochs["red"])

    def test_missing_label_raises(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        bad_task = {"name": "bad", "pos_labels": ["blue"], "neg_labels": ["red"]}
        with pytest.raises(ValueError, match="not found"):
            ev._get_task_data(bad_task)

    def test_single_class_raises(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        # All three labels are positive → y will be all 1s
        all_pos_task = {
            "name": "all_pos",
            "pos_labels": ["red", "green", "yellow"],
            "neg_labels": [],
        }
        with pytest.raises(ValueError, match="one class"):
            ev._get_task_data(all_pos_task)


# ── _build_classifier ─────────────────────────────────────────────────────────

class TestBuildClassifier:
    def test_returns_pipeline(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        clf = ev._build_classifier()
        assert isinstance(clf, Pipeline)

    def test_last_step_is_lda(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        clf = ev._build_classifier()
        assert isinstance(clf[-1], LinearDiscriminantAnalysis)

    def test_lda_params_applied(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        clf = ev._build_classifier()
        lda = clf[-1]
        assert lda.solver == "lsqr"
        assert lda.shrinkage == "auto"


# ── run_evaluation ────────────────────────────────────────────────────────────

class TestRunEvaluation:
    @pytest.fixture
    def result(self, synthetic_epochs, evaluator_settings):
        ev = ModelEvaluator(synthetic_epochs, evaluator_settings)
        return ev.run_evaluation()

    def test_top_level_keys(self, result):
        assert set(result.keys()) == {"times", "suggested_timepoint", "average_peak_auc", "tasks"}

    def test_times_matches_epochs(self, result, synthetic_epochs):
        np.testing.assert_array_equal(result["times"], synthetic_epochs.times)

    def test_task_keys_present(self, result, evaluator_settings):
        expected = {t["name"] for t in evaluator_settings["tasks"]}
        assert set(result["tasks"].keys()) == expected

    def test_per_task_keys(self, result, evaluator_settings):
        required = {"diagonal_auc", "tgm_matrix", "peak_auc", "chance_level"}
        for task_data in result["tasks"].values():
            assert set(task_data.keys()) == required

    def test_diagonal_auc_shape(self, result, synthetic_epochs):
        n_times = len(synthetic_epochs.times)
        for task_data in result["tasks"].values():
            assert task_data["diagonal_auc"].shape == (n_times,)

    def test_tgm_is_square(self, result, synthetic_epochs):
        n_times = len(synthetic_epochs.times)
        for task_data in result["tasks"].values():
            assert task_data["tgm_matrix"].shape == (n_times, n_times)

    def test_peak_auc_equals_max_diagonal(self, result):
        for task_data in result["tasks"].values():
            assert task_data["peak_auc"] == pytest.approx(np.max(task_data["diagonal_auc"]))

    def test_chance_level_is_half(self, result):
        for task_data in result["tasks"].values():
            assert task_data["chance_level"] == pytest.approx(0.5)

    def test_suggested_timepoint_in_times(self, result):
        assert result["suggested_timepoint"] in result["times"]

    def test_average_peak_auc_is_float_in_range(self, result):
        auc = result["average_peak_auc"]
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_empty_tasks_raises(self, synthetic_epochs):
        ev = ModelEvaluator(synthetic_epochs, {"tasks": []})
        with pytest.raises(ValueError, match="no tasks"):
            ev.run_evaluation()

    def test_single_task_config(self, synthetic_epochs):
        settings = {
            "model": "LDA",
            "params": {"solver": "lsqr", "shrinkage": "auto"},
            "cv": {"k": 3},
            "tasks": [
                {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow"]}
            ],
        }
        result = ModelEvaluator(synthetic_epochs, settings).run_evaluation()
        assert len(result["tasks"]) == 1
        assert "red decoder" in result["tasks"]
