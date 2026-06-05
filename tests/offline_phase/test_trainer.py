"""
Unit tests for ModelTrainer.

Each test class covers one logical concern. Private methods are called directly
where they are the simplest target; run_training() is used for integration-level
checks that require the full pipeline.
"""

import warnings

import numpy as np
import pytest
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from backend.offline_phase.trainer import ModelTrainer


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_epochs_stored(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        assert tr.epochs is synthetic_epochs

    def test_settings_stored(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        assert tr.settings is evaluator_settings

    def test_times_match_epochs(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        np.testing.assert_array_equal(tr.times, synthetic_epochs.times)


# ── _extract_features ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_X_t_shape_is_2d(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]
        X_t, _ = tr._extract_features(task, timepoint=0.0)
        assert X_t.ndim == 2
        assert X_t.shape[1] == synthetic_epochs.info["nchan"]

    def test_n_trials_matches_selected_epochs(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]  # pos: red (30), neg: green+yellow (60)
        X_t, y = tr._extract_features(task, timepoint=0.0)
        assert X_t.shape[0] == len(y)
        assert X_t.shape[0] == 90  # 30 red + 30 green + 30 yellow

    def test_y_is_binary(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        for task in evaluator_settings["tasks"]:
            _, y = tr._extract_features(task, timepoint=0.0)
            assert set(y).issubset({0, 1})

    def test_pos_trials_labeled_one(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]  # pos: red (30 trials)
        _, y = tr._extract_features(task, timepoint=0.0)
        assert np.sum(y == 1) == len(synthetic_epochs["red"])

    def test_missing_label_raises(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        bad_task = {"name": "bad", "pos_labels": ["blue"], "neg_labels": ["red"]}
        with pytest.raises(ValueError, match="not found"):
            tr._extract_features(bad_task, timepoint=0.0)

    def test_out_of_bounds_timepoint_raises(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]
        with pytest.raises(ValueError, match="out of bounds"):
            tr._extract_features(task, timepoint=99.0)


# ── _train_classifier ─────────────────────────────────────────────────────────

class TestTrainClassifier:
    def _get_Xy(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        task = evaluator_settings["tasks"][0]
        return tr._extract_features(task, timepoint=0.0)

    def test_returns_fitted_pipeline_lda(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        X_t, y = self._get_Xy(synthetic_epochs, evaluator_settings)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model, Pipeline)
        assert isinstance(model[-1], LinearDiscriminantAnalysis)

    def test_lda_has_coef_after_fit(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        X_t, y = self._get_Xy(synthetic_epochs, evaluator_settings)
        model = tr._train_classifier(X_t, y)
        assert hasattr(model[-1], "coef_")

    def test_returns_fitted_pipeline_logistic(self, synthetic_epochs, logistic_evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, logistic_evaluator_settings)
        task = logistic_evaluator_settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model[-1], LogisticRegression)

    def test_logistic_fit_emits_no_penalty_deprecation(
        self, synthetic_epochs, logistic_evaluator_settings
    ):
        # sklearn 1.8 deprecated penalty=; fitting must not trip the FutureWarning.
        tr = ModelTrainer(synthetic_epochs, logistic_evaluator_settings)
        task = logistic_evaluator_settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            tr._train_classifier(X_t, y)

    def test_returns_fitted_pipeline_svm(self, synthetic_epochs, svm_evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, svm_evaluator_settings)
        task = svm_evaluator_settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model[-1], SVC)
        assert model[-1].probability is True

    def test_standard_scaler_wrapping(self, synthetic_epochs, evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        X_t, y = self._get_Xy(synthetic_epochs, evaluator_settings)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model[0], StandardScaler)

    def test_robust_scaler_wrapping(self, synthetic_epochs, svm_evaluator_settings):
        tr = ModelTrainer(synthetic_epochs, svm_evaluator_settings)
        task = svm_evaluator_settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model[0], RobustScaler)

    def test_no_scaler_returns_bare_classifier(self, synthetic_epochs, evaluator_settings):
        settings = {**evaluator_settings, "scale_method": None}
        tr = ModelTrainer(synthetic_epochs, settings)
        task = evaluator_settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        model = tr._train_classifier(X_t, y)
        assert isinstance(model, LinearDiscriminantAnalysis)


# ── _calculate_spatial_patterns ───────────────────────────────────────────────

class TestCalculateSpatialPatterns:
    def _fit_model(self, synthetic_epochs, settings):
        tr = ModelTrainer(synthetic_epochs, settings)
        task = settings["tasks"][0]
        X_t, y = tr._extract_features(task, timepoint=0.0)
        model = tr._train_classifier(X_t, y)
        return tr, X_t, model

    def test_output_shape(self, synthetic_epochs, evaluator_settings):
        tr, X_t, model = self._fit_model(synthetic_epochs, evaluator_settings)
        pattern = tr._calculate_spatial_patterns(X_t, model)
        assert pattern.shape == (synthetic_epochs.info["nchan"],)

    def test_pattern_is_nonzero(self, synthetic_epochs, evaluator_settings):
        tr, X_t, model = self._fit_model(synthetic_epochs, evaluator_settings)
        pattern = tr._calculate_spatial_patterns(X_t, model)
        assert np.any(pattern != 0)

    def test_bare_classifier_without_pipeline(self, synthetic_epochs, evaluator_settings):
        settings = {**evaluator_settings, "scale_method": None}
        tr, X_t, model = self._fit_model(synthetic_epochs, settings)
        pattern = tr._calculate_spatial_patterns(X_t, model)
        assert pattern.shape == (synthetic_epochs.info["nchan"],)

    def test_no_coef_raises(self, synthetic_epochs, evaluator_settings):
        from sklearn.neighbors import KNeighborsClassifier
        tr = ModelTrainer(synthetic_epochs, evaluator_settings)
        knn = KNeighborsClassifier().fit(
            np.zeros((10, synthetic_epochs.info["nchan"])), np.array([0]*5 + [1]*5)
        )
        with pytest.raises(ValueError, match="coef_"):
            tr._calculate_spatial_patterns(
                np.zeros((10, synthetic_epochs.info["nchan"])), knn
            )


# ── run_training ──────────────────────────────────────────────────────────────

class TestRunTraining:
    @pytest.fixture
    def result(self, synthetic_epochs, evaluator_settings):
        return ModelTrainer(synthetic_epochs, evaluator_settings).run_training(0.0)

    def test_top_level_keys(self, result):
        assert set(result.keys()) == {"models", "spatial_patterns", "mne_info"}

    def test_models_keys_match_tasks(self, result, evaluator_settings):
        expected = {t["name"] for t in evaluator_settings["tasks"]}
        assert set(result["models"].keys()) == expected

    def test_models_are_fitted(self, result):
        for model in result["models"].values():
            assert hasattr(model[-1], "coef_")

    def test_spatial_patterns_keys_match_tasks(self, result, evaluator_settings):
        expected = {t["name"] for t in evaluator_settings["tasks"]}
        assert set(result["spatial_patterns"].keys()) == expected

    def test_spatial_patterns_shape(self, result, synthetic_epochs):
        n_ch = synthetic_epochs.info["nchan"]
        for pattern in result["spatial_patterns"].values():
            assert pattern.shape == (n_ch,)

    def test_mne_info_type(self, result):
        assert isinstance(result["mne_info"], mne.Info)

    def test_empty_tasks_raises(self, synthetic_epochs):
        settings = {
            "model": "LDA",
            "params": {"solver": "lsqr", "shrinkage": "auto"},
            "scale_method": "standard",
            "random_state": 42,
            "tasks": [],
        }
        with pytest.raises(ValueError, match="no tasks"):
            ModelTrainer(synthetic_epochs, settings).run_training(0.0)

    def test_single_task_config(self, synthetic_epochs):
        settings = {
            "model": "LDA",
            "params": {"solver": "lsqr", "shrinkage": "auto"},
            "scale_method": "standard",
            "random_state": 42,
            "tasks": [
                {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow"]}
            ],
        }
        result = ModelTrainer(synthetic_epochs, settings).run_training(0.0)
        assert len(result["models"]) == 1
        assert "red decoder" in result["models"]
