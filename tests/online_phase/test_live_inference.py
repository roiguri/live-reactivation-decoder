from __future__ import annotations

import numpy as np
import pytest

from backend.online_phase.live_inference import LiveInferenceEngine


class FakeModel:
    def __init__(
        self,
        *,
        n_features_in_=64,
        classes=(0, 1),
        probabilities=None,
    ):
        if n_features_in_ is not None:
            self.n_features_in_ = n_features_in_
        if classes is not None:
            self.classes_ = np.array(classes)
        self.probabilities = probabilities

    def predict_proba(self, features):
        if self.probabilities is not None:
            if callable(self.probabilities):
                return self.probabilities(features)
            return np.asarray(self.probabilities)

        n_rows = np.asarray(features).shape[0]
        positive = np.full(n_rows, 0.75)
        return np.column_stack([1.0 - positive, positive])


class ModelWithoutPredictProba:
    classes_ = np.array([0, 1])


def _features(n_rows=3, n_features=64):
    return np.ones((n_rows, n_features))


def test_engine_stores_models_and_metadata():
    models = {"red decoder": FakeModel()}
    metadata = {"source": "test", "feature_width": 64}

    engine = LiveInferenceEngine(models=models, metadata=metadata)

    assert engine.models is models
    assert engine.metadata is metadata


def test_engine_does_not_expose_online_state():
    engine = LiveInferenceEngine(
        models={"red decoder": FakeModel()},
        metadata={"feature_width": 64},
    )

    assert not hasattr(engine, "online_state")


def test_feature_width_comes_from_metadata():
    engine = LiveInferenceEngine(
        models={"red decoder": FakeModel(n_features_in_=None)},
        metadata={"feature_width": 128},
    )

    assert engine.feature_width == 128


def test_feature_width_falls_back_to_model_n_features_in():
    engine = LiveInferenceEngine(
        models={"red decoder": FakeModel(n_features_in_=32)},
        metadata={},
    )

    assert engine.feature_width == 32


def test_inconsistent_model_feature_widths_raise_value_error():
    models = {
        "red decoder": FakeModel(n_features_in_=32),
        "green decoder": FakeModel(n_features_in_=64),
    }

    with pytest.raises(ValueError, match="inconsistent n_features_in_"):
        LiveInferenceEngine(models=models, metadata={})


def test_model_feature_width_must_match_metadata_feature_width():
    models = {"red decoder": FakeModel(n_features_in_=32)}

    with pytest.raises(ValueError, match="do not match metadata"):
        LiveInferenceEngine(models=models, metadata={"feature_width": 64})


@pytest.mark.parametrize("feature_width", ["64", 0, -1, True])
def test_non_integer_or_non_positive_feature_width_raises_value_error(feature_width):
    models = {"red decoder": FakeModel()}

    with pytest.raises(ValueError, match="positive integer"):
        LiveInferenceEngine(models=models, metadata={"feature_width": feature_width})


def test_missing_feature_width_and_model_width_raise_value_error():
    models = {"red decoder": FakeModel(n_features_in_=None)}

    with pytest.raises(ValueError, match="metadata must include feature_width"):
        LiveInferenceEngine(models=models, metadata={})


def test_missing_feature_width_with_partially_untyped_models_raises_value_error():
    models = {
        "red decoder": FakeModel(n_features_in_=32),
        "green decoder": FakeModel(n_features_in_=None),
    }

    with pytest.raises(ValueError, match="some models do not expose n_features_in_"):
        LiveInferenceEngine(models=models, metadata={})


def test_empty_models_raise_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        LiveInferenceEngine(models={}, metadata={"feature_width": 64})


def test_non_dict_models_raise_value_error():
    with pytest.raises(ValueError, match="models must be a non-empty dictionary"):
        LiveInferenceEngine(models=["not", "a", "dict"], metadata={"feature_width": 64})


def test_model_without_callable_predict_proba_raises_value_error():
    models = {"red decoder": ModelWithoutPredictProba()}

    with pytest.raises(ValueError, match="predict_proba"):
        LiveInferenceEngine(models=models, metadata={"feature_width": 64})


def test_non_dict_metadata_raises_value_error():
    models = {"red decoder": FakeModel()}

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        LiveInferenceEngine(models=models, metadata=["not", "a", "dict"])


def test_predict_rejects_non_2d_input():
    engine = LiveInferenceEngine(
        models={"red decoder": FakeModel()},
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="2D array"):
        engine.predict(np.ones(64))


def test_predict_rejects_wrong_feature_width():
    engine = LiveInferenceEngine(
        models={"red decoder": FakeModel()},
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="feature width"):
        engine.predict(_features(n_features=63))


def test_predict_empty_feature_batch_returns_empty_vectors_for_each_task():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(),
            "green decoder": FakeModel(),
        },
        metadata={"feature_width": 64},
    )

    predictions = engine.predict(_features(n_rows=0))

    assert set(predictions) == {"red decoder", "green decoder"}
    assert predictions["red decoder"].shape == (0,)
    assert predictions["green decoder"].shape == (0,)


def test_predict_returns_one_positive_probability_vector_per_task():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                probabilities=np.array([[0.1, 0.9], [0.7, 0.3]])
            ),
            "green decoder": FakeModel(
                probabilities=np.array([[0.8, 0.2], [0.4, 0.6]])
            ),
        },
        metadata={"feature_width": 64},
    )

    predictions = engine.predict(_features(n_rows=2))

    assert set(predictions) == {"red decoder", "green decoder"}
    np.testing.assert_allclose(predictions["red decoder"], [0.9, 0.3])
    np.testing.assert_allclose(predictions["green decoder"], [0.2, 0.6])
    assert predictions["red decoder"].shape == (2,)


def test_predict_uses_configured_positive_class_metadata():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                classes=("negative", "target"),
                probabilities=np.array([[0.35, 0.65]]),
            )
        },
        metadata={"feature_width": 64, "positive_class": "target"},
    )

    predictions = engine.predict(_features(n_rows=1))

    np.testing.assert_allclose(predictions["red decoder"], [0.65])


def test_predict_falls_back_to_binary_target_label_one():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                classes=(0, 1),
                probabilities=np.array([[0.25, 0.75]]),
            )
        },
        metadata={"feature_width": 64},
    )

    predictions = engine.predict(_features(n_rows=1))

    np.testing.assert_allclose(predictions["red decoder"], [0.75])


def test_predict_raises_when_positive_class_is_not_identifiable():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                classes=(0, 2),
                probabilities=np.array([[0.4, 0.6]]),
            )
        },
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="positive class"):
        engine.predict(_features(n_rows=1))


def test_predict_raises_when_model_has_no_classes():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                classes=None,
                probabilities=np.array([[0.4, 0.6]]),
            )
        },
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="classes_"):
        engine.predict(_features(n_rows=1))


def test_predict_rejects_probability_matrix_with_wrong_row_count():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                probabilities=np.array([[0.1, 0.9], [0.2, 0.8]])
            )
        },
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="row count"):
        engine.predict(_features(n_rows=1))


def test_predict_rejects_probability_matrix_with_too_few_columns():
    engine = LiveInferenceEngine(
        models={
            "red decoder": FakeModel(
                classes=(0, 1),
                probabilities=np.array([[0.9]]),
            )
        },
        metadata={"feature_width": 64},
    )

    with pytest.raises(ValueError, match="at least two probability columns"):
        engine.predict(_features(n_rows=1))
