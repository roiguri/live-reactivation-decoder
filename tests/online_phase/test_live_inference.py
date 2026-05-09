from __future__ import annotations

import joblib
import pytest

from backend.online_phase.live_inference import LiveInferenceEngine


class FakeModel:
    def predict_proba(self, features):
        return features


class ModelWithoutPredictProba:
    pass


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "decoder_pipeline.joblib"
    joblib.dump(artifact, path)
    return path


def _valid_artifact(online_state=None):
    return {
        "models": {"red decoder": FakeModel()},
        "online_state": online_state if online_state is not None else {"ch_names": []},
        "metadata": {"source": "test"},
    }


def test_valid_artifact_loads_successfully(tmp_path):
    path = _write_artifact(tmp_path, _valid_artifact())
    engine = LiveInferenceEngine(path)

    engine.load_pipeline()

    assert isinstance(engine.models, dict)
    assert isinstance(engine.metadata, dict)


def test_load_pipeline_returns_exact_online_state_object(tmp_path):
    online_state = {"bad_channels": ["Fz"], "nested": {"kept": True}}
    artifact = _valid_artifact(online_state=online_state)
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    loaded_online_state = engine.load_pipeline()

    assert loaded_online_state is engine.online_state
    assert loaded_online_state == online_state


def test_load_pipeline_stores_models_and_metadata(tmp_path):
    artifact = _valid_artifact()
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    engine.load_pipeline()

    assert set(engine.models) == {"red decoder"}
    assert isinstance(engine.models["red decoder"], FakeModel)
    assert engine.metadata == artifact["metadata"]


def test_missing_artifact_raises_file_not_found(tmp_path):
    engine = LiveInferenceEngine(tmp_path / "missing.joblib")

    with pytest.raises(FileNotFoundError, match="Decoder pipeline artifact not found"):
        engine.load_pipeline()


def test_non_dict_artifact_raises_value_error(tmp_path):
    path = _write_artifact(tmp_path, ["not", "a", "dict"])
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match="must be a dictionary"):
        engine.load_pipeline()


@pytest.mark.parametrize("missing_key", ["models", "online_state", "metadata"])
def test_missing_required_top_level_keys_raise_value_error(tmp_path, missing_key):
    artifact = _valid_artifact()
    artifact.pop(missing_key)
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match=missing_key):
        engine.load_pipeline()


def test_empty_models_raise_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = {}
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match="must not be empty"):
        engine.load_pipeline()


def test_non_dict_models_raise_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = ["not", "a", "dict"]
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match="models must be a non-empty dictionary"):
        engine.load_pipeline()


def test_model_without_callable_predict_proba_raises_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = {"red decoder": ModelWithoutPredictProba()}
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match="predict_proba"):
        engine.load_pipeline()


def test_non_dict_metadata_raises_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["metadata"] = ["not", "a", "dict"]
    path = _write_artifact(tmp_path, artifact)
    engine = LiveInferenceEngine(path)

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        engine.load_pipeline()
