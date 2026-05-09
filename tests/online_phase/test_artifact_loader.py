from __future__ import annotations

import joblib
import pytest

from backend.online_phase import (
    DecoderPipelineArtifact,
    LiveInferenceEngine,
    load_decoder_pipeline_artifact,
)


def _write_artifact(tmp_path, artifact):
    path = tmp_path / "decoder_pipeline.joblib"
    joblib.dump(artifact, path)
    return path


def _valid_artifact(online_state=None):
    return {
        "models": {"red decoder": "opaque model"},
        "online_state": online_state if online_state is not None else {"ch_names": []},
        "metadata": {"source": "test", "feature_width": 64},
    }


def test_valid_artifact_loads_successfully(tmp_path):
    path = _write_artifact(tmp_path, _valid_artifact())

    artifact = load_decoder_pipeline_artifact(path)

    assert isinstance(artifact, DecoderPipelineArtifact)
    assert artifact.models == {"red decoder": "opaque model"}
    assert artifact.metadata == {"source": "test", "feature_width": 64}


def test_loader_returns_exact_online_state_object_from_loaded_artifact(
    tmp_path,
    monkeypatch,
):
    online_state = {"bad_channels": ["Fz"], "nested": {"kept": True}}
    loaded_payload = _valid_artifact(online_state=online_state)
    path = tmp_path / "decoder_pipeline.joblib"
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "backend.online_phase.artifact_loader.joblib.load",
        lambda loaded_path: loaded_payload,
    )

    artifact = load_decoder_pipeline_artifact(path)

    assert artifact.online_state is online_state


def test_missing_artifact_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Decoder pipeline artifact not found"):
        load_decoder_pipeline_artifact(tmp_path / "missing.joblib")


def test_non_dict_artifact_raises_value_error(tmp_path):
    path = _write_artifact(tmp_path, ["not", "a", "dict"])

    with pytest.raises(ValueError, match="must be a dictionary"):
        load_decoder_pipeline_artifact(path)


@pytest.mark.parametrize("missing_key", ["models", "online_state", "metadata"])
def test_missing_required_top_level_keys_raise_value_error(tmp_path, missing_key):
    artifact = _valid_artifact()
    artifact.pop(missing_key)
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(ValueError, match=missing_key):
        load_decoder_pipeline_artifact(path)


def test_empty_models_raise_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = {}
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(ValueError, match="must not be empty"):
        load_decoder_pipeline_artifact(path)


def test_non_dict_models_raise_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = ["not", "a", "dict"]
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(ValueError, match="models must be a non-empty dictionary"):
        load_decoder_pipeline_artifact(path)


def test_non_dict_metadata_raises_value_error(tmp_path):
    artifact = _valid_artifact()
    artifact["metadata"] = ["not", "a", "dict"]
    path = _write_artifact(tmp_path, artifact)

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        load_decoder_pipeline_artifact(path)


def test_loader_does_not_validate_model_runtime_interface(tmp_path):
    artifact = _valid_artifact()
    artifact["models"] = {"red decoder": object()}
    path = _write_artifact(tmp_path, artifact)

    loaded = load_decoder_pipeline_artifact(path)

    assert "red decoder" in loaded.models


def test_online_phase_package_exports_runtime_components():
    assert LiveInferenceEngine is not None
    assert DecoderPipelineArtifact is not None
    assert load_decoder_pipeline_artifact is not None
