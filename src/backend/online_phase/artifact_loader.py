from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class DecoderPipelineArtifact:
    """Unwrapped Phase 1 decoder pipeline artifact."""

    models: dict[str, Any]
    online_state: Any
    metadata: dict[str, Any]


_REQUIRED_ARTIFACT_KEYS = ("models", "online_state", "metadata")


def load_decoder_pipeline_artifact(path: str | Path) -> DecoderPipelineArtifact:
    """Load and validate the top-level Phase 1 decoder artifact envelope."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Decoder pipeline artifact not found: {artifact_path}"
        )

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Decoder pipeline artifact must be a dictionary.")

    missing = [key for key in _REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(
            "Decoder pipeline artifact missing required keys: "
            + ", ".join(missing)
        )

    models = artifact["models"]
    if not isinstance(models, dict):
        raise ValueError("Decoder pipeline models must be a non-empty dictionary.")
    if not models:
        raise ValueError("Decoder pipeline models must not be empty.")

    metadata = artifact["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Decoder pipeline metadata must be a dictionary.")

    return DecoderPipelineArtifact(
        models=models,
        online_state=artifact["online_state"],
        metadata=metadata,
    )
