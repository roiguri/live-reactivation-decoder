from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class LiveInferenceEngine:
    """Load the Phase 1 decoder artifact envelope for live inference."""

    _REQUIRED_ARTIFACT_KEYS = ("models", "online_state", "metadata")

    def __init__(self, pipeline_filepath: str | Path) -> None:
        self.pipeline_filepath = Path(pipeline_filepath)
        self.models: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None
        self.online_state: dict[str, Any] | None = None

    def load_pipeline(self) -> dict[str, Any]:
        """Load and validate the top-level decoder artifact envelope."""
        if not self.pipeline_filepath.exists():
            raise FileNotFoundError(
                f"Decoder pipeline artifact not found: {self.pipeline_filepath}"
            )

        artifact = joblib.load(self.pipeline_filepath)
        if not isinstance(artifact, dict):
            raise ValueError("Decoder pipeline artifact must be a dictionary.")

        missing = [
            key for key in self._REQUIRED_ARTIFACT_KEYS if key not in artifact
        ]
        if missing:
            raise ValueError(
                "Decoder pipeline artifact missing required keys: "
                + ", ".join(missing)
            )

        models = artifact["models"]
        metadata = artifact["metadata"]
        self._validate_models(models)
        if not isinstance(metadata, dict):
            raise ValueError("Decoder pipeline metadata must be a dictionary.")

        # TODO: Once the Phase 1 artifact metadata contract is locked, validate
        # feature width and positive-class metadata before prediction is added.
        self.models = models
        self.metadata = metadata
        # TODO: Validate online_state internals only after Phase 1 locks that
        # schema; for now it is an opaque payload for OnlinePreprocessor.
        self.online_state = artifact["online_state"]
        return self.online_state

    @staticmethod
    def _validate_models(models: Any) -> None:
        if not isinstance(models, dict):
            raise ValueError("Decoder pipeline models must be a non-empty dictionary.")
        if not models:
            raise ValueError("Decoder pipeline models must not be empty.")

        invalid = [
            task_name
            for task_name, model in models.items()
            if not callable(getattr(model, "predict_proba", None))
        ]
        if invalid:
            raise ValueError(
                "Decoder models must expose callable predict_proba: "
                + ", ".join(str(name) for name in invalid)
            )
