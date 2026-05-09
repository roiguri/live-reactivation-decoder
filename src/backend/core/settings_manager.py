from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .config_models import ExperimentConfig


class SettingsManager:
    """Loads and provides read-only access to the shared experiment configuration (YAML)."""

    def __init__(self, config_filepath: str | Path) -> None:
        self.config_filepath = Path(config_filepath)
        with open(self.config_filepath) as f:
            raw = yaml.safe_load(f)
        try:
            self._config = ExperimentConfig.model_validate(raw)
        except ValidationError as e:
            raise ValueError(
                f"Invalid config '{self.config_filepath}':\n{e}"
            ) from e

    def get_preprocessing_params(self) -> dict[str, Any]:
        """Returns the 'preprocessing' block as a plain dict, with top-level random_state injected."""
        return {"random_state": self._config.random_state, **self._config.preprocessing.model_dump()}

    def get_decoder_settings(self) -> dict[str, Any]:
        """Returns the 'decoders' block as a plain dict, with top-level random_state injected."""
        return {"random_state": self._config.random_state, **self._config.decoders.model_dump()}

    def get_event_mapping(self) -> dict[str, int]:
        """Returns event name → trigger ID (e.g. {'red': 1}), ready for mne.Epochs event_id."""
        return {e.name: e.id for e in self._config.markers_mapping.events}
