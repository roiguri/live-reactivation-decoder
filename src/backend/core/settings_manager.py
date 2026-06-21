from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from . import preprocessing_constants as pc
from .config_models import ExperimentConfig

logger = logging.getLogger(__name__)


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
        logger.info(
            "Loaded config %s: model=%s, %d decoder(s)",
            self.config_filepath.name,
            self._config.decoders.model,
            len(self._config.decoders.tasks),
        )

    def get_preprocessing_params(self) -> dict[str, Any]:
        """Returns the configurable 'preprocessing' block as a plain dict.

        This is the *backend pipeline* input — only the fields still carried in
        the config (``random_state`` and any not-yet-hardcoded blocks). The
        hardcoded recipe lives in
        :mod:`backend.core.preprocessing_constants` and is read there directly by
        the preprocessors, so it is intentionally absent here. For the full
        effective recipe (config + constants) used by the UI, see
        :meth:`get_settings`.
        """
        return self._config.preprocessing.model_dump()

    def get_decoder_settings(self) -> dict[str, Any]:
        """Returns the 'decoders' block as a plain dict (random_state is a model field)."""
        return self._config.decoders.model_dump()

    def get_event_mapping(self) -> dict[str, int]:
        """Returns event name → trigger ID (e.g. {'red': 1}), ready for mne.Epochs event_id."""
        return {e.name: e.id for e in self._config.markers_mapping.events}

    def get_settings(self) -> dict[str, Any]:
        """Returns the full effective settings in one dict for the UI.

        The ``preprocessing`` section is the *effective recipe*: the configurable
        fields (from the YAML) merged with the hardcoded blocks re-attached from
        :mod:`backend.core.preprocessing_constants`, in their historical shape.
        This keeps the frontend's view shape-stable across the block-by-block
        hardcoding migration — values move from config to constants under the
        hood, but consumers keep reading the same dict. The raw config stays
        decapsulated behind :meth:`get_preprocessing_params` (backend pipeline).
        """
        preprocessing = self.get_preprocessing_params()
        preprocessing.update(self._hardcoded_recipe())
        return {
            "preprocessing": preprocessing,
            "decoders":      self.get_decoder_settings(),
            "event_mapping": self.get_event_mapping(),
        }

    @staticmethod
    def _hardcoded_recipe() -> dict[str, Any]:
        """The preprocessing blocks now fixed as constants, in config-dict shape.

        Single source of truth: :mod:`backend.core.preprocessing_constants`. Each
        block migrated out of the YAML schema is re-attached here so the effective
        recipe surfaced by :meth:`get_settings` stays complete.
        """
        return {
            "highpass": {"l_freq": pc.HIGHPASS_L_FREQ, "method": pc.HIGHPASS_METHOD},
            "notch": {"freq": pc.NOTCH_FREQ},
            "lowpass": {"h_freq": pc.LOWPASS_H_FREQ, "method": pc.LOWPASS_METHOD},
            "final_resample": {"target_rate": pc.FINAL_RESAMPLE_RATE},
        }
