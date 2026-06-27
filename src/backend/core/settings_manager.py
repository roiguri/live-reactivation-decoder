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

    def get_random_state(self) -> int:
        """The top-level reproducibility seed (consumed by the offline ICA fit)."""
        return self._config.random_state

    def get_decoder_settings(self) -> dict[str, Any]:
        """Returns the 'decoders' block as a plain dict (random_state is a model field)."""
        return self._config.decoders.model_dump()

    def get_event_mapping(self) -> dict[str, int]:
        """Returns event name → trigger ID (e.g. {'red': 1}), ready for mne.Epochs event_id."""
        return {e.name: e.id for e in self._config.markers_mapping.events}

    def get_intervals(self) -> list[dict[str, str]]:
        """Returns the interval specs as plain dicts ({name, start, stop} each).

        Each interval defines a class tiled from fixed-size windows between a
        start and stop marker; consumed by the offline preprocessor's epoching.
        """
        return [s.model_dump() for s in self._config.intervals]

    def get_settings(self) -> dict[str, Any]:
        """Returns the full effective settings in one dict for the UI.

        The ``preprocessing`` section is the hardcoded recipe, assembled from
        :mod:`backend.core.preprocessing_constants` (the recipe is no longer in the
        config — see the migration in docs/plans/minimize_settings_plan.md). The
        ``decoders`` / ``event_mapping`` sections come from the YAML config.
        """
        return {
            "preprocessing": self._hardcoded_recipe(),
            "decoders":      self.get_decoder_settings(),
            "event_mapping": self.get_event_mapping(),
            "intervals":     self.get_intervals(),
        }

    @staticmethod
    def _hardcoded_recipe() -> dict[str, Any]:
        """The full preprocessing recipe as constants, in config-dict shape.

        Single source of truth: :mod:`backend.core.preprocessing_constants`. This
        is what the UI reads as ``session.settings["preprocessing"]``.
        """
        return {
            "channel_hygiene": {
                "drop_emg": pc.CHANNEL_DROP_EMG,
                "rename_hegoc_to_heog": pc.CHANNEL_RENAME_HEGOC_TO_HEOG,
                "montage_name": pc.CHANNEL_MONTAGE_NAME,
                "afz_case_fix": pc.CHANNEL_AFZ_CASE_FIX,
            },
            "ica": {
                "method": pc.ICA_METHOD,
                "extended": pc.ICA_EXTENDED,
                "n_components": pc.ICA_N_COMPONENTS,
                "fit_l_freq": pc.ICA_FIT_L_FREQ,
                "iclabel": {
                    "enabled": pc.ICLABEL_ENABLED,
                    "drop_labels": list(pc.ICLABEL_DROP_LABELS),
                },
            },
            "highpass": {"l_freq": pc.HIGHPASS_L_FREQ, "method": pc.HIGHPASS_METHOD},
            "notch": {"freq": pc.NOTCH_FREQ},
            "lowpass": {"h_freq": pc.LOWPASS_H_FREQ, "method": pc.LOWPASS_METHOD},
            "final_resample": {"target_rate": pc.FINAL_RESAMPLE_RATE},
            "epochs": {
                "tmin": pc.EPOCH_TMIN,
                "tmax": pc.EPOCH_TMAX,
                "baseline": pc.EPOCH_BASELINE,
            },
        }
