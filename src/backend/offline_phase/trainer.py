from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np

from backend.offline_phase.utils import build_classifier, get_task_data

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains the final decoders at the user-selected timepoint and calculates
    biological spatial patterns for verification.

    Single entry point: run_training().
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: dict[str, Any]) -> None:
        self.epochs = epochs
        self.settings = decoder_settings
        self.times: np.ndarray = epochs.times

    # ── Public API ────────────────────────────────────────────────────────────

    def run_training(self, timepoint: float) -> dict[str, Any]:
        """
        Train one classifier per task at the given timepoint.

        Args:
            timepoint: Time in seconds to extract features from (e.g. 0.350).

        Returns:
            {
                "models":           {task_name: fitted_sklearn_pipeline},
                "spatial_patterns": {task_name: np.ndarray},  # (n_channels,) each
                "mne_info":         mne.Info,
            }

        Raises:
            ValueError: If settings contain no tasks, or if any task's labels
                        are missing from the epochs or resolve to a single class.
        """
        tasks = self.settings["tasks"]
        if not tasks:
            raise ValueError("decoder_settings contains no tasks.")

        models: dict[str, Any] = {}
        spatial_patterns: dict[str, np.ndarray] = {}

        for task_cfg in tasks:
            name = task_cfg["name"]
            logger.info("Training task: %s", name)
            X_t, y = self._extract_features(task_cfg, timepoint)
            model = self._train_classifier(X_t, y)
            models[name] = model
            spatial_patterns[name] = self._calculate_spatial_patterns(X_t, model)

        return {
            "models": models,
            "spatial_patterns": spatial_patterns,
            "mne_info": self.epochs.info,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_features(
        self, task_cfg: dict[str, Any], timepoint: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return X_t (n_trials, n_channels) and binary y at the given timepoint."""
        X, y = get_task_data(self.epochs, task_cfg)
        t_idx = self.epochs.time_as_index(timepoint)[0]
        if t_idx < 0 or t_idx >= X.shape[2]:
            raise ValueError(
                f"Timepoint {timepoint}s is out of bounds "
                f"[{self.epochs.tmin:.3f}, {self.epochs.tmax:.3f}]."
            )
        return X[:, :, t_idx], y

    def _train_classifier(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Build and fit a classifier on 100% of the data."""
        model = build_classifier(self.settings)
        model.fit(X, y)
        return model

    def _calculate_spatial_patterns(
        self, X: np.ndarray, model: Any
    ) -> np.ndarray:
        """
        Compute Haufe et al. 2014 activation pattern: A = Cov(X) @ w / Var(X @ w).

        Transforms raw classifier weights into biological activation patterns so
        that GUI topomaps accurately reflect brain activity rather than filter
        suppression directions.
        """
        clf_step = model[-1] if hasattr(model, "__len__") else model
        if not hasattr(clf_step, "coef_"):
            raise ValueError(
                f"Model step {type(clf_step).__name__} has no coef_; "
                "cannot compute spatial patterns."
            )
        w = clf_step.coef_.flatten()  # (n_channels,)

        # Transform weights to original feature space when a scaler is in the pipeline.
        # Both StandardScaler and RobustScaler expose .scale_ with shape (n_channels,).
        if hasattr(model, "steps") and len(model.steps) > 1:
            w = w / model[0].scale_

        X_c = X - X.mean(axis=0)
        cov_X = (X_c.T @ X_c) / (len(X) - 1)  # (n_ch, n_ch)
        s = X @ w                                # projections (n_trials,)
        return cov_X @ w / np.var(s, ddof=1)    # (n_channels,)
