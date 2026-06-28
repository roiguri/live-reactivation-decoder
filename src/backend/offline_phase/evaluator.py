from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold

from backend.offline_phase.utils import build_classifier, get_task_data

logger = logging.getLogger(__name__)

# TODO: consider computing an empiric change level - this depend if weights are balanced
_CHANCE_LEVEL = 0.5


class ModelEvaluator:
    """
    Evaluates offline decoder performance using Temporal Generalization CV.
    Runs one GeneralizingEstimator pass per task to produce both the TGM and
    the diagonal AUC curve, then surfaces a suggested inference timepoint.

    Single entry point: run_evaluation().
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: dict[str, Any]) -> None:
        self.epochs = epochs
        self.settings = decoder_settings
        self.times: np.ndarray = epochs.times

    # ── Public API ────────────────────────────────────────────────────────────

    def run_evaluation(self, on_progress=None) -> dict[str, Any]:
        """
        Run full evaluation for all decoder tasks defined in settings.

        Args:
            on_progress: Optional callback invoked once after each decoder's
                Temporal-Generalization CV completes, as
                ``on_progress(completed: int, total: int, task_name: str)``
                where ``completed`` counts decoders finished so far (1-based)
                and ``total`` is the decoder count. Purely a progress hook —
                the return value is ignored and exceptions are not caught here.
                Runs synchronously on the calling thread; a GUI caller is
                responsible for marshalling onto its own thread. Default
                ``None`` leaves behaviour unchanged.

        Returns:
            {
                "times": np.ndarray,
                "suggested_timepoint": float,
                "average_peak_auc": float,
                "tasks": {
                    "<task_name>": {
                        "diagonal_auc":  np.ndarray,   # shape (n_times,)
                        "tgm_matrix":    np.ndarray,   # shape (n_times, n_times)
                        "peak_auc":      float,
                        "peak_timepoint": float,       # time (s) of diagonal argmax
                        "chance_level":  float,
                    },
                    ...
                },
            }

        Raises:
            ValueError: If settings contain no tasks, or if any task's labels
                        are missing from the epochs or resolve to a single class.
        """
        tasks = self.settings["tasks"]
        if not tasks:
            raise ValueError("decoder_settings contains no tasks.")

        total = len(tasks)
        task_results: dict[str, Any] = {}
        for i, task_cfg in enumerate(tasks):
            name = task_cfg["name"]
            logger.info("Evaluating task: %s", name)
            X, y = self._get_task_data(task_cfg)
            tgm = self._run_tgm_cv(X, y)
            diagonal = np.diag(tgm)
            task_results[name] = {
                "diagonal_auc": diagonal,
                "tgm_matrix": tgm,
                "peak_auc": float(np.max(diagonal)),
                "peak_timepoint": float(self.times[int(np.argmax(diagonal))]),
                "chance_level": _CHANCE_LEVEL,
            }
            logger.info(
                "Task '%s': peak AUC %.3f at t=%.3fs",
                name,
                task_results[name]["peak_auc"],
                task_results[name]["peak_timepoint"],
            )
            if on_progress is not None:
                on_progress(i + 1, total, name)

        suggested_idx = self._compute_suggested_idx(task_results)
        return {
            "times": self.times,
            "suggested_timepoint": float(self.times[suggested_idx]),
            "average_peak_auc": float(
                np.mean([v["diagonal_auc"][suggested_idx] for v in task_results.values()])
            ),
            "tasks": task_results,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_task_data(
        self, task_cfg: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return X (n_trials, n_ch, n_times) and binary y for one task."""
        return get_task_data(self.epochs, task_cfg)

    def _build_classifier(self) -> Any:
        """Build scaler + classifier pipeline from settings."""
        return build_classifier(self.settings)

    def _run_tgm_cv(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Run GeneralizingEstimator with StratifiedKFold CV.
        Returns the mean TGM (n_times, n_times) averaged over folds.
        """
        k: int = self.settings["cv"]["k"]
        min_class_count = int(np.min(np.bincount(y)))
        if min_class_count < k:
            raise ValueError(
                f"Too few epochs for {k}-fold CV: the minority class has only "
                f"{min_class_count} sample(s). Each class needs at least {k} epochs "
                f"(one per fold). Either collect more data or reduce cv.k in the config."
            )
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.settings["random_state"])
        # Parallelize the TGM over train-timepoints (n_jobs=-1 uses all cores,
        # so this stays machine-agnostic). Timepoints (~121) far outnumber CV
        # folds (k=3), so distributing them keeps every core busy and scales
        # better than fold-parallelism. cross_val_multiscore stays serial
        # (n_jobs=1) on purpose: nesting loky inside loky oversubscribes and,
        # because it changes the BLAS thread count seen by LogisticRegression's
        # iterative solver, perturbs the scores past tolerance. With only the
        # estimator parallelized the TGM is bit-for-bit identical to the serial
        # result — pure speedup, no behavior change. (See issue #43.)
        estimator = GeneralizingEstimator(
            self._build_classifier(), scoring="roc_auc", n_jobs=-1, verbose=False
        )
        # scores: (n_folds, n_train_times, n_test_times)
        scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=1)
        return np.mean(scores, axis=0)

    # Consider correct way of computing suggested - should probably be average peak.
    def _compute_suggested_idx(self, task_results: dict[str, Any]) -> int:
        """Return the time index where the mean diagonal AUC across tasks peaks."""
        diagonals = np.stack([v["diagonal_auc"] for v in task_results.values()])
        return int(np.argmax(np.mean(diagonals, axis=0)))
