from __future__ import annotations

from typing import Any

import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


def build_classifier(decoder_settings: dict[str, Any]) -> Any:
    """Build an unfitted scaler+classifier pipeline from decoder settings."""
    model_type: str = decoder_settings["model"]
    scale_method: str | None = decoder_settings["scale_method"]
    params: dict = decoder_settings["params"]
    random_state: int = decoder_settings["random_state"]

    if model_type == "LDA":
        clf = LinearDiscriminantAnalysis(**params)
    elif model_type == "Logistic":
        clf = LogisticRegression(random_state=random_state, **params)
    elif model_type == "SVM":
        clf = SVC(probability=True, random_state=random_state, **params)
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")

    if scale_method == "standard":
        return make_pipeline(StandardScaler(), clf)
    elif scale_method == "median":
        return make_pipeline(RobustScaler(), clf)
    return clf


def get_task_data(
    epochs: mne.Epochs, task_cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return X (n_trials, n_ch, n_times) and binary y for one task.

    Raises:
        ValueError: If any label is missing from epochs.event_id, or if only
                    one class is present after filtering.
    """
    pos_labels: list[str] = task_cfg["pos_labels"]
    neg_labels: list[str] = task_cfg["neg_labels"]
    all_labels = pos_labels + neg_labels

    missing = [lbl for lbl in all_labels if lbl not in epochs.event_id]
    if missing:
        raise ValueError(
            f"Task '{task_cfg['name']}': labels not found in epochs: {missing}"
        )

    selected = epochs[all_labels]
    pos_codes = {selected.event_id[lbl] for lbl in pos_labels}
    y = np.where(np.isin(selected.events[:, 2], list(pos_codes)), 1, 0)

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Task '{task_cfg['name']}': only one class present after label filtering."
        )

    return selected.get_data(), y
