from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


class LiveInferenceEngine:
    """Run live decoder inference from already-unwrapped models."""

    def __init__(
        self,
        models: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Decoder pipeline metadata must be a dictionary.")

        self._validate_models(models)
        feature_width = self._derive_feature_width(models, metadata)

        self._models = models
        self._metadata = metadata
        self._feature_width = feature_width

    @property
    def models(self) -> dict[str, Any]:
        return self._models

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def feature_width(self) -> int:
        return self._feature_width

    def predict(self, model_features: Any) -> dict[str, np.ndarray]:
        """Predict positive-class probabilities for every decoder task."""
        features = np.asarray(model_features)
        if features.ndim != 2:
            raise ValueError("Live inference features must be a 2D array.")
        if features.shape[1] != self._feature_width:
            raise ValueError(
                "Live inference feature width does not match decoder "
                f"feature_width={self._feature_width}: got {features.shape[1]}"
            )

        predictions: dict[str, np.ndarray] = {}
        for task_name, model in self._models.items():
            probabilities = self._validate_probability_matrix(
                task_name,
                model.predict_proba(features),
                expected_rows=features.shape[0],
            )
            positive_idx = self._positive_class_index(str(task_name), model)
            if positive_idx >= probabilities.shape[1]:
                raise ValueError(
                    f"Decoder '{task_name}' returned too few probability columns "
                    f"for positive class index {positive_idx}."
                )
            predictions[str(task_name)] = probabilities[:, positive_idx]
        return predictions

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

    @staticmethod
    def _validate_probability_matrix(
        task_name: Any,
        probabilities: Any,
        *,
        expected_rows: int,
    ) -> np.ndarray:
        probabilities_array = np.asarray(probabilities)
        if probabilities_array.ndim != 2:
            raise ValueError(
                f"Decoder '{task_name}' predict_proba output must be a 2D array."
            )
        if probabilities_array.shape[0] != expected_rows:
            raise ValueError(
                f"Decoder '{task_name}' predict_proba row count "
                f"{probabilities_array.shape[0]} does not match input row count "
                f"{expected_rows}."
            )
        if probabilities_array.shape[1] < 2:
            raise ValueError(
                f"Decoder '{task_name}' predict_proba output must include at "
                "least two probability columns."
            )
        return probabilities_array

    def _positive_class_index(self, task_name: str, model: Any) -> int:
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise ValueError(
                f"Decoder '{task_name}' must expose classes_ for positive-class "
                "probability selection."
            )

        target_label = self._metadata.get("positive_class", 1)
        class_labels = list(classes)
        for idx, class_label in enumerate(class_labels):
            if class_label == target_label:
                return idx

        raise ValueError(
            f"Decoder '{task_name}' classes_ does not contain an identifiable "
            f"positive class. Tried: {target_label}"
        )

    @classmethod
    def _derive_feature_width(
        cls,
        models: dict[str, Any],
        metadata: dict[str, Any],
    ) -> int:
        metadata_width = metadata.get("feature_width")
        model_widths = cls._model_feature_widths(models)

        # TODO: Remove the model-derived fallback if Phase 1 always writes
        # feature_width into the final decoder_pipeline.joblib metadata.
        if metadata_width is not None:
            feature_width = cls._validate_feature_width(metadata_width)
            mismatched = {
                task_name: width
                for task_name, width in model_widths.items()
                if width != feature_width
            }
            if mismatched:
                raise ValueError(
                    "Decoder model feature widths do not match metadata "
                    f"feature_width={feature_width}: {mismatched}"
                )
            return feature_width

        if not model_widths:
            raise ValueError(
                "Decoder pipeline metadata must include feature_width when "
                "models do not expose n_features_in_."
            )
        missing_widths = [
            str(task_name)
            for task_name, model in models.items()
            if getattr(model, "n_features_in_", None) is None
        ]
        if missing_widths:
            raise ValueError(
                "Decoder pipeline metadata must include feature_width when "
                "some models do not expose n_features_in_: "
                + ", ".join(missing_widths)
            )

        unique_widths = set(model_widths.values())
        if len(unique_widths) != 1:
            raise ValueError(
                "Decoder models expose inconsistent n_features_in_ values: "
                f"{model_widths}"
            )
        return unique_widths.pop()

    @classmethod
    def _model_feature_widths(cls, models: dict[str, Any]) -> dict[str, int]:
        widths: dict[str, int] = {}
        for task_name, model in models.items():
            width = getattr(model, "n_features_in_", None)
            if width is not None:
                widths[str(task_name)] = cls._validate_feature_width(width)
        return widths

    @staticmethod
    def _validate_feature_width(feature_width: Any) -> int:
        if not isinstance(feature_width, Integral) or isinstance(feature_width, bool):
            raise ValueError("Decoder pipeline feature_width must be a positive integer.")
        if feature_width <= 0:
            raise ValueError("Decoder pipeline feature_width must be a positive integer.")
        return int(feature_width)
