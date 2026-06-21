from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RANDOM_STATE: int = 42


class CVSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=5, ge=2)


class DecoderTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pos_labels: list[str]
    neg_labels: list[str]

    @model_validator(mode="after")
    def _labels_do_not_overlap(self) -> DecoderTask:
        overlap = set(self.pos_labels) & set(self.neg_labels)
        if overlap:
            raise ValueError(
                f"Task '{self.name}': pos_labels and neg_labels overlap: {sorted(overlap)}"
            )
        return self


_VALID_PARAMS_BY_MODEL: dict[str, set[str]] = {
    "LDA":      {"solver", "shrinkage", "n_components", "priors"},
    "Logistic": {"C", "l1_ratio", "solver", "class_weight", "max_iter"},
    "SVM":      {"C", "kernel", "gamma", "class_weight", "max_iter"},
}

_CLASSIFIER_DEFAULTS: dict[str, dict] = {
    "LDA":      {},
    # sklearn 1.8 deprecated penalty=; l1_ratio=1 == penalty="l1" (liblinear only).
    "Logistic": {"solver": "liblinear", "class_weight": "balanced",
                 "C": 1000, "l1_ratio": 1, "max_iter": 1000},
    "SVM":      {"kernel": "linear", "class_weight": "balanced", "C": 1.0, "max_iter": 1000},
}


class DecoderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_state: int = DEFAULT_RANDOM_STATE
    model:        Literal["LDA", "Logistic", "SVM"] = "LDA"
    params:       dict[str, Any] = Field(default_factory=dict)
    scale_method: Literal["standard", "median"] | None = "standard"
    cv:           CVSettings = Field(default_factory=CVSettings)
    tasks:        list[DecoderTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_and_apply_defaults(self) -> DecoderSettings:
        valid = _VALID_PARAMS_BY_MODEL[self.model]
        invalid = set(self.params.keys()) - valid
        if invalid:
            raise ValueError(
                f"Invalid params for model '{self.model}': {sorted(invalid)}. "
                f"Valid: {sorted(valid)}"
            )
        self.params = {**_CLASSIFIER_DEFAULTS[self.model], **self.params}
        return self


class EventEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str


class MarkersMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventEntry]


class ExperimentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_info: ExperimentInfo
    random_state: int = DEFAULT_RANDOM_STATE
    decoders: DecoderSettings = Field(default_factory=DecoderSettings)
    markers_mapping: MarkersMapping

    @model_validator(mode="before")
    @classmethod
    def _propagate_random_state(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        rs = data.get("random_state", DEFAULT_RANDOM_STATE)
        sub = data.get("decoders")
        if isinstance(sub, dict):
            if "random_state" in sub:
                raise ValueError(
                    "'random_state' must be set at the top level only, "
                    "not under 'decoders'"
                )
            sub["random_state"] = rs
        return data

    @model_validator(mode="after")
    def _task_labels_exist_in_event_mapping(self) -> ExperimentConfig:
        known_names = {e.name for e in self.markers_mapping.events}
        for task in self.decoders.tasks:
            for label in task.pos_labels + task.neg_labels:
                if label not in known_names:
                    raise ValueError(
                        f"Task '{task.name}': label '{label}' not found in "
                        f"markers_mapping.events. Known names: {sorted(known_names)}"
                    )
        return self
