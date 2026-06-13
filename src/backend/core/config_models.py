from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RANDOM_STATE: int = 42


class ChannelHygieneSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drop_emg: bool = True
    rename_hegoc_to_heog: bool = True
    montage_name: str = "easycap-M1"
    afz_case_fix: bool = True


class HighpassSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l_freq: float = Field(default=0.1, gt=0)
    method: Literal["iir", "fir"] = "iir"


class NotchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # null in YAML disables the notch filter entirely.
    freq: Optional[float] = Field(default=50.0, gt=0)


# TODO: verify this is correct and up-to-date with mne-icalabel. We want to be sure we catch any typos in the config's ``iclabel.drop_labels`` list, which would otherwise silently let real artifacts through (as discovered when comparing against the ``tomer_preprocessing_new`` reference).
# The exact strings ``mne-icalabel`` returns from
# ``label_components(..., method='iclabel')`` — see
# ``mne_icalabel/iclabel/_config.py::ICLABEL_NUMERICAL_TO_STRING``.
# We pin them here to catch ``iclabel.drop_labels`` typos at config-load
# time (any string outside this set yields zero matches downstream and
# would silently let real artifacts through, as discovered when comparing
# against the ``tomer_preprocessing_new`` reference).
_ICLABEL_VALID_LABELS: frozenset[str] = frozenset({
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
})


class IclabelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Components whose ICLabel class is in this set are pre-selected for
    # exclusion. Default excludes the five confident-artifact categories;
    # "other" (ICLabel's low-confidence catch-all) is intentionally NOT
    # in the default — the operator decides on those manually in the
    # topomap review window.
    drop_labels: list[str] = Field(
        default_factory=lambda: [
            "muscle artifact", "eye blink", "heart beat",
            "line noise", "channel noise",
        ]
    )

    @model_validator(mode="after")
    def _drop_labels_are_known(self) -> IclabelSettings:
        unknown = [lbl for lbl in self.drop_labels if lbl not in _ICLABEL_VALID_LABELS]
        if unknown:
            raise ValueError(
                f"iclabel.drop_labels contains string(s) ICLabel never returns: "
                f"{unknown}. Valid labels (from mne-icalabel): "
                f"{sorted(_ICLABEL_VALID_LABELS)}."
            )
        return self


class ICASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["infomax", "picard", "fastica"] = "infomax"
    extended: bool = True
    # null → let MNE/infomax decide (rank = n_electrodes - 1 after avg ref).
    n_components: Optional[int] = Field(default=None, ge=1)
    fit_l_freq: float = Field(default=1.0, gt=0)  # HP-only freq for the ICA fitting copy
    iclabel: IclabelSettings = Field(default_factory=IclabelSettings)


class EpochSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 1.0
    # null in YAML → None (paper-aligned, baseline correction omitted).
    # [null, 0] → (None, 0.0); passed directly to mne.Epochs.
    baseline: Optional[tuple[Optional[float], Optional[float]]] = None

    @model_validator(mode="after")
    def _tmin_below_tmax(self) -> EpochSettings:
        if self.tmin >= self.tmax:
            raise ValueError(
                f"tmin ({self.tmin}) must be less than tmax ({self.tmax})"
            )
        return self


class PreprocessingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_state: int = DEFAULT_RANDOM_STATE
    # Where to place LPF + decimation in the pipeline.
    #   "early" → on raw before ICA (faster ICA, paper-aligned)
    #   "late"  → on epochs after ICA (reference order, ICA on full-rate data)
    resample_filter_stage: Literal["early", "late"] = "early"
    channel_hygiene: ChannelHygieneSettings = Field(default_factory=ChannelHygieneSettings)
    highpass: HighpassSettings = Field(default_factory=HighpassSettings)
    notch: NotchSettings = Field(default_factory=NotchSettings)
    ica: ICASettings = Field(default_factory=ICASettings)
    epochs: EpochSettings = Field(default_factory=EpochSettings)


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
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    decoders: DecoderSettings = Field(default_factory=DecoderSettings)
    markers_mapping: MarkersMapping

    @model_validator(mode="before")
    @classmethod
    def _propagate_random_state(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        rs = data.get("random_state", DEFAULT_RANDOM_STATE)
        for section in ("preprocessing", "decoders"):
            sub = data.get(section)
            if isinstance(sub, dict):
                if "random_state" in sub:
                    raise ValueError(
                        f"'random_state' must be set at the top level only, "
                        f"not under '{section}'"
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
