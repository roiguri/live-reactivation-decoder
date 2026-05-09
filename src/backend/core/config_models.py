from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BandpassSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l_freq: float = Field(default=0.1, gt=0)
    h_freq: float = 40.0
    method: Literal["iir", "fir"] = "iir"
    notch: Optional[float] = 50.0

    @model_validator(mode="after")
    def _l_freq_below_h_freq(self) -> BandpassSettings:
        if self.l_freq >= self.h_freq:
            raise ValueError(
                f"l_freq ({self.l_freq}) must be less than h_freq ({self.h_freq})"
            )
        return self


class ResampleSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_rate: int = Field(default=256, ge=1)


class ICASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_components: int = Field(default=25, ge=1)
    method: Literal["fastica", "infomax", "picard"] = "fastica"
    fit_l_freq: float = Field(default=1.0, gt=0)  # HP freq for the ICA fitting copy


class EpochSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 1.0
    # [null, 0] in YAML → (None, 0.0) — passed directly to mne.Epochs
    baseline: tuple[Optional[float], Optional[float]] = (None, 0.0)

    @model_validator(mode="after")
    def _tmin_below_tmax(self) -> EpochSettings:
        if self.tmin >= self.tmax:
            raise ValueError(
                f"tmin ({self.tmin}) must be less than tmax ({self.tmax})"
            )
        return self


class RejectCriteriaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hard_amplitude: float = Field(default=150e-6, gt=0)  # epoch amplitude pre-filter before AutoReject (V)
    flat_threshold: float = Field(default=0.5e-6, gt=0)  # channel std below this → flat channel (V)
    noisy_z_score: float = Field(default=3.0, gt=0)      # channel std z-score above this → noisy channel


class PreprocessingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_state: int = 42
    bandpass: BandpassSettings = Field(default_factory=BandpassSettings)
    resample: ResampleSettings = Field(default_factory=ResampleSettings)
    reject_criteria: RejectCriteriaSettings = Field(default_factory=RejectCriteriaSettings)
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


class DecoderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["LDA"] = "LDA"
    params: dict[str, Any] = Field(default_factory=dict)
    cv: CVSettings = Field(default_factory=CVSettings)
    tasks: list[DecoderTask] = Field(default_factory=list)


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
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    decoders: DecoderSettings = Field(default_factory=DecoderSettings)
    markers_mapping: MarkersMapping

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
