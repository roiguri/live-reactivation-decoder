from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BandpassSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    l_freq: float = Field(default=1.0, gt=0)
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

    target_rate: int = Field(default=250, ge=1)


class ICASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_components: int = Field(default=20, ge=1)
    method: Literal["fastica", "infomax", "picard"] = "fastica"
    random_state: int = 42


class EpochSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 0.8
    # [null, 0] in YAML → (None, 0.0) — passed directly to mne.Epochs
    baseline: tuple[Optional[float], Optional[float]] = (None, 0.0)
    reject: float = Field(default=1.0e-4, gt=0)

    @model_validator(mode="after")
    def _tmin_below_tmax(self) -> EpochSettings:
        if self.tmin >= self.tmax:
            raise ValueError(
                f"tmin ({self.tmin}) must be less than tmax ({self.tmax})"
            )
        return self


class AutoRejectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_state: int = 42


class PreprocessingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bandpass: BandpassSettings = Field(default_factory=BandpassSettings)
    resample: ResampleSettings = Field(default_factory=ResampleSettings)
    ica: ICASettings = Field(default_factory=ICASettings)
    epochs: EpochSettings = Field(default_factory=EpochSettings)
    autoreject: AutoRejectSettings = Field(default_factory=AutoRejectSettings)


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
