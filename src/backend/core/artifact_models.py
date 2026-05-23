"""Producer-side contract for ``decoder_pipeline.joblib``.

The on-disk artifact is consumed by
:func:`~backend.online_phase.artifact_loader.load_decoder_pipeline_artifact`,
which expects a dict with top-level keys ``{"models", "online_state",
"metadata"}``. Constructing this via :class:`DecoderPipelineArtifactSpec`
ensures shape and cross-field consistency are validated at training-end
— bugs surface in :class:`OfflineOrchestrator` rather than three layers
deep on the first LSL batch.

:meth:`pydantic.BaseModel.model_dump` emits the exact dict shape the
consumer validates against; the consumer code stays untouched.

TODO: the consumer's dataclass in
``backend/online_phase/artifact_loader.py`` is also named
``DecoderPipelineArtifact``. The ``Spec`` suffix here disambiguates by
type (producer-side schema vs. consumer-side loaded dataclass). Revisit
naming on both sides once we're ready to unify.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.pipeline import Pipeline


class DecoderPipelineMetadata(BaseModel):
    """Runtime metadata the live pipeline needs."""

    model_config = ConfigDict(extra="forbid")

    feature_width: int = Field(
        gt=0,
        description="Number of EEG channels after offline hygiene (i.e. length of online_state['eeg_chunk_indices']).",
    )
    decoding_timepoint: float = Field(
        description="Post-stimulus time (seconds) the decoder was trained at.",
    )


class DecoderPipelineArtifactSpec(BaseModel):
    """Producer-side schema for ``decoder_pipeline.joblib``.

    :meth:`model_dump` produces the dict
    :func:`~backend.online_phase.artifact_loader.load_decoder_pipeline_artifact`
    validates against (top-level keys ``models``, ``online_state``,
    ``metadata``).

    Notes
    -----
    ``online_state`` is kept as ``dict[str, Any]`` rather than a nested
    BaseModel for now — we only validate top-level presence and the
    cross-field ``feature_width`` invariant. Tighter shape validation
    (per-key dtypes / shapes) can be added later if the live runtime
    surfaces silent shape bugs.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    models: dict[str, Pipeline] = Field(min_length=1)
    online_state: dict[str, Any] = Field(min_length=1)
    metadata: DecoderPipelineMetadata

    @model_validator(mode="after")
    def _validate_feature_width(self) -> "DecoderPipelineArtifactSpec":
        if "eeg_chunk_indices" not in self.online_state:
            raise ValueError(
                "online_state must contain 'eeg_chunk_indices' "
                "(positional EEG channel mask exported by OfflinePreprocessor)."
            )
        n_channels = len(self.online_state["eeg_chunk_indices"])
        if n_channels != self.metadata.feature_width:
            raise ValueError(
                f"metadata.feature_width ({self.metadata.feature_width}) does not "
                f"match online_state['eeg_chunk_indices'] length ({n_channels})."
            )
        return self
