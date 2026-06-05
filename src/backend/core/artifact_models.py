"""Producer-side contract for ``decoder_pipeline.joblib``.

``DecoderPipelineArtifactSpec.model_dump()`` emits the dict shape
:func:`~backend.online_phase.artifact_loader.load_decoder_pipeline_artifact`
validates against — the consumer stays untouched.

TODO: the consumer dataclass is also named ``DecoderPipelineArtifact``;
the ``Spec`` suffix disambiguates the producer-side schema. Revisit
naming on both sides once we're ready to unify.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.pipeline import Pipeline


class DecoderPipelineMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_width: int = Field(gt=0)
    # Per-decoder training timepoints (``{task_name: seconds}``) — authoritative.
    # Each decoder is trained at its own timepoint; consumers that want a single
    # representative compute the mean themselves.
    decoding_timepoints: dict[str, float] = Field(min_length=1)


class DecoderPipelineArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    models: dict[str, Pipeline] = Field(min_length=1)
    online_state: dict[str, Any] = Field(min_length=1)
    metadata: DecoderPipelineMetadata

    @model_validator(mode="after")
    def _validate_feature_width(self) -> "DecoderPipelineArtifactSpec":
        if "eeg_chunk_indices" not in self.online_state:
            raise ValueError("online_state must contain 'eeg_chunk_indices'.")
        n_channels = len(self.online_state["eeg_chunk_indices"])
        if n_channels != self.metadata.feature_width:
            raise ValueError(
                f"metadata.feature_width ({self.metadata.feature_width}) does not "
                f"match online_state['eeg_chunk_indices'] length ({n_channels})."
            )
        return self
