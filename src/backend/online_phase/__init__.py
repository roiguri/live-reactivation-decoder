"""Phase 2 online inference components."""

from backend.online_phase.artifact_loader import (
    DecoderPipelineArtifact,
    load_decoder_pipeline_artifact,
)
from backend.online_phase.live_inference import LiveInferenceEngine

__all__ = [
    "DecoderPipelineArtifact",
    "LiveInferenceEngine",
    "load_decoder_pipeline_artifact",
]
