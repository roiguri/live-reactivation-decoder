from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.lsl_receiver import LSLReceiver
from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.prediction_logger import PredictionLogger
from backend.online_phase.stream_worker import StreamWorker


@dataclass
class LiveStreamSession:
    """Lifecycle wrapper for one composed live decoding run."""

    _receiver: LSLReceiver
    _worker: StreamWorker
    _logger: PredictionLogger | None = None

    def __post_init__(self) -> None:
        self._started = False
        self._stopped = False

    @property
    def prediction_ready(self) -> Any:
        """Forward the worker signal without exposing worker internals."""
        return self._worker.prediction_ready

    def start(self) -> None:
        """Start receiver and worker. Safe to call more than once."""
        if self._started:
            return
        if self._stopped:
            raise RuntimeError("Cannot restart a stopped live stream session.")

        self._receiver.start()
        try:
            self._worker.start()
        except Exception:
            self._receiver.stop()
            raise
        self._started = True

    def stop(self) -> None:
        """Stop worker, close logger, and stop receiver. Safe to call more than once."""
        if self._stopped:
            return

        if self._started:
            self._worker.stop()
            self._worker.wait()

        if self._logger is not None:
            self._logger.close()
        self._receiver.stop()
        self._started = False
        self._stopped = True


class AppSession:
    """Single entry point for the frontend. Owns SettingsManager lifetime.

    Two-stage initialisation:
      1. AppSession(config_path)          — loads and validates config; session.settings
                                            becomes available immediately.
      2. session.configure_output(dir)    — creates OfflineOrchestrator; session.offline
                                            becomes available for pipeline steps.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._settings = SettingsManager(config_path)
        self.offline: OfflineOrchestrator | None = None

    def configure_output(self, output_dir: str | Path) -> None:
        """Create the OfflineOrchestrator. Must be called before session.offline is used."""
        self.offline = OfflineOrchestrator(self._settings, Path(output_dir))

    def build_live_stream_session(
        self,
        decoder_pipeline_path: str | Path,
        log_path: str | Path | None = None,
        batch_size_samples: int = 40,
    ) -> LiveStreamSession:
        """Construct the live backend pipeline without starting it."""
        # TODO(open): Avoid unnecessary disk reload when Phase 1 already has
        # an in-memory DecoderPipelineArtifact; see stream_worker_design.md Open §2.
        artifact = load_decoder_pipeline_artifact(decoder_pipeline_path)
        preprocessing_settings = self._settings.get_preprocessing_params()

        # TODO(open): Stop hardcoding default LSLReceiver settings once Phase 2
        # config defines stream name/type and runtime sampling parameters; see
        # stream_worker_design.md Open §3.
        receiver = LSLReceiver()
        preprocessor = OnlinePreprocessor(
            preprocessing_settings=preprocessing_settings,
            online_state=artifact.online_state,
        )
        inference_engine = LiveInferenceEngine(
            models=artifact.models,
            metadata=artifact.metadata,
        )
        worker = StreamWorker(
            receiver=receiver,
            preprocessor=preprocessor,
            inference_engine=inference_engine,
            batch_size_samples=batch_size_samples,
        )

        logger = None
        if log_path is not None:
            logger = PredictionLogger(
                out_path=log_path,
                task_names=list(artifact.models.keys()),
                target_sfreq=preprocessor.target_sfreq,
            )
            worker.prediction_ready.connect(logger.on_predictions)

        return LiveStreamSession(
            _receiver=receiver,
            _worker=worker,
            _logger=logger,
        )

    @property
    def settings(self) -> dict[str, Any]:
        """All config sections in one dict: preprocessing, decoders, event_mapping."""
        return self._settings.get_settings()
