from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.lsl_receiver import LSLReceiver
from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.prediction_logger import PredictionLogger
from backend.online_phase.stream_source import LslProxySource, StreamSource
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

    @property
    def error_occurred(self) -> Any:
        """Forward worker runtime errors without exposing worker internals."""
        return self._worker.error_occurred

    @property
    def latency_ready(self) -> Any:
        """Forward worker runtime diagnostics without exposing worker internals."""
        return self._worker.latency_ready

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
        # TODO(#1): Rethink the locking approach on the stream source.
        self._stream_source: StreamSource | None = None
        self._source_lock = threading.Lock()

    def configure_output(self, output_dir: str | Path) -> None:
        """Create the OfflineOrchestrator. Must be called before session.offline is used."""
        self.offline = OfflineOrchestrator(self._settings, Path(output_dir))

    # ── live stream source lifecycle ──────────────────────────────────────────

    def _ensure_proxy_source(self, *, start: bool) -> LslProxySource:
        """Return the active proxy source, creating/starting it as needed.

        Reuses an already-running proxy (e.g. one started during discovery) so
        the NeurOne connection is not churned between discovery and the run.
        """
        with self._source_lock:
            if not isinstance(self._stream_source, LslProxySource):
                if self._stream_source is not None:
                    self._stream_source.stop()
                self._stream_source = LslProxySource()
            if start:
                self._stream_source.start()
            return self._stream_source

    def start_stream_source(self) -> None:
        """Start (or reuse) the proxy source ahead of a live run."""
        self._ensure_proxy_source(start=True)

    def stop_stream_source(self) -> None:
        """Stop and clear the active stream source. Safe to call when idle."""
        with self._source_lock:
            if self._stream_source is not None:
                self._stream_source.stop()
                self._stream_source = None

    def discover_streams(self, timeout_sec: float = 3.0) -> list[str]:
        """Return the names of LSL streams currently visible on the network.

        Ensures the proxy is running (so the live NeurOne stream is published)
        and leaves it running so the following run reuses it. The proxy
        lifecycle stays behind ``AppSession`` so the frontend never touches
        backend internals.
        """
        self._ensure_proxy_source(start=True)
        return LSLReceiver().discover_streams(timeout_sec=timeout_sec)

    def build_live_stream_session(
        self,
        decoder_pipeline_path: str | Path,
        log_path: str | Path | None = None,
        batch_size_samples: int = 40,
        *,
        stream_name: str | None = None,
    ) -> LiveStreamSession:
        """Construct the live backend pipeline without starting it.

        The receiver is a pure consumer; making the stream appear is the job of
        the active ``StreamSource`` (start it via ``start_stream_source`` before
        ``LiveStreamSession.start``). The resolve timeout is picked from the
        active source kind — replay needs longer to advertise after MNE preload.
        """
        # TODO(open): Avoid unnecessary disk reload when Phase 1 already has
        # an in-memory DecoderPipelineArtifact; see stream_worker_design.md Open §2.
        artifact = load_decoder_pipeline_artifact(decoder_pipeline_path)
        preprocessing_settings = self._settings.get_preprocessing_params()

        # Replay sources need longer to advertise (MNE preload before PlayerLSL).
        is_replay = self._stream_source is not None and not isinstance(
            self._stream_source, LslProxySource
        )
        resolve_timeout_sec = 15.0 if is_replay else 5.0
        receiver = LSLReceiver(
            stream_name=stream_name,
            resolve_timeout_sec=resolve_timeout_sec,
        )
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
            worker.prediction_ready.connect(
                logger.on_predictions,
                Qt.ConnectionType.DirectConnection,
            )

        return LiveStreamSession(
            _receiver=receiver,
            _worker=worker,
            _logger=logger,
        )

    @property
    def settings(self) -> dict[str, Any]:
        """All config sections in one dict: preprocessing, decoders, event_mapping."""
        return self._settings.get_settings()
