from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.lsl_receiver import LSLReceiver
from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.session_logger import LiveSessionLogger
from backend.online_phase.stream_source import LslProxySource, StreamSource
from backend.online_phase.stream_worker import StreamWorker


@dataclass
class LiveStreamSession:
    """Lifecycle wrapper for one composed live decoding run."""

    _receiver: LSLReceiver
    _worker: StreamWorker
    _logger: LiveSessionLogger | None = None

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

    @staticmethod
    def resolve_phase2_log_dir(decoder_pipeline_path: str | Path) -> Path:
        """Return a fresh timestamped run directory for this run's Phase 2 logs.

        Logs land in ``<artifact_root>/phase2_live/<YYYYMMDD_HHMMSS>/`` (mirroring
        the PRD layout), where ``<artifact_root>`` is the directory holding the
        artifact's ``models/`` folder. Derived from the pipeline path so it works
        in both Go-Live and debug-profile paths (neither has an offline
        ``output_dir``). A fresh timestamp per call keeps each Start
        self-contained (file per Start). The directory is created on demand.
        """
        artifact_root = Path(decoder_pipeline_path).resolve().parent.parent
        run_dir = artifact_root / "phase2_live" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def build_live_stream_session(
        self,
        decoder_pipeline_path: str | Path,
        log_dir: str | Path | None = None,
        batch_size_samples: int = 40,
        *,
        stream_name: str | None = None,
    ) -> LiveStreamSession:
        """Construct the live backend pipeline without starting it.

        The receiver is a pure consumer; making the stream appear is the job of
        the active ``StreamSource`` (start it via ``start_stream_source`` before
        ``LiveStreamSession.start``). The resolve timeout is picked from the
        active source kind — replay needs longer to advertise after MNE preload.

        When ``log_dir`` is given, a :class:`LiveSessionLogger` persists the run
        there (predictions/markers CSVs + manifest + npz on close).
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
        if log_dir is not None:
            # Receiver emits {name: id}; the log records every edge by code and
            # resolves the name where one is configured (empty otherwise).
            event_names = {
                int(code): str(name)
                for name, code in self._settings.get_event_mapping().items()
            }
            logger = LiveSessionLogger(
                run_dir=log_dir,
                task_names=list(artifact.models.keys()),
                event_names=event_names,
                metadata={
                    "target_sfreq": preprocessor.target_sfreq,
                    "input_sfreq": preprocessor.input_sfreq,
                    "config": self._settings.config_filepath.name,
                },
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
