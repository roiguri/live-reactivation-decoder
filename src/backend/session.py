from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt

from backend.core.session_paths import SessionPaths
from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.lsl_receiver import LSLReceiver
from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.session_logger import LiveSessionLogger
from backend.online_phase.stream_source import LslProxySource, StreamSource
from backend.online_phase.stream_worker import StreamWorker

logger = logging.getLogger(__name__)


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

    @property
    def batch_size_samples(self) -> int:
        """Micro-batch size the worker accumulates before each inference.

        Exposed so diagnostics consumers (the header's buffer-health chip)
        can scale the backlog threshold to the batch size without reaching
        into ``_worker``.
        """
        return self._worker.batch_size_samples

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
      2. session.configure_output(dir)    — sets the session workspace (session.paths)
                                            and creates OfflineOrchestrator; session.offline
                                            becomes available for pipeline steps.

    ``session.paths`` (a :class:`SessionPaths`) is the single source of truth for
    the on-disk layout. Every phase derives its locations from it — Phase 1
    epochs/models, Phase 2 live logs — so nothing reverse-engineers a path from
    another file. ``configure_output`` sets it for Go-Live; a debug Phase 2 jump
    assigns ``session.paths`` directly (no offline orchestrator).
    """

    def __init__(self, config_path: str | Path) -> None:
        self._settings = SettingsManager(config_path)
        self.offline: OfflineOrchestrator | None = None
        self.paths: SessionPaths | None = None
        # TODO(#1): Rethink the locking approach on the stream source.
        self._stream_source: StreamSource | None = None
        self._source_lock = threading.Lock()

    def configure_output(self, output_dir: str | Path) -> None:
        """Set the session workspace and create the OfflineOrchestrator.

        Also drops a copy of the experiment config at the workspace root, so the
        run is self-documenting. The copy happens once per distinct workspace:
        re-configuring the same directory does not rewrite it (the config a run
        was built with shouldn't change underfoot), while pointing at a new — or
        a previously used — directory copies in (overwriting any stale copy).

        Must be called before session.offline is used.
        """
        new_paths = SessionPaths(Path(output_dir))
        is_new_workspace = self.paths is None or self.paths.root != new_paths.root
        self.paths = new_paths
        self.offline = OfflineOrchestrator(self._settings, self.paths)
        if is_new_workspace:
            self._copy_config_to_workspace()
        logger.info("Workspace configured: %s", self.paths.root)

    def _copy_config_to_workspace(self) -> None:
        """Copy the source config verbatim to ``paths.experiment_config_path``.

        Copies the original YAML (preserving comments) rather than re-serialising
        the parsed model. Best-effort: a copy failure must not block configuring
        the workspace, so it is logged and swallowed.
        """
        assert self.paths is not None
        dest = self.paths.experiment_config_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self._settings.config_filepath, dest)
        except OSError as exc:
            logger.warning("Could not copy experiment config to %s: %s", dest, exc)

    def new_phase2_log_dir(self) -> Path | None:
        """Return (and create) a fresh run directory for Phase 2 logs, or ``None``.

        The layout + run-naming live on :class:`SessionPaths`; the only thing here
        is the ``None`` case — ``session.paths`` is optional (a ``SessionPaths``
        instance never is), and an unset workspace lets live inference run
        unlogged rather than failing.
        """
        return self.paths.new_phase2_run_dir() if self.paths is not None else None

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
