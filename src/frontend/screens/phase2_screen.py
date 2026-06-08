from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from backend.session import AppSession, LiveStreamSession
from frontend.styles.theme import (
    BG_LIGHT,
    BORDER_GRAY,
    CARD_WHITE,
    SUCCESS_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from frontend.widgets.live_probability_chart import LiveProbabilityChart
from frontend.widgets.phase2 import (
    FrozenEventView,
    Phase2Header,
    Phase2SettingsPanel,
    StartHaltButton,
    TargetSelectionDialog,
)

logger = logging.getLogger(__name__)

# TODO: wire a Back button. Unresolved: which screen Back lands on
# (Node 5 results vs. journey reset) and what to do with a running
# stream (auto-halt vs. confirm). One-way for now — restart the app
# to leave Phase 2.
#
# TODO: threshold is hardcoded; the config schema has no
# ``decoders.threshold`` field yet. Once it does, read it from
# ``session.settings["decoders"]["threshold"]``.
_DEFAULT_THRESHOLD = 0.85
# Fixed rolling-window width for the live probability chart. Operator
# control over this (5 / 10 / 30 / 60 s) is Goal 15 in the M2 plan; until
# then it's a single knob here, same altitude as _DEFAULT_THRESHOLD.
_DEFAULT_WINDOW_SECONDS = 5.0
_CHART_MAX_HEIGHT = 420

# Live diagnostics: rolling window of per-batch total_ms used for the header's
# p50/p95 readout, refreshed on a slow timer decoupled from the ~25 Hz
# latency_ready stream.
_LATENCY_WINDOW = 100
_DIAG_REFRESH_HZ = 5
# Fallback batch size for the buffer-health threshold if the session doesn't
# expose one (e.g. the test fakes). Matches StreamWorker's default.
_DEFAULT_BATCH_SIZE = 40


class Phase2Screen(QWidget):
    """Live-inference screen. Layout glue only — each panel lives in its
    own module under ``frontend.widgets.phase2``.

    Owns the :class:`LiveStreamSession` lifecycle: builds the session on each
    Start bound to the chosen target (the session is one-shot per
    ``LiveStreamSession.stop``), and tears it down defensively on error, halt,
    and screen-close.
    """

    def __init__(
        self,
        session: AppSession,
        decoder_pipeline_path: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.decoder_pipeline_path = Path(decoder_pipeline_path)
        self.setObjectName("phase2_screen")

        settings = session.settings
        task_names = [t["name"] for t in settings["decoders"]["tasks"]]
        target_sfreq = float(settings["preprocessing"]["final_resample"]["target_rate"])
        # settings["event_mapping"] is {name: id}; invert to {id: name} so the
        # chart can resolve each trigger code to its configured event name.
        event_names = {
            int(code): str(name)
            for name, code in settings.get("event_mapping", {}).items()
        }

        self._chart = LiveProbabilityChart(
            task_names=task_names,
            window_seconds=_DEFAULT_WINDOW_SECONDS,
            target_sfreq=target_sfreq,
            threshold=_DEFAULT_THRESHOLD,
            event_names=event_names,
        )
        # Event-locked snapshot view (Goals 9 + 11): epochs each trigger
        # event into a fixed window and keeps a browsable history. Shares the
        # decoder palette + visibility with the live chart.
        self._frozen = FrozenEventView(
            task_names=task_names,
            target_sfreq=target_sfreq,
            threshold=_DEFAULT_THRESHOLD,
            event_names=event_names,
        )
        # Live target chosen by the operator via the header. None until a
        # target is selected; Start is guarded against a missing target.
        self._target: dict | None = None
        self._header = Phase2Header()
        self._header.choose_target_clicked.connect(self._on_choose_target)
        self._settings_panel = Phase2SettingsPanel(task_colors=self._chart.task_colors)
        # Decoder show/hide drives both charts so they stay in sync.
        self._settings_panel.task_visibility_toggled.connect(
            self._chart.set_task_visible
        )
        self._settings_panel.task_visibility_toggled.connect(
            self._frozen.set_task_visible
        )

        self._start_halt_button = StartHaltButton()
        self._start_halt_button.start_clicked.connect(self._on_start_clicked)
        self._start_halt_button.halt_clicked.connect(self._on_halt_clicked)
        self._settings_panel.footer_layout.addWidget(self._start_halt_button)

        # The session is built lazily on Start, bound to the chosen target.
        self._live: LiveStreamSession | None = None

        # Live diagnostics: a rolling window of per-batch total_ms and the
        # latest backlog, summarised onto the header by a 5 Hz timer that only
        # runs while a stream is live.
        self._latency_window: deque[float] = deque(maxlen=_LATENCY_WINDOW)
        self._pending_samples: int = 0
        self._batch_size: int = _DEFAULT_BATCH_SIZE
        self._diag_timer = QTimer(self)
        self._diag_timer.setInterval(int(round(1000 / _DIAG_REFRESH_HZ)))
        self._diag_timer.timeout.connect(self._update_diagnostics)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header)

        body = QWidget()
        body.setStyleSheet(f"background: {BG_LIGHT};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._settings_panel)
        body_layout.addWidget(self._build_chart_panel(), 1)
        root.addWidget(body, 1)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _wire_session(self, live: LiveStreamSession) -> None:
        """Connect cross-thread signals from the worker. Queued connection
        ensures the slot runs on the UI thread regardless of where the
        signal was emitted from."""
        live.error_occurred.connect(self._on_error, Qt.ConnectionType.QueuedConnection)
        live.prediction_ready.connect(
            self._on_predictions, Qt.ConnectionType.QueuedConnection
        )
        live.latency_ready.connect(
            self._on_latency, Qt.ConnectionType.QueuedConnection
        )

    def _on_latency(self, timing: dict) -> None:
        """Data-only hot path for the ~25 Hz timing stream: record the latest
        backlog and append the batch latency. The header is updated separately
        on the slow ``_diag_timer`` so this never does percentile work."""
        self._pending_samples = int(timing.get("pending_samples", 0))
        total_ms = timing.get("total_ms")
        if total_ms is not None:
            self._latency_window.append(float(total_ms))

    def _update_diagnostics(self) -> None:
        """5 Hz tick: summarise the latency window + backlog onto the header.

        Buffer is healthy while the backlog stays under twice the batch size —
        i.e. the worker has at most one batch queued behind the current one.
        """
        if not self._latency_window:
            return
        window = np.fromiter(self._latency_window, dtype=np.float64)
        p50, p95 = np.percentile(window, [50, 95])
        self._header.set_latency(float(p50), float(p95))
        self._header.set_buffer_health(
            self._pending_samples < self._batch_size * 2
        )

    def _on_predictions(
        self,
        predictions: dict,
        out_ts,
        markers: list,
    ) -> None:
        self._chart.append_predictions(predictions, out_ts)
        self._chart.append_markers(markers)
        self._frozen.append_predictions(predictions, out_ts)
        self._frozen.append_markers(markers)

    # ── target selection ───────────────────────────────────────────────────────

    def _on_choose_target(self) -> None:
        dialog = TargetSelectionDialog(self.session, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.selected_target()
        if target is None:
            return
        self._target = target
        self._header.set_target_text(self._describe_target(target))

    @staticmethod
    def _describe_target(target: dict) -> str:
        if target.get("source") == "lsl":
            return f"Target: {target.get('stream_name')} (LSL)"
        return "Target: (unknown)"

    def _on_start_clicked(self) -> None:
        if self._target is None:
            # No stream picked yet — open the target picker instead of
            # erroring, so Start doubles as "pick, then start".
            self._on_choose_target()
            if self._target is None:
                return  # operator cancelled the picker

        # Drop any frozen tail from the previous session so the new one
        # starts visually blank.
        self._chart.reset_buffers()
        self._frozen.reset_buffers()
        self._start_halt_button.set_connecting()
        # Force the disabled "Connecting…" repaint before the blocking
        # LSL resolve so the operator sees the state change.
        self.repaint()

        try:
            # Ensure the publishing source (proxy) is up — reuses the one
            # started during discovery — then build a fresh one-shot session
            # bound to the chosen stream and start it.
            self.session.start_stream_source()
            log_dir = self.session.new_phase2_log_dir()
            self._live = self.session.build_live_stream_session(
                self.decoder_pipeline_path,
                log_dir=log_dir,
                stream_name=self._target.get("stream_name"),
            )
            self._wire_session(self._live)
            self._live.start()
        except Exception as exc:
            logger.exception("Live inference failed to start")
            self._safely_stop()
            QMessageBox.critical(self, "Could not start live inference", str(exc))
            return

        self._start_halt_button.set_live()
        self._header.set_status("LIVE INFERENCE", color=SUCCESS_GREEN)
        # Begin live diagnostics: scale the backlog threshold to this session's
        # batch size, start fresh, and run the 5 Hz header refresh.
        self._batch_size = getattr(
            self._live, "batch_size_samples", _DEFAULT_BATCH_SIZE
        )
        self._latency_window.clear()
        self._pending_samples = 0
        self._diag_timer.start()

    def _on_halt_clicked(self) -> None:
        self._safely_stop()

    def _on_error(self, message: str) -> None:
        # Stop first so the receiver/proxy/worker clean up before any
        # modal blocks the event loop.
        self._safely_stop()
        QMessageBox.critical(self, "Live inference error", message)

    def _safely_stop(self) -> None:
        """Idempotent teardown. After this returns, ``self._live is None``."""
        # Stop live diagnostics first so no stale percentile lingers on the
        # header after Halt. Idempotent: a no-op when already stopped.
        self._diag_timer.stop()
        self._latency_window.clear()
        self._pending_samples = 0
        self._header.clear_diagnostics()
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                logger.warning(
                    "Failed to stop live session during teardown", exc_info=True
                )
            self._live = None
        try:
            # Stop the publishing source (proxy/replay). AppSession owns its
            # lifetime; a subsequent Start relaunches it.
            self.session.stop_stream_source()
        except Exception:
            logger.warning(
                "Failed to stop stream source during teardown", exc_info=True
            )
        self._start_halt_button.set_idle()
        self._header.set_status("INFERENCE HALTED", color=TEXT_PRIMARY)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Guard against app-close mid-stream leaving the LSL proxy + worker
        # thread orphaned.
        self._safely_stop()
        super().closeEvent(event)

    # ── center panel ──────────────────────────────────────────────────────────

    def _build_chart_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG_LIGHT};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(8)

        title = QLabel("PROBABILITY ANALYSIS")
        tf = title.font()
        tf.setPointSize(9)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)
        title.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent; letter-spacing: 1px;"
        )
        layout.addWidget(title)

        chart_card = QFrame()
        chart_card.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; }}"
        )
        chart_card.setMaximumHeight(_CHART_MAX_HEIGHT)
        card_layout = QVBoxLayout(chart_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(self._chart)
        layout.addWidget(chart_card)

        # Event-locked snapshot, below the rolling chart. Scratch placement
        # (a stacked second card) — final layout is Goal 13 (modular panels).
        event_title = QLabel("EVENT-LOCKED VIEW")
        etf = event_title.font()
        etf.setPointSize(9)
        etf.setWeight(QFont.Weight.DemiBold)
        event_title.setFont(etf)
        event_title.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent; letter-spacing: 1px;"
        )
        layout.addSpacing(8)
        layout.addWidget(event_title)

        event_card = QFrame()
        event_card.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; }}"
        )
        event_card.setMaximumHeight(_CHART_MAX_HEIGHT)
        event_card_layout = QVBoxLayout(event_card)
        event_card_layout.setContentsMargins(8, 8, 8, 8)
        event_card_layout.addWidget(self._frozen)
        layout.addWidget(event_card)

        layout.addStretch(1)
        return panel
