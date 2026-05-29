from __future__ import annotations

from pathlib import Path

from backend.session import AppSession, LiveStreamSession
from PyQt6.QtCore import Qt
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
    Phase2Header,
    Phase2SettingsPanel,
    StartHaltButton,
    TargetSelectionDialog,
)

# TODO: wire a Back button. Unresolved: which screen Back lands on
# (Node 5 results vs. journey reset) and what to do with a running
# stream (auto-halt vs. confirm). One-way for now — restart the app
# to leave Phase 2.
#
# TODO: threshold is hardcoded; the config schema has no
# ``decoders.threshold`` field yet. Once it does, read it from
# ``session.settings["decoders"]["threshold"]``.
_DEFAULT_THRESHOLD = 0.85
_CHART_MAX_HEIGHT = 420


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

        self._chart = LiveProbabilityChart(
            task_names=task_names,
            target_sfreq=target_sfreq,
            threshold=_DEFAULT_THRESHOLD,
        )
        # Live target chosen by the operator via the header. None until a
        # target is selected; Start is guarded against a missing target.
        self._target: dict | None = None
        self._header = Phase2Header()
        self._header.choose_target_clicked.connect(self._on_choose_target)
        self._settings_panel = Phase2SettingsPanel(task_colors=self._chart.task_colors)
        self._settings_panel.task_visibility_toggled.connect(
            self._chart.set_task_visible
        )

        self._start_halt_button = StartHaltButton()
        self._start_halt_button.start_clicked.connect(self._on_start_clicked)
        self._start_halt_button.halt_clicked.connect(self._on_halt_clicked)
        self._settings_panel.footer_layout.addWidget(self._start_halt_button)

        # The session is built lazily on Start, bound to the chosen target.
        self._live: LiveStreamSession | None = None

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

    def _on_predictions(
        self,
        predictions: dict,
        out_ts,
        markers: list,
    ) -> None:
        self._chart.append_predictions(predictions, out_ts)

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
            QMessageBox.information(
                self,
                "No target selected",
                "Choose a target before starting inference.",
            )
            return

        # Drop any frozen tail from the previous session so the new one
        # starts visually blank.
        self._chart.reset_buffers()
        self._start_halt_button.set_connecting()
        # Force the disabled "Connecting…" repaint before the blocking
        # LSL resolve so the operator sees the state change.
        self.repaint()

        try:
            # Ensure the publishing source (proxy) is up — reuses the one
            # started during discovery — then build a fresh one-shot session
            # bound to the chosen stream and start it.
            self.session.start_stream_source()
            self._live = self.session.build_live_stream_session(
                self.decoder_pipeline_path,
                stream_name=self._target.get("stream_name"),
            )
            self._wire_session(self._live)
            self._live.start()
        except Exception as exc:
            self._safely_stop()
            QMessageBox.critical(self, "Could not start live inference", str(exc))
            return

        self._start_halt_button.set_live()
        self._header.set_status("LIVE INFERENCE", color=SUCCESS_GREEN)

    def _on_halt_clicked(self) -> None:
        self._safely_stop()

    def _on_error(self, message: str) -> None:
        # Stop first so the receiver/proxy/worker clean up before any
        # modal blocks the event loop.
        self._safely_stop()
        QMessageBox.critical(self, "Live inference error", message)

    def _safely_stop(self) -> None:
        """Idempotent teardown. After this returns, ``self._live is None``."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                # Best-effort: surface the failure via header text but
                # don't re-raise — we're already in a cleanup path.
                pass
            self._live = None
        try:
            # Stop the publishing source (proxy/replay). AppSession owns its
            # lifetime; a subsequent Start relaunches it.
            self.session.stop_stream_source()
        except Exception:
            pass
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
        # Stretch keeps the chart card pinned to the top at its
        # intended height; leaves room below for future content.
        layout.addStretch(1)
        return panel
