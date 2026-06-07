from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, SUCCESS_GREEN, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.shared import FilePicker
from frontend.workers.load_worker import LoadWorker

logger = logging.getLogger(__name__)


class LoadDataView(QWidget):
    """Node 2 workspace: data directory picker; load runs off-thread."""

    # Loading overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 2 action button
    ready_changed = Signal(bool)
    # Emitted after the raw data has been loaded successfully
    data_loaded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._data_dir: str | None = None
        self._was_ready: bool = False
        self._loading: bool = False
        self._load_thread: QThread | None = None
        self._load_worker: LoadWorker | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 24)
        outer.setSpacing(0)
        outer.addStretch()

        card = QFrame()
        card.setObjectName("load_card")
        card.setStyleSheet(
            f"QFrame#load_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 6px; }}"
        )
        card.setFixedWidth(520)

        body = QVBoxLayout(card)
        body.setContentsMargins(40, 32, 40, 32)
        body.setSpacing(12)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Select EEG Data File")
        f = title.font()
        f.setPointSize(14)
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(title)

        desc = QLabel("Locate the directory containing the subject's .vhdr file.")
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        body.addWidget(desc)
        body.addSpacing(8)

        picker_row = QHBoxLayout()
        picker_row.setSpacing(10)
        picker_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._picker = FilePicker("Browse Folder", mode="dir")
        self._picker.path_selected.connect(self._on_dir_selected)
        picker_row.addWidget(self._picker)
        body.addLayout(picker_row)

        self._status_lbl = QLabel("✓  Data directory selected")
        self._status_lbl.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-size: 11px; font-weight: 600;"
        )
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.hide()
        body.addWidget(self._status_lbl)

        outer.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch()

    # ── public ───────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        """Provide the AppSession built by Node 1. Called by Phase1Screen."""
        self._session = session

    def trigger_load(self) -> None:
        """Start the background load. Wired to the journey-panel Node 2 button.

        Node 2 is only reachable after Node 1 emits session_ready (session set +
        offline configured), and the button is gated on a selected data dir — so
        prerequisites are guaranteed; an unready call is a wiring bug that fails
        loudly rather than being silently swallowed.
        """
        if self._loading:
            # The button is disabled while a load is in flight, so re-entry
            # shouldn't happen; if it does, something fired this slot
            # unexpectedly — warn and don't start a second load thread.
            logger.warning(
                "trigger_load re-entered while a load is in flight; ignoring"
            )
            return

        self._session.offline.set_file_path(self._data_dir)

        self._picker.setEnabled(False)
        self._loading = True
        self._update_ready_state()
        self.loading_requested.emit("Loading data…")

        self._load_thread = QThread()
        self._load_worker = LoadWorker(self._session.offline)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.result_ready.connect(self._on_load_done)
        self._load_worker.error_occurred.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.finished.connect(self._on_load_thread_finished)

        self._load_thread.start()

    # ── private slots ────────────────────────────────────────────────────────

    def _on_dir_selected(self, path: str) -> None:
        self._data_dir = path
        self._update_ready_state()

    def _on_load_done(self, _payload) -> None:
        self._loading = False
        self.loading_done.emit()
        self.data_loaded.emit()
        # Node 2 advances to "complete" via data_loaded; ready state on
        # the now-inactive node is moot but keep the flag consistent.
        self._update_ready_state()

    def _on_load_error(self, message: str) -> None:
        self._loading = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Load Error", message)
        self._picker.setEnabled(True)
        self._update_ready_state()

    def _on_load_thread_finished(self) -> None:
        """Drop Python refs only after the QThread is fully stopped."""
        self._load_thread = None
        self._load_worker = None

    def _update_ready_state(self) -> None:
        ready = bool(self._data_dir) and not self._loading
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
