from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.ica_component_card import ICAComponentCard
from frontend.workers.preprocessing_worker import (
    PreprocessingStep1Worker, PreprocessingStep2Worker,
)


class PreprocessingView(QWidget):
    """Node 3 workspace: 2-page stack (ICA review + complete stub).

    Step 1 (``run_step1_prepare_ica``) is triggered by the journey-panel Node 3
    action button and uses the shared ``LoadingOverlay`` while running. On
    success the ICA grid renders on Page 0. The in-view "Confirm" button then
    fires Step 2 (placeholder here — Step 7 will wire the real worker).
    """

    # Loading-overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 3 action button
    ready_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._data_loaded: bool = False
        self._step1_running: bool = False
        self._step1_done: bool = False
        self._was_ready: bool = False
        self._step1_thread: QThread | None = None
        self._step1_worker: PreprocessingStep1Worker | None = None
        self._step2_thread: QThread | None = None
        self._step2_worker: PreprocessingStep2Worker | None = None
        self._cards: list[ICAComponentCard] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_ready_page())
        self._pages.addWidget(self._build_review_page())
        self._pages.addWidget(self._build_complete_stub_page())
        outer.addWidget(self._pages)

    # ── public ───────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        """Provide the AppSession built by Node 1. Called by Phase1Screen."""
        self._session = session
        self._update_ready_state()

    def on_data_loaded(self) -> None:
        """Slot connected by Phase1Screen to LoadDataView.data_loaded."""
        self._data_loaded = True
        self._update_ready_state()

    def trigger_start(self) -> None:
        """Start Step 1. Wired to the journey-panel Node 3 button."""
        if (
            self._session is None
            or self._session.offline is None
            or not self._data_loaded
            or self._step1_running
            or self._step1_done
        ):
            return

        self._step1_running = True
        self._update_ready_state()
        self.loading_requested.emit("Running preprocessing…")

        self._step1_thread = QThread()
        self._step1_worker = PreprocessingStep1Worker(self._session.offline)
        self._step1_worker.moveToThread(self._step1_thread)

        self._step1_thread.started.connect(self._step1_worker.run)
        self._step1_worker.result_ready.connect(self._on_step1_done)
        self._step1_worker.error_occurred.connect(self._on_step1_error)
        self._step1_worker.finished.connect(self._step1_thread.quit)
        self._step1_thread.finished.connect(self._step1_worker.deleteLater)
        self._step1_thread.finished.connect(self._step1_thread.deleteLater)
        self._step1_thread.finished.connect(self._on_step1_thread_finished)

        self._step1_thread.start()

    # ── page builders ────────────────────────────────────────────────────────

    def _build_ready_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("▶")
        icon.setFixedSize(96, 96)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(
            f"background: #EFF6FF; color: {PRIMARY_BLUE}; "
            f"border-radius: 48px; font-size: 36px;"
        )
        center.addWidget(icon, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        title = QLabel("Ready to Preprocess")
        f = title.font()
        f.setPointSize(16)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel(
            "Settings configured. Click Start to begin the preprocessing pipeline."
        )
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(360)
        desc.setMinimumHeight(36)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(24)

        self._start_btn = QPushButton("Start Preprocessing")
        self._start_btn.setProperty("class", "primary")
        self._start_btn.clicked.connect(self.trigger_start)
        center.addWidget(self._start_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("ICA Component Review")
        f = title.font()
        f.setPointSize(14)
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        header.addWidget(title)

        subtitle = QLabel(
            "Toggle components to reject; suggested rejections are pre-selected."
        )
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Scrollable grid container
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll.setWidget(self._grid_host)
        layout.addWidget(self._scroll, 1)

        # Confirm row
        action_row = QHBoxLayout()
        action_row.addStretch()
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setProperty("class", "primary")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm_clicked)
        action_row.addWidget(self._confirm_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        return page

    def _build_complete_stub_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel("Step 2 complete — stats coming next.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        layout.addWidget(label)
        return page

    # ── private slots ────────────────────────────────────────────────────────

    def _populate_grid(self, ica, suggested: list[int]) -> None:
        suggested_set = set(suggested)
        # Clear any previous cards (defensive: rerun after error)
        for card in self._cards:
            self._grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        n_components = ica.get_components().shape[1]
        for i in range(n_components):
            card = ICAComponentCard(ica, i, is_suggested=(i in suggested_set))
            self._cards.append(card)
            self._grid.addWidget(card, i // 4, i % 4)

    def _on_step1_done(self, payload) -> None:
        ica, suggested = payload
        self._step1_running = False
        self._step1_done = True
        self.loading_done.emit()
        self._populate_grid(ica, suggested)
        self._confirm_btn.setEnabled(True)
        self._pages.setCurrentIndex(1)  # Ready → ICA review
        self._update_ready_state()

    def _on_step1_error(self, message: str) -> None:
        self._step1_running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Preprocessing Error", message)
        self._update_ready_state()

    def _on_step1_thread_finished(self) -> None:
        self._step1_thread = None
        self._step1_worker = None

    def _on_confirm_clicked(self) -> None:
        if self._session is None or self._session.offline is None:
            return
        rejected = [c._index for c in self._cards if c.is_rejected]
        self._confirm_btn.setEnabled(False)
        self.loading_requested.emit("Finishing preprocessing pipeline…")

        self._step2_thread = QThread()
        self._step2_worker = PreprocessingStep2Worker(self._session.offline, rejected)
        self._step2_worker.moveToThread(self._step2_thread)

        self._step2_thread.started.connect(self._step2_worker.run)
        self._step2_worker.result_ready.connect(self._on_step2_done)
        self._step2_worker.error_occurred.connect(self._on_step2_error)
        self._step2_worker.finished.connect(self._step2_thread.quit)
        self._step2_thread.finished.connect(self._step2_worker.deleteLater)
        self._step2_thread.finished.connect(self._step2_thread.deleteLater)
        self._step2_thread.finished.connect(self._on_step2_thread_finished)

        self._step2_thread.start()

    def _on_step2_done(self, _payload) -> None:
        self.loading_done.emit()
        self._pages.setCurrentIndex(2)  # ICA review → complete stub

    def _on_step2_error(self, message: str) -> None:
        self.loading_done.emit()
        QMessageBox.critical(self, "Preprocessing Error", message)
        self._confirm_btn.setEnabled(True)

    def _on_step2_thread_finished(self) -> None:
        self._step2_thread = None
        self._step2_worker = None

    def _update_ready_state(self) -> None:
        ready = (
            self._session is not None
            and self._data_loaded
            and not self._step1_running
            and not self._step1_done
        )
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
