from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, SUCCESS_GREEN, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.ica_component_card import ICAComponentCard
from frontend.workers.preprocessing_worker import (
    PreprocessingStep1Worker, PreprocessingStep2Worker,
)


class PreprocessingView(QWidget):
    """Node 3 workspace: 3-page stack (Ready → ICA review → Preprocessing Complete).

    All advance actions live on the journey-panel Node 3 button (no in-view
    primary buttons). ``Phase1Screen`` rebinds that button as the user moves
    through the pages:
      Page 0 → ``trigger_start``    (run_step1_prepare_ica)
      Page 1 → ``trigger_confirm``  (run_step2_finish_pipeline)
      Page 2 → ``trigger_continue`` (advance trail to Node 4)
    Both Step 1 and Step 2 use the shared ``LoadingOverlay`` while running.
    The in-view play button on Page 0 mirrors ``trigger_start`` for affordance.
    """

    # Loading-overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 3 action button
    ready_changed = Signal(bool)
    # Emitted once Step 1 finished and the ICA review page is displayed.
    # Phase1Screen handles this by re-binding the Node 3 panel button to
    # trigger_confirm (label stays "Confirm && Continue").
    step1_complete = Signal()
    # Emitted once Step 2 finished and the complete page is displayed.
    # Phase1Screen handles this by re-binding the Node 3 panel button to
    # trigger_continue and updating its label to "Continue to Evaluation".
    step2_complete = Signal()
    # Emitted when the user clicks the rebound panel button. Phase1Screen
    # handles this by calling journey_panel.advance(3).
    preprocessing_complete = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._data_loaded: bool = False
        self._step1_running: bool = False
        self._step1_done: bool = False
        self._step2_running: bool = False
        self._step2_done: bool = False
        self._excluded_count: int = 0
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
        self._pages.addWidget(self._build_complete_page())
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

    def trigger_continue(self) -> None:
        """Wired by Phase1Screen as the Node 3 action handler once Step 2 finishes.

        Emits ``preprocessing_complete`` so the journey trail advances to Node 4.
        """
        if not self._step2_done:
            return
        self.preprocessing_complete.emit()

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

        self._start_btn = QPushButton("▶")
        self._start_btn.setFixedSize(96, 96)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(
            f"QPushButton {{ background: #EFF6FF; color: {PRIMARY_BLUE}; "
            f"border: none; border-radius: 48px; font-size: 36px; }}"
            f"QPushButton:hover {{ background: #DBEAFE; }}"
            f"QPushButton:pressed {{ background: #BFDBFE; }}"
        )
        self._start_btn.clicked.connect(self.trigger_start)
        center.addWidget(self._start_btn, 0, Qt.AlignmentFlag.AlignHCenter)
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
            "Settings configured. Click the play button to begin "
            "the preprocessing pipeline."
        )
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(360)
        desc.setMinimumHeight(36)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)

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
        return page

    def _build_complete_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 32, 48, 32)
        layout.setSpacing(0)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        badge = QLabel("✓")
        badge.setFixedSize(72, 72)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"QLabel {{ background: #F0FDF4; color: {SUCCESS_GREEN}; "
            f"border: 1px solid #DCFCE7; border-radius: 36px; font-size: 32px; }}"
        )
        center.addWidget(badge, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(24)

        title = QLabel("Preprocessing Complete")
        f = title.font()
        f.setPointSize(18)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel("Cleaned epochs are ready for model evaluation.")
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(440)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        stats_card = QFrame()
        stats_card.setFixedWidth(460)
        stats_card.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; "
            f"border-radius: 6px; }}"
        )
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(24, 8, 24, 8)
        stats_layout.setSpacing(0)

        self._epochs_value = QLabel("—")
        self._components_value = QLabel("—")
        # AutoReject drop count is intentionally omitted until the backend TODO
        # in OfflineOrchestrator.run_step2_finish_pipeline surfaces the value
        # (it's computed but discarded today). Restore the row here once the
        # data is available.

        self._append_stat_row(stats_layout, "Epochs retained", self._epochs_value)
        self._append_separator(stats_layout)
        self._append_stat_row(
            stats_layout, "ICA components removed", self._components_value
        )

        center.addWidget(stats_card, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _append_stat_row(
        self, layout: QVBoxLayout, caption: str, value_label: QLabel
    ) -> None:
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 14, 0, 14)
        row_layout.setSpacing(12)

        cap = QLabel(caption)
        cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; background: transparent; border: none;"
        )
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; "
            f"background: transparent; border: none;"
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(cap, 1)
        row_layout.addWidget(value_label, 0)
        layout.addWidget(row)

    def _append_separator(self, layout: QVBoxLayout) -> None:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background: {BORDER_GRAY}; border: none;"
        )
        layout.addWidget(sep)

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
        self._pages.setCurrentIndex(1)  # Ready → ICA review
        self._update_ready_state()
        self.step1_complete.emit()

    def _on_step1_error(self, message: str) -> None:
        self._step1_running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Preprocessing Error", message)
        self._update_ready_state()

    def _on_step1_thread_finished(self) -> None:
        self._step1_thread = None
        self._step1_worker = None

    def trigger_confirm(self) -> None:
        """Confirm ICA rejections and run Step 2. Wired to the journey-panel
        Node 3 button once Step 1 has finished and the review page is shown.
        """
        if self._session is None or self._session.offline is None:
            return
        if not self._step1_done or self._step2_running or self._step2_done:
            return
        rejected = [c._index for c in self._cards if c.is_rejected]
        self._excluded_count = len(rejected)
        self._step2_running = True
        self._update_ready_state()
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

    def _on_step2_done(self, payload) -> None:
        self.loading_done.emit()
        n_epochs = int(payload.get("n_epochs", 0)) if isinstance(payload, dict) else 0
        self._epochs_value.setText(str(n_epochs))
        self._components_value.setText(str(self._excluded_count))
        self._pages.setCurrentIndex(2)  # ICA review → complete
        self._step2_running = False
        self._step2_done = True
        self._update_ready_state()
        self.step2_complete.emit()

    def _on_step2_error(self, message: str) -> None:
        self._step2_running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Preprocessing Error", message)
        self._update_ready_state()

    def _on_step2_thread_finished(self) -> None:
        self._step2_thread = None
        self._step2_worker = None

    def _update_ready_state(self) -> None:
        # Three pages, three actions on the panel button:
        #   Page 0 → trigger_start    (ready when data is loaded)
        #   Page 1 → trigger_confirm  (ready once Step 1 finished, before Step 2)
        #   Page 2 → trigger_continue (ready once Step 2 finished)
        page0_ready = (
            self._session is not None
            and self._data_loaded
            and not self._step1_running
            and not self._step1_done
        )
        page1_ready = (
            self._step1_done
            and not self._step2_running
            and not self._step2_done
        )
        page2_ready = self._step2_done
        ready = page0_ready or page1_ready or page2_ready
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
