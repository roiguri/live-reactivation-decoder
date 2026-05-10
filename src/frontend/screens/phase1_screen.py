from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.widgets.journey_panel import JourneyPanel
from frontend.widgets.loading_overlay import LoadingOverlay
from frontend.views.settings_view import SettingsView
from frontend.views.load_data_view import LoadDataView
from frontend.views.preprocessing_view import PreprocessingView
from frontend.views.evaluation_view import EvaluationView
from frontend.views.train_view import TrainView
from frontend.styles.theme import BG_LIGHT, BORDER_GRAY, CARD_WHITE, TEXT_PRIMARY

_NODE_TITLES = [
    "Pipeline Settings",
    "Data Ingestion",
    "Preprocessing",
    "Model Evaluation",
    "Train & Save",
]


class Phase1Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.setObjectName("phase1_screen")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: gray surround + white card ──────────────────────────────────
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_LIGHT};")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(24, 24, 24, 24)
        outer_layout.setSpacing(0)

        card = QFrame()
        card.setStyleSheet(
            f"QFrame#workspace_card {{"
            f"  background: {CARD_WHITE};"
            f"  border: 1px solid {BORDER_GRAY};"
            f"  border-radius: 6px;"
            f"}}"
        )
        card.setObjectName("workspace_card")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 30))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Header bar
        header_bar = QWidget()
        header_bar.setFixedHeight(48)
        header_bar.setStyleSheet(
            "background: #FAFAFA; border-bottom: 1px solid #F3F4F6;"
        )
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(24, 0, 24, 0)
        self._header_title = QLabel(_NODE_TITLES[0])
        f = self._header_title.font()
        f.setPointSize(13)
        f.setWeight(QFont.Weight.Medium)
        self._header_title.setFont(f)
        self._header_title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        header_layout.addWidget(self._header_title)
        card_layout.addWidget(header_bar)

        # Workspace stack (views own their internal padding)
        self._settings_view = SettingsView()
        self._load_data_view = LoadDataView()

        self._workspace = QStackedWidget()
        self._workspace.setObjectName("workspace_stack")
        self._workspace.setStyleSheet(f"background: {CARD_WHITE};")
        self._workspace.addWidget(self._settings_view)
        self._workspace.addWidget(self._load_data_view)
        self._workspace.addWidget(PreprocessingView())
        self._workspace.addWidget(EvaluationView())
        self._workspace.addWidget(TrainView())
        card_layout.addWidget(self._workspace, 1)

        outer_layout.addWidget(card, 1)
        root.addWidget(outer, 1)

        # ── Loading overlay (covers workspace card only) ──────────────────────
        self._loading_overlay = LoadingOverlay(card)

        # ── Right: journey panel ───────────────────────────────────────────────
        self._journey_panel = JourneyPanel()
        root.addWidget(self._journey_panel)

        # ── Wiring ────────────────────────────────────────────────────────────
        # Node 1: Pipeline Settings
        self._journey_panel.set_node_action(0, self._settings_view.trigger_continue)
        self._journey_panel.set_node_ready(0, False)
        self._settings_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(0, ready)
        )
        self._settings_view.session_ready.connect(self._on_session_ready)
        self._settings_view.loading_requested.connect(self.show_loading)
        self._settings_view.loading_done.connect(self.hide_loading)

        # Node 2: Load Data
        self._journey_panel.set_node_action(1, self._load_data_view.trigger_load)
        self._journey_panel.set_node_ready(1, False)
        self._load_data_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(1, ready)
        )
        self._load_data_view.loading_requested.connect(self.show_loading)
        self._load_data_view.loading_done.connect(self.hide_loading)
        self._load_data_view.data_loaded.connect(lambda: self._journey_panel.advance(2))

        self._journey_panel.node_changed.connect(self._on_node_changed)

    # ── public ────────────────────────────────────────────────────────────────

    def show_loading(self, message: str) -> None:
        self._loading_overlay.show_with_message(message)

    def hide_loading(self) -> None:
        self._loading_overlay.hide()

    def _on_session_ready(self, session) -> None:
        self.session = session
        self._load_data_view.set_session(session)
        self._journey_panel.advance(1)

    def _on_node_changed(self, completed_node: int) -> None:
        next_idx = completed_node  # node_changed emits 1-indexed completed node
        if next_idx < self._workspace.count():
            self._workspace.setCurrentIndex(next_idx)
            self._header_title.setText(_NODE_TITLES[next_idx])
