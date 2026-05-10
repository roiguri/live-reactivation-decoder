from PyQt6.QtWidgets import QWidget, QHBoxLayout, QStackedWidget

from frontend.widgets.journey_panel import JourneyPanel
from frontend.views.settings_view import SettingsView
from frontend.views.load_data_view import LoadDataView
from frontend.views.preprocessing_view import PreprocessingView
from frontend.views.evaluation_view import EvaluationView
from frontend.views.train_view import TrainView
from frontend.styles.theme import BG_LIGHT, CARD_WHITE


class Phase1Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.setObjectName("phase1_screen")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._workspace = QStackedWidget()
        self._workspace.setObjectName("workspace_stack")
        self._workspace.addWidget(SettingsView())
        self._workspace.addWidget(LoadDataView())
        self._workspace.addWidget(PreprocessingView())
        self._workspace.addWidget(EvaluationView())
        self._workspace.addWidget(TrainView())
        layout.addWidget(self._workspace, 1)

        self._journey_panel = JourneyPanel()
        layout.addWidget(self._journey_panel)

        self._journey_panel.node_changed.connect(self._on_node_changed)

    def _on_node_changed(self, completed_node: int) -> None:
        next_idx = completed_node  # node_changed emits 1-indexed completed node; next view is at that index
        if next_idx < self._workspace.count():
            self._workspace.setCurrentIndex(next_idx)
