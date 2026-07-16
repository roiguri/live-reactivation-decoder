from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QWidget

# App icon lives alongside the other shared assets. Loaded defensively so a
# missing file just leaves the default Qt icon rather than crashing startup.
_APP_ICON = Path(__file__).resolve().parent / "styles" / "assets" / "app_icon.png"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Reactivation Decoder")
        if _APP_ICON.exists():
            self.setWindowIcon(QIcon(str(_APP_ICON)))
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

    def show_screen(self, widget: QWidget) -> None:
        if self._stack.indexOf(widget) == -1:
            self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)
