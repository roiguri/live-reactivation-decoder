from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Reactivation Decoder")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

    def show_screen(self, widget: QWidget) -> None:
        if self._stack.indexOf(widget) == -1:
            self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)
