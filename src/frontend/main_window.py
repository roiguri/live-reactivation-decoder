from PyQt6.QtWidgets import QMainWindow, QStackedWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Decoder — Phase 1")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

    def add_screen(self, widget):
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)
