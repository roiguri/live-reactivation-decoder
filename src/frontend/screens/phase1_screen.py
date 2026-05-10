from PyQt6.QtWidgets import QWidget


class Phase1Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        self.setObjectName("phase1_screen")
