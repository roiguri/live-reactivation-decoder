from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from frontend.styles.theme import TEXT_PRIMARY, progress_bar_qss


class LoadingOverlay(QWidget):
    """Semi-transparent overlay with a message and indeterminate progress bar.

    Parented to a host widget and resized to fill it via an installed
    event filter. Hidden by default — call ``show_with_message(text)``
    to display, ``hide()`` to dismiss.
    """

    def __init__(self, host: QWidget):
        super().__init__(host)
        self.setObjectName("loading_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"""
            QWidget#loading_overlay {{
                background-color: rgba(255, 255, 255, 215);
            }}
            QLabel#loading_message {{
                color: {TEXT_PRIMARY};
                background: transparent;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        self._message = QLabel("")
        self._message.setObjectName("loading_message")
        f: QFont = self._message.font()
        f.setPointSize(12)
        f.setWeight(QFont.Weight.Medium)
        self._message.setFont(f)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._message)

        self._progress = QProgressBar()
        self._progress.setObjectName("loading_bar")
        self._progress.setStyleSheet(progress_bar_qss("loading_bar"))
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(240)
        self._progress.setFixedHeight(8)
        layout.addWidget(self._progress, alignment=Qt.AlignmentFlag.AlignCenter)

        host.installEventFilter(self)
        self.setGeometry(host.rect())
        self.hide()

    def show_with_message(self, text: str) -> None:
        self._message.setText(text)
        self.raise_()
        self.show()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(obj.rect())
        return super().eventFilter(obj, event)
