"""Phase 2 status header: status indicator + target selector."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.styles.theme import (
    BG_LIGHT,
    BORDER_GRAY,
    PRIMARY_BLUE,
    TEXT_PRIMARY,
)


class Phase2Header(QWidget):
    """Header bar: ``[status] | [Choose target… / Target: <name>]``.

    The target is a clickable control: it starts as *Choose target…* and,
    when clicked, emits :attr:`choose_target_clicked` so the screen can open
    the target-selection dialog.
    """

    choose_target_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(
            f"background: {BG_LIGHT}; border-bottom: 1px solid {BORDER_GRAY};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("INFERENCE HALTED")
        sf = self._status_label.font()
        sf.setPointSize(12)
        sf.setWeight(QFont.Weight.DemiBold)
        self._status_label.setFont(sf)
        self._status_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent;"
        )
        layout.addWidget(self._status_label)

        divider = QFrame()
        divider.setFixedSize(1, 16)
        divider.setStyleSheet(f"background: {BORDER_GRAY};")
        layout.addWidget(divider)

        self._target_button = QPushButton("Choose target…")
        self._target_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._target_button.setFlat(True)
        self._target_button.setStyleSheet(
            "QPushButton {"
            f"  color: {PRIMARY_BLUE}; background: transparent; border: none;"
            "  padding: 0; font-size: 10pt; text-align: left;"
            "}"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._target_button.clicked.connect(self.choose_target_clicked)
        layout.addWidget(self._target_button)

        layout.addStretch(1)

        self._latency_label = QLabel("")
        self._latency_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent; font-size: 10pt;"
        )
        layout.addWidget(self._latency_label)

    def set_status(self, text: str, *, color: str = TEXT_PRIMARY) -> None:
        """Update the status label text and tint."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; background: transparent;"
        )

    def set_target_text(self, text: str) -> None:
        """Update the target selector's label (e.g. ``Target: X (LSL)``)."""
        self._target_button.setText(text)

    def set_latency_text(self, text: str) -> None:
        """Update the latency comparison label (empty string hides it)."""
        self._latency_label.setText(text)
