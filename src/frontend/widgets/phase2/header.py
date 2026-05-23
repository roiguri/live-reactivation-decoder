"""Phase 2 status header: status indicator + target hardware label."""
from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from backend.online_phase.lsl_receiver import DEFAULT_STREAM_NAME
from frontend.styles.theme import (
    BG_LIGHT,
    BORDER_GRAY,
    TEXT_MUTED,
    TEXT_PRIMARY,
)


class Phase2Header(QWidget):
    """Header bar: ``[status] | Target: <stream> (LSL)``."""

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

        self._target_label = QLabel(f"Target: {DEFAULT_STREAM_NAME} (LSL)")
        tf = self._target_label.font()
        tf.setPointSize(10)
        self._target_label.setFont(tf)
        self._target_label.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent;"
        )
        layout.addWidget(self._target_label)

        layout.addStretch(1)

    def set_status(self, text: str, *, color: str = TEXT_PRIMARY) -> None:
        """Update the status label text and tint."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; background: transparent;"
        )
