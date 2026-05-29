"""Start / Halt action button with three visual states."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import QPushButton, QWidget

from frontend.styles.theme import ALERT_RED, SUCCESS_GREEN, TEXT_MUTED


class StartHaltButton(QPushButton):
    """Action button driven by a state machine: idle → connecting → live → idle.

    Idle and live are clickable; connecting is disabled. The button
    only emits the two semantic events — ``start_clicked`` (from idle)
    and ``halt_clicked`` (from live). The parent owns transitions to
    the connecting state.
    """

    start_clicked = Signal()
    halt_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state: str = "idle"
        self.clicked.connect(self._on_self_clicked)
        self.set_idle()

    def set_idle(self) -> None:
        self._state = "idle"
        self.setEnabled(True)
        self.setText("▶  Start Inference")
        self._apply_style(SUCCESS_GREEN, hover="#1E7A1E")

    def set_connecting(self) -> None:
        self._state = "connecting"
        self.setEnabled(False)
        self.setText("Connecting…")
        self._apply_style("#D1D5DB", hover="#D1D5DB", fg=TEXT_MUTED)

    def set_live(self) -> None:
        self._state = "live"
        self.setEnabled(True)
        self.setText("■  Halt Inference")
        self._apply_style(ALERT_RED, hover="#A31830")

    # ── internals ─────────────────────────────────────────────────────────────

    def _on_self_clicked(self) -> None:
        if self._state == "idle":
            self.start_clicked.emit()
        elif self._state == "live":
            self.halt_clicked.emit()
        # connecting is disabled; this branch is unreachable

    def _apply_style(self, bg: str, *, hover: str, fg: str = "white") -> None:
        self.setStyleSheet(
            f"QPushButton {{"
            f"  background: {bg}; color: {fg}; border: none;"
            f"  border-radius: 2px; padding: 8px 16px;"
            f"  font-size: 13px; font-weight: 700; letter-spacing: 0.5px;"
            f"}}"
            f"QPushButton:hover {{ background: {hover}; }}"
            f"QPushButton:disabled {{ background: #D1D5DB; color: {TEXT_MUTED}; }}"
        )
