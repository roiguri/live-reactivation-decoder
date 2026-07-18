"""Shared debug toolbar used across the dev-only debug screens.

The welcome (:class:`DebugLaunchScreen`), Phase 1
(:class:`DebugPhase1Screen`), and Phase 2 (:class:`DebugPhase2Screen`) debug
screens all pin the same full-width bar to the top of the window: a centered
``[DEBUG] …`` label on the left and one or more action buttons on the right.
This module is the single source of truth for that bar's look and its button
styles, so the three screens can't drift apart.

Dev-only — nothing here is imported by production ``frontend.main``.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.styles.theme import BORDER_GRAY, PRIMARY_BLUE, TEXT_PRIMARY

# Prefix stamped on every debug bar label (and the Phase 1 header title) so the
# mode is always visible.
DEBUG_PREFIX = "[DEBUG] "

# Right-side action button styles. "primary" = the blue Next button (with a
# muted disabled state); "outline" = a bordered secondary action (e.g. the
# Phase 1 "Live →" jump); "default" = the plain Qt button (Reset).
_PRIMARY_QSS = (
    f"QPushButton {{ background: {PRIMARY_BLUE}; color: white; "
    f"border: none; border-radius: 4px; padding: 4px 12px; font-weight: 600; }}"
    f"QPushButton:disabled {{ background: #C7D2FE; }}"
)
_OUTLINE_QSS = (
    f"QPushButton {{ background: white; color: {PRIMARY_BLUE}; "
    f"border: 1px solid {PRIMARY_BLUE}; border-radius: 4px; "
    f"padding: 4px 12px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: #EFF6FF; }}"
)


class DebugBar(QFrame):
    """A full-width debug toolbar: a stretched centered label + action buttons.

    Buttons are added left-to-right via :meth:`add_button`; callers add them in
    display order (e.g. ``Reset`` then ``Next →`` to pin Next to the far right).
    """

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("debug_toolbar")
        self.setStyleSheet(
            f"QFrame#debug_toolbar {{ background: #F5F3FF; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
        )
        self.setFixedHeight(40)

        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(16, 4, 16, 4)
        self._row.setSpacing(8)

        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._row.addWidget(self._label, 1)

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def add_button(
        self,
        text: str,
        *,
        kind: str = "default",
        enabled: bool = True,
        on_click: Callable[[], object] | None = None,
    ) -> QPushButton:
        """Append a right-side action button and return it.

        ``kind`` is ``"primary"`` (blue Next), ``"outline"`` (bordered
        secondary), or ``"default"`` (plain Reset).
        """
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        if kind == "primary":
            btn.setStyleSheet(_PRIMARY_QSS)
        elif kind == "outline":
            btn.setStyleSheet(_OUTLINE_QSS)
        btn.setEnabled(enabled)
        if on_click is not None:
            btn.clicked.connect(on_click)
        self._row.addWidget(btn)
        return btn
