"""Shared debug toolbar used across the dev-only debug screens.

The welcome (:class:`DebugLaunchScreen`), Phase 1
(:class:`DebugPhase1Screen`), and Phase 2 (:class:`DebugPhase2Screen`) debug
screens all pin the same full-width bar to the top of the window: a solid
amber ``DEBUG`` tag chip on the far left, a centered label, and one or more
action buttons on the right. This module is the single source of truth for
that bar's look and its button styles, so the three screens can't drift apart.

The amber palette is deliberately distinct from the production blue
(``PRIMARY_BLUE``) so debug mode is unmistakable at a glance.

Dev-only — nothing here is imported by production ``frontend.main``.
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.styles.theme import BORDER_GRAY, PRIMARY_BLUE, TEXT_PRIMARY

# All bar buttons share one height so Reset / Next / Live line up; width is left
# to each button's text.
_BTN_HEIGHT = 22

# Amber "debug mode" palette — distinct from the production blue so the bar
# reads as clearly non-production.
_BAR_BG = "#FEF3C7"      # soft amber bar background
_BAR_BORDER = "#FCD34D"  # slightly stronger amber for the bottom border
_TAG_BG = "#F59E0B"      # solid amber chip

# Right-side action button styles, sharing one box model (1px border, same
# radius/padding) so heights match. "outline" = a bordered blue secondary
# action (e.g. the Phase 1 "Live →" jump); "default"/anything else = the plain
# neutral button used for both Reset and Next.
_NEUTRAL_QSS = (
    f"QPushButton {{ background: white; color: {TEXT_PRIMARY}; "
    f"border: 1px solid {BORDER_GRAY}; border-radius: 4px; "
    f"padding: 0 12px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: #F9FAFB; }}"
    f"QPushButton:disabled {{ color: #9CA3AF; border-color: #F3F4F6; "
    f"background: #F9FAFB; }}"
)
_OUTLINE_QSS = (
    f"QPushButton {{ background: white; color: {PRIMARY_BLUE}; "
    f"border: 1px solid {PRIMARY_BLUE}; border-radius: 4px; "
    f"padding: 0 12px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: #EFF6FF; }}"
)
# Compact solid amber chip.
_TAG_QSS = (
    f"QLabel {{ background: {_TAG_BG}; color: white; border-radius: 3px; "
    f"padding: 0 5px; font-size: 9px; font-weight: 700; }}"
)


class DebugBar(QFrame):
    """A full-width debug toolbar: a ``DEBUG`` tag chip, a centered label, and
    action buttons.

    Buttons are added left-to-right via :meth:`add_button`; callers add them in
    display order (e.g. ``Reset`` then ``Next →`` to pin Next to the far right).
    """

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("debug_toolbar")
        self.setStyleSheet(
            f"QFrame#debug_toolbar {{ background: {_BAR_BG}; "
            f"border-top: 1px solid {_BAR_BORDER}; "
            f"border-bottom: 1px solid {_BAR_BORDER}; }}"
        )
        self.setFixedHeight(40)

        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(16, 4, 16, 4)
        self._row.setSpacing(8)

        # Solid amber "DEBUG" chip pinned to the far left — replaces the old
        # "[DEBUG] " text prefix that used to live inside the label.
        tag = QLabel("DEBUG")
        tag.setStyleSheet(_TAG_QSS)
        tag.setFixedHeight(_BTN_HEIGHT)
        self._row.addWidget(tag)

        self._label = QLabel(label)
        self._label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600; "
            "background: transparent;"
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

        ``kind`` is ``"outline"`` (bordered blue, e.g. "Live →") or ``"default"``
        (neutral, used for both Reset and Next). All buttons share one height.
        """
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedHeight(_BTN_HEIGHT)
        btn.setStyleSheet(_OUTLINE_QSS if kind == "outline" else _NEUTRAL_QSS)
        btn.setEnabled(enabled)
        if on_click is not None:
            btn.clicked.connect(on_click)
        self._row.addWidget(btn)
        return btn
