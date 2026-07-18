"""Dev-only welcome/launch pre-screen for the debug entry point.

Invoked automatically by ``python -m frontend.debug.main`` (and its
``--phase2`` / ``--profile`` variants): the debug app now boots on the
same **welcome screen** production shows, then continues into whichever
debug screen the CLI selected.

Reuses the production :class:`LaunchScreen` verbatim for its visuals, so
you can iterate on the real welcome screen in debug. Two behavioural
changes, both dev-only:

* A **debug toolbar** is added at the top of the screen (mirroring
  :class:`DebugPhase1Screen`) with a ``Next →`` button — the canonical way
  to move forward — plus the window-scoped ``Ctrl+Right`` shortcut.
* Debug has **no branching** — the launch screen's two production entry
  buttons ("Start New Training" / "Open Live from Existing Output") are
  also rewired to the same continue action, so clicking either one goes to
  the CLI-selected debug screen instead of the production Phase 1 / Phase 2
  paths. The choice of debug destination is made by the ``--phase2`` flag,
  not on this screen.

Production ``frontend.main`` never imports this module — it only imports
:class:`LaunchScreen` (this subclass lives under ``frontend/debug/``).
"""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QWidget,
)

from frontend.screens.launch_screen import LaunchScreen
from frontend.styles.theme import BORDER_GRAY, PRIMARY_BLUE, TEXT_PRIMARY

_DEBUG_PREFIX = "[DEBUG] "


class DebugLaunchScreen(LaunchScreen):
    """:class:`LaunchScreen` with a debug toolbar that continues to a debug screen.

    ``build_next`` is a zero-arg factory that constructs the debug screen to
    show on continue (e.g. ``lambda: DebugPhase1Screen(profile)``). It's called
    lazily on ``Next`` so the (potentially heavier) debug screen is only built
    when the operator actually proceeds.
    """

    def __init__(
        self, build_next: Callable[[], QWidget], parent=None
    ) -> None:
        super().__init__(parent)
        self._build_next = build_next

        # Debug bar at the very top — the canonical "move forward" affordance,
        # consistent with the Phase 1 walkthrough toolbar.
        self.layout().insertWidget(0, self._build_debug_toolbar())
        self._install_shortcuts()

        # No branching: repoint both production entry buttons at the single
        # CLI-selected debug screen too. disconnect() drops the production
        # handlers (_on_start_new_clicked / _on_open_live_clicked) so a click
        # no longer routes to the production Phase 1 / Phase 2 screens.
        for btn in (self._start_btn, self._live_btn):
            btn.clicked.disconnect()
            btn.clicked.connect(self._continue)

    # ── toolbar + shortcuts ──────────────────────────────────────────────────

    def _build_debug_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("debug_toolbar")
        bar.setStyleSheet(
            f"QFrame#debug_toolbar {{ background: #F5F3FF; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
        )
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(8)

        lbl = QLabel(f"{_DEBUG_PREFIX}Welcome — press Next to continue")
        lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl, 1)

        next_btn = QPushButton("Next →")
        next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        next_btn.setStyleSheet(
            f"QPushButton {{ background: {PRIMARY_BLUE}; color: white; "
            f"border: none; border-radius: 4px; padding: 4px 12px; "
            f"font-weight: 600; }}"
            f"QPushButton:disabled {{ background: #C7D2FE; }}"
        )
        next_btn.clicked.connect(self._continue)
        layout.addWidget(next_btn)

        return bar

    def _install_shortcuts(self) -> None:
        sc = QShortcut(QKeySequence("Ctrl+Right"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._continue)

    # ── navigation ────────────────────────────────────────────────────────────

    def _continue(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        mw.show_screen(self._build_next())
