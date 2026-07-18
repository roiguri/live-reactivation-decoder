"""Dev-only welcome/launch pre-screen for the debug entry point.

Boots the debug app on the production welcome screen (:class:`LaunchScreen`)
and acts as the **hub** into the two debug destinations, both built from the
selected debug *profile*:

* **Next →** (and the production "Start New Training" card) → the Phase 1
  debug walkthrough (:class:`DebugPhase1Screen`).
* **Live →** (and the production "Open Live from Existing Output" card) → the
  Phase 2 live debug screen (:func:`build_debug_phase2`).

The heavy debug screens are imported lazily inside the button handlers —
mirroring production ``LaunchScreen`` — so this module has no top-level debug
cross-imports and can't form an import cycle with the Phase 1 / Phase 2 debug
modules (which both import back toward this one).

Production ``frontend.main`` never imports this module — it only imports
:class:`LaunchScreen` (this subclass lives under ``frontend/debug/``).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QWidget

from frontend.debug.debug_bar import DebugBar
from frontend.debug.profiles import DebugProfile
from frontend.screens.launch_screen import LaunchScreen


class DebugLaunchScreen(LaunchScreen):
    """:class:`LaunchScreen` wired as the debug hub.

    ``Next →`` opens the Phase 1 walkthrough; ``Live →`` opens the Phase 2 live
    screen. Both are built from ``profile``; the walkthrough is the default
    forward path (also bound to Ctrl+Right).
    """

    def __init__(self, profile: DebugProfile, parent=None) -> None:
        super().__init__(parent)
        self._profile = profile

        # Debug bar at the very top — the canonical affordance, consistent with
        # the Phase 1 / Phase 2 debug bars.
        self.layout().insertWidget(0, self._build_debug_toolbar())
        self._install_shortcuts()

        # Repoint the production entry cards at the matching debug destination
        # (their labels already say as much): Start New Training → Phase 1,
        # Open Live → Phase 2. disconnect() drops the production handlers so a
        # click no longer routes to the production screens / file dialog.
        self._start_btn.clicked.disconnect()
        self._start_btn.clicked.connect(self._go_phase1)
        self._live_btn.clicked.disconnect()
        self._live_btn.clicked.connect(self._go_phase2)

    # ── toolbar + shortcuts ──────────────────────────────────────────────────

    def _build_debug_toolbar(self) -> QWidget:
        bar = DebugBar("Welcome · Next: Phase 1 · Live: Phase 2")
        # Live sits left of Next; Next (the default walkthrough) stays far right.
        bar.add_button("Live →", kind="outline", on_click=self._go_phase2)
        bar.add_button("Next →", on_click=self._go_phase1)
        return bar

    def _install_shortcuts(self) -> None:
        sc = QShortcut(QKeySequence("Ctrl+Right"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._go_phase1)

    # ── navigation ────────────────────────────────────────────────────────────

    def _go_phase1(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        from frontend.debug.phase1_screen_debug import DebugPhase1Screen

        mw.show_screen(DebugPhase1Screen(self._profile))

    def _go_phase2(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        from frontend.debug.phase2_screen_debug import build_debug_phase2

        mw.show_screen(build_debug_phase2(self._profile))
