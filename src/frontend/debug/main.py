"""Debug entry point: ``python -m frontend.debug.main``.

Builds the production app with :class:`DebugPhase1Screen` instead of
:class:`Phase1Screen`. Production ``frontend.main`` is **byte-for-byte
unaffected** and imports nothing from this package.
"""
from __future__ import annotations

import logging
import sys

import mne
from PyQt6.QtWidgets import QApplication

from frontend.debug.phase1_screen_debug import DebugPhase1Screen
from frontend.main_window import MainWindow
from frontend.styles.theme import GLOBAL_QSS


def _configure_logging() -> None:
    """Surface backend logger.info messages in the terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _select_mne_browser_backend() -> None:
    """Use the Qt-native MNE browser (same rationale as production main)."""
    try:
        mne.viz.set_browser_backend("qt")
    except Exception as exc:  # pragma: no cover — startup guard
        print(
            f"WARNING: Qt browser backend unavailable ({exc}). "
            "Install mne-qt-browser to enable.",
            file=sys.stderr,
        )


def main() -> None:
    _configure_logging()
    _select_mne_browser_backend()

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow()
    window.add_screen(DebugPhase1Screen())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
