"""Debug entry point: ``python -m frontend.debug.main``.

Builds the production app with :class:`DebugPhase1Screen` instead of
:class:`Phase1Screen`. Production ``frontend.main`` is **byte-for-byte
unaffected** and imports nothing from this package.

CLI flags:
* (no flag) — opens the Phase 1 debug walkthrough.
* ``--phase2`` — opens :class:`Phase2Screen` directly with a session
  built from the default config + the snapshot training step's
  decoder pipeline. Skips the whole Phase 1 click-through.
"""
from __future__ import annotations

import argparse
import logging
import sys

import mne
from PyQt6.QtWidgets import QApplication

from frontend.debug.phase1_screen_debug import DebugPhase1Screen
from frontend.debug.phase2_screen_debug import build_debug_phase2
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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="frontend.debug.main")
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Skip the Phase 1 walkthrough and open Phase 2 directly.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    _configure_logging()
    _select_mne_browser_backend()

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow()
    if args.phase2:
        window.show_screen(build_debug_phase2())
    else:
        window.show_screen(DebugPhase1Screen())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
