import argparse
import sys

import mne
from PyQt6.QtWidgets import QApplication

from backend.core.logging_setup import configure_logging
from frontend.main_window import MainWindow
from frontend.screens.phase1_screen import Phase1Screen
from frontend.styles.theme import GLOBAL_QSS


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="frontend.main")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging verbosity (DEBUG/INFO/WARNING/...). Overrides the "
        "LRD_LOG_LEVEL env var; defaults to INFO.",
    )
    # parse_known_args so unrelated Qt args on sys.argv don't trip the parser.
    return parser.parse_known_args(argv)[0]


def _select_mne_browser_backend() -> None:
    """Use the Qt-native MNE browser so raw.plot / ica.plot_sources blocking
    calls work inside the live QApplication event loop.

    The matplotlib fallback starts its own event loop on top of Qt's, which
    Qt rejects ("event loop is already running") — see the bad-channel
    review step in preprocessing_view.py.
    """
    try:
        mne.viz.set_browser_backend("qt")
    except Exception as exc:  # pragma: no cover — startup guard
        print(
            f"WARNING: Qt browser backend unavailable ({exc}). MNE will fall "
            "back to matplotlib, which crashes inside the Qt event loop at "
            "the bad-channel / ICA review windows. Install with: "
            "pip install mne-qt-browser",
            file=sys.stderr,
        )


def main():
    args = _parse_args(sys.argv[1:])
    configure_logging(args.log_level)
    _select_mne_browser_backend()

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow()
    screen = Phase1Screen()
    window.show_screen(screen)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
