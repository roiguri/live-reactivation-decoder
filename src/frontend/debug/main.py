"""Debug entry point: ``python -m frontend.debug.main``.

Builds the production app with :class:`DebugPhase1Screen` instead of
:class:`Phase1Screen`. Production ``frontend.main`` is **byte-for-byte
unaffected** and imports nothing from this package.

Entry points:
* (no flag) — boots on the welcome hub (:class:`DebugLaunchScreen`):
  **Next →** opens the Phase 1 debug walkthrough, **Live →** opens the Phase 2
  live screen. Both use the selected profile.
* ``--phase2`` — the direct/separate access: opens the Phase 2 live screen
  immediately, skipping the welcome hub (its Reset returns to the hub). Builds
  a session from the profile's config + its ``models/decoder_pipeline.joblib``.
* ``--profile <name>`` — selects a debug profile (see
  ``frontend.debug.profiles``); applies to both entry points. Defaults to a
  profile named ``default``, or the sole profile if only one exists.
* ``--list-profiles`` — print the discovered profiles and exit.
* ``--config`` / ``--data`` — override the profile's config / raw-data path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
from PyQt6.QtWidgets import QApplication

from backend.core.logging_setup import configure_logging
from frontend.debug.launch_screen_debug import DebugLaunchScreen
from frontend.debug.phase2_screen_debug import build_debug_phase2
from frontend.debug.profiles import list_profiles, resolve_profile
from frontend.main_window import MainWindow
from frontend.styles.theme import GLOBAL_QSS


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
    parser.add_argument(
        "--profile",
        default=None,
        help="Debug profile name (default: 'default', or the sole profile).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the discovered debug profiles and exit.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Override the profile's experiment config path.",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Override the profile's raw-data directory.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging verbosity (DEBUG/INFO/WARNING/...). Overrides the "
        "LRD_LOG_LEVEL env var; defaults to INFO.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    configure_logging(args.log_level)

    if args.list_profiles:
        names = list_profiles()
        print("\n".join(names) if names else "(no profiles — run the seeder)")
        return

    profile = resolve_profile(args.profile, config=args.config, data=args.data)

    _select_mne_browser_backend()

    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    window = MainWindow()
    # --phase2 is the direct/separate access straight into the live screen;
    # otherwise boot on the welcome hub (Next → Phase 1, Live → Phase 2).
    if args.phase2:
        window.show_screen(build_debug_phase2(profile))
    else:
        window.show_screen(DebugLaunchScreen(profile))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
