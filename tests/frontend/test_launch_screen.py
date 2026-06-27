"""Headless tests for the startup pre-screen.

``LaunchScreen`` offers the two entry paths as alternatives and routes through
``MainWindow.show_screen``. We patch the lazily-imported launch helpers so no
real output folder or decoder pipeline is needed — the test only asserts the
navigation wiring (validation gate, error dialog, screen handoff).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.main_window import MainWindow  # noqa: E402
from frontend.screens.launch_screen import LaunchScreen  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _mount(screen: LaunchScreen) -> MainWindow:
    """Put the screen in a MainWindow so ``self.window()`` resolves."""
    mw = MainWindow()
    mw.show_screen(screen)
    return mw


def test_exposes_two_actions(qapp):
    screen = LaunchScreen()
    mw = _mount(screen)  # keep the window alive so Qt doesn't GC the screen
    assert screen._start_btn.text() == "Start New Training"
    assert "Open Live" in screen._live_btn.text()


def test_start_new_opens_phase1(qapp):
    screen = LaunchScreen()
    mw = _mount(screen)

    screen._on_start_new_clicked()

    from frontend.screens.phase1_screen import Phase1Screen
    assert isinstance(mw._stack.currentWidget(), Phase1Screen)


def test_open_live_missing_artifacts_shows_error_and_stays(qapp, monkeypatch):
    screen = LaunchScreen()
    mw = _mount(screen)

    monkeypatch.setattr(
        "frontend.screens.launch_screen.QFileDialog.getExistingDirectory",
        lambda *a, **k: "/some/folder",
    )
    monkeypatch.setattr(
        "frontend.screens.phase2_launch.missing_live_artifacts",
        lambda _path: ["experiment_config.yaml"],
    )
    critical_calls: list[tuple] = []
    monkeypatch.setattr(
        "frontend.screens.launch_screen.QMessageBox.critical",
        lambda *a, **k: critical_calls.append(a),
    )

    screen._on_open_live_clicked()

    assert critical_calls, "expected an error dialog for a non-live-ready folder"
    assert mw._stack.currentWidget() is screen  # stayed on the launch screen


def test_open_live_ready_folder_navigates(qapp, monkeypatch):
    screen = LaunchScreen()
    mw = _mount(screen)
    target = QWidget()

    monkeypatch.setattr(
        "frontend.screens.launch_screen.QFileDialog.getExistingDirectory",
        lambda *a, **k: "/ready/folder",
    )
    monkeypatch.setattr(
        "frontend.screens.phase2_launch.missing_live_artifacts",
        lambda _path: [],
    )
    monkeypatch.setattr(
        "frontend.screens.phase2_launch.build_phase2_from_output",
        lambda _path: target,
    )

    screen._on_open_live_clicked()

    assert mw._stack.currentWidget() is target


def test_open_live_cancelled_dialog_is_noop(qapp, monkeypatch):
    screen = LaunchScreen()
    mw = _mount(screen)

    monkeypatch.setattr(
        "frontend.screens.launch_screen.QFileDialog.getExistingDirectory",
        lambda *a, **k: "",  # user cancelled
    )

    screen._on_open_live_clicked()

    assert mw._stack.currentWidget() is screen
