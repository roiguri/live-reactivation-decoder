"""Headless tests for CVProgressView — the per-decoder evaluation progress
screen.

Exercises the public lifecycle (set_decoders → start → update_progress →
mark_all_complete / reset) without leaning on real wall-clock timing: the
QTimer-driven easing is incidental; what matters is that card states and the
overall bar track the real backend completion events truthfully (never 100 %
before mark_all_complete, never a card "done" ahead of its event).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.widgets.cv_progress_view import CVProgressView  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def view(qapp) -> CVProgressView:
    return CVProgressView()


def test_set_decoders_builds_one_card_each(view: CVProgressView) -> None:
    view.set_decoders(["red", "green", "yellow"])
    assert set(view._cards) == {"red", "green", "yellow"}
    assert all(view._status[n] == "pending" for n in view._cards)
    assert view._total == 3


def test_set_decoders_rebuilds_clean(view: CVProgressView) -> None:
    view.set_decoders(["a", "b", "c"])
    view.set_decoders(["x", "y"])  # second build replaces the first
    assert set(view._cards) == {"x", "y"}
    assert view._total == 2


def test_start_marks_first_running(view: CVProgressView) -> None:
    view.set_decoders(["red", "green"])
    view.start()
    assert view._status["red"] == "running"
    assert view._status["green"] == "pending"


def test_update_progress_advances_serially(view: CVProgressView) -> None:
    view.set_decoders(["red", "green", "yellow"])
    view.start()

    view.update_progress(1, 3, "red")
    assert view._status["red"] == "done"
    assert view._status["green"] == "running"   # next decoder advanced
    assert view._status["yellow"] == "pending"

    view.update_progress(2, 3, "green")
    assert view._status["green"] == "done"
    assert view._status["yellow"] == "running"  # last one now running


def test_overall_bar_never_completes_before_mark_all(view: CVProgressView) -> None:
    view.set_decoders(["red", "green"])
    view.start()
    view.update_progress(1, 2, "red")
    # One of two decoders done → bar is at the 50 % floor, never 100 %.
    assert view._overall_bar.value() < 100
    assert view._overall_bar.value() >= 50


def test_mark_all_complete_fills_everything(view: CVProgressView) -> None:
    view.set_decoders(["red", "green"])
    view.start()
    view.mark_all_complete()
    assert all(view._status[n] == "done" for n in view._cards)
    assert view._overall_bar.value() == 100
    assert view._pct_lbl.text() == "100%"
    assert not view._anim_timer.isActive()


def test_mark_all_complete_safe_without_start(view: CVProgressView) -> None:
    # The owning view calls _on_eval_done (→ mark_all_complete) even on the
    # direct/test path where start() never ran.
    view.set_decoders(["red"])
    view.mark_all_complete()
    assert view._status["red"] == "done"
    assert view._overall_bar.value() == 100


def test_reset_clears_and_stops(view: CVProgressView) -> None:
    view.set_decoders(["red", "green"])
    view.start()
    assert view._anim_timer.isActive()

    view.reset()
    assert view._cards == {}
    assert view._total == 0
    assert view._overall_bar.value() == 0
    assert not view._anim_timer.isActive()


def test_update_progress_ignores_unknown_decoder(view: CVProgressView) -> None:
    view.set_decoders(["red"])
    view.start()
    view.update_progress(1, 1, "ghost")  # not a real card → no-op, no raise
    assert view._status["red"] == "running"
