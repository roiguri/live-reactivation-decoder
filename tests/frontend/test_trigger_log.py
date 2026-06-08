"""Headless tests for the Phase 2 Trigger Log (Goal 4).

Covers the :class:`TriggerLog` widget in isolation (marker formatting, name
resolution, line cap, lifecycle lines, reset) and its wiring into
``Phase2Screen`` (markers forwarded; start / halt / error logged).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the Phase 2 lifecycle harness (fake session + stub app session).
from tests.frontend.test_phase2_lifecycle import (  # noqa: E402
    _FakeLiveStreamSession,
    _StubAppSession,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


# ── widget in isolation ─────────────────────────────────────────────────────


def _lines(widget) -> list[str]:
    text = widget.toPlainText()
    return text.splitlines() if text else []


def test_append_markers_resolves_names(qapp) -> None:
    from frontend.widgets.phase2.trigger_log import TriggerLog

    log = TriggerLog(event_names={11: "red", 12: "green"})
    log.append_markers([(100.0, 11), (100.5, 12)])

    lines = _lines(log)
    assert len(lines) == 2
    # First marker sets t0 → +0.00s; second is +0.50s. Names resolved.
    assert "+0.00s" in lines[0] and "TRIG  11" in lines[0] and lines[0].endswith("red")
    assert "+0.50s" in lines[1] and "green" in lines[1]


def test_unmapped_code_is_logged_without_name(qapp) -> None:
    from frontend.widgets.phase2.trigger_log import TriggerLog

    log = TriggerLog(event_names={11: "red"})
    log.append_markers([(5.0, 99)])

    line = _lines(log)[0]
    assert "TRIG  99" in line
    # No configured name → the line ends at the code (no trailing name).
    assert line.rstrip().endswith("99")


def test_line_cap_trims_oldest(qapp) -> None:
    from frontend.widgets.phase2.trigger_log import TriggerLog
    from frontend.widgets.phase2 import trigger_log as mod

    log = TriggerLog(event_names={11: "red"})
    n = mod._MAX_LINES + 50
    log.append_markers([(float(i), 11) for i in range(n)])

    assert len(_lines(log)) <= mod._MAX_LINES


def test_log_event_and_reset(qapp) -> None:
    from frontend.widgets.phase2.trigger_log import TriggerLog

    log = TriggerLog(event_names={11: "red"})
    log.log_event("Stream started")
    assert any("Stream started" in ln for ln in _lines(log))

    log.append_markers([(2.0, 11)])
    log.reset()
    assert _lines(log) == []
    # t0 cleared → the next marker restarts the clock at +0.00s.
    log.append_markers([(50.0, 11)])
    assert "+0.00s" in _lines(log)[0]


# ── Phase2Screen wiring ─────────────────────────────────────────────────────


@pytest.fixture
def screen_and_session(qapp):
    from frontend.screens.phase2_screen import Phase2Screen

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    with patch("frontend.screens.phase2_screen.QMessageBox.critical"):
        screen = Phase2Screen(
            session=app_session,
            decoder_pipeline_path=Path("/nonexistent.joblib"),
        )
        screen._target = {"source": "lsl", "stream_name": "NeuroneStream"}
        yield screen, fake


def test_predictions_forward_markers_to_log(screen_and_session) -> None:
    import numpy as np

    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    task = next(iter(screen._chart.task_colors.keys()))

    fake.prediction_ready.emit(
        {task: np.array([0.1, 0.2])}, np.array([0.0, 0.01]), [(0.0, 11)]
    )
    QApplication.processEvents()

    assert any("TRIG  11" in ln for ln in screen._trigger_log.toPlainText().splitlines())


def test_start_halt_error_logged(screen_and_session) -> None:
    screen, fake = screen_and_session

    screen._start_halt_button.start_clicked.emit()
    assert "Stream started" in screen._trigger_log.toPlainText()

    screen._start_halt_button.halt_clicked.emit()
    assert "Inference halted" in screen._trigger_log.toPlainText()


def test_error_logged_then_halted(screen_and_session) -> None:
    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    fake.error_occurred.emit("worker exploded")
    QApplication.processEvents()

    text = screen._trigger_log.toPlainText()
    assert "Error: worker exploded" in text
    assert "Inference halted" in text


def test_start_resets_log(screen_and_session) -> None:
    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    screen._start_halt_button.halt_clicked.emit()
    lines_before = len(screen._trigger_log.toPlainText().splitlines())
    assert lines_before > 0

    # A fresh Start clears the previous session's lines.
    screen._start_halt_button.start_clicked.emit()
    text = screen._trigger_log.toPlainText()
    assert "Inference halted" not in text  # old lines gone
    assert "Stream started" in text  # only the new start line
