"""Headless tests for the Phase 2 Latency Display + Buffer Health.

Covers the :class:`Phase2Header` diagnostics surface in isolation (latency
text, chip state, clear) and its wiring into ``Phase2Screen`` (latency_ready
buffered, percentiles pushed to the header, backlog threshold, timer
lifecycle).
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

from tests.frontend.test_phase2_lifecycle import (  # noqa: E402
    _FakeLiveStreamSession,
    _StubAppSession,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


# ── header in isolation ─────────────────────────────────────────────────────


def test_header_latency_text(qapp) -> None:
    from frontend.widgets.phase2.header import Phase2Header

    header = Phase2Header()
    header.set_latency(3.2, 7.8)
    # Rounded to whole milliseconds.
    assert header._latency_label.text() == "Latency: 3 / 8 ms"


def test_header_buffer_chip_states(qapp) -> None:
    from frontend.widgets.phase2.header import Phase2Header

    header = Phase2Header()
    header.set_buffer_health(True)
    assert "OK" in header._buffer_chip.text()
    assert "228B22" in header._buffer_chip.styleSheet().upper()  # SUCCESS_GREEN

    header.set_buffer_health(False)
    assert "BACKLOG" in header._buffer_chip.text()
    assert "228B22" not in header._buffer_chip.styleSheet().upper()


def test_header_clear_diagnostics(qapp) -> None:
    from frontend.widgets.phase2.header import Phase2Header

    header = Phase2Header()
    header.set_latency(5.0, 10.0)
    header.set_buffer_health(False)
    header.clear_diagnostics()
    assert header._latency_label.text() == ""
    assert header._buffer_chip.text() == ""


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


def test_latency_ready_buffers_and_summarises(screen_and_session) -> None:
    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    for ms in (2.0, 4.0, 6.0, 8.0):
        fake.latency_ready.emit({"total_ms": ms, "pending_samples": 5})
    QApplication.processEvents()

    assert len(screen._latency_window) == 4
    assert screen._pending_samples == 5

    # Slow timer is what paints the header — drive it directly.
    screen._update_diagnostics()
    assert "Latency:" in screen._header._latency_label.text()
    assert "OK" in screen._header._buffer_chip.text()  # 5 < 40*2


def test_backlog_flips_pill_amber(screen_and_session) -> None:
    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    # pending_samples >= batch_size (40 fallback) * 2 → backlog.
    fake.latency_ready.emit({"total_ms": 5.0, "pending_samples": 200})
    QApplication.processEvents()
    screen._update_diagnostics()

    assert "BACKLOG" in screen._header._buffer_chip.text()


def test_diag_timer_lifecycle(screen_and_session) -> None:
    screen, fake = screen_and_session
    assert not screen._diag_timer.isActive()

    screen._start_halt_button.start_clicked.emit()
    assert screen._diag_timer.isActive()

    screen._start_halt_button.halt_clicked.emit()
    assert not screen._diag_timer.isActive()
    # Cleared on Halt so no stale numbers linger.
    assert len(screen._latency_window) == 0
    assert screen._header._latency_label.text() == ""


def test_diagnostics_cleared_on_error(screen_and_session) -> None:
    screen, fake = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    fake.latency_ready.emit({"total_ms": 5.0, "pending_samples": 5})
    QApplication.processEvents()

    fake.error_occurred.emit("worker exploded")
    QApplication.processEvents()

    assert not screen._diag_timer.isActive()
    assert screen._header._buffer_chip.text() == ""


def test_update_diagnostics_noop_when_empty(screen_and_session) -> None:
    """No latency samples yet → no crash, header stays blank."""
    screen, _ = screen_and_session
    screen._update_diagnostics()
    assert screen._header._latency_label.text() == ""
