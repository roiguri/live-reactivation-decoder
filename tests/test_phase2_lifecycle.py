"""Tier 1 headless tests for the Phase 2 live-stream lifecycle.

Exercises Phase2Screen's button state machine, error-path cleanup, and
``closeEvent`` halt — all with a fake :class:`LiveStreamSession` so no
LSL stream is required.

The fake records start/stop calls and exposes the same ``error_occurred``
signal shape Phase2Screen connects to, so we can simulate worker-side
errors and assert the screen halts before showing its dialog.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import pyqtSignal as Signal, QObject  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv)
    return app


class _FakeLiveStreamSession(QObject):
    """Drop-in for :class:`backend.session.LiveStreamSession` with the
    same Qt signal surface (``prediction_ready``, ``error_occurred``,
    ``latency_ready``) and idempotent ``start``/``stop`` semantics.
    """

    prediction_ready = Signal(dict, object, list)
    error_occurred = Signal(str)
    latency_ready = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0
        self.raise_on_start: Exception | None = None
        self._started = False
        self._stopped = False

    def start(self) -> None:
        self.start_calls += 1
        if self._stopped:
            raise RuntimeError("Cannot restart a stopped live stream session.")
        if self.raise_on_start is not None:
            raise self.raise_on_start
        self._started = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._started = False
        self._stopped = True


def _make_session_settings() -> dict[str, Any]:
    return {
        "preprocessing": {"final_resample": {"target_rate": 100.0}},
        "decoders": {
            "tasks": [
                {"name": "alpha", "pos_labels": ["a"], "neg_labels": ["b"]},
            ]
        },
    }


class _StubAppSession:
    """Stand-in for AppSession that returns the supplied fake live session."""

    def __init__(self, live: _FakeLiveStreamSession, settings: dict | None = None) -> None:
        self._live = live
        self.settings = settings or _make_session_settings()

    def build_live_stream_session(self, decoder_pipeline_path):
        # Hand out a fresh fake on each call to mirror the real one-shot semantics
        # (the screen rebuilds after every stop).
        if getattr(self._live, "_handed_out", False):
            new_fake = _FakeLiveStreamSession()
            new_fake.raise_on_start = self._live.raise_on_start
            self._live = new_fake
        self._live._handed_out = True
        return self._live


@pytest.fixture
def screen_and_session(qapp):
    """Construct a Phase2Screen wired to a fake LiveStreamSession.

    Patches the QMessageBox.critical so dialogs don't block tests.
    """
    from frontend.screens.phase2_screen import Phase2Screen

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    with patch("frontend.screens.phase2_screen.QMessageBox.critical") as mock_box:
        screen = Phase2Screen(
            session=app_session,
            decoder_pipeline_path=Path("/nonexistent.joblib"),
        )
        yield screen, fake, app_session, mock_box


# ── Construction ──────────────────────────────────────────────────────────────


def test_constructor_builds_session_eagerly(screen_and_session) -> None:
    screen, fake, app_session, _ = screen_and_session
    # The eager construction in __init__ should have invoked
    # build_live_stream_session and stored the result.
    assert screen._live is fake
    # No start() yet — the operator hasn't clicked.
    assert fake.start_calls == 0


def test_constructor_error_propagates(qapp) -> None:
    """If build_live_stream_session raises (e.g. artifact missing), the
    exception escapes ``__init__`` so the caller (Phase1Screen) can
    show its own dialog and stay on Phase 1."""
    from frontend.screens.phase2_screen import Phase2Screen

    class _RaisingSession:
        settings = _make_session_settings()

        def build_live_stream_session(self, _):
            raise FileNotFoundError("artifact missing")

    with pytest.raises(FileNotFoundError, match="artifact missing"):
        Phase2Screen(
            session=_RaisingSession(),
            decoder_pipeline_path=Path("/nope.joblib"),
        )


# ── Button state machine ──────────────────────────────────────────────────────


def test_start_then_halt_state_transitions(screen_and_session) -> None:
    screen, fake, _, _ = screen_and_session
    btn = screen._start_halt_button

    # Initial: idle (Start, green)
    assert btn._state == "idle"
    assert btn.isEnabled()
    assert "Start" in btn.text()

    # Click Start → start() called, button live, status updated
    btn.start_clicked.emit()
    assert fake.start_calls == 1
    assert btn._state == "live"
    assert "Halt" in btn.text()
    assert screen._header._status_label.text() == "LIVE INFERENCE"

    # Click Halt → stop() called, session cleared, button idle
    btn.halt_clicked.emit()
    assert fake.stop_calls == 1
    assert screen._live is None
    assert btn._state == "idle"
    assert "Start" in btn.text()
    assert screen._header._status_label.text() == "INFERENCE HALTED"


def test_start_failure_returns_to_idle_and_calls_stop(screen_and_session) -> None:
    screen, fake, _, mock_box = screen_and_session
    fake.raise_on_start = RuntimeError("LSL resolve timed out")

    screen._start_halt_button.start_clicked.emit()

    # start() raised; the screen must have called stop() defensively then
    # surfaced a critical dialog and returned to idle.
    assert fake.start_calls == 1
    assert fake.stop_calls == 1
    assert mock_box.called
    assert screen._live is None
    assert screen._start_halt_button._state == "idle"
    assert screen._header._status_label.text() == "INFERENCE HALTED"


def test_halt_then_start_rebuilds_session(screen_and_session) -> None:
    """A fresh build_live_stream_session call must happen on restart
    because LiveStreamSession is one-shot."""
    screen, fake_initial, app_session, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    screen._start_halt_button.halt_clicked.emit()
    assert screen._live is None

    # Re-start should construct a NEW fake (the stub returns a fresh one
    # after the first hand-out).
    screen._start_halt_button.start_clicked.emit()
    assert screen._live is not None
    assert screen._live is not fake_initial
    assert screen._live.start_calls == 1
    assert screen._start_halt_button._state == "live"


# ── Error path ────────────────────────────────────────────────────────────────


def test_error_signal_stops_session_before_dialog(screen_and_session) -> None:
    """``error_occurred`` must trigger stop() BEFORE the modal so resources
    free up regardless of how long the operator takes to dismiss it."""
    screen, fake, _, mock_box = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    assert screen._live is fake
    call_order: list[str] = []

    original_stop = fake.stop

    def recording_stop():
        call_order.append("stop")
        original_stop()

    fake.stop = recording_stop  # type: ignore[assignment]
    mock_box.side_effect = lambda *a, **kw: call_order.append("dialog")

    fake.error_occurred.emit("worker exploded")
    QApplication.processEvents()

    assert call_order == ["stop", "dialog"], (
        f"expected stop before dialog, got {call_order}"
    )
    assert screen._live is None
    assert screen._start_halt_button._state == "idle"


# ── Close path ────────────────────────────────────────────────────────────────


def test_close_event_halts_running_session(screen_and_session) -> None:
    screen, fake, _, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    assert fake._started

    screen.close()
    assert fake.stop_calls >= 1
    assert screen._live is None


def test_close_event_safe_after_halt(screen_and_session) -> None:
    """closeEvent must be a no-op when the session was already halted."""
    screen, fake, _, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    screen._start_halt_button.halt_clicked.emit()
    stops_before = fake.stop_calls
    assert screen._live is None

    screen.close()
    # _safely_stop on an already-None _live is a no-op for stop() calls.
    assert fake.stop_calls == stops_before


# ── Thread-leak check ─────────────────────────────────────────────────────────


def test_rapid_start_halt_cycles_no_thread_leak(qapp) -> None:
    """5 start/halt cycles against the fake session must leave no extra
    threads behind (with the real session, this would catch leaked
    QThread instances)."""
    from frontend.screens.phase2_screen import Phase2Screen

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    threads_before = len(threading.enumerate())

    with patch("frontend.screens.phase2_screen.QMessageBox.critical"):
        screen = Phase2Screen(
            session=app_session,
            decoder_pipeline_path=Path("/nonexistent.joblib"),
        )
        for _ in range(5):
            screen._start_halt_button.start_clicked.emit()
            screen._start_halt_button.halt_clicked.emit()
        screen.close()

    threads_after = len(threading.enumerate())
    assert threads_after == threads_before, (
        f"thread count drifted: {threads_before} → {threads_after}"
    )


# ── prediction_ready → chart ──────────────────────────────────────────────────


def test_prediction_ready_forwards_to_chart(screen_and_session) -> None:
    """prediction_ready emissions land in the chart's ring buffer via
    append_predictions; markers are ignored for M1."""
    screen, fake, _, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    task_name = next(iter(screen._chart.task_colors.keys()))
    out_ts = np.arange(4, dtype=np.float64) * 0.01
    predictions = {task_name: np.array([0.1, 0.3, 0.5, 0.7])}

    fake.prediction_ready.emit(predictions, out_ts, [])
    QApplication.processEvents()

    assert screen._chart._write_idx == 4
    assert screen._chart._latest_ts == out_ts[-1]
    cap = screen._chart._capacity
    assert screen._chart._buffers[task_name][0] == 0.1
    assert screen._chart._buffers[task_name][cap] == 0.1  # double-length mirror


def test_start_resets_chart_buffers(screen_and_session) -> None:
    """A fresh Start blanks any stale tail from a previous session."""
    screen, fake, _, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    # Push some data through.
    task_name = next(iter(screen._chart.task_colors.keys()))
    fake.prediction_ready.emit(
        {task_name: np.array([0.8, 0.9])},
        np.array([0.0, 0.01]),
        [],
    )
    QApplication.processEvents()
    assert screen._chart._write_idx == 2

    # Halt then Start again — the chart should be blank.
    screen._start_halt_button.halt_clicked.emit()
    screen._start_halt_button.start_clicked.emit()
    assert screen._chart._write_idx == 0
    assert screen._chart._latest_ts is None
    assert np.all(np.isnan(screen._chart._buffers[task_name]))
