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
        # Mirrors SettingsManager.get_settings(): markers_mapping is flattened
        # to event_mapping as {name: id}.
        "event_mapping": {"a": 11, "b": 12},
    }


class _StubAppSession:
    """Stand-in for AppSession that returns the supplied fake live session."""

    def __init__(self, live: _FakeLiveStreamSession, settings: dict | None = None) -> None:
        self._live = live
        self.settings = settings or _make_session_settings()
        self.start_source_calls = 0
        self.stop_source_calls = 0
        self.last_stream_name: str | None = None

    def build_live_stream_session(self, decoder_pipeline_path, *, stream_name=None):
        self.last_stream_name = stream_name
        # Hand out a fresh fake on each call to mirror the real one-shot
        # semantics (the screen builds a new session on every Start).
        if getattr(self._live, "_handed_out", False):
            new_fake = _FakeLiveStreamSession()
            new_fake.raise_on_start = self._live.raise_on_start
            self._live = new_fake
        self._live._handed_out = True
        return self._live

    def start_stream_source(self) -> None:
        self.start_source_calls += 1

    def stop_stream_source(self) -> None:
        self.stop_source_calls += 1

    def discover_streams(self, timeout_sec: float = 3.0) -> list[str]:
        return ["NeuroneStream", "OtherStream"]


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
        # Pre-select a target so the state-machine tests exercise Start
        # without tripping the "no target" guard. Target-selection itself is
        # covered by dedicated tests below.
        screen._target = {"source": "lsl", "stream_name": "NeuroneStream"}
        yield screen, fake, app_session, mock_box


# ── Construction ──────────────────────────────────────────────────────────────


def test_constructor_does_not_build_session(screen_and_session) -> None:
    screen, fake, app_session, _ = screen_and_session
    # The session is built lazily on Start (bound to the chosen target), so
    # __init__ leaves _live unset and starts nothing.
    assert screen._live is None
    assert fake.start_calls == 0


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
        screen._target = {"source": "lsl", "stream_name": "NeuroneStream"}
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


# ── target selection (Step 1b) ─────────────────────────────────────────────────


class _FakeTargetDialog:
    """Stand-in for TargetSelectionDialog: always accepts with a fixed target."""

    last_session = None

    def __init__(self, session, parent=None) -> None:
        _FakeTargetDialog.last_session = session

    def exec(self):
        from PyQt6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def selected_target(self):
        return {"source": "lsl", "stream_name": "PickedStream"}


def test_header_starts_without_target_and_emits_signal(qapp) -> None:
    # Exercise the header in isolation: a real screen would open a modal
    # dialog on click, which would deadlock the offscreen test.
    from frontend.widgets.phase2.header import Phase2Header

    header = Phase2Header()
    assert "Choose target" in header._target_button.text()

    fired: list[bool] = []
    header.choose_target_clicked.connect(lambda: fired.append(True))
    header._target_button.click()
    assert fired == [True]


def test_choose_target_updates_header_and_state(screen_and_session) -> None:
    screen, _, _, _ = screen_and_session
    with patch(
        "frontend.screens.phase2_screen.TargetSelectionDialog", _FakeTargetDialog
    ):
        screen._header.choose_target_clicked.emit()

    assert screen._target == {"source": "lsl", "stream_name": "PickedStream"}
    assert screen._header._target_button.text() == "Target: PickedStream (LSL)"
    assert _FakeTargetDialog.last_session is screen.session


def test_start_without_target_is_guarded(qapp) -> None:
    """Start with no target must not build/start a session — just prompt."""
    from frontend.screens.phase2_screen import Phase2Screen

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    with patch("frontend.screens.phase2_screen.QMessageBox.critical"), patch(
        "frontend.screens.phase2_screen.QMessageBox.information"
    ) as mock_info:
        screen = Phase2Screen(
            session=app_session,
            decoder_pipeline_path=Path("/nonexistent.joblib"),
        )
        assert screen._target is None
        screen._start_halt_button.start_clicked.emit()

        assert mock_info.called
        assert fake.start_calls == 0
        assert app_session.start_source_calls == 0
        assert screen._start_halt_button._state == "idle"


def test_start_uses_selected_stream_and_starts_source(screen_and_session) -> None:
    screen, fake, app_session, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()

    assert app_session.start_source_calls == 1
    assert app_session.last_stream_name == "NeuroneStream"
    assert fake.start_calls == 1
    assert screen._start_halt_button._state == "live"


def test_halt_stops_stream_source(screen_and_session) -> None:
    screen, _, app_session, _ = screen_and_session
    screen._start_halt_button.start_clicked.emit()
    screen._start_halt_button.halt_clicked.emit()

    assert app_session.stop_source_calls >= 1


# ── discovery worker + dialog ───────────────────────────────────────────────────


def test_stream_discovery_worker_emits_names(qapp) -> None:
    from frontend.workers.stream_discovery_worker import StreamDiscoveryWorker

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    worker = StreamDiscoveryWorker(app_session, timeout_sec=1.0)

    results: list[list[str]] = []
    worker.result_ready.connect(results.append)
    worker.run()

    assert results == [["NeuroneStream", "OtherStream"]]


def test_target_dialog_accept_returns_descriptor(qapp) -> None:
    from frontend.widgets.phase2.target_dialog import TargetSelectionDialog

    fake = _FakeLiveStreamSession()
    app_session = _StubAppSession(fake)
    dialog = TargetSelectionDialog(app_session)

    # No streams yet → OK disabled, no result.
    assert dialog.selected_target() is None

    # Simulate a completed discovery, pick the second stream, accept.
    dialog._on_streams_found(["StreamA", "StreamB"])
    dialog._combo.setCurrentText("StreamB")
    dialog._on_accept()

    assert dialog.selected_target() == {"source": "lsl", "stream_name": "StreamB"}
