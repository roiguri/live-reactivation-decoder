"""Headless tests for the event-locked (frozen) chart + browsing view.

Drives :class:`FrozenEventChart` directly (calling ``_maybe_freeze`` instead
of waiting on its QTimer) to exercise epoching, the pending-capture queue,
history accumulation, and follow-vs-browse. Then covers the
:class:`FrozenEventView` dropdown wiring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.widgets.phase2.frozen_event_chart import FrozenEventChart  # noqa: E402
from frontend.widgets.phase2.frozen_event_view import FrozenEventView  # noqa: E402

TASKS = ["alpha", "beta"]
EVENTS = {11: "red", 12: "green"}
SFREQ = 100.0
DT = 1.0 / SFREQ
PRE, POST = 0.2, 1.0


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _make_chart() -> FrozenEventChart:
    return FrozenEventChart(
        TASKS,
        pre_seconds=PRE,
        post_seconds=POST,
        target_sfreq=SFREQ,
        event_names=EVENTS,
    )


def _stream(chart, t_start: float, t_end: float) -> None:
    """Feed a continuous ramp from t_start to t_end. Each task gets a distinct
    value (alpha = ts, beta = ts + 0.5 mod) so we can check curve alignment."""
    ts = np.arange(round(t_start / DT), round(t_end / DT)) * DT
    preds = {
        "alpha": np.sin(ts) * 0.5 + 0.5,
        "beta": np.cos(ts) * 0.5 + 0.5,
    }
    chart.append_predictions(preds, ts)


# ── epoching ────────────────────────────────────────────────────────────────


def test_freeze_after_post_window(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 0.30)  # pre-event baseline + onset
    chart.append_markers([(0.30, 11)])
    # Not enough post-event data yet → no capture.
    chart._maybe_freeze()
    assert chart.frozen_event is None
    assert len(chart._history) == 0

    _stream(chart, 0.30, 1.35)  # stream past t_event + post (1.30)
    chart._maybe_freeze()
    assert chart.frozen_event is not None
    assert chart.frozen_event["name"] == "red"
    assert chart.frozen_event["ts"] == pytest.approx(0.30)
    assert len(chart._history) == 1


def test_epoch_window_bounds_and_onset_at_zero(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 1.40)
    chart.append_markers([(0.30, 11)])
    chart._maybe_freeze()

    snap = chart.frozen_event
    x = snap["x"]
    # x is rebased so onset sits at 0; window is [-pre, +post].
    assert x.min() == pytest.approx(-PRE, abs=DT)
    assert x.max() == pytest.approx(POST, abs=DT)
    assert (np.diff(x) > 0).all()  # sorted ascending
    # Curve length matches the window sample count for both tasks.
    assert len(snap["ys"]["alpha"]) == len(x)
    assert len(snap["ys"]["beta"]) == len(x)


def test_unmapped_code_dropped(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 1.40)
    chart.append_markers([(0.30, 99)])  # 99 not in EVENTS
    chart._maybe_freeze()
    assert chart.frozen_event is None
    assert len(chart._history) == 0


# ── pending queue: every event lands in history ───────────────────────────────


def test_two_close_events_both_captured(qapp) -> None:
    """Events 0.4 s apart (< post window) must BOTH become history snapshots —
    the queue does not drop the earlier one."""
    chart = _make_chart()
    _stream(chart, 0.0, 0.60)
    chart.append_markers([(0.30, 11), (0.70, 12)])
    _stream(chart, 0.60, 1.80)  # past 0.70 + post = 1.70
    chart._maybe_freeze()

    assert len(chart._history) == 2
    # Newest-first: green (0.70) at index 0, red (0.30) at index 1.
    assert chart._history[0]["name"] == "green"
    assert chart._history[1]["name"] == "red"
    # Auto-follow leaves the newest on screen.
    assert chart.frozen_event["name"] == "green"


def test_history_labels_newest_first(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 0.60)
    chart.append_markers([(0.30, 11), (0.70, 12)])
    _stream(chart, 0.60, 1.80)
    chart._maybe_freeze()

    labels = chart.history_labels()
    assert len(labels) == 2
    assert "green" in labels[0] and "#2" in labels[0]
    assert "red" in labels[1] and "#1" in labels[1]


# ── follow vs browse ──────────────────────────────────────────────────────────


def test_auto_follow_renders_newest(qapp) -> None:
    """While following, each completed event replaces the on-screen view."""
    chart = _make_chart()
    _stream(chart, 0.0, 1.40)
    chart.append_markers([(0.30, 11)])
    chart._maybe_freeze()
    assert chart.frozen_event["name"] == "red"
    assert chart._current_index == 0

    chart.append_markers([(1.60, 12)])
    _stream(chart, 1.40, 2.70)
    chart._maybe_freeze()
    # Followed onto the newest event.
    assert chart.frozen_event["name"] == "green"
    assert chart._current_index == 0
    assert len(chart._history) == 2


def test_browse_back_then_new_event_keeps_view(qapp) -> None:
    chart = _make_chart()
    # Two events captured.
    _stream(chart, 0.0, 0.60)
    chart.append_markers([(0.30, 11), (0.70, 12)])
    _stream(chart, 0.60, 1.80)
    chart._maybe_freeze()
    assert len(chart._history) == 2

    # Browse to the older event (red, index 1) → follow cleared.
    chart.show_event(1)
    assert chart.frozen_event["name"] == "red"
    assert chart._following is False

    # A third event arrives and completes.
    chart.append_markers([(2.00, 11)])
    _stream(chart, 1.80, 3.10)
    chart._maybe_freeze()

    # Display still on red; its index shifted from 1 to 2; history grew to 3.
    assert chart.frozen_event["name"] == "red"
    assert len(chart._history) == 3
    assert chart._current_index == 2


def test_show_event_out_of_range_is_noop(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 1.40)
    chart.append_markers([(0.30, 11)])
    chart._maybe_freeze()
    before = chart.frozen_event
    chart.show_event(5)  # out of range
    chart.show_event(-1)
    assert chart.frozen_event is before


# ── reset ─────────────────────────────────────────────────────────────────────


def test_reset_clears_history_and_pending(qapp) -> None:
    chart = _make_chart()
    _stream(chart, 0.0, 1.40)
    chart.append_markers([(0.30, 11)])
    chart._maybe_freeze()
    assert len(chart._history) == 1

    chart.reset_buffers()
    assert chart._history == []
    assert chart._pending == []
    assert chart.frozen_event is None
    assert chart._latest_ts is None
    assert chart._following is True


# ── composite view: dropdown wiring ───────────────────────────────────────────


def _make_view() -> FrozenEventView:
    return FrozenEventView(
        TASKS, pre_seconds=PRE, post_seconds=POST, target_sfreq=SFREQ, event_names=EVENTS
    )


def test_view_combo_populates_on_capture(qapp) -> None:
    view = _make_view()
    assert not view._combo.isEnabled()  # empty until first event

    view.append_predictions(
        {"alpha": np.full(60, 0.5), "beta": np.full(60, 0.5)},
        np.arange(60) * DT,
    )
    view.append_markers([(0.10, 11)])
    view.append_predictions(
        {"alpha": np.full(80, 0.5), "beta": np.full(80, 0.5)},
        (np.arange(60, 140)) * DT,
    )
    view.chart._maybe_freeze()

    assert view._combo.isEnabled()
    assert view._combo.count() == 1
    assert "red" in view._combo.currentText()
    # On the latest, the "Latest" button shows its active state.
    assert view._latest_btn.property("active") is True


def test_view_selecting_past_event_shows_it(qapp) -> None:
    view = _make_view()
    # Capture two events.
    view.append_predictions(
        {"alpha": np.full(60, 0.5), "beta": np.full(60, 0.5)}, np.arange(60) * DT
    )
    view.append_markers([(0.10, 11), (0.50, 12)])
    view.append_predictions(
        {"alpha": np.full(120, 0.5), "beta": np.full(120, 0.5)},
        (np.arange(60, 180)) * DT,
    )
    view.chart._maybe_freeze()
    assert view._combo.count() == 2

    # On the latest (green): Latest button is in its active state; older IS
    # available (2 events), newer is not.
    assert view._latest_btn.property("active") is True
    assert view._older_btn.isEnabled()
    assert not view._newer_btn.isEnabled()

    # Select the older event (index 1) via the combo.
    view._combo.setCurrentIndex(1)
    assert view.chart.frozen_event["name"] == "red"
    assert view._latest_btn.property("active") is False  # reviewing → normal
    assert view._newer_btn.isEnabled()  # can step back toward newer
    assert not view._older_btn.isEnabled()  # already the oldest

    # Step "newer" returns to the latest.
    view._show_newer()
    assert view._combo.currentIndex() == 0
    assert view.chart.frozen_event["name"] == "green"
    assert view._latest_btn.property("active") is True

    # Browse back, then jump to latest.
    view._combo.setCurrentIndex(1)
    view._jump_to_latest()
    assert view._combo.currentIndex() == 0
    assert view.chart.frozen_event["name"] == "green"
    assert view._latest_btn.property("active") is True


def _feed_view(view, a: int, b: int) -> None:
    n = b - a
    view.append_predictions(
        {"alpha": np.full(n, 0.5), "beta": np.full(n, 0.5)},
        np.arange(a, b) * DT,
    )


def test_view_pin_latest_deactivates_follow(qapp) -> None:
    """Clicking the active Latest button pins the current event: follow turns
    off, the view stays put, and a new event no longer advances it."""
    view = _make_view()
    _feed_view(view, 0, 140)  # 0 .. 1.39 s
    view.append_markers([(0.10, 11)])
    view.chart._maybe_freeze()
    assert view.chart.following is True
    assert view._latest_btn.property("active") is True

    # Click while active → deactivate, staying on the (still newest) event.
    view._on_latest_clicked()
    assert view.chart.following is False
    assert view.chart._current_index == 0
    assert view._latest_btn.property("active") is False

    # A new event arrives — must NOT jump the view forward.
    _feed_view(view, 140, 280)  # 1.40 .. 2.79 s
    view.append_markers([(1.60, 12)])
    view.chart._maybe_freeze()
    assert view.chart.following is False
    assert view.chart.frozen_event["name"] == "red"  # still the pinned event
    assert view.chart._current_index == 1  # shifted as history grew
    assert view._combo.currentIndex() == 1

    # Click while inactive → go live: newest event, follow back on.
    view._on_latest_clicked()
    assert view.chart.following is True
    assert view.chart._current_index == 0
    assert view.chart.frozen_event["name"] == "green"
    assert view._latest_btn.property("active") is True


def test_view_reset_clears_combo(qapp) -> None:
    view = _make_view()
    view.append_predictions(
        {"alpha": np.full(60, 0.5), "beta": np.full(60, 0.5)}, np.arange(60) * DT
    )
    view.append_markers([(0.10, 11)])
    view.append_predictions(
        {"alpha": np.full(80, 0.5), "beta": np.full(80, 0.5)},
        (np.arange(60, 140)) * DT,
    )
    view.chart._maybe_freeze()
    assert view._combo.isEnabled()

    view.reset_buffers()
    assert not view._combo.isEnabled()
    assert view.chart.frozen_event is None
    # Empty state: every control disabled.
    assert not view._older_btn.isEnabled()
    assert not view._newer_btn.isEnabled()
    assert not view._latest_btn.isEnabled()


def test_history_label_time_is_session_relative(qapp) -> None:
    """The events-list time is seconds since the stream started, not the raw
    LSL clock — so a stream beginning at a large LSL timestamp still reads as
    a small ``+X.Xs``."""
    chart = _make_chart()
    base = 48000.0  # simulate a large pylsl local_clock() origin
    ts = base + np.arange(140) * DT
    chart.append_predictions(
        {"alpha": np.full(140, 0.5), "beta": np.full(140, 0.5)}, ts
    )
    chart.append_markers([(base + 0.30, 11)])
    chart._maybe_freeze()

    assert chart.frozen_event["t_rel"] == pytest.approx(0.30, abs=DT)
    assert "+0.3s" in chart.history_labels()[0]
