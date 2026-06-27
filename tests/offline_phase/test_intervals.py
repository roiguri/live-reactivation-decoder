"""Unit tests for the pure interval-tiling helper, build_interval_events.

The helper takes an MNE-style events array and tiles synthetic epoch events
inside each [start, stop] span. These tests exercise it without any MNE objects.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.core.preprocessing_constants import EPOCH_TMAX, EPOCH_TMIN
from backend.offline_phase.preprocessor import build_interval_events

SFREQ = 100.0
WIN = round((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)   # window length in samples
OFFSET = round(EPOCH_TMIN * SFREQ)               # negative; ev = window_start - OFFSET

EVENT_MAPPING = {"red": 11, "trial_start": 20, "trial_end": 21}
REST = [{"name": "rest", "start": "trial_start", "stop": "trial_end"}]


def _events(rows):
    return np.array(rows, dtype=int)


def test_empty_intervals_returns_empty():
    rows, mapping = build_interval_events(_events([[1000, 0, 20]]), SFREQ, [], EVENT_MAPPING)
    assert rows.shape == (0, 3)
    assert mapping == {}


def test_window_count_is_floor_span_over_win():
    span = 500  # samples; floor(500/120) = 4
    events = _events([[1000, 0, 20], [1000 + span, 0, 21]])
    rows, mapping = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == math.floor(span / WIN) == 4
    assert set(rows[:, 2]) == {mapping["rest"]}


def test_windows_are_contiguous_and_inside_span():
    s, e = 1000, 1500
    events = _events([[s, 0, 20], [e, 0, 21]])
    rows, _ = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    # Synthetic event sample = window_start - OFFSET → window_start = ev + OFFSET.
    window_starts = sorted(int(r[0]) + OFFSET for r in rows)
    assert window_starts == [s + k * WIN for k in range(len(rows))]
    # Every window's data span [window_start, window_start + WIN] stays within [s, e].
    for ws in window_starts:
        assert s <= ws and ws + WIN <= e


def test_trailing_partial_is_dropped():
    # span of 1.5 windows → only 1 full window fits.
    s = 1000
    events = _events([[s, 0, 20], [s + WIN + WIN // 2, 0, 21]])
    rows, _ = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == 1


def test_multiple_spans_each_tiled():
    events = _events([
        [1000, 0, 20], [1500, 0, 21],   # span 500 → 4
        [3000, 0, 20], [3300, 0, 21],   # span 300 → 2
    ])
    rows, _ = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == math.floor(500 / WIN) + math.floor(300 / WIN) == 6


def test_start_without_following_stop_is_skipped():
    events = _events([[5000, 0, 20]])  # start, no stop
    rows, mapping = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == 0
    assert "rest" in mapping  # name still mapped, just no windows


def test_start_with_no_stop_before_next_start_is_skipped():
    # First start has another start before any stop → skipped; second pairs normally.
    events = _events([
        [1000, 0, 20],          # start A (no stop before start B) → skipped
        [1100, 0, 20],          # start B
        [1600, 0, 21],          # stop for B (span 500 → 4)
    ])
    rows, _ = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == math.floor(500 / WIN) == 4


def test_synthetic_code_is_disjoint_from_config_and_recording():
    # A real code 22 in the config must push the interval code past it.
    em = {**EVENT_MAPPING, "feedback": 22}
    events = _events([[1000, 0, 20], [1500, 0, 21], [1200, 0, 22]])
    _, mapping = build_interval_events(events, SFREQ, REST, em)
    assert mapping["rest"] not in set(em.values())
    assert mapping["rest"] not in {20, 21, 22}


def test_two_intervals_get_distinct_codes():
    specs = [
        {"name": "rest", "start": "trial_start", "stop": "trial_end"},
        {"name": "fixation", "start": "trial_start", "stop": "trial_end"},
    ]
    events = _events([[1000, 0, 20], [1500, 0, 21]])
    _, mapping = build_interval_events(events, SFREQ, specs, EVENT_MAPPING)
    assert mapping["rest"] != mapping["fixation"]
    assert len({mapping["rest"], mapping["fixation"]}) == 2


def test_span_shorter_than_one_window_yields_nothing():
    events = _events([[1000, 0, 20], [1000 + WIN - 1, 0, 21]])
    rows, _ = build_interval_events(events, SFREQ, REST, EVENT_MAPPING)
    assert len(rows) == 0
