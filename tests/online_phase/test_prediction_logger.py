from __future__ import annotations

import csv

import numpy as np

from backend.online_phase.prediction_logger import PredictionLogger


def _read_rows(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


def test_writes_header_and_rows_with_empty_markers(tmp_path):
    path = tmp_path / "session.csv"
    logger = PredictionLogger(path, ["object", "scene"], target_sfreq=100.0)

    logger.on_predictions(
        {
            "object": np.array([0.1, 0.2, 0.3]),
            "scene": np.array([0.4, 0.5, 0.6]),
        },
        np.array([1.0, 1.01, 1.02]),
        [],
    )
    logger.close()

    rows = _read_rows(path)
    assert rows == [
        ["timestamp", "marker_code", "object", "scene"],
        ["1.0", "", "0.1", "0.4"],
        ["1.01", "", "0.2", "0.5"],
        ["1.02", "", "0.3", "0.6"],
    ]


def test_marker_code_attaches_to_nearest_timestamp_row(tmp_path):
    path = tmp_path / "session.csv"
    logger = PredictionLogger(path, ["object"], target_sfreq=100.0)

    logger.on_predictions(
        {"object": np.array([0.1, 0.2, 0.3])},
        np.array([2.0, 2.01, 2.02]),
        [(2.0101, 7)],
    )
    logger.close()

    rows = _read_rows(path)
    assert rows[1][1] == ""
    assert rows[2][1] == "7"
    assert rows[3][1] == ""


def test_two_batches_accumulate_in_input_order(tmp_path):
    path = tmp_path / "session.csv"
    logger = PredictionLogger(path, ["object"], target_sfreq=100.0)

    logger.on_predictions(
        {"object": np.array([0.1])},
        np.array([1.0]),
        [],
    )
    logger.on_predictions(
        {"object": np.array([0.2, 0.3])},
        np.array([1.01, 1.02]),
        [],
    )
    logger.close()

    rows = _read_rows(path)
    assert [row[0] for row in rows[1:]] == ["1.0", "1.01", "1.02"]
    assert [row[2] for row in rows[1:]] == ["0.1", "0.2", "0.3"]


def test_close_is_idempotent(tmp_path):
    path = tmp_path / "session.csv"
    logger = PredictionLogger(path, ["object"], target_sfreq=100.0)

    logger.close()
    logger.close()


def test_header_column_order_matches_task_names(tmp_path):
    path = tmp_path / "session.csv"
    logger = PredictionLogger(path, ["scene", "object"], target_sfreq=100.0)
    logger.close()

    rows = _read_rows(path)
    assert rows[0] == ["timestamp", "marker_code", "scene", "object"]
