from __future__ import annotations

import csv
import json

import numpy as np

from backend.online_phase.session_logger import (
    LiveSessionLogger,
    export_session_npz,
)


def _read_rows(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


# ── predictions CSV ──────────────────────────────────────────────────────────


def test_predictions_csv_has_timestamp_t_sec_and_tasks(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object", "scene"])
    logger.on_predictions(
        {"object": np.array([0.1, 0.2, 0.3]), "scene": np.array([0.4, 0.5, 0.6])},
        np.array([10.0, 10.01, 10.02]),
        [],
    )
    logger.close()

    rows = _read_rows(tmp_path / "predictions.csv")
    assert rows[0] == ["lsl_timestamp", "t_sec", "object", "scene"]
    # t_sec rebased to the first sample (10.0 → 0.0).
    assert [r[1] for r in rows[1:]] == ["0.0", "0.01", "0.02"]
    assert [r[2] for r in rows[1:]] == ["0.1", "0.2", "0.3"]


def test_probabilities_rounded_in_csv_full_precision_in_npz(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object"])
    logger.on_predictions({"object": np.array([0.123456789])}, np.array([1.0]), [])
    logger.close()

    rows = _read_rows(tmp_path / "predictions.csv")
    assert rows[1][2] == "0.12346"  # CSV rounded to 5 dp

    with np.load(tmp_path / "predictions.npz") as data:
        # npz keeps the raw in-memory value, not the rounded CSV one.
        assert data["predictions"][0, 0] == 0.123456789


# ── markers sidecar ──────────────────────────────────────────────────────────


def test_markers_go_to_sidecar_with_resolved_names(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object"], event_names={11: "red", 12: "green"})
    logger.on_predictions(
        {"object": np.array([0.1, 0.2, 0.3])},
        np.array([2.0, 2.01, 2.02]),
        [(2.013, 11)],
    )
    logger.close()

    rows = _read_rows(tmp_path / "markers.csv")
    assert rows[0] == ["lsl_timestamp", "t_sec", "code", "name"]
    assert rows[1] == ["2.013", "0.013", "11", "red"]


def test_marker_off_the_grid_is_not_dropped(tmp_path):
    """A marker far from any prediction sample is recorded verbatim (no snap)."""
    logger = LiveSessionLogger(tmp_path, ["object"])
    logger.on_predictions(
        {"object": np.array([0.1, 0.2])},
        np.array([5.0, 5.01]),
        [(5.004, 99)],
    )
    logger.close()

    rows = _read_rows(tmp_path / "markers.csv")
    assert len(rows) == 2  # header + the one marker
    assert rows[1] == ["5.004", "0.004", "99", ""]  # unmapped code → empty name


def test_simultaneous_markers_both_recorded(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object"])
    logger.on_predictions(
        {"object": np.array([0.1])},
        np.array([3.0]),
        [(3.0001, 11), (3.0002, 12)],
    )
    logger.close()

    rows = _read_rows(tmp_path / "markers.csv")
    assert [r[2] for r in rows[1:]] == ["11", "12"]


# ── manifest ─────────────────────────────────────────────────────────────────


def test_manifest_written_at_start_and_finalized_at_close(tmp_path):
    logger = LiveSessionLogger(
        tmp_path,
        ["object", "scene"],
        event_names={11: "red"},
        metadata={"target_sfreq": 100.0, "config": "experiment_config.yaml"},
    )
    # Preliminary manifest exists before any data / close (crash-safe).
    prelim = json.loads((tmp_path / "manifest.json").read_text())
    assert prelim["schema_version"] == 1
    assert prelim["lsl_t0"] is None
    assert prelim["wall_clock_end"] is None
    assert prelim["event_map"] == {"11": "red"}
    assert prelim["target_sfreq"] == 100.0

    logger.on_predictions(
        {"object": np.array([0.1]), "scene": np.array([0.2])},
        np.array([42.0]),
        [(42.0, 11)],
    )
    logger.close()

    final = json.loads((tmp_path / "manifest.json").read_text())
    assert final["lsl_t0"] == 42.0
    assert final["wall_clock_end"] is not None
    assert final["n_predictions"] == 1
    assert final["n_markers"] == 1
    assert final["config"] == "experiment_config.yaml"


# ── npz bundle ───────────────────────────────────────────────────────────────


def test_npz_bundle_contents(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object", "scene"], event_names={11: "red"})
    logger.on_predictions(
        {"object": np.array([0.1, 0.2]), "scene": np.array([0.4, 0.5])},
        np.array([1.0, 1.01]),
        [(1.005, 11)],
    )
    logger.close()

    with np.load(tmp_path / "predictions.npz") as data:
        assert data["predictions"].shape == (2, 2)
        assert list(data["task_names"]) == ["object", "scene"]
        np.testing.assert_allclose(data["lsl_timestamp"], [1.0, 1.01])
        np.testing.assert_allclose(data["t_sec"], [0.0, 0.01])
        markers = data["markers"]
        assert markers.shape == (1,)
        assert markers["code"][0] == 11
        assert markers["name"][0] == "red"
        manifest = json.loads(str(data["manifest_json"]))
        assert manifest["n_predictions"] == 2


def test_empty_session_writes_valid_artifacts(tmp_path):
    """A Start with no batches (e.g. immediate Halt) still closes cleanly."""
    logger = LiveSessionLogger(tmp_path, ["object"])
    logger.close()

    assert _read_rows(tmp_path / "predictions.csv") == [["lsl_timestamp", "t_sec", "object"]]
    with np.load(tmp_path / "predictions.npz") as data:
        assert data["predictions"].shape == (0, 1)
        assert data["lsl_timestamp"].shape == (0,)


def test_close_is_idempotent(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["object"])
    logger.close()
    logger.close()


# ── recovery exporter ────────────────────────────────────────────────────────


def test_export_session_npz_rebuilds_from_csvs(tmp_path):
    """A crashed session (CSVs present, no npz) can be re-exported from disk."""
    logger = LiveSessionLogger(tmp_path, ["object"], event_names={11: "red"})
    logger.on_predictions(
        {"object": np.array([0.1, 0.2])},
        np.array([7.0, 7.01]),
        [(7.004, 11)],
    )
    logger.close()
    # Simulate "npz lost / never written" then recover purely from the CSVs.
    (tmp_path / "predictions.npz").unlink()

    npz_path = export_session_npz(tmp_path)

    with np.load(npz_path) as data:
        np.testing.assert_allclose(data["predictions"][:, 0], [0.1, 0.2])
        np.testing.assert_allclose(data["lsl_timestamp"], [7.0, 7.01])
        assert data["markers"]["name"][0] == "red"
