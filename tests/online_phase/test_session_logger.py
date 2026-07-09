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


# ── decisions sink ───────────────────────────────────────────────────────────


from types import SimpleNamespace  # noqa: E402

from backend.online_phase.session_logger import episodes_from_decisions  # noqa: E402

_INITIAL_CONFIG = {
    "threshold": 0.85,
    "sustain_timepoints": 10,
    "release_timepoints": 1,
}


def _decision_logger(tmp_path):
    return LiveSessionLogger(
        tmp_path,
        ["animate decoder", "inanimate decoder"],
        decision_config=_INITIAL_CONFIG,
    )


def _result(timestamps, active, version, change=None):
    return SimpleNamespace(
        timestamps=np.asarray(timestamps, dtype=float),
        active={k: np.asarray(v, dtype=bool) for k, v in active.items()},
        config_version=version,
        config_change=change,
    )


def _read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_decision_logging_off_by_default(tmp_path):
    logger = LiveSessionLogger(tmp_path, ["a"])  # no decision_config
    logger.on_decisions(_result([1.0], {"a": [True]}, 0))  # no-op
    logger.close()
    assert not (tmp_path / "decisions.csv").exists()
    assert not (tmp_path / "decision_config.jsonl").exists()
    assert "n_decision_samples" not in json.loads((tmp_path / "manifest.json").read_text())


def test_decisions_csv_dense_rows_with_config_version(tmp_path):
    logger = _decision_logger(tmp_path)
    logger.on_decisions(
        _result(
            [10.0, 10.01],
            {"animate decoder": [True, True], "inanimate decoder": [False, True]},
            version=0,
        )
    )
    logger.close()

    rows = _read_rows(tmp_path / "decisions.csv")
    assert rows[0] == [
        "lsl_timestamp", "t_sec", "animate decoder", "inanimate decoder", "config_version"
    ]
    assert rows[1] == ["10.0", "0.0", "True", "False", "0"]
    assert rows[2] == ["10.01", "0.01", "True", "True", "0"]


def test_version0_snapshot_written_at_construction(tmp_path):
    logger = _decision_logger(tmp_path)
    # Crash-safe: the timeline exists before any decisions / close().
    lines = _read_jsonl(tmp_path / "decision_config.jsonl")
    assert lines == [
        {"config_version": 0, "lsl_timestamp": None, "config": _INITIAL_CONFIG}
    ]
    logger.close()


def test_config_change_appends_version_and_rows_carry_it(tmp_path):
    logger = _decision_logger(tmp_path)
    logger.on_decisions(_result([1.0], {"animate decoder": [False], "inanimate decoder": [False]}, 0))

    new_config = {
        "threshold": 0.70,
        "sustain_timepoints": 7,
        "release_timepoints": 1,
    }
    change = SimpleNamespace(version=1, lsl_timestamp=2.0, config=new_config)
    logger.on_decisions(
        _result(
            [2.0], {"animate decoder": [True], "inanimate decoder": [False]}, 1, change
        )
    )
    logger.close()

    timeline = _read_jsonl(tmp_path / "decision_config.jsonl")
    assert [ln["config_version"] for ln in timeline] == [0, 1]
    assert timeline[1] == {"config_version": 1, "lsl_timestamp": 2.0, "config": new_config}

    rows = _read_rows(tmp_path / "decisions.csv")
    assert [r[-1] for r in rows[1:]] == ["0", "1"]  # version boundary at ts 2.0

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["decision_schema_version"] == 1
    assert manifest["decision_initial_config"] == _INITIAL_CONFIG
    assert manifest["n_decision_samples"] == 2


def test_episodes_from_decisions_pairs_edges_and_trailing_open(tmp_path):
    logger = _decision_logger(tmp_path)
    # animate: on at 1.0, off at 3.0 (closed); on again at 4.0 and never closes.
    # inanimate: never on.
    logger.on_decisions(
        _result(
            [1.0, 2.0, 3.0, 4.0],
            {
                "animate decoder": [True, True, False, True],
                "inanimate decoder": [False, False, False, False],
            },
            version=0,
        )
    )
    logger.close()

    episodes = episodes_from_decisions(tmp_path)
    animate = [e for e in episodes if e.decoder == "animate decoder"]
    assert (animate[0].onset_ts, animate[0].offset_ts) == (1.0, 3.0)
    assert (animate[1].onset_ts, animate[1].offset_ts) == (4.0, None)  # open
    assert animate[0].config_version_at_onset == 0
    assert not [e for e in episodes if e.decoder == "inanimate decoder"]


def test_decision_files_flushed_before_close(tmp_path):
    """Crash-safe: rows are on disk before close() (line-buffered append)."""
    logger = _decision_logger(tmp_path)
    logger.on_decisions(
        _result([1.0], {"animate decoder": [True], "inanimate decoder": [False]}, 0)
    )
    # Read without closing the logger.
    rows = _read_rows(tmp_path / "decisions.csv")
    assert len(rows) == 2  # header + one decision row
    logger.close()
