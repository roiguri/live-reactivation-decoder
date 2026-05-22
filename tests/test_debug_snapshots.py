"""Round-trip tests for ``frontend.debug.snapshots``.

The debug-mode snapshot helpers are dev tooling, but the on-disk
format is what binds the seeder script (writer) to the debug screen
(reader). These tests guard that contract against regressions —
keys ↔ attributes round-trip, the inferred phase tag matches what was
saved, and empty/None state is omitted (so partial snapshots don't
masquerade as later phases).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frontend.debug.snapshots import load_snapshot, save_snapshot


class _FakeOrchestrator:
    """Minimal stand-in matching the attributes ``snapshots._ATTRS`` touches.

    A real ``OfflineOrchestrator`` would expose the same surface plus
    a lot more; the snapshot module only reads/writes these five names.
    """

    def __init__(self) -> None:
        self._data_dir: Path | None = None
        self._raw = None
        self._epochs = None
        self._eval_results: dict | None = None
        self.online_state: dict = {}


# ── save / load contract ──────────────────────────────────────────────────────


def test_round_trip_eval_done_phase(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._data_dir = Path("/tmp/sub_001")
    orig._epochs = "EPOCHS_SENTINEL"
    orig._eval_results = {
        "suggested_timepoint": 0.35,
        "average_peak_auc": 0.72,
        "tasks": {"red": {"diagonal_auc": np.array([0.5, 0.6, 0.71])}},
    }

    out = save_snapshot(orig, tmp_path / "eval_done.joblib")
    assert out.exists()

    restored = _FakeOrchestrator()
    payload = load_snapshot(restored, out)

    assert payload["_phase"] == "eval_done"
    assert restored._data_dir == orig._data_dir
    assert restored._epochs == "EPOCHS_SENTINEL"
    assert restored._eval_results["suggested_timepoint"] == 0.35
    np.testing.assert_array_equal(
        restored._eval_results["tasks"]["red"]["diagonal_auc"],
        np.array([0.5, 0.6, 0.71]),
    )


def test_round_trip_train_done_phase(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._data_dir = Path("/tmp/sub_001")
    orig._epochs = "EPOCHS_SENTINEL"
    orig._eval_results = {"suggested_timepoint": 0.42}
    orig.online_state = {
        "models": "FAKE_MODELS",
        "spatial_patterns": {"red": np.array([1.0, 2.0, 3.0])},
        "ica_unmixing": np.eye(3),
    }

    out = save_snapshot(orig, tmp_path / "train_done.joblib")
    restored = _FakeOrchestrator()
    payload = load_snapshot(restored, out)

    assert payload["_phase"] == "train_done"
    assert restored.online_state["models"] == "FAKE_MODELS"
    np.testing.assert_array_equal(
        restored.online_state["spatial_patterns"]["red"], np.array([1.0, 2.0, 3.0])
    )


# ── empty-state pruning so partial snapshots don't mis-phase ─────────────────


def test_empty_online_state_does_not_promote_to_train_done(tmp_path: Path) -> None:
    """A populated eval result with default-empty online_state must read
    as ``eval_done`` — not ``train_done``."""
    orig = _FakeOrchestrator()
    orig._eval_results = {"suggested_timepoint": 0.1, "tasks": {}}
    # online_state stays at its default {}; should be dropped.

    payload = load_snapshot(_FakeOrchestrator(), save_snapshot(orig, tmp_path / "e.joblib"))
    assert payload["_phase"] == "eval_done"
    assert "online_state" not in payload


def test_none_attrs_are_omitted(tmp_path: Path) -> None:
    """Unpopulated attributes (None) should not appear in the dump."""
    orig = _FakeOrchestrator()  # all defaults: nothing populated
    payload = load_snapshot(_FakeOrchestrator(), save_snapshot(orig, tmp_path / "blank.joblib"))
    assert payload["_phase"] == "unknown"
    for k in ("_data_dir", "_raw", "_epochs", "_eval_results", "online_state"):
        assert k not in payload, f"unpopulated {k} should be omitted"


# ── include_raw flag ─────────────────────────────────────────────────────────


def test_raw_omitted_by_default(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._raw = "RAW_SENTINEL"
    orig._epochs = "EPOCHS_SENTINEL"

    out = save_snapshot(orig, tmp_path / "no_raw.joblib")
    restored = _FakeOrchestrator()
    restored._raw = "untouched"  # should stay untouched if _raw not in payload
    load_snapshot(restored, out)
    assert restored._raw == "untouched"
    assert restored._epochs == "EPOCHS_SENTINEL"


def test_raw_included_when_flag_set(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._raw = "RAW_SENTINEL"

    out = save_snapshot(orig, tmp_path / "with_raw.joblib", include_raw=True)
    restored = _FakeOrchestrator()
    load_snapshot(restored, out)
    assert restored._raw == "RAW_SENTINEL"


# ── output path handling ─────────────────────────────────────────────────────


def test_save_creates_missing_parent_dirs(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._data_dir = Path("/tmp/x")
    out = save_snapshot(orig, tmp_path / "a" / "b" / "c" / "snap.joblib")
    assert out.exists()
