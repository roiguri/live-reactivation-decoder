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
    """Minimal stand-in matching the attributes ``snapshots._ATTRS`` touches."""

    def __init__(self) -> None:
        self._data_dir: Path | None = None
        self._raw = None
        self._preprocessor = None
        self._epochs = None
        self._eval_results: dict | None = None
        self._live_artifact_spec = None
        self._ui_state: dict | None = None


class _FakePreprocessor:
    """Stand-in for ``OfflinePreprocessor``: just the fields we expect
    save_snapshot to preserve, plus a ``raw`` we expect it to strip."""

    def __init__(self) -> None:
        self.raw = "RAW_TO_BE_STRIPPED"
        self.epochs = "EPOCHS_SENTINEL"
        self.ica = "ICA_SENTINEL"
        self._bad_channels = ["F4", "C3"]
        self._interp_weights = np.array([[1.0, 2.0]])
        self._post_hygiene_eeg_names = ["Fp1", "Fp2"]


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


def test_round_trip_preproc_done_phase(tmp_path: Path) -> None:
    """A populated preprocessor + epochs (no eval results yet) tags as
    ``preproc_done`` and round-trips the preprocessor fields the
    debug walkthrough needs."""
    orig = _FakeOrchestrator()
    orig._data_dir = Path("/tmp/sub_001")
    orig._preprocessor = _FakePreprocessor()
    orig._epochs = "EPOCHS_SENTINEL"

    out = save_snapshot(orig, tmp_path / "preproc_done.joblib")
    restored = _FakeOrchestrator()
    payload = load_snapshot(restored, out)

    assert payload["_phase"] == "preproc_done"
    assert restored._preprocessor is not None
    # ``.raw`` is the only field stripped before pickling.
    assert restored._preprocessor.raw is None
    # Everything else round-trips verbatim.
    assert restored._preprocessor.ica == "ICA_SENTINEL"
    assert restored._preprocessor._bad_channels == ["F4", "C3"]
    np.testing.assert_array_equal(
        restored._preprocessor._interp_weights, np.array([[1.0, 2.0]])
    )
    assert restored._preprocessor._post_hygiene_eeg_names == ["Fp1", "Fp2"]
    # The original instance is **not** mutated by save_snapshot; we
    # shallow-copied before stripping `.raw`.
    assert orig._preprocessor.raw == "RAW_TO_BE_STRIPPED"


def test_round_trip_train_done_phase(tmp_path: Path) -> None:
    orig = _FakeOrchestrator()
    orig._data_dir = Path("/tmp/sub_001")
    orig._epochs = "EPOCHS_SENTINEL"
    orig._eval_results = {"suggested_timepoint": 0.42}
    # ``_live_artifact_spec`` would normally be a Pydantic model; for the
    # snapshot round-trip we just need a non-None sentinel (joblib pickles
    # arbitrary objects).
    orig._live_artifact_spec = {"sentinel": "FAKE_SPEC"}
    orig._ui_state = {
        "spatial_patterns": {"red": np.array([1.0, 2.0, 3.0])},
        "mne_info": "MNE_INFO_SENTINEL",
    }

    out = save_snapshot(orig, tmp_path / "train_done.joblib")
    restored = _FakeOrchestrator()
    payload = load_snapshot(restored, out)

    assert payload["_phase"] == "train_done"
    assert restored._live_artifact_spec == {"sentinel": "FAKE_SPEC"}
    np.testing.assert_array_equal(
        restored._ui_state["spatial_patterns"]["red"], np.array([1.0, 2.0, 3.0])
    )
    assert restored._ui_state["mne_info"] == "MNE_INFO_SENTINEL"


# ── empty-state pruning so partial snapshots don't mis-phase ─────────────────


def test_missing_live_artifact_does_not_promote_to_train_done(tmp_path: Path) -> None:
    """A populated eval result with no live artifact spec must read as
    ``eval_done`` — not ``train_done``."""
    orig = _FakeOrchestrator()
    orig._eval_results = {"suggested_timepoint": 0.1, "tasks": {}}
    # _live_artifact_spec stays at its default None; should be dropped.

    payload = load_snapshot(_FakeOrchestrator(), save_snapshot(orig, tmp_path / "e.joblib"))
    assert payload["_phase"] == "eval_done"
    assert "_live_artifact_spec" not in payload


def test_none_attrs_are_omitted(tmp_path: Path) -> None:
    """Unpopulated attributes (None) should not appear in the dump."""
    orig = _FakeOrchestrator()  # all defaults: nothing populated
    payload = load_snapshot(_FakeOrchestrator(), save_snapshot(orig, tmp_path / "blank.joblib"))
    assert payload["_phase"] == "unknown"
    for k in ("_data_dir", "_raw", "_epochs", "_eval_results", "_live_artifact_spec", "_ui_state"):
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
