from __future__ import annotations

from pathlib import Path

from backend.core.session_paths import SessionPaths


def test_layout_accessors_hang_off_root(tmp_path):
    paths = SessionPaths(tmp_path)

    assert paths.epochs_dir == tmp_path / "epochs"
    assert paths.evaluation_dir == tmp_path / "evaluation"
    assert paths.models_dir == tmp_path / "models"
    assert paths.decoder_pipeline_path == tmp_path / "models" / "decoder_pipeline.joblib"
    assert paths.phase2_live_dir == tmp_path / "phase2_live"


def test_accepts_string_root(tmp_path):
    paths = SessionPaths(str(tmp_path))
    assert isinstance(paths.root, Path)
    assert paths.models_dir == tmp_path / "models"


def test_accessors_are_pure_no_io(tmp_path):
    paths = SessionPaths(tmp_path)
    # Reading path accessors must not create anything on disk.
    _ = (paths.epochs_dir, paths.models_dir, paths.phase2_live_dir)
    assert list(tmp_path.iterdir()) == []


def test_new_phase2_run_dir_creates_timestamped_run_directory(tmp_path):
    paths = SessionPaths(tmp_path)

    run_dir = paths.new_phase2_run_dir()

    assert run_dir.parent == tmp_path / "phase2_live"
    assert run_dir.is_dir()
    # Named by timestamp (YYYYMMDD_HHMMSS) — digits with one underscore.
    assert run_dir.name.replace("_", "").isdigit()
