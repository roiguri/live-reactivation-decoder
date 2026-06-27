"""Headless tests for opening Phase 2 directly from an existing output folder.

Covers the production launch helpers in ``frontend.screens.phase2_launch``:
``missing_live_artifacts`` (up-front validation) and ``build_phase2_from_output``
(the debug-quick-jump analog pointed at a real output dir). No LSL stream is
required — the decoder pipeline is loaded lazily at Start, so an empty stub file
suffices for construction.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.core.session_paths import SessionPaths  # noqa: E402
from frontend.screens.phase2_launch import (  # noqa: E402
    build_phase2_from_output,
    missing_live_artifacts,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _make_output_folder(root: Path, sample_config: Path, *, with_pipeline: bool) -> Path:
    """Lay out a production-style output folder under ``root``."""
    paths = SessionPaths(root)
    paths.experiment_config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_config, paths.experiment_config_path)
    if with_pipeline:
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        paths.decoder_pipeline_path.touch()  # lazy load → empty stub is fine
    return root


def test_missing_live_artifacts_empty_folder(tmp_path):
    assert missing_live_artifacts(tmp_path) == [
        "experiment_config.yaml",
        "models/decoder_pipeline.joblib",
    ]


def test_missing_live_artifacts_config_only(tmp_path, sample_config_path):
    _make_output_folder(tmp_path, sample_config_path, with_pipeline=False)
    assert missing_live_artifacts(tmp_path) == ["models/decoder_pipeline.joblib"]


def test_missing_live_artifacts_ready_folder(tmp_path, sample_config_path):
    _make_output_folder(tmp_path, sample_config_path, with_pipeline=True)
    assert missing_live_artifacts(tmp_path) == []


def test_build_phase2_from_output(qapp, tmp_path, sample_config_path):
    _make_output_folder(tmp_path, sample_config_path, with_pipeline=True)
    paths = SessionPaths(tmp_path)

    screen = build_phase2_from_output(tmp_path)

    assert screen.session.paths.root == tmp_path
    assert Path(screen.decoder_pipeline_path) == paths.decoder_pipeline_path
    # Live-only entry: no offline orchestrator is created.
    assert screen.session.offline is None
