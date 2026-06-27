"""Open :class:`Phase2Screen` directly from an existing production output folder.

A production output directory written by a prior Go-Live already contains
everything Phase 2 needs — ``experiment_config.yaml`` and
``models/decoder_pipeline.joblib`` — so live inference can be re-entered without
walking the whole Phase 1 journey again.

This is the production analog of ``frontend.debug.phase2_screen_debug``: same
three-line construction (build an :class:`AppSession`, assign ``session.paths``
directly — no :class:`OfflineOrchestrator`, which Phase 2 never uses — and hand
both to :class:`Phase2Screen`), just pointed at a real output folder instead of a
debug profile. :class:`SessionPaths` derives both the config and the artifact path
from the chosen root, so nothing here guesses a path.
"""
from __future__ import annotations

from pathlib import Path

from backend.core.session_paths import SessionPaths
from backend.session import AppSession
from frontend.screens.phase2_screen import Phase2Screen


def missing_live_artifacts(output_dir: str | Path) -> list[str]:
    """Return human-readable names of the required-but-absent files.

    Empty list means the folder is ready for live mode. The names match the
    on-disk layout (relative to the output root) so they read well in an error
    dialog.
    """
    paths = SessionPaths(Path(output_dir))
    missing: list[str] = []
    if not paths.experiment_config_path.exists():
        missing.append("experiment_config.yaml")
    if not paths.decoder_pipeline_path.exists():
        missing.append("models/decoder_pipeline.joblib")
    return missing


def build_phase2_from_output(output_dir: str | Path) -> Phase2Screen:
    """Build a :class:`Phase2Screen` from an existing production output folder.

    Callers should first check :func:`missing_live_artifacts` is empty so they can
    show a friendly message; this function still fails loudly (config load) if the
    folder is incomplete or corrupt.

    Mirrors ``build_debug_phase2``: a real, validated :class:`AppSession` with
    ``session.paths`` assigned directly (no ``OfflineOrchestrator`` — Phase 2 is
    live-only). The decoder pipeline is loaded lazily when the operator clicks
    Start, so construction is cheap and a stale/missing artifact does not block the
    shell from opening.
    """
    paths = SessionPaths(Path(output_dir))
    session = AppSession(paths.experiment_config_path)
    # Live-only: the output dir is the session workspace root; Phase 2 logs land
    # under it (output_dir/phase2_live/...) via session.paths, exactly as Go-Live.
    session.paths = paths
    return Phase2Screen(
        session=session,
        decoder_pipeline_path=paths.decoder_pipeline_path,
    )
