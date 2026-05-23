"""Dev-only quick-jump to :class:`Phase2Screen`.

Invoked via ``python -m frontend.debug.main --phase2``. Skips the whole
Phase 1 walkthrough by constructing an :class:`AppSession` from a known
``experiment_config.yaml`` and pointing :class:`Phase2Screen` at the
``decoder_pipeline.joblib`` written by the Phase 1 debug walkthrough's
training step.

This module imports only existing production APIs — production code
is **byte-for-byte unaffected**. If a quick-jump ever needs a hook in
production, that's a signal the Phase 2 surface is wrong, not that
the debug helper needs to grow.
"""
from __future__ import annotations

from pathlib import Path

from backend.session import AppSession
from frontend.screens.phase2_screen import Phase2Screen

_DEFAULT_CONFIG = Path("experiment_config.yaml")
_DEFAULT_PIPELINE = Path("debug_snapshots/models/decoder_pipeline.joblib")


def build_debug_phase2(
    config_path: Path = _DEFAULT_CONFIG,
    decoder_pipeline_path: Path = _DEFAULT_PIPELINE,
) -> Phase2Screen:
    """Build an :class:`AppSession` and a :class:`Phase2Screen` ready to
    embed in :class:`MainWindow`.

    The session is real (config is loaded and validated). The decoder
    pipeline path is stored on the screen but not yet read in Commit 2's
    shell — later commits load the artifact via
    ``session.build_live_stream_session(...)``.
    """
    session = AppSession(config_path)
    return Phase2Screen(
        session=session,
        decoder_pipeline_path=decoder_pipeline_path,
    )
