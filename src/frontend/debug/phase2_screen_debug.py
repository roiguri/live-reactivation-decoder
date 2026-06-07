"""Dev-only quick-jump to :class:`Phase2Screen`.

Invoked via ``python -m frontend.debug.main --phase2 [--profile <name>]``.
Skips the whole Phase 1 walkthrough by constructing an :class:`AppSession`
from a debug *profile*'s ``experiment_config.yaml`` and pointing
:class:`Phase2Screen` at that profile's ``models/decoder_pipeline.joblib``.

This module imports only existing production APIs — production code
is **byte-for-byte unaffected**. If a quick-jump ever needs a hook in
production, that's a signal the Phase 2 surface is wrong, not that
the debug helper needs to grow.
"""
from __future__ import annotations

from backend.core.session_paths import SessionPaths
from backend.session import AppSession
from frontend.debug.profiles import DebugProfile, resolve_profile
from frontend.screens.phase2_screen import Phase2Screen


def build_debug_phase2(profile: DebugProfile | None = None) -> Phase2Screen:
    """Build an :class:`AppSession` and a :class:`Phase2Screen` ready to
    embed in :class:`MainWindow`.

    ``profile`` resolves the config + decoder-pipeline paths; when ``None``,
    the default profile is selected (see :func:`resolve_profile`).

    The session is real (config is loaded and validated). The decoder
    pipeline path is stored on the screen; the artifact itself is loaded
    later by ``session.build_live_stream_session(...)`` when live inference
    starts — so a missing pipeline file does not block the shell opening.
    """
    if profile is None:
        profile = resolve_profile()
    session = AppSession(profile.config_path)
    # The profile dir is the session workspace root; Phase 2 logs land under it
    # (profile_dir/phase2_live/...) via session.paths, like Go-Live under output_dir.
    session.paths = SessionPaths(profile.root_dir)
    return Phase2Screen(
        session=session,
        decoder_pipeline_path=profile.pipeline_path,
    )
