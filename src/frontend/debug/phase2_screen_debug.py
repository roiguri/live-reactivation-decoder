"""Dev-only quick-jump to :class:`Phase2Screen`, with a debug toolbar.

Invoked via ``python -m frontend.debug.main --phase2 [--profile <name>]``
and by the Phase 1 walkthrough's in-app "Live →" button. Skips the whole
Phase 1 walkthrough by constructing an :class:`AppSession` from a debug
*profile*'s ``experiment_config.yaml`` and pointing the screen at that
profile's ``models/decoder_pipeline.joblib``.

:class:`DebugPhase2Screen` subclasses the production :class:`Phase2Screen`
purely to add a full-width debug toolbar (matching the welcome / Phase 1
debug bars): a disabled ``Next →`` kept for visual parity, and a ``Reset``
that returns to the welcome screen. Production code is **byte-for-byte
unaffected** — the only extra imports are sibling ``frontend.debug`` helpers.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from backend.core.session_paths import SessionPaths
from backend.session import AppSession
from frontend.debug.debug_bar import DEBUG_PREFIX, DebugBar
from frontend.debug.launch_screen_debug import DebugLaunchScreen
from frontend.debug.profiles import DebugProfile, resolve_profile
from frontend.screens.phase2_screen import Phase2Screen


class DebugPhase2Screen(Phase2Screen):
    """:class:`Phase2Screen` + a full-width debug toolbar.

    The bar carries a disabled ``Next →`` (kept for parity with the welcome /
    Phase 1 debug bars — there is no forward step past live inference) and an
    enabled ``Reset`` that returns to the welcome screen. ``profile`` is kept
    so Reset can rebuild a fresh live screen when the operator continues past
    the welcome screen again.
    """

    def __init__(
        self,
        profile: DebugProfile,
        *,
        session: AppSession,
        decoder_pipeline_path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            session=session,
            decoder_pipeline_path=decoder_pipeline_path,
            parent=parent,
        )
        self._profile = profile
        # Debug bar pinned full-width to the very top, above the Phase 2
        # header — same placement as the welcome / Phase 1 debug bars. The
        # Phase 2 root is already a vertical [header, body], so a plain insert
        # at index 0 spans the full width; no layout steal needed here.
        self.layout().insertWidget(0, self._build_debug_toolbar())

    # ── toolbar ──────────────────────────────────────────────────────────────

    def _build_debug_toolbar(self) -> QWidget:
        bar = DebugBar(f"{DEBUG_PREFIX}Live inference")
        # Reset sits left of Next; Next stays pinned to the far right.
        self._reset_btn = bar.add_button("Reset", on_click=self._on_reset)
        # Next is kept for parity with the other debug bars, but there is no
        # step past live inference — so it is permanently disabled, not removed.
        self._next_btn = bar.add_button("Next →", kind="primary", enabled=False)
        return bar

    # ── navigation ───────────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        """Return to the welcome screen.

        The Phase 2 screen isn't closed (MainWindow keeps it in its stack), so
        mirror ``closeEvent``'s teardown explicitly — stop the live session and
        the publishing source — to avoid a dangling stream source/proxy. The
        welcome screen's Next continues back into a fresh live screen.
        """
        mw = self.window()
        if mw is None or not hasattr(mw, "show_screen"):
            return
        self._safely_stop()
        try:
            self.session.stop_stream_source()
        except Exception:  # pragma: no cover — defensive teardown
            pass
        mw.show_screen(DebugLaunchScreen(self._profile))


def build_debug_phase2(profile: DebugProfile | None = None) -> DebugPhase2Screen:
    """Build an :class:`AppSession` and a :class:`DebugPhase2Screen` ready to
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
    return DebugPhase2Screen(
        profile,
        session=session,
        decoder_pipeline_path=profile.pipeline_path,
    )
