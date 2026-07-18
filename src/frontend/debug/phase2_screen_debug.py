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

import subprocess
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QLabel, QWidget

from backend.core.session_paths import SessionPaths
from backend.session import AppSession
from frontend.debug.debug_bar import DebugBar
from frontend.debug.launch_screen_debug import DebugLaunchScreen
from frontend.debug.profiles import DebugProfile, resolve_profile
from frontend.screens.phase2_screen import Phase2Screen

# scripts/replay_vhdr_to_lsl.py, resolved from this file:
# debug → frontend → src → repo root.
_REPLAY_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "replay_vhdr_to_lsl.py"
)


class DebugPhase2Screen(Phase2Screen):
    """:class:`Phase2Screen` + a full-width debug toolbar.

    The bar carries a disabled ``Next →`` (kept for parity with the welcome /
    Phase 1 debug bars — there is no forward step past live inference) and an
    enabled ``Reset`` that returns to the welcome screen. ``profile`` is kept
    so Reset can rebuild a fresh live screen when the operator continues past
    the welcome screen again.
    """

    # Emitted from the replay stdout-watcher thread when the child prints its
    # "Streaming '<name>' ..." line — i.e. the LSL outlet is up and
    # discoverable. Queued to the UI thread via the default auto-connection.
    _replay_ready = pyqtSignal()

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
        # Handle to the background VHDR→LSL replay subprocess (None when not
        # running). Owned here so Reset/close can always tear it down.
        self._replay_proc: subprocess.Popen | None = None
        # Debug bar pinned full-width to the very top, above the Phase 2
        # header — same placement as the welcome / Phase 1 debug bars. The
        # Phase 2 root is already a vertical [header, body], so a plain insert
        # at index 0 spans the full width; no layout steal needed here.
        self.layout().insertWidget(0, self._build_debug_toolbar())
        self._replay_ready.connect(self._on_replay_ready)

    # ── toolbar ──────────────────────────────────────────────────────────────

    def _build_debug_toolbar(self) -> QWidget:
        self._bar = DebugBar("Live inference")
        # Replay control cluster, grouped just right of the DEBUG chip: the
        # button then its status pill, so the toggle and its state read as a
        # unit. "Start replay" launches scripts/replay_vhdr_to_lsl.py on the
        # profile's recording dir, publishing a NeurOne-like LSL stream to feed
        # the live screen without hardware. Outline (secondary) style.
        self._replay_btn = self._bar.add_button(
            "Start replay ▶",
            kind="outline",
            side="left",
            on_click=self._on_replay_toggle,
        )
        # Hidden until a replay is launched; amber while the stream spins up,
        # green once the child reports the LSL outlet is live.
        self._replay_status = QLabel()
        self._replay_status.setFixedHeight(22)
        self._replay_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._replay_status.setVisible(False)
        self._bar.insert_left_widget(self._replay_status)
        # Reset sits left of Next; Next stays pinned to the far right.
        self._reset_btn = self._bar.add_button("Reset", on_click=self._on_reset)
        # Next is kept for parity with the other debug bars, but there is no
        # step past live inference — so it is permanently disabled, not removed.
        self._next_btn = self._bar.add_button("Next →", enabled=False)
        return self._bar

    # ── VHDR → LSL replay ────────────────────────────────────────────────────

    def _replay_running(self) -> bool:
        return self._replay_proc is not None and self._replay_proc.poll() is None

    def _on_replay_toggle(self) -> None:
        """Start or stop the background VHDR→LSL replay subprocess."""
        if self._replay_running():
            self._stop_replay()
        else:
            self._start_replay()

    def _replay_start_args(self) -> list[str]:
        """Translate the profile's ``replay_start`` manifest field into replay
        CLI flags: ``"first_event"`` → ``--start-at-first-event``; a positive
        number → ``--start-sec N``; anything else (incl. unset) → start at 0.
        """
        start = self._profile.replay_start
        if isinstance(start, str) and start.strip().lower() == "first_event":
            return ["--start-at-first-event"]
        if isinstance(start, (int, float)) and not isinstance(start, bool) and start > 0:
            return ["--start-sec", str(start)]
        return []

    def _start_replay(self) -> None:
        rec_dir = self._profile.raw_data_dir
        if not rec_dir.is_dir():
            self._bar.set_label(f"Replay: recording dir not found — {rec_dir}")
            return
        # -u so the child's prints aren't block-buffered when piped — otherwise
        # the "Streaming ..." readiness line would sit in its buffer. stdout is
        # piped (stderr merged in) and echoed by the watcher thread so console
        # visibility is preserved. Loops forever until we terminate it.
        proc = subprocess.Popen(
            [sys.executable, "-u", str(_REPLAY_SCRIPT), str(rec_dir),
             *self._replay_start_args()],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._replay_proc = proc
        threading.Thread(
            target=self._watch_replay, args=(proc,), daemon=True
        ).start()
        self._replay_btn.setText("Stop replay ■")
        self._set_replay_status("starting")

    def _watch_replay(self, proc: subprocess.Popen) -> None:
        """Echo the replay's output to the console and flip the ready indicator
        when it advertises the stream. Runs on a daemon thread; terminating the
        process closes ``stdout`` and ends the loop."""
        if proc.stdout is None:  # pragma: no cover — defensive
            return
        for line in proc.stdout:
            sys.stdout.write(line)
            if line.lstrip().startswith("Streaming '"):
                self._replay_ready.emit()

    def _on_replay_ready(self) -> None:
        # A late signal from a just-stopped run must not light the indicator.
        if self._replay_running():
            self._set_replay_status("ready")

    def _set_replay_status(self, state: str) -> None:
        """``state`` is ``"off"`` (hidden), ``"starting"`` (amber), or
        ``"ready"`` (green)."""
        if state == "off":
            self._replay_status.setVisible(False)
            return
        text, color = {
            "starting": ("● replay starting…", "#B45309"),
            "ready": ("● replay live", "#16A34A"),
        }[state]
        self._replay_status.setText(text)
        self._replay_status.setStyleSheet(
            f"QLabel {{ color: {color}; background: transparent; "
            "font-size: 11px; font-weight: 700; }}"
        )
        self._replay_status.setVisible(True)

    def _stop_replay(self) -> None:
        proc, self._replay_proc = self._replay_proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover — defensive
                proc.kill()
        self._replay_btn.setText("Start replay ▶")
        self._bar.set_label("Live inference")
        self._set_replay_status("off")

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
        self._stop_replay()
        try:
            self.session.stop_stream_source()
        except Exception:  # pragma: no cover — defensive teardown
            pass
        mw.show_screen(DebugLaunchScreen(self._profile))

    def closeEvent(self, event: QCloseEvent) -> None:
        # Kill the replay subprocess before the base teardown stops the
        # live session and the publishing source.
        self._stop_replay()
        super().closeEvent(event)


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
