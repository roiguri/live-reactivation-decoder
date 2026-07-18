"""Phase1Screen subclass with a Next/Prev walkthrough toolbar.

Dev-only. Production ``frontend.main`` does **not** import this module —
it's invoked via ``frontend.debug.main``.

What it adds over :class:`Phase1Screen`:

* A visible ``[DEBUG] `` prefix in the workspace header so the operator
  always knows the mode.
* A **debug toolbar** at the top of the workspace card with
  ``Step N/M: <name>   [Next →]   [Reset]`` controls.
* One keyboard shortcut (window-scoped): **Ctrl+Right** = Next.

The walkthrough is a linear sequence of atomic actions that mimic
each user interaction on the Phase 1 pipeline — picking the config,
output dir, demo data, etc. — and skip the slow compute (data load,
preprocessing, evaluation, training) by emitting completion signals
directly or loading on-disk snapshots written by
``scripts/demo_seed_debug_snapshots.py``.

Each Next runs the current step's action and advances. Reset rewinds
to step 0 + clears file-picker visuals + switches the workspace to
Settings; for a truly clean restart, relaunch the debug app. There
is no Prev — irreversible actions (snapshot injected, signals fired)
can't be cleanly rewound without re-init.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QMessageBox, QVBoxLayout, QWidget,
)

from backend.session import AppSession
from frontend.debug.debug_bar import DebugBar
from frontend.debug.mne_review import review_bad_channels, review_ica_components
from frontend.debug.phase2_screen_debug import build_debug_phase2
from frontend.debug.profiles import DebugProfile, resolve_profile
from frontend.debug.snapshots import load_snapshot
from frontend.screens.phase1_screen import Phase1Screen, _NODE_TITLES

logger = logging.getLogger(__name__)

# The workspace header keeps a text "[DEBUG] " prefix (a separate widget from
# the debug bar, which now carries the amber DEBUG chip instead).
_DEBUG_PREFIX = "[DEBUG] "

# The bad-channel step loads + filters the raw synchronously on the GUI
# thread (production runs it in a worker); crop to this many seconds first so
# the walkthrough stays responsive. Set to ``None`` to review the full raw.
_BADCHAN_PREVIEW_SECONDS: float | None = 120.0


@dataclass
class _Step:
    name: str
    action: Callable[[], object]  # returns False to "don't advance"


class DebugPhase1Screen(Phase1Screen):
    """Phase1Screen + Next/Prev walkthrough toolbar."""

    def __init__(self, profile: DebugProfile | None = None, parent=None) -> None:
        super().__init__(parent)

        self._profile: DebugProfile = (
            profile if profile is not None else resolve_profile()
        )
        self._step_idx: int = 0
        self._steps: list[_Step] = [
            _Step("Load config",              self._step_load_config),
            _Step("Pick output directory",    self._step_pick_output),
            _Step("Continue → Load Data",     self._step_continue_settings),
            _Step("Pick demo data folder",    self._step_pick_data),
            _Step("Skip data load",           self._step_skip_load),
            _Step("Bad-channel review (MNE)",  self._step_bad_channels),
            _Step("ICA component review (MNE)", self._step_ica_review),
            _Step("Skip preprocessing",       self._step_skip_preproc),
            _Step("Continue → Evaluation",    self._step_continue_preproc),
            _Step("Skip evaluation",          self._step_skip_eval),
            _Step("Continue → Train",         self._step_continue_eval),
            _Step("Skip training",            self._step_skip_train),
        ]

        # Always-visible mode indicator, even for the initial node.
        self._header_title.setText(_DEBUG_PREFIX + _NODE_TITLES[0])

        # Debug toolbar pinned full-width to the very top of the window —
        # above BOTH the workspace card and the journey (flow) side panel —
        # matching the welcome screen's debug bar
        # (frontend.debug.launch_screen_debug).
        #
        # Phase1Screen installs a horizontal root layout on ``self``
        # ([card area, journey panel]). To seat the bar above that whole row we
        # steal that layout onto a content widget and give ``self`` a fresh
        # vertical root of [toolbar, content]. Qt's setLayout() reparents an
        # existing layout (and its child widgets) off its old widget, so no
        # manual child juggling is needed. Dev-mode-acceptable coupling to the
        # parent's layout shape.
        toolbar = self._build_debug_toolbar()
        content = QWidget()
        content.setLayout(self.layout())
        new_root = QVBoxLayout(self)
        new_root.setContentsMargins(0, 0, 0, 0)
        new_root.setSpacing(0)
        new_root.addWidget(toolbar)
        new_root.addWidget(content, 1)

        self._install_shortcuts()
        self._update_toolbar()

    # ── parent hook override ─────────────────────────────────────────────────

    def _on_node_changed(self, completed_node: int) -> None:
        """Keep the [DEBUG] prefix on every header change."""
        next_idx = completed_node
        if next_idx < self._workspace.count():
            self._workspace.setCurrentIndex(next_idx)
            self._header_title.setText(_DEBUG_PREFIX + _NODE_TITLES[next_idx])

    # ── toolbar + shortcuts ──────────────────────────────────────────────────

    def _build_debug_toolbar(self) -> QWidget:
        self._debug_bar = DebugBar()
        # Left-to-right: Live jump, Reset, then Next pinned to the far right.
        # "Live →" is a direct hop to Phase 2 for this profile, available at
        # any point in the walkthrough (see _jump_to_phase2).
        self._phase2_btn = self._debug_bar.add_button(
            "Live →", kind="outline", on_click=self._jump_to_phase2
        )
        self._reset_btn = self._debug_bar.add_button("Reset", on_click=self._on_reset)
        self._next_btn = self._debug_bar.add_button(
            "Next →", on_click=self._on_next
        )
        return self._debug_bar

    def _install_shortcuts(self) -> None:
        sc = QShortcut(QKeySequence("Ctrl+Right"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_next)

    def _jump_to_phase2(self) -> None:
        """Hop straight to the live (Phase 2) screen for this profile.

        Reuses the same builder as the ``--phase2`` entry point
        (:func:`build_debug_phase2`), so it works at any point in the
        walkthrough regardless of how far the operator has clicked — it
        builds a fresh session from the profile's config + pipeline rather
        than depending on the walkthrough's trained-in session. The
        ``--phase2`` CLI access stays as-is; this is an in-app shortcut to
        the same destination.
        """
        mw = self.window()
        if mw is None or not hasattr(mw, "show_screen"):
            return
        try:
            phase2 = build_debug_phase2(self._profile)
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Debug jump to Phase 2 failed")
            QMessageBox.critical(
                self, "Could not open Phase 2",
                f"Failed to build the live screen for profile "
                f"'{self._profile.name}':\n\n{exc}",
            )
            return
        mw.show_screen(phase2)

    def _update_toolbar(self) -> None:
        total = len(self._steps)
        if self._step_idx == 0:
            label = f"Step 0/{total}: (press Next to start)"
        elif self._step_idx <= total:
            done_step = self._steps[self._step_idx - 1]
            label = f"Step {self._step_idx}/{total}: {done_step.name} ✓"
        self._debug_bar.set_label(label)
        self._next_btn.setEnabled(self._step_idx < total)

    # ── Next / Reset ─────────────────────────────────────────────────────────

    def _on_next(self) -> None:
        if self._step_idx >= len(self._steps):
            return
        step = self._steps[self._step_idx]
        logger.info("Debug walkthrough → step %d: %s",
                    self._step_idx + 1, step.name)
        try:
            result = step.action()
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Walkthrough step %d failed", self._step_idx + 1)
            QMessageBox.critical(
                self,
                "Walkthrough error",
                f"Step {self._step_idx + 1}: {step.name}\n\n{exc}",
            )
            return
        if result is False:
            return  # action explicitly signalled "don't advance"
        self._step_idx += 1
        self._update_toolbar()

    def _on_reset(self) -> None:
        """Rewind the walkthrough back to the *empty* Settings page.

        Clears every piece of Settings state the walkthrough populated so
        the screen looks identical to a fresh launch — pickers reset,
        Subject/Session/etc. blanked, ``✓  Config loaded`` hidden, the
        journey-panel Continue button regated as 'not ready', and the
        trail rewound (node 1 active, the rest inactive).
        """
        logger.info("Debug walkthrough reset to step 0")
        self._step_idx = 0

        sv = self._settings_view
        sv._config_picker.clear()
        sv._output_picker.clear()
        sv._config_status_lbl.hide()
        sv._temp_session = None
        sv._config_path = None
        sv._output_dir = None
        sv._update_settings_display(None)
        sv._update_continue_state()  # emits ready_changed(False) — regate Continue

        self._load_data_view._picker.clear()
        self._load_data_view._picker.setEnabled(True)
        self._load_data_view._data_dir = None

        self.session = None  # ``Phase1Screen.session`` lives on the parent

        self._reset_journey_panel()

        self._workspace.setCurrentIndex(0)
        self._header_title.setText(_DEBUG_PREFIX + _NODE_TITLES[0])
        self._update_toolbar()

    def _reset_journey_panel(self) -> None:
        """Best-effort rewind of the side-panel trail to fresh-launch state.

        JourneyPanel has no production reset API, so we poke a couple of
        private fields (``_completed_segments``, ``_animating_segment``)
        and re-drive each ``JourneyNode``'s state — dev-mode acceptable.
        """
        jp = self._journey_panel
        jp._anim.stop()  # no-op if not running
        jp._completed_segments = 0
        jp._animating_segment = -1
        for i, node in enumerate(jp._nodes):
            node.set_state("active" if i == 0 else "inactive")
        jp.update()

    # ── walkthrough actions ──────────────────────────────────────────────────

    def _step_load_config(self) -> None:
        """1: build an AppSession ourselves, populate SettingsView's display."""
        session = AppSession(self._profile.config_path)
        self._fake_picker(
            self._settings_view._config_picker,
            str(self._profile.config_path),
        )
        # _on_config_loaded sets _temp_session + _config_path and populates
        # the read-only display fields — same effect as the ConfigLoaderWorker
        # finishing successfully, but instant.
        self._settings_view._on_config_loaded(session)

    def _step_pick_output(self) -> None:
        """2: fake the output-directory picker."""
        self._profile.root_dir.mkdir(parents=True, exist_ok=True)
        path = str(self._profile.root_dir.resolve())
        self._fake_picker(self._settings_view._output_picker, path)
        self._settings_view._on_output_dir_selected(path)

    def _step_continue_settings(self) -> None:
        """3: fire the Settings 'Continue' action (advances trail → LoadData)."""
        self._settings_view.trigger_continue()

    def _step_pick_data(self) -> None:
        """4: fake the demo data folder pick."""
        path = str(self._profile.raw_data_dir)
        self._fake_picker(self._load_data_view._picker, path)
        self._load_data_view._on_dir_selected(path)

    def _step_skip_load(self) -> None:
        """5: fake the post-load state instantly — no LoadWorker.

        Production fires ``data_loaded`` only when the worker completes;
        we emit it directly, which (per ``phase1_screen.py``) auto-advances
        the trail to Preprocessing and primes the Preprocessing view's
        ready state. The orchestrator's ``_raw`` stays ``None`` —
        downstream views only read snapshot state, never raw.
        """
        self._load_data_view._picker.setEnabled(False)
        self._load_data_view.data_loaded.emit()

    def _step_bad_channels(self) -> bool | None:
        """Pop MNE's interactive bad-channel window — the real production screen.

        The raw is loaded live from the profile's original recording folder
        (``manifest.raw_data_dir``), not a stored snapshot, so no raw is baked
        into ``debug_snapshots/``. It's cropped to ``_BADCHAN_PREVIEW_SECONDS``
        before filtering to keep this (main-thread) step responsive. The bads
        the operator marks are logged but not persisted — the seeded snapshot
        already carries the finished pipeline; this step just demonstrates the
        screen.
        """
        data_dir = self._profile.raw_data_dir
        if not Path(data_dir).exists():
            QMessageBox.warning(
                self,
                "Raw data missing",
                f"{data_dir} not found.\n\nThe profile's original recording "
                "folder is required for the bad-channel window (it is loaded "
                "live, not from a snapshot).",
            )
            return False
        orch = self.session.offline
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            orch.set_file_path(data_dir)
            orch.load_raw_data()
            if _BADCHAN_PREVIEW_SECONDS is not None and orch._raw is not None:
                tmax = min(_BADCHAN_PREVIEW_SECONDS, float(orch._raw.times[-1]))
                orch._raw.crop(tmax=tmax)
            raw = orch.run_step1a_filter()
        finally:
            QApplication.restoreOverrideCursor()
        review_bad_channels(raw)
        return None

    def _step_ica_review(self) -> bool | None:
        """Pop MNE's interactive ICA topomap grid — the real production screen.

        Uses the fitted ICA + cleaned epochs from ``preproc_done.joblib`` (both
        already in the snapshot), so no recompute. Toggling reject/keep here is
        cosmetic — the seeded pipeline's exclusions stand.
        """
        preproc_snap = self._profile.snapshot_paths["preproc"]
        if not self._require_snapshot(preproc_snap):
            return False
        load_snapshot(self.session.offline, preproc_snap)
        pre = self.session.offline._preprocessor
        epochs = self.session.offline.epochs
        if pre is None or pre.ica is None or epochs is None:
            QMessageBox.warning(
                self,
                "Bad snapshot",
                f"{preproc_snap} has no fitted ICA / epochs; re-run the seeder.",
            )
            return False
        labels = self.session.offline.ica_component_labels()
        review_ica_components(pre.ica, epochs, labels)
        return None

    def _step_skip_preproc(self) -> bool | None:
        """Load preproc snapshot + force Preprocessing view into Complete state."""
        preproc_snap = self._profile.snapshot_paths["preproc"]
        if not self._require_snapshot(preproc_snap):
            return False
        load_snapshot(self.session.offline, preproc_snap)
        pre = self.session.offline._preprocessor
        epochs = self.session.offline.epochs
        n_epochs = len(epochs) if epochs is not None else 0
        n_excluded = len(pre.ica.exclude) if pre is not None and pre.ica is not None else 0

        pv = self._preprocessing_view
        pv._data_loaded = True
        pv._done = True
        pv._excluded_count = n_excluded
        pv._epochs_value.setText(str(n_epochs))
        pv._components_value.setText(str(n_excluded))
        pv._render_per_class(pv._per_class_counts(epochs))
        pv._pages.setCurrentIndex(1)  # complete page
        pv._update_ready_state()
        pv.step2_complete.emit()
        return None

    def _step_continue_preproc(self) -> None:
        """8: fire preprocessing_complete → advance(3) (wired in phase1_screen)."""
        self._preprocessing_view.preprocessing_complete.emit()

    def _step_skip_eval(self) -> bool | None:
        """9: load eval snapshot + populate Evaluation results."""
        eval_snap = self._profile.snapshot_paths["eval"]
        if not self._require_snapshot(eval_snap):
            return False
        snap = load_snapshot(self.session.offline, eval_snap)
        eval_results = snap.get("_eval_results")
        if eval_results is None:
            QMessageBox.warning(
                self,
                "Bad snapshot",
                f"{eval_snap} has no '_eval_results'; re-run the seeder.",
            )
            return False
        self._evaluation_view._preproc_done = True
        self._evaluation_view._on_eval_done(eval_results)
        return None

    def _step_continue_eval(self) -> None:
        """10: fire evaluation_complete(timepoints) → advance(4) (wired).

        ``_on_eval_done`` above pre-fills ``_selected_timepoints`` with each
        decoder's suggested peak, so the walkthrough emits those (the gated
        per-decoder confirm is bypassed in dev tooling).
        """
        timepoints = dict(self._evaluation_view._selected_timepoints)
        self._evaluation_view.evaluation_complete.emit(timepoints)

    def _step_skip_train(self) -> bool | None:
        """11: load train snapshot + drive ``TrainView._on_train_done``
        with a fake result dict shaped exactly like ``run_training``
        returns.
        """
        train_snap = self._profile.snapshot_paths["train"]
        if not self._require_snapshot(train_snap):
            return False
        load_snapshot(self.session.offline, train_snap)
        ui_state = self.session.offline._ui_state or {}
        spec = self.session.offline._live_artifact_spec
        # In a real run the file is written before _on_train_done; the
        # walkthrough fakes the same string so the path field has
        # something to show.
        fake_path = self._profile.pipeline_path
        result = {
            "model_filepath": fake_path,
            "spatial_patterns": ui_state.get("spatial_patterns", {}),
            "mne_info": ui_state.get("mne_info"),
        }
        self._workspace.setCurrentIndex(4)
        self._header_title.setText(_DEBUG_PREFIX + _NODE_TITLES[4])
        self._train_view.set_session(self.session)
        # Eval step earlier already set the per-decoder timepoints; the
        # snapshot also carries them via the spec's metadata. Set them
        # explicitly so TrainView's ready-gating doesn't block.
        if spec is not None:
            self._train_view.set_timepoints(dict(spec.metadata.decoding_timepoints))
        self._train_view._on_train_done(result)  # noqa: SLF001 — dev tooling
        return None

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fake_picker(picker, path: str) -> None:
        """Drive a ``FilePicker`` programmatically.

        ``FilePicker`` has no public setter; the production flow is:
        ``_open_dialog`` selects → sets ``_path`` → updates the label →
        emits ``path_selected``. Mirror that here without a dialog.
        """
        picker._path = path
        picker._path_lbl.setText(path)
        picker.path_selected.emit(path)

    def _require_snapshot(self, path: Path) -> bool:
        if not path.exists():
            QMessageBox.warning(
                self,
                "Snapshot missing",
                f"{path} not found.\n\nRun:\n"
                f"  python -m scripts.demo_seed_debug_snapshots\n"
                "first to generate it (see src/frontend/debug/README.md).",
            )
            return False
        return True
