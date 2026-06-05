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
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget,
)

from backend.session import AppSession
from frontend.debug.profiles import DebugProfile, resolve_profile
from frontend.debug.snapshots import load_snapshot
from frontend.screens.phase1_screen import Phase1Screen, _NODE_TITLES
from frontend.styles.theme import (
    BORDER_GRAY, PRIMARY_BLUE, TEXT_MUTED, TEXT_PRIMARY,
)

logger = logging.getLogger(__name__)

_DEBUG_PREFIX = "[DEBUG] "


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
            _Step("Skip preprocessing",       self._step_skip_preproc),
            _Step("Continue → Evaluation",    self._step_continue_preproc),
            _Step("Skip evaluation",          self._step_skip_eval),
            _Step("Continue → Train",         self._step_continue_eval),
            _Step("Skip training",            self._step_skip_train),
        ]

        # Always-visible mode indicator, even for the initial node.
        self._header_title.setText(_DEBUG_PREFIX + _NODE_TITLES[0])

        # Toolbar sits between the workspace header bar and the
        # QStackedWidget (relies on phase1_screen.py's card layout
        # being [header_bar, workspace], a dev-mode-acceptable coupling).
        toolbar = self._build_debug_toolbar()
        card = self._workspace.parentWidget()
        card.layout().insertWidget(1, toolbar)

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
        bar = QFrame()
        bar.setObjectName("debug_toolbar")
        bar.setStyleSheet(
            f"QFrame#debug_toolbar {{ background: #F5F3FF; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
        )
        bar.setFixedHeight(40)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(8)

        self._step_lbl = QLabel("")
        self._step_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        self._step_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_lbl, 1)

        self._next_btn = QPushButton("Next →")
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn.setStyleSheet(
            f"QPushButton {{ background: {PRIMARY_BLUE}; color: white; "
            f"border: none; border-radius: 4px; padding: 4px 12px; "
            f"font-weight: 600; }}"
            f"QPushButton:disabled {{ background: #C7D2FE; }}"
        )
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._next_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self._reset_btn)

        return bar

    def _install_shortcuts(self) -> None:
        sc = QShortcut(QKeySequence("Ctrl+Right"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_next)

    def _update_toolbar(self) -> None:
        total = len(self._steps)
        if self._step_idx == 0:
            label = f"Step 0/{total}: (press Next to start)"
        elif self._step_idx <= total:
            done_step = self._steps[self._step_idx - 1]
            label = f"Step {self._step_idx}/{total}: {done_step.name} ✓"
        self._step_lbl.setText(label)
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

    def _step_skip_preproc(self) -> bool | None:
        """7: load preproc snapshot + force Preprocessing view into Complete state."""
        preproc_snap = self._profile.snapshot_paths["preproc"]
        if not self._require_snapshot(preproc_snap):
            return False
        load_snapshot(self.session.offline, preproc_snap)
        pre = self.session.offline._preprocessor
        epochs = self.session.offline._epochs
        n_epochs = len(epochs) if epochs is not None else 0
        n_excluded = len(pre.ica.exclude) if pre is not None and pre.ica is not None else 0

        pv = self._preprocessing_view
        pv._data_loaded = True
        pv._done = True
        pv._excluded_count = n_excluded
        pv._epochs_value.setText(str(n_epochs))
        pv._components_value.setText(str(n_excluded))
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
        """10: fire evaluation_complete(timepoint) → advance(4) (wired)."""
        t = self._evaluation_view._selected_timepoint
        if t is None:
            t = 0.0  # defensive; _on_eval_done above sets this
        self._evaluation_view.evaluation_complete.emit(t)

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
        # Eval step earlier already locked the timepoint; the snapshot
        # also carries it via the spec's metadata. Set it explicitly so
        # set_timepoint's validation doesn't block.
        if spec is not None:
            self._train_view.set_timepoint(float(spec.metadata.decoding_timepoint))
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
