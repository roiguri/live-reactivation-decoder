import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.widgets.journey_panel import JourneyPanel
from frontend.widgets.loading_overlay import LoadingOverlay
from frontend.views.settings_view import SettingsView
from frontend.views.load_data_view import LoadDataView
from frontend.views.preprocessing_view import PreprocessingView
from frontend.views.evaluation_view import EvaluationView
from frontend.views.train_view import TrainView
from frontend.styles.theme import BG_LIGHT, BORDER_GRAY, CARD_WHITE, TEXT_PRIMARY

logger = logging.getLogger(__name__)

_NODE_TITLES = [
    "Pipeline Settings",
    "Data Ingestion",
    "Preprocessing",
    "Model Evaluation",
    "Train & Save",
]


class Phase1Screen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = None
        # Per-decoder timepoints chosen by the operator on Node 4
        # (``{task_name: seconds}``); Node 5's Training worker reads these
        # when it fires. None until eval is confirmed.
        self._selected_timepoints: dict[str, float] | None = None
        # Path to the decoder_pipeline.joblib emitted by Node 5; consumed
        # by the Go-Live handoff to Phase 2. None until training succeeds.
        self._decoder_pipeline_path: Path | None = None
        self.setObjectName("phase1_screen")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: gray surround + white card ──────────────────────────────────
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_LIGHT};")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(24, 24, 24, 24)
        outer_layout.setSpacing(0)

        card = QFrame()
        card.setStyleSheet(
            f"QFrame#workspace_card {{"
            f"  background: {CARD_WHITE};"
            f"  border: 1px solid {BORDER_GRAY};"
            f"  border-radius: 6px;"
            f"}}"
        )
        card.setObjectName("workspace_card")

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 30))
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Header bar
        header_bar = QWidget()
        header_bar.setFixedHeight(48)
        header_bar.setStyleSheet(
            "background: #FAFAFA; border-bottom: 1px solid #F3F4F6;"
        )
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(24, 0, 24, 0)
        self._header_title = QLabel(_NODE_TITLES[0])
        f = self._header_title.font()
        f.setPointSize(13)
        f.setWeight(QFont.Weight.Medium)
        self._header_title.setFont(f)
        self._header_title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        header_layout.addWidget(self._header_title)
        card_layout.addWidget(header_bar)

        # Workspace stack (views own their internal padding)
        self._settings_view = SettingsView()
        self._load_data_view = LoadDataView()
        self._preprocessing_view = PreprocessingView()
        self._evaluation_view = EvaluationView()

        self._workspace = QStackedWidget()
        self._workspace.setObjectName("workspace_stack")
        self._workspace.setStyleSheet(f"background: {CARD_WHITE};")
        self._train_view = TrainView()
        self._workspace.addWidget(self._settings_view)
        self._workspace.addWidget(self._load_data_view)
        self._workspace.addWidget(self._preprocessing_view)
        self._workspace.addWidget(self._evaluation_view)
        self._workspace.addWidget(self._train_view)
        card_layout.addWidget(self._workspace, 1)

        outer_layout.addWidget(card, 1)
        root.addWidget(outer, 1)

        # ── Loading overlay (covers workspace card only) ──────────────────────
        self._loading_overlay = LoadingOverlay(card)

        # ── Right: journey panel ───────────────────────────────────────────────
        self._journey_panel = JourneyPanel()
        root.addWidget(self._journey_panel)

        # ── Wiring ────────────────────────────────────────────────────────────
        # Node 1: Pipeline Settings
        self._journey_panel.set_node_action(0, self._settings_view.trigger_continue)
        self._journey_panel.set_node_ready(0, False)
        self._settings_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(0, ready)
        )
        self._settings_view.session_ready.connect(self._on_session_ready)
        self._settings_view.loading_requested.connect(self.show_loading)
        self._settings_view.loading_done.connect(self.hide_loading)

        # Node 2: Load Data
        self._journey_panel.set_node_action(1, self._load_data_view.trigger_load)
        self._journey_panel.set_node_ready(1, False)
        self._load_data_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(1, ready)
        )
        self._load_data_view.loading_requested.connect(self.show_loading)
        self._load_data_view.loading_done.connect(self.hide_loading)
        self._load_data_view.data_loaded.connect(self._on_data_loaded)

        # Node 3: Preprocessing
        self._journey_panel.set_node_action(2, self._preprocessing_view.trigger_start)
        self._journey_panel.set_node_ready(2, False)
        self._preprocessing_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(2, ready)
        )
        self._preprocessing_view.loading_requested.connect(self.show_loading)
        self._preprocessing_view.loading_done.connect(self.hide_loading)
        self._load_data_view.data_loaded.connect(self._preprocessing_view.on_data_loaded)
        self._preprocessing_view.step2_complete.connect(
            self._on_preprocessing_complete_displayed
        )
        self._preprocessing_view.preprocessing_complete.connect(
            lambda: self._journey_panel.advance(3)
        )

        # Node 4: Evaluation
        self._journey_panel.set_node_action(3, self._evaluation_view.trigger_run)
        self._journey_panel.set_node_ready(3, False)
        self._evaluation_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(3, ready)
        )
        self._evaluation_view.loading_requested.connect(self.show_loading)
        self._evaluation_view.loading_done.connect(self.hide_loading)
        # Preprocessing's preprocessing_complete fires when the operator
        # clicks "Continue to Evaluation"; that's exactly when Node 4 is
        # eligible to run, so use it as the eval view's enable gate.
        self._preprocessing_view.preprocessing_complete.connect(
            self._evaluation_view.on_preprocessing_complete
        )
        self._evaluation_view.results_displayed.connect(
            self._on_eval_results_displayed
        )
        self._evaluation_view.evaluation_complete.connect(
            self._on_evaluation_confirmed
        )

        # Node 5: Train & Save
        self._journey_panel.set_node_action(4, self._train_view.trigger_run)
        self._journey_panel.set_node_ready(4, False)
        self._train_view.ready_changed.connect(
            lambda ready: self._journey_panel.set_node_ready(4, ready)
        )
        self._train_view.loading_requested.connect(self.show_loading)
        self._train_view.loading_done.connect(self.hide_loading)
        self._train_view.results_displayed.connect(
            self._on_train_results_displayed
        )
        self._train_view.training_complete.connect(self._on_training_complete)

        self._journey_panel.node_changed.connect(self._on_node_changed)

    # ── public ────────────────────────────────────────────────────────────────

    def show_loading(self, message: str) -> None:
        self._loading_overlay.show_with_message(message)

    def hide_loading(self) -> None:
        self._loading_overlay.hide()

    def _on_session_ready(self, session) -> None:
        self.session = session
        self._load_data_view.set_session(session)
        self._preprocessing_view.set_session(session)
        self._evaluation_view.set_session(session)
        self._train_view.set_session(session)
        self._journey_panel.set_node_summary(0, self._format_settings_summary())
        self._journey_panel.advance(1)

    def _on_data_loaded(self) -> None:
        self._journey_panel.set_node_summary(1, self._format_load_summary())
        self._journey_panel.advance(2)

    def _on_preprocessing_complete_displayed(self) -> None:
        self._journey_panel.set_node_action(2, self._preprocessing_view.trigger_continue)
        self._journey_panel.set_node_action_label(2, "Continue to Evaluation")
        self._journey_panel.set_node_summary(2, self._format_preprocessing_summary())

    def _on_eval_results_displayed(self) -> None:
        # Eval results page is showing; the journey-panel Node 4 button
        # now confirms the operator's timepoint choice (defaults to the
        # ICLabel-suggested timepoint until they click on a chart in a
        # future plan step).
        self._journey_panel.set_node_action(3, self._evaluation_view.trigger_confirm)
        self._journey_panel.set_node_action_label(3, "Approve && Continue")

    def _on_evaluation_confirmed(self, timepoints: dict) -> None:
        # Stash for Node 5's training worker, then advance the trail.
        self._selected_timepoints = dict(timepoints)
        self._train_view.set_timepoints(self._selected_timepoints)
        self._journey_panel.set_node_summary(
            3, self._format_evaluation_summary(self._selected_timepoints)
        )
        self._journey_panel.advance(4)

    def _on_training_complete(self, result: dict) -> None:
        path = result.get("model_filepath")
        if path is not None:
            self._decoder_pipeline_path = Path(path)
        self._journey_panel.set_node_summary(
            4, self._format_training_summary(result)
        )

    def _on_train_results_displayed(self) -> None:
        # Topomaps are showing; relabel Node 5's button and swap its
        # action from "run training" to "go live to Phase 2".
        self._journey_panel.set_node_action(4, self._on_go_live)
        self._journey_panel.set_node_action_label(4, "Go Live")

    def _on_go_live(self) -> None:
        if self.session is None or self._decoder_pipeline_path is None:
            return
        mw = self.window()
        if mw is None or not hasattr(mw, "show_screen"):
            return
        from PyQt6.QtWidgets import QMessageBox
        from frontend.screens.phase2_screen import Phase2Screen
        try:
            phase2 = Phase2Screen(
                session=self.session,
                decoder_pipeline_path=self._decoder_pipeline_path,
            )
        except Exception as exc:
            # Artifact load / live-stream construction failed. Keep the
            # operator on Phase 1 so they can re-train or pick a different
            # artifact rather than landing on a half-built screen.
            logger.exception("Failed to open live inference (decoder pipeline load)")
            QMessageBox.critical(
                self, "Could not open live inference",
                f"Failed to load the decoder pipeline:\n{exc}",
            )
            return
        mw.show_screen(phase2)

    def _on_node_changed(self, completed_node: int) -> None:
        next_idx = completed_node  # node_changed emits 1-indexed completed node
        if next_idx < self._workspace.count():
            self._workspace.setCurrentIndex(next_idx)
            self._header_title.setText(_NODE_TITLES[next_idx])

    # ── summary formatters ────────────────────────────────────────────────────

    def _format_settings_summary(self) -> str:
        cfg_path = self._settings_view._config_path
        out_dir = self._settings_view._output_dir
        cfg_name = Path(cfg_path).name if cfg_path else "—"
        out_name = Path(out_dir).name if out_dir else "—"
        settings = self.session.settings
        model = settings["decoders"]["model"]
        n_decoders = len(settings["decoders"].get("tasks", []))
        return (
            f"Config: {cfg_name}\n"
            f"Output: {out_name}\n"
            f"Model: {model} · {n_decoders} decoder(s)"
        )

    def _format_load_summary(self) -> str:
        info = None
        if self.session is not None and self.session.offline is not None:
            info = self.session.offline.get_loaded_data_summary()
        if info is None:
            return "Data loaded"
        return (
            f"File: {info['file_name'] or '—'}\n"
            f"{info['n_channels']} ch · {info['sfreq']:.0f} Hz · "
            f"{info['duration_s']:.0f} s\n"
            f"{info['n_events']} event markers"
        )

    def _format_preprocessing_summary(self) -> str:
        view = self._preprocessing_view
        n_epochs = view._epochs_count
        n_excluded = view._excluded_count
        bads = view._bad_channels
        if bads:
            shown = ", ".join(bads[:3]) + ("…" if len(bads) > 3 else "")
            bads_line = f"Bads dropped: {len(bads)} ({shown})"
        else:
            bads_line = "Bads dropped: 0"
        return (
            f"Epochs: {n_epochs}\n"
            f"{bads_line}\n"
            f"ICs removed: {n_excluded}"
        )

    def _format_evaluation_summary(self, timepoints: dict[str, float]) -> str:
        view = self._evaluation_view
        result = view._result or {}
        tasks = result.get("tasks", {}) or {}
        times = result.get("times")

        import numpy as np
        arr = np.asarray(times) if times is not None else None

        vals: list[float] = []
        for name, task in tasks.items():
            diag = task.get("diagonal_auc")
            t = timepoints.get(name)
            if diag is None or arr is None or t is None:
                continue
            idx = int(np.argmin(np.abs(arr - t)))
            if len(diag) > idx:
                vals.append(float(diag[idx]))
        avg_str = f"{sum(vals) / len(vals):.2f}" if vals else "—"

        # Per-decoder ms, compact (e.g. "red 220 · green 180").
        tp_str = " · ".join(
            f"{name} {t * 1000.0:.0f}" for name, t in timepoints.items()
        )
        return (
            f"Decoders: {len(timepoints)}\n"
            f"Timepoints: {tp_str} ms\n"
            f"Avg AUC @t: {avg_str}"
        )

    def _format_training_summary(self, result: dict) -> str:
        patterns = result.get("spatial_patterns", {}) or {}
        n_models = len(patterns)
        timepoints = self._selected_timepoints or {}
        tp_str = (
            " · ".join(f"{name} {t * 1000.0:.0f}" for name, t in timepoints.items())
            if timepoints
            else "—"
        )
        path = result.get("model_filepath")
        path_name = Path(path).name if path else "decoder_pipeline.joblib"
        return (
            f"Models trained: {n_models}\n"
            f"Trained @ {tp_str} ms\n"
            f"Saved: {path_name}"
        )
