from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QStackedWidget,
    QTabWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, PRIMARY_BLUE, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.workers.evaluation_worker import EvaluationWorker

logger = logging.getLogger(__name__)


class EvaluationView(QWidget):
    """Node 4 workspace: 2-page stack (Ready → Results).

    Step 1 scaffolding: Page 0 triggers ``orchestrator.run_evaluation()``
    off-thread; Page 1 currently shows a placeholder summary
    ("N tasks evaluated, suggested timepoint = X ms"). Subsequent plan
    steps replace the placeholder with the QTabWidget + chart widgets.

    ``_selected_timepoint`` is initialised to the evaluator's
    ``suggested_timepoint`` so the journey-panel "Approve && Continue"
    button is immediately clickable — later steps let the operator
    override it via the AUC chart.
    """

    # Loading-overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 4 action button
    ready_changed = Signal(bool)
    # Emitted once results render and the trigger should rebind to confirm.
    results_displayed = Signal()
    # Emitted when the operator confirms the chosen timepoint; payload is
    # the timepoint (seconds) Phase 1 Training will use.
    evaluation_complete = Signal(float)
    # Emitted when the operator picks a timepoint on a chart (later steps).
    timepoint_selected = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._preproc_done: bool = False
        self._running: bool = False
        self._done: bool = False
        self._was_ready: bool = False
        self._result: Optional[dict[str, Any]] = None
        self._selected_timepoint: Optional[float] = None
        self._thread: QThread | None = None
        self._worker = None
        # Populated when results land. Tabs are built once per eval run.
        self._tabs: Optional[QTabWidget] = None
        self._per_decoder_tabs: dict[str, QWidget] = {}
        self._suggested_lbl: Optional[QLabel] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_ready_page())
        self._pages.addWidget(self._build_results_page())
        outer.addWidget(self._pages)

    # ── public ────────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        """Provide the AppSession built by Node 1. Called by Phase1Screen."""
        self._session = session
        self._update_ready_state()

    def on_preprocessing_complete(self) -> None:
        """Slot connected by Phase1Screen to ``preprocessing_complete``.

        Enables Page 0's trigger button — evaluation requires the cleaned
        epochs that Step 2 of preprocessing produces.
        """
        self._preproc_done = True
        self._update_ready_state()

    def trigger_run(self) -> None:
        """Start the evaluation. Wired to the journey-panel Node 4 button."""
        if (
            self._session is None
            or self._session.offline is None
            or not self._preproc_done
            or self._running
            or self._done
        ):
            return
        self._running = True
        self._update_ready_state()
        self._start_worker(
            EvaluationWorker(self._session.offline),
            "Running evaluation…",
            self._on_eval_done,
        )

    def trigger_confirm(self) -> None:
        """Confirm the operator's selected timepoint, advance to Node 5.

        Wired by Phase1Screen as the Node 4 panel action once results
        render. Guarded by ``_update_ready_state`` so the panel button
        won't fire until ``_selected_timepoint is not None``.
        """
        if not self._done or self._selected_timepoint is None:
            return
        logger.info(
            "Evaluation confirmed; operator selected timepoint = %.3f s "
            "(suggested = %.3f s)",
            self._selected_timepoint,
            self._result.get("suggested_timepoint") if self._result else float("nan"),
        )
        self.evaluation_complete.emit(self._selected_timepoint)

    # ── worker plumbing ──────────────────────────────────────────────────────

    def _start_worker(self, worker, message: str, on_done) -> None:
        self.loading_requested.emit(message)
        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.result_ready.connect(on_done)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _on_error(self, message: str) -> None:
        self._running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Evaluation Error", message)
        self._update_ready_state()

    # ── eval result hand-off ──────────────────────────────────────────────────

    def _on_eval_done(self, result: dict) -> None:
        self.loading_done.emit()
        self._result = result
        self._selected_timepoint = float(result["suggested_timepoint"])
        self._populate_results(result)
        self._pages.setCurrentIndex(1)
        self._running = False
        self._done = True
        self._update_ready_state()
        self.results_displayed.emit()

    def _populate_results(self, result: dict) -> None:
        """Update the suggested-time header and (re)build per-decoder tabs.

        Step 2: tabs hold dashed-border placeholders in the chart/stats/
        table slots. Steps 3–6 replace each placeholder with its widget.
        """
        suggested_ms = float(result["suggested_timepoint"]) * 1000.0
        avg_peak = float(result.get("average_peak_auc", float("nan")))
        n_tasks = len(result.get("tasks", {}))
        if self._suggested_lbl is not None:
            self._suggested_lbl.setText(
                f"{n_tasks} decoder(s) evaluated · suggested timepoint = "
                f"{suggested_ms:.0f} ms · average peak AUC = {avg_peak:.3f}"
            )

        # Drop any per-decoder tabs left over from a previous run; the
        # Summary tab (index 0) stays put.
        assert self._tabs is not None
        for name, tab in list(self._per_decoder_tabs.items()):
            idx = self._tabs.indexOf(tab)
            if idx >= 0:
                self._tabs.removeTab(idx)
            tab.deleteLater()
        self._per_decoder_tabs.clear()

        for task_name in result.get("tasks", {}):
            tab = self._build_decoder_tab(task_name)
            self._per_decoder_tabs[task_name] = tab
            self._tabs.addTab(tab, task_name)

    # ── page builders ────────────────────────────────────────────────────────

    def _build_ready_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._start_btn = QPushButton("▶")
        self._start_btn.setFixedSize(96, 96)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(
            # padding-left compensates for the ▶ glyph's intrinsic
            # left-bias inside its character cell — without it the
            # triangle looks shifted left of centre, especially against
            # the more-saturated hover/press background.
            f"QPushButton {{ background: #EFF6FF; color: {PRIMARY_BLUE}; "
            f"border: none; border-radius: 48px; font-size: 36px; "
            f"padding: 0 0 0 6px; }}"
            f"QPushButton:hover {{ background: #DBEAFE; }}"
            f"QPushButton:pressed {{ background: #BFDBFE; }}"
        )
        self._start_btn.clicked.connect(self.trigger_run)
        center.addWidget(self._start_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        title = QLabel("Ready to Evaluate")
        f = title.font()
        f.setPointSize(16)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel(
            "Click play to run temporal-generalization cross-validation "
            "on the cleaned epochs. Each configured decoder is evaluated "
            "across all timepoints — the resulting AUC curves let you "
            "pick the timepoint that will train the production model."
        )
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(420)
        desc.setMinimumHeight(48)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _build_results_page(self) -> QWidget:
        """Step 2: QTabWidget skeleton with Summary tab in place.

        The per-decoder tabs are added dynamically in ``_populate_results``
        once we know the task names from the eval result. Each placeholder
        is a dashed-border QLabel sitting in the slot its real widget
        (AUCChart / TGMChart / stats panel / decoder table) will occupy
        in subsequent plan steps.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(10)

        # One-line header above the tabs — populated in _populate_results.
        self._suggested_lbl = QLabel("—")
        self._suggested_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px;"
        )
        layout.addWidget(self._suggested_lbl)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._build_summary_tab(), "Summary")
        layout.addWidget(self._tabs, 1)
        return page

    def _build_summary_tab(self) -> QWidget:
        """Summary tab skeleton: chart slot + stats slot on top, table below."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(
            self._make_placeholder(
                "AUC chart goes here\n(one coloured line per decoder; "
                "click a timepoint to select)",
                min_h=320,
            ),
            1,
        )
        stats = self._make_placeholder(
            "Stats panel\n· selected time\n· per-decoder AUC\n"
            "· average AUC\n· Reset to suggested",
            min_h=320,
        )
        stats.setFixedWidth(220)
        top.addWidget(stats, 0)
        layout.addLayout(top)

        layout.addWidget(
            self._make_placeholder(
                "Decoder summary table goes here\n"
                "(Decoder · Positive · Negative · Peak AUC · Peak (ms) "
                "— click a row to jump to that decoder's tab)",
                min_h=160,
            )
        )
        return tab

    def _build_decoder_tab(self, task_name: str) -> QWidget:
        """Per-decoder tab skeleton: header chip row + AUC + TGM slots."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel(f"Decoder: {task_name}")
        f = header.font()
        f.setPointSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        header.setFont(f)
        header.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(header)

        charts = QHBoxLayout()
        charts.setSpacing(12)
        charts.addWidget(
            self._make_placeholder(
                "AUC chart goes here\n(single decoder)",
                min_h=320,
            ),
            1,
        )
        charts.addWidget(
            self._make_placeholder(
                "TGM heatmap goes here\n(Train time × Test time, "
                "crosshair at selected timepoint)",
                min_h=320,
            ),
            1,
        )
        layout.addLayout(charts)
        return tab

    @staticmethod
    def _make_placeholder(text: str, min_h: int = 200) -> QLabel:
        """Dashed-border slot label. Replaced by real widgets in Steps 3–6."""
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setMinimumHeight(min_h)
        lbl.setStyleSheet(
            f"QLabel {{ border: 1px dashed {BORDER_GRAY}; "
            f"background: #FAFAFA; color: {TEXT_MUTED}; "
            f"padding: 18px; font-size: 12px; }}"
        )
        return lbl

    # ── ready gating ─────────────────────────────────────────────────────────

    def _update_ready_state(self) -> None:
        # Page 0 → trigger_run (ready when session set, preprocessing done,
        # nothing running)
        # Page 1 → trigger_confirm (ready once results are displayed and a
        # timepoint is selected — defaults to suggested, so immediate)
        page0_ready = (
            self._session is not None
            and self._preproc_done
            and not self._running
            and not self._done
        )
        page1_ready = self._done and self._selected_timepoint is not None
        ready = page0_ready or page1_ready
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
