from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    AMBER, BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, PRIMARY_BLUE_HOVER,
    SUCCESS_GREEN, TEXT_MUTED, TEXT_PRIMARY, chart_line_color,
)
from frontend.widgets.charts import AUCChart
from frontend.workers.evaluation_worker import EvaluationWorker

_DEVIATION_WARN_MS = 50.0  # |selected − suggested| above this → amber hint shown

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
        # Locally-confirmed timepoint. None = no commitment yet; equal to
        # ``_selected_timepoint`` = the operator has pressed "Confirm
        # Timepoint" for the currently selected time. Gates the
        # journey-panel "Approve && Continue" button.
        self._confirmed_timepoint: Optional[float] = None
        self._thread: QThread | None = None
        self._worker = None
        # Populated when results land. Tabs are built once per eval run.
        self._tabs: Optional[QTabWidget] = None
        self._per_decoder_tabs: dict[str, QWidget] = {}
        # Summary-tab live widgets (built up-front, populated on result arrival).
        self._stats_selected_lbl: Optional[QLabel] = None
        self._stats_dev_lbl: Optional[QLabel] = None
        self._stats_decoder_rows_layout: Optional[QVBoxLayout] = None
        self._stats_avg_row: Optional[QFrame] = None
        self._stats_avg_lbl: Optional[QLabel] = None
        # Per-decoder AUC value labels — kept around so we can rewrite
        # them when the operator picks a different timepoint (the AUC
        # column always reads "AUC @ selected timepoint", not peak).
        self._stats_auc_lbls: dict[str, QLabel] = {}
        self._table_auc_lbls: dict[str, QLabel] = {}
        # AUC chart instances: one on the Summary tab plotting all
        # decoders, plus one inside each per-decoder tab plotting just
        # that decoder's curve.
        self._summary_auc_chart: Optional[AUCChart] = None
        self._decoder_auc_charts: dict[str, AUCChart] = {}
        self._stats_suggested_hint_lbl: Optional[QLabel] = None
        self._stats_confirm_btn: Optional[QPushButton] = None
        self._stats_reset_btn: Optional[QPushButton] = None
        self._summary_table_body: Optional[QVBoxLayout] = None

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
        """Advance to Node 5 once the operator has locally confirmed a
        timepoint via the in-panel button.

        Wired by Phase1Screen as the Node 4 panel "Approve && Continue"
        action. Guarded by ``_update_ready_state`` so the panel button
        won't fire until ``_confirmed_timepoint is not None``.
        """
        if not self._done or self._confirmed_timepoint is None:
            return
        logger.info(
            "Evaluation confirmed; operator selected timepoint = %.3f s "
            "(suggested = %.3f s)",
            self._confirmed_timepoint,
            self._result.get("suggested_timepoint") if self._result else float("nan"),
        )
        self.evaluation_complete.emit(self._confirmed_timepoint)

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
        self._running = False
        self._done = True  # set before _populate_results so the in-panel
                           # Confirm button gating sees the right state.
        self._selected_timepoint = float(result["suggested_timepoint"])
        self._populate_results(result)
        self._pages.setCurrentIndex(1)
        self._update_ready_state()
        self.results_displayed.emit()

    def _populate_results(self, result: dict) -> None:
        """Fill in everything the eval result determines: per-decoder tabs
        (with coloured dot icons), stats panel rows, and decoder summary
        table rows.

        The AUC chart / TGM heatmap slots stay as dashed placeholders —
        the chart widgets land in subsequent plan steps.
        """
        suggested_s = float(result["suggested_timepoint"])
        suggested_ms = suggested_s * 1000.0
        tasks_result: dict[str, dict] = result.get("tasks", {})
        task_names = list(tasks_result.keys())

        # Drop any per-decoder tabs left over from a previous run; the
        # Summary tab (index 0) stays put. Also drop the chart refs since
        # the chart widgets are destroyed along with the tabs.
        assert self._tabs is not None
        for name, tab in list(self._per_decoder_tabs.items()):
            idx = self._tabs.indexOf(tab)
            if idx >= 0:
                self._tabs.removeTab(idx)
            tab.deleteLater()
        self._per_decoder_tabs.clear()
        self._decoder_auc_charts.clear()

        for task_name in task_names:
            tab = self._build_decoder_tab(task_name)  # also fills _decoder_auc_charts[task_name]
            self._per_decoder_tabs[task_name] = tab
            self._tabs.addTab(tab, task_name)

        # Clearing _confirmed_timepoint first ensures the Confirm button
        # comes back as primary-blue "Confirm Timepoint" on a fresh result.
        self._confirmed_timepoint = None
        # Build the per-decoder AUC label widgets + chart curves in
        # BOTH places before calling _set_selected_timepoint — that
        # method writes into the labels and moves the chart markers.
        self._rebuild_stats_decoder_rows(tasks_result)
        self._rebuild_summary_table(tasks_result, suggested_s)
        self._populate_auc_charts(result.get("times"), tasks_result)
        if self._stats_suggested_hint_lbl is not None:
            self._stats_suggested_hint_lbl.setText(
                f"Suggested: {suggested_ms:.0f} ms (avg peak)"
            )
        self._set_selected_timepoint(suggested_s)

    def _populate_auc_charts(self, times, tasks_result: dict[str, dict]) -> None:
        """Push the diagonal AUC curves into the Summary + per-decoder charts.

        Builds a stable ``name → colour`` map up front and threads it
        through every chart, so a decoder's line keeps the same hue in
        the Summary view and in its own per-decoder tab. Also seeds the
        stationary dashed suggested-timepoint marker on every chart.
        """
        if times is None or not tasks_result:
            return
        curves = {name: t["diagonal_auc"] for name, t in tasks_result.items()}
        colors = {
            name: chart_line_color(i) for i, name in enumerate(curves.keys())
        }
        suggested_t = (
            float(self._result["suggested_timepoint"]) if self._result else 0.0
        )
        if self._summary_auc_chart is not None:
            self._summary_auc_chart.set_curves(times, curves, colors=colors)
            self._summary_auc_chart.set_suggested_timepoint(suggested_t)
        for name, diag in curves.items():
            chart = self._decoder_auc_charts.get(name)
            if chart is not None:
                chart.set_curves(times, {name: diag}, colors={name: colors[name]})
                chart.set_suggested_timepoint(suggested_t)

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

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._build_summary_tab(), "SUMMARY")
        layout.addWidget(self._tabs, 1)
        return page

    def _build_summary_tab(self) -> QWidget:
        """Summary tab: AUC chart + stats panel on top, decoder table below.

        Chart slot is still a dashed placeholder (the AUCChart widget lands
        in a follow-up step). Stats panel and table are real widgets that
        ``_populate_results`` fills in once the eval result arrives.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self._build_auc_chart_slot(), 1)
        top.addWidget(self._build_stats_panel(), 0)
        layout.addLayout(top)

        layout.addWidget(self._build_summary_table())
        layout.addStretch()
        return tab

    def _build_auc_chart_slot(self) -> QWidget:
        """Bordered card holding the multi-decoder AUC chart for Summary."""
        card = QFrame()
        card.setObjectName("auc_card")
        card.setStyleSheet(
            f"QFrame#auc_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
        )
        body = QVBoxLayout(card)
        body.setContentsMargins(10, 10, 10, 10)
        body.setSpacing(6)

        caption = QLabel("AUC OVER TIME — ALL DECODERS")
        caption.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        body.addWidget(caption)

        self._summary_auc_chart = AUCChart(show_legend=True)
        self._summary_auc_chart.setMinimumHeight(300)
        # Click on the Summary chart → drive the same _set_selected_timepoint
        # path used by the stats-panel/table updates so every view stays
        # in sync.
        self._summary_auc_chart.timepoint_clicked.connect(self._set_selected_timepoint)
        body.addWidget(self._summary_auc_chart, 1)
        return card

    def _build_stats_panel(self) -> QWidget:
        """Right-side stats card.

        Order (top → bottom):
          • Suggested timepoint hint.
          • AUC PER DECODER mini-table (header row + body rows + Avg).
          • SELECTED TIMEPOINT value + amber deviation hint (hidden ≤ ±50 ms).
          • Confirm Timepoint button (toggles blue ↔ green) + Reset.

        All child labels carry ``background: transparent`` so when the
        card or row hovers another colour the labels don't punch
        white rectangles through.
        """
        card = QFrame()
        card.setObjectName("stats_card")
        card.setStyleSheet(
            f"QFrame#stats_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
            "QFrame#stats_card QLabel { background: transparent; }"
        )
        card.setFixedWidth(240)
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        body = QVBoxLayout(card)
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(8)

        # ─── Suggested hint ────────────────────────────────────────────────
        self._stats_suggested_hint_lbl = QLabel("Suggested: — ms (avg peak)")
        self._stats_suggested_hint_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;"
        )
        body.addWidget(self._stats_suggested_hint_lbl)

        # ─── AUC PER DECODER mini-table ────────────────────────────────────
        auc_cap = QLabel("AUC PER DECODER")
        auc_cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        body.addWidget(auc_cap)
        body.addWidget(self._build_auc_minitable_header())

        self._stats_decoder_rows_layout = QVBoxLayout()
        self._stats_decoder_rows_layout.setSpacing(2)
        self._stats_decoder_rows_layout.setContentsMargins(0, 0, 0, 0)
        body.addLayout(self._stats_decoder_rows_layout)

        # Avg row sits inside the mini-table visually, separated by a
        # thin top border + bolder type. Built as a two-cell row so the
        # value column lines up with the per-decoder AUCs above.
        self._stats_avg_row = QFrame()
        self._stats_avg_row.setObjectName("auc_minitable_avg")
        self._stats_avg_row.setStyleSheet(
            "QFrame#auc_minitable_avg { background: transparent; "
            f"border-top: 1px solid {BORDER_GRAY}; }}"
            "QFrame#auc_minitable_avg QLabel { background: transparent; }"
        )
        avg_h = QHBoxLayout(self._stats_avg_row)
        avg_h.setContentsMargins(0, 4, 0, 2)
        avg_h.setSpacing(6)
        avg_left = QLabel("Avg")
        avg_left.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 700;"
        )
        avg_h.addWidget(avg_left, 1)
        self._stats_avg_lbl = QLabel("—")
        self._stats_avg_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 700;"
        )
        self._stats_avg_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        avg_h.addWidget(self._stats_avg_lbl)
        self._stats_avg_row.hide()
        body.addWidget(self._stats_avg_row)

        body.addStretch()

        # ─── SELECTED TIMEPOINT ───────────────────────────────────────────
        sel_cap = QLabel("SELECTED TIMEPOINT")
        sel_cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        body.addWidget(sel_cap)

        self._stats_selected_lbl = QLabel("—")
        f = self._stats_selected_lbl.font()
        f.setPointSize(16)
        f.setWeight(QFont.Weight.Bold)
        self._stats_selected_lbl.setFont(f)
        self._stats_selected_lbl.setStyleSheet(f"color: {PRIMARY_BLUE};")
        body.addWidget(self._stats_selected_lbl)

        self._stats_dev_lbl = QLabel("")
        self._stats_dev_lbl.setStyleSheet(f"color: {AMBER}; font-size: 10px;")
        self._stats_dev_lbl.hide()
        body.addWidget(self._stats_dev_lbl)

        # ─── Buttons ─────────────────────────────────────────────────────
        self._stats_confirm_btn = QPushButton("Confirm Timepoint")
        self._stats_confirm_btn.setEnabled(False)
        self._stats_confirm_btn.clicked.connect(self._toggle_confirm)
        body.addWidget(self._stats_confirm_btn)
        # Style applied here — and re-applied (with the same inline rule)
        # by _refresh_confirm_button on every confirm/unconfirm toggle.
        self._refresh_confirm_button()

        self._stats_reset_btn = QPushButton("Reset to suggested")
        self._stats_reset_btn.setProperty("class", "secondary")
        self._stats_reset_btn.clicked.connect(self._reset_to_suggested)
        # Always visible — gating is via setEnabled so the row keeps a
        # stable height and the panel doesn't shift when selected ↔
        # suggested.
        self._stats_reset_btn.setEnabled(False)
        body.addWidget(self._stats_reset_btn)

        return card

    def _build_auc_minitable_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("auc_minitable_header")
        header.setStyleSheet(
            "QFrame#auc_minitable_header { background: transparent; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
            "QFrame#auc_minitable_header QLabel { background: transparent; }"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(6)
        for text, stretch, align in (
            ("Decoder", 1, Qt.AlignmentFlag.AlignLeft),
            ("AUC", 0, Qt.AlignmentFlag.AlignRight),
        ):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 9px; font-weight: 700; "
                "letter-spacing: 0.4px;"
            )
            lbl.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            h.addWidget(lbl, stretch)
        return header

    def _build_summary_table(self) -> QWidget:
        """Decoder summary table. Header row up-front; body rows are added
        in ``_populate_results``. Rows are clickable → jump to that
        decoder's tab.
        """
        card = QFrame()
        card.setObjectName("table_card")
        card.setStyleSheet(
            f"QFrame#table_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
        )
        outer = QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_table_header())

        body_holder = QWidget()
        self._summary_table_body = QVBoxLayout(body_holder)
        self._summary_table_body.setContentsMargins(0, 0, 0, 0)
        self._summary_table_body.setSpacing(0)
        outer.addWidget(body_holder)
        return card

    def _build_table_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("table_header")
        # Child-QLabel rule keeps the column titles transparent so the
        # grey header background shows through cleanly. Same trick used
        # on the body rows so the hover-blue shines through there.
        header.setStyleSheet(
            "QFrame#table_header { background: #F9FAFB; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
            "QFrame#table_header QLabel { background: transparent; }"
        )
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)
        for text, stretch, align in (
            ("Decoder",  2, Qt.AlignmentFlag.AlignLeft),
            ("Positive", 2, Qt.AlignmentFlag.AlignLeft),
            ("Negative", 2, Qt.AlignmentFlag.AlignLeft),
            ("AUC",      1, Qt.AlignmentFlag.AlignRight),
            ("Peak (ms)", 1, Qt.AlignmentFlag.AlignRight),
        ):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700;"
            )
            lbl.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lbl, stretch)
        return header

    def _build_decoder_tab(self, task_name: str) -> QWidget:
        """Per-decoder tab skeleton: name header + AUC + TGM slots."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel(task_name)
        f = header.font()
        f.setPointSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        header.setFont(f)
        header.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(header)

        charts = QHBoxLayout()
        charts.setSpacing(12)
        chart = AUCChart(show_legend=False)
        chart.setMinimumHeight(320)
        chart.timepoint_clicked.connect(self._set_selected_timepoint)
        self._decoder_auc_charts[task_name] = chart
        charts.addWidget(chart, 1)
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

    # ── populate helpers (called by _populate_results) ───────────────────────

    def _rebuild_stats_decoder_rows(self, tasks_result: dict[str, dict]) -> None:
        layout = self._stats_decoder_rows_layout
        assert layout is not None
        # Wipe any existing rows from a previous run.
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._stats_auc_lbls.clear()

        for name in tasks_result:
            wrapper = QFrame()
            wrapper.setObjectName("auc_minitable_row")
            wrapper.setStyleSheet(
                "QFrame#auc_minitable_row { background: transparent; }"
                "QFrame#auc_minitable_row QLabel { background: transparent; }"
            )
            row = QHBoxLayout(wrapper)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(6)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px;")
            name_lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            row.addWidget(name_lbl, 1)
            # AUC value is rewritten on every selected-timepoint change
            # by ``_update_per_decoder_aucs`` — it always reads AUC@t,
            # not the per-decoder peak (which is shown in the table's
            # Peak (ms) column instead).
            auc_lbl = QLabel("—")
            auc_lbl.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 600;"
            )
            auc_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            row.addWidget(auc_lbl)
            self._stats_auc_lbls[name] = auc_lbl
            layout.addWidget(wrapper)

        if (
            self._stats_avg_row is not None
            and tasks_result
        ):
            self._stats_avg_row.show()

    def _rebuild_summary_table(
        self, tasks_result: dict[str, dict], _suggested_s: float
    ) -> None:
        body = self._summary_table_body
        assert body is not None
        while body.count():
            item = body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._table_auc_lbls.clear()

        task_cfg_by_name = self._decoder_task_config_by_name()
        times = self._result.get("times") if self._result else None

        for name, task in tasks_result.items():
            cfg = task_cfg_by_name.get(name, {})
            diagonal = task.get("diagonal_auc")
            peak_ms = "—"
            if diagonal is not None and times is not None and len(diagonal) > 0:
                import numpy as np
                peak_ms = f"{float(times[int(np.argmax(diagonal))] * 1000.0):.0f}"

            row_widget, auc_lbl = self._build_table_row(
                name=name,
                positives=list(cfg.get("pos_labels", [])),
                negatives=list(cfg.get("neg_labels", [])),
                peak_ms=peak_ms,
                on_click=lambda _n=name: self._jump_to_decoder_tab(_n),
            )
            self._table_auc_lbls[name] = auc_lbl
            body.addWidget(row_widget)

    def _build_table_row(
        self,
        *,
        name: str,
        positives: list[str],
        negatives: list[str],
        peak_ms: str,
        on_click,
    ) -> tuple[QWidget, QLabel]:
        """Return the row widget AND the AUC value label.

        Caller stashes the AUC label so ``_update_per_decoder_aucs``
        can rewrite it whenever the selected timepoint changes.
        """
        row = QFrame()
        row.setObjectName("table_row")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        # Child rule: every direct QLabel + the chip-strip wrapper paint
        # transparent so the row's own background (white default → blue
        # on :hover) shows through. Without this the operator sees a
        # white rectangle behind each cell on hover.
        row.setStyleSheet(
            f"QFrame#table_row {{ border-bottom: 1px solid {BORDER_GRAY}; }}"
            "QFrame#table_row:hover { background: #EFF6FF; }"
            "QFrame#table_row QLabel { background: transparent; }"
            "QFrame#table_row QWidget#chip_strip { background: transparent; }"
        )

        def _mouse_press(_ev, _cb=on_click):
            _cb()

        row.mousePressEvent = _mouse_press  # type: ignore[assignment]

        h = QHBoxLayout(row)
        h.setContentsMargins(12, 10, 12, 10)
        h.setSpacing(8)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 500;"
        )
        h.addWidget(name_lbl, 2)

        h.addWidget(self._chip_strip(positives, kind="pos"), 2)
        h.addWidget(self._chip_strip(negatives, kind="neg"), 2)

        # AUC text + colour rewritten in _update_per_decoder_aucs.
        auc_lbl = QLabel("—")
        auc_lbl.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            f"font-weight: 600; color: {TEXT_PRIMARY};"
        )
        auc_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        h.addWidget(auc_lbl, 1)

        peak_lbl = QLabel(peak_ms)
        peak_lbl.setStyleSheet(
            f"font-family: monospace; font-size: 11px; color: {TEXT_MUTED};"
        )
        peak_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        h.addWidget(peak_lbl, 1)
        return row, auc_lbl

    def _chip_strip(self, labels: list[str], kind: str) -> QWidget:
        """Coloured chips for the Positive / Negative table cells.

        ``kind="pos"`` → blue chips, ``kind="neg"`` → red chips. Empty list
        renders a muted em-dash so the cell isn't blank.
        """
        wrap = QWidget()
        wrap.setObjectName("chip_strip")
        h = QHBoxLayout(wrap)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        if not labels:
            dash = QLabel("—")
            dash.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
            h.addWidget(dash)
            h.addStretch()
            return wrap

        if kind == "pos":
            bg, fg = "#DBEAFE", "#1D4ED8"
        else:
            bg, fg = "#FEE2E2", "#B91C1C"
        for lbl in labels:
            chip = QLabel(lbl)
            chip.setStyleSheet(
                f"background: {bg}; color: {fg}; "
                "padding: 1px 6px; border-radius: 2px; "
                "font-size: 10px;"
            )
            h.addWidget(chip)
        h.addStretch()
        return wrap

    def _set_selected_timepoint(self, t_seconds: float) -> None:
        """Update the Selected display + deviation hint + button states.

        Called once when results arrive (with the suggested timepoint),
        and by the AUC charts on operator click.

        Picking a new timepoint **unconfirms** any previous commitment —
        the operator has to re-press Confirm to re-lock.
        """
        self._selected_timepoint = t_seconds
        if (
            self._confirmed_timepoint is not None
            and abs(self._confirmed_timepoint - t_seconds) > 1e-9
        ):
            self._confirmed_timepoint = None

        suggested = (
            float(self._result["suggested_timepoint"]) if self._result else t_seconds
        )
        sel_ms = t_seconds * 1000.0
        dev_ms = abs(sel_ms - suggested * 1000.0)

        if self._stats_selected_lbl is not None:
            self._stats_selected_lbl.setText(f"{sel_ms:.0f} ms")
            warn = dev_ms > _DEVIATION_WARN_MS
            self._stats_selected_lbl.setStyleSheet(
                f"color: {AMBER if warn else PRIMARY_BLUE}; "
                "background: transparent;"
            )

        if self._stats_dev_lbl is not None:
            if dev_ms > _DEVIATION_WARN_MS:
                self._stats_dev_lbl.setText(f"±{dev_ms:.0f} ms from suggested")
                self._stats_dev_lbl.show()
            else:
                self._stats_dev_lbl.hide()

        if self._stats_reset_btn is not None:
            # Always rendered (avoids layout shifts) — disabled at suggested.
            self._stats_reset_btn.setEnabled(dev_ms > 1e-6)

        self._update_per_decoder_aucs(t_seconds)
        self._sync_chart_markers(t_seconds)
        self._refresh_confirm_button()
        self._update_ready_state()

    def _sync_chart_markers(self, t_seconds: float) -> None:
        """Move the vertical marker on every AUC chart to ``t_seconds``."""
        if self._summary_auc_chart is not None:
            self._summary_auc_chart.set_selected_timepoint(t_seconds)
        for chart in self._decoder_auc_charts.values():
            chart.set_selected_timepoint(t_seconds)

    def _update_per_decoder_aucs(self, t_seconds: float) -> None:
        """Rewrite the per-decoder AUC values to show AUC @ ``t_seconds``.

        The "AUC PER DECODER" mini-table and the summary table's AUC
        column always read **AUC at the selected timepoint**, never the
        per-decoder peak (which is shown separately as ``Peak (ms)`` in
        the summary table). Avg row = mean across decoders at ``t_seconds``.
        """
        if self._result is None:
            return
        times = self._result.get("times")
        tasks_result = self._result.get("tasks", {})
        if times is None or not tasks_result:
            return
        import numpy as np
        idx = int(np.argmin(np.abs(np.asarray(times) - t_seconds)))

        aucs_at_t: list[float] = []
        for name, task in tasks_result.items():
            diag = task.get("diagonal_auc")
            if diag is None or len(diag) <= idx:
                continue
            v = float(diag[idx])
            aucs_at_t.append(v)
            text = f"{v:.2f}"
            colour = "#16A34A" if v >= 0.70 else "#DC2626"
            stats_lbl = self._stats_auc_lbls.get(name)
            if stats_lbl is not None:
                stats_lbl.setText(text)
            tbl_lbl = self._table_auc_lbls.get(name)
            if tbl_lbl is not None:
                tbl_lbl.setText(text)
                tbl_lbl.setStyleSheet(
                    "font-family: monospace; font-size: 11px; "
                    f"font-weight: 600; color: {colour};"
                )

        if self._stats_avg_lbl is not None and aucs_at_t:
            self._stats_avg_lbl.setText(f"{sum(aucs_at_t) / len(aucs_at_t):.2f}")

    def _toggle_confirm(self) -> None:
        """Lock / unlock the currently selected timepoint locally.

        Doesn't advance to Node 5 by itself — that's the job of the
        journey-panel "Approve && Continue" button, which is gated by
        ``_confirmed_timepoint is not None``. Re-clicking the in-panel
        button unconfirms.
        """
        if not self._done or self._selected_timepoint is None:
            return
        if (
            self._confirmed_timepoint is not None
            and abs(self._confirmed_timepoint - self._selected_timepoint) < 1e-9
        ):
            self._confirmed_timepoint = None
        else:
            self._confirmed_timepoint = self._selected_timepoint
        self._refresh_confirm_button()
        self._update_ready_state()

    def _refresh_confirm_button(self) -> None:
        """Flip Confirm Timepoint between primary-blue and success-green.

        Uses an inline stylesheet for **both** states. The first attempt
        toggled `setProperty("class", ...)` and relied on the global
        QSS to re-apply, but Qt doesn't re-polish on its own — the button
        was left unstyled (invisible on the white card, but still
        clickable) after unconfirming.
        """
        btn = self._stats_confirm_btn
        if btn is None:
            return
        confirmed = (
            self._confirmed_timepoint is not None
            and self._selected_timepoint is not None
            and abs(self._confirmed_timepoint - self._selected_timepoint) < 1e-9
        )
        btn.setEnabled(self._done)
        if confirmed:
            btn.setText("Timepoint Confirmed ✓")
            btn.setStyleSheet(
                f"QPushButton {{ background: {SUCCESS_GREEN}; color: white; "
                "border: none; border-radius: 2px; padding: 6px 20px; "
                "font-size: 13px; font-weight: 600; }"
            )
        else:
            btn.setText("Confirm Timepoint")
            btn.setStyleSheet(
                f"QPushButton {{ background: {PRIMARY_BLUE}; color: white; "
                "border: none; border-radius: 2px; padding: 6px 20px; "
                "font-size: 13px; font-weight: 600; }"
                f"QPushButton:hover {{ background: {PRIMARY_BLUE_HOVER}; }}"
                f"QPushButton:disabled {{ background: #D1D5DB; "
                f"color: {TEXT_MUTED}; }}"
            )

    def _reset_to_suggested(self) -> None:
        if self._result is None:
            return
        self._set_selected_timepoint(float(self._result["suggested_timepoint"]))

    def _jump_to_decoder_tab(self, task_name: str) -> None:
        tab = self._per_decoder_tabs.get(task_name)
        if tab is None or self._tabs is None:
            return
        idx = self._tabs.indexOf(tab)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _decoder_task_config_by_name(self) -> dict[str, dict]:
        if self._session is None:
            return {}
        try:
            tasks = self._session.settings["decoders"]["tasks"]
        except (KeyError, TypeError, AttributeError):
            return {}
        return {t["name"]: t for t in tasks}

    # ── tiny visual helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_placeholder(text: str, min_h: int = 200) -> QLabel:
        """Dashed-border slot label. Used for the AUC / TGM chart slots
        until the chart widgets land in a follow-up step.
        """
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
        # Page 1 → trigger_confirm (ready once results are displayed AND
        # the operator has locally confirmed a timepoint via the in-panel
        # Confirm button — the journey-panel "Approve && Continue" stays
        # disabled until this).
        page0_ready = (
            self._session is not None
            and self._preproc_done
            and not self._running
            and not self._done
        )
        page1_ready = self._done and self._confirmed_timepoint is not None
        ready = page0_ready or page1_ready
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
