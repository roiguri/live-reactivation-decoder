from __future__ import annotations

import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy,
    QSpinBox, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    AMBER, BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, PRIMARY_BLUE_HOVER,
    SUCCESS_GREEN, TEXT_MUTED, TEXT_PRIMARY, chart_line_color,
)
from frontend.widgets.charts import AUCChart, TGMChart
from frontend.workers.evaluation_worker import EvaluationWorker

_DEVIATION_WARN_MS = 50.0  # |selected − suggested| above this → amber hint shown

# Summary-roster column widths (px) — header + rows share them so cells line up.
_ROSTER_SPIN_W = 78
_ROSTER_PEAK_W = 42
_ROSTER_AUC_W = 36
_ROSTER_CONFIRM_W = 84

# Tooltip explaining the cross-task suggested timepoint (not a plain average).
_SUGGESTED_TOOLTIP = (
    "Suggested timepoint = peak of the mean AUC curve.\n"
    "\n"
    "The single moment where decoding accuracy, averaged across all\n"
    "decoders, is highest - the best shared timepoint if one had to\n"
    "serve every decoder.\n"
    "\n"
    "This is NOT the mean of the per-decoder peaks. Each decoder\n"
    "below is pre-filled with its own peak instead."
)

logger = logging.getLogger(__name__)


class EvaluationView(QWidget):
    """Node 4 workspace: 2-page stack (Ready → Results).

    Page 0 triggers ``orchestrator.run_evaluation()`` off-thread; Page 1
    shows a Summary tab (per-decoder roster + overlay AUC chart + decoder
    table) plus one tab per decoder (AUC curve + TGM heatmap).

    Each decoder gets its OWN selected timepoint (``_selected_timepoints``),
    pre-filled with its evaluator ``peak_timepoint`` and confirmed
    independently from the roster. The journey-panel "Approve && Continue"
    button unlocks only once every decoder is confirmed (``_all_confirmed``).
    """

    # Loading-overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 4 action button
    ready_changed = Signal(bool)
    # Emitted once results render and the trigger should rebind to confirm.
    results_displayed = Signal()
    # Emitted when the operator confirms every decoder's timepoint; payload
    # is the per-decoder ``{task_name: seconds}`` map Phase 1 Training will use.
    evaluation_complete = Signal(dict)
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
        # Per-decoder selected timepoints (task → seconds) and per-decoder
        # confirm flags. The operator picks/locks each decoder independently;
        # the journey-panel "Approve && Continue" button unlocks only once
        # every decoder is confirmed (see ``_all_confirmed``).
        self._selected_timepoints: dict[str, float] = {}
        self._confirmed: dict[str, bool] = {}
        # Each decoder's suggested peak (task → seconds), from the evaluator's
        # per-task ``peak_timepoint``. Pre-fills the spinboxes/markers and
        # drives the per-decoder amber deviation hint.
        self._suggested_timepoints: dict[str, float] = {}
        self._thread: QThread | None = None
        self._worker = None
        # Populated when results land. Tabs are built once per eval run.
        self._tabs: Optional[QTabWidget] = None
        self._per_decoder_tabs: dict[str, QWidget] = {}
        # Summary-tab roster: one row per decoder, each with its own
        # timepoint spinbox, AUC@t label and Confirm button. Built when the
        # eval result lands (task names known then).
        self._roster_rows_layout: Optional[QVBoxLayout] = None
        self._roster_suggested_lbl: Optional[QLabel] = None
        self._roster_spins: dict[str, QSpinBox] = {}
        self._roster_auc_lbls: dict[str, QLabel] = {}
        self._roster_confirm_btns: dict[str, QPushButton] = {}
        self._roster_status_lbl: Optional[QLabel] = None
        self._roster_reset_btn: Optional[QPushButton] = None
        # Bottom summary-table AUC labels — rewritten when a decoder's
        # timepoint changes (reads AUC @ that decoder's own timepoint).
        self._table_auc_lbls: dict[str, QLabel] = {}
        # AUC chart instances: one on the Summary tab plotting all
        # decoders, plus one inside each per-decoder tab plotting just
        # that decoder's curve.
        self._summary_auc_chart: Optional[AUCChart] = None
        self._decoder_auc_charts: dict[str, AUCChart] = {}
        # Per-decoder TGM heatmaps + the stats card widgets that live
        # inside each per-decoder tab. ``_decoder_stats`` maps
        # ``task_name → {"timepoint": QSpinBox, "auc_at_t": QLabel,
        # "peak_auc": QLabel, "peak_ms": QLabel}`` so the populate /
        # update flow can rewrite the values from one place.
        self._decoder_tgm_charts: dict[str, TGMChart] = {}
        self._decoder_stats: dict[str, dict[str, QWidget]] = {}
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
        """Advance to Node 5 once the operator has confirmed every decoder's
        timepoint via the Summary-tab roster.

        Wired by Phase1Screen as the Node 4 panel "Approve && Continue"
        action. Guarded by ``_update_ready_state`` so the panel button won't
        fire until ``_all_confirmed()``.
        """
        if not self._done or not self._all_confirmed():
            return
        logger.info(
            "Evaluation confirmed; per-decoder timepoints (s) = %s",
            {k: round(v, 3) for k, v in self._selected_timepoints.items()},
        )
        self.evaluation_complete.emit(dict(self._selected_timepoints))

    def _all_confirmed(self) -> bool:
        """True once every decoder has a confirmed timepoint."""
        names = list(self._selected_timepoints.keys())
        return bool(names) and all(self._confirmed.get(n, False) for n in names)

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
        self._done = True  # set before _populate_results so the per-decoder
                           # Confirm button gating sees the right state.
        self._populate_results(result)
        self._pages.setCurrentIndex(1)
        self._update_ready_state()
        self.results_displayed.emit()

    def _populate_results(self, result: dict) -> None:
        """Fill in everything the eval result determines: per-decoder tabs,
        the Summary-tab roster (per-decoder timepoint + Confirm), and the
        decoder summary table rows.

        Each decoder is pre-filled with its own suggested peak
        (``tasks[name]['peak_timepoint']``); the operator overrides any of
        them and confirms each from the roster.
        """
        tasks_result: dict[str, dict] = result.get("tasks", {})
        task_names = list(tasks_result.keys())

        # Per-decoder suggested peaks → pre-filled, unconfirmed selections.
        self._suggested_timepoints = {
            name: float(task.get("peak_timepoint", 0.0))
            for name, task in tasks_result.items()
        }
        self._selected_timepoints = dict(self._suggested_timepoints)
        self._confirmed = {name: False for name in task_names}

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
        self._decoder_tgm_charts.clear()
        self._decoder_stats.clear()

        for task_name in task_names:
            tab = self._build_decoder_tab(task_name)  # also fills the per-decoder dicts
            self._per_decoder_tabs[task_name] = tab
            self._tabs.addTab(tab, task_name)

        # Cross-task suggestion line (peak of the mean AUC curve).
        if self._roster_suggested_lbl is not None:
            suggested_ms = float(result.get("suggested_timepoint", 0.0)) * 1000.0
            self._roster_suggested_lbl.setText(
                f"Suggested {suggested_ms:.0f} ms · peak of mean AUC"
            )

        # Build the roster + table + chart widgets BEFORE the per-decoder
        # sync loop below — that loop writes into the labels and moves the
        # chart markers.
        self._rebuild_roster(tasks_result)
        self._rebuild_summary_table(tasks_result, 0.0)
        self._populate_charts(result.get("times"), tasks_result)
        self._populate_decoder_peak_stats(result.get("times"), tasks_result)

        # Constrain every timepoint spinbox to the evaluator's time range,
        # with a step matching the sample spacing — so the spinbox arrows
        # walk one sample at a time.
        times = result.get("times")
        if times is not None and len(times) > 1:
            step_ms = int(round((times[1] - times[0]) * 1000.0))
            t_lo = int(round(times[0] * 1000.0))
            t_hi = int(round(times[-1] * 1000.0))
            for spin in self._all_timepoint_spinboxes():
                spin.setRange(t_lo, t_hi)
                spin.setSingleStep(max(1, step_ms))

        # Sync every decoder's display (spinboxes, AUC labels, markers,
        # confirm button) to its pre-filled suggested timepoint.
        for name in task_names:
            self._set_decoder_timepoint(name, self._selected_timepoints[name])
        self._refresh_roster_status()

    def _populate_charts(self, times, tasks_result: dict[str, dict]) -> None:
        """Push curves into the Summary + per-decoder AUC charts AND
        seed each per-decoder TGM heatmap with its matrix.

        Builds a stable ``name → colour`` map up front and threads it
        through every AUC chart, so a decoder's line keeps the same hue
        across views. Also seeds the stationary dashed suggested-
        timepoint marker on every chart.
        """
        if times is None or not tasks_result:
            return
        curves = {name: t["diagonal_auc"] for name, t in tasks_result.items()}
        colors = {
            name: chart_line_color(i) for i, name in enumerate(curves.keys())
        }
        # Summary overlay shows the cross-task average peak as a single
        # reference line (the overlay is for comparing curves, not picking
        # a per-decoder timepoint).
        avg_suggested = (
            float(self._result["suggested_timepoint"]) if self._result else 0.0
        )
        if self._summary_auc_chart is not None:
            self._summary_auc_chart.set_curves(times, curves, colors=colors)
            self._summary_auc_chart.set_suggested_timepoint(avg_suggested)
        # Per-decoder charts show that decoder's OWN suggested peak.
        for name, task in tasks_result.items():
            suggested_t = self._suggested_timepoints.get(name, avg_suggested)
            auc = self._decoder_auc_charts.get(name)
            if auc is not None:
                auc.set_curves(times, {name: task["diagonal_auc"]},
                               colors={name: colors[name]})
                auc.set_suggested_timepoint(suggested_t)
            tgm = self._decoder_tgm_charts.get(name)
            if tgm is not None and task.get("tgm_matrix") is not None:
                tgm.set_matrix(times, task["tgm_matrix"])
                tgm.set_suggested_timepoint(suggested_t)

    def _populate_decoder_peak_stats(
        self, times, tasks_result: dict[str, dict]
    ) -> None:
        """Fill the per-decoder stats card's ``Peak AUC`` and ``Peak time``
        rows. These are timepoint-independent (won't change as the
        operator moves the selected marker around)."""
        if times is None:
            return
        import numpy as np
        for name, task in tasks_result.items():
            stats = self._decoder_stats.get(name)
            if stats is None:
                continue
            diag = task.get("diagonal_auc")
            peak_auc = float(task.get("peak_auc", float("nan")))
            peak_ms = "—"
            if diag is not None and len(diag) > 0:
                peak_ms = f"{float(times[int(np.argmax(diag))] * 1000.0):.0f}"
            self._set_stat_value(stats["peak_auc"], f"{peak_auc:.2f}")
            self._set_stat_value(stats["peak_ms"], peak_ms)

    def _set_stat_value(self, value_lbl: QWidget, text: str) -> None:
        """Apply the stashed ``suffix`` (e.g. ``" ms"``) to a stat-row value."""
        suffix = value_lbl.property("suffix") or ""
        value_lbl.setText(f"{text}{suffix}")

    def _all_timepoint_spinboxes(self) -> list[QSpinBox]:
        """Every timepoint spinbox in the Eval results screen — the Summary
        roster rows plus the per-decoder tab cards."""
        spins: list[QSpinBox] = []
        for spin in self._roster_spins.values():
            if isinstance(spin, QSpinBox):
                spins.append(spin)
        for stats in self._decoder_stats.values():
            spin = stats.get("timepoint")
            if isinstance(spin, QSpinBox):
                spins.append(spin)
        return spins

    def _decoder_spinboxes(self, task_name: str) -> list[QSpinBox]:
        """Both spinboxes that edit ``task_name`` — its Summary roster row
        and its per-decoder tab card — kept in sync with each other."""
        spins: list[QSpinBox] = []
        roster = self._roster_spins.get(task_name)
        if isinstance(roster, QSpinBox):
            spins.append(roster)
        stats = self._decoder_stats.get(task_name)
        if stats is not None:
            tab_spin = stats.get("timepoint")
            if isinstance(tab_spin, QSpinBox):
                spins.append(tab_spin)
        return spins

    def _snap_to_sample(self, t_seconds: float) -> float:
        """Snap a time (seconds) to the nearest evaluator sample."""
        if self._result is None:
            return t_seconds
        times = self._result.get("times")
        if times is None or len(times) == 0:
            return t_seconds
        import numpy as np
        idx = int(np.argmin(np.abs(np.asarray(times) - t_seconds)))
        return float(times[idx])

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
        # Inspection-only: a click on the all-decoder overlay is ambiguous
        # (which decoder?). Per-decoder selection happens in the roster or
        # the individual decoder tabs, so this chart has no click-to-set.
        body.addWidget(self._summary_auc_chart, 1)
        return card

    def _build_stats_panel(self) -> QWidget:
        """Right-side per-decoder roster — the control center.

        One row per decoder: ``[name] [timepoint spinbox (ms)] [peak (ms)]
        [AUC @ t] [Confirm]``. Each decoder is pre-filled with its own
        suggested peak and edited/confirmed independently; the read-only
        ``peak`` column shows that decoder's own peak for comparison. A
        caption line above shows the cross-task suggested timepoint (peak of
        the mean AUC curve). Footer shows an ``N / M confirmed`` status and a
        ``Reset all to suggested`` button.

        All child labels carry ``background: transparent`` so when the card
        or a row hovers another colour the labels don't punch white
        rectangles through.
        """
        card = QFrame()
        card.setObjectName("stats_card")
        card.setStyleSheet(
            f"QFrame#stats_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
            "QFrame#stats_card QLabel { background: transparent; }"
        )
        card.setFixedWidth(380)
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        body = QVBoxLayout(card)
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(8)

        cap = QLabel("DECODER TIMEPOINTS")
        cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        body.addWidget(cap)

        # Cross-task suggestion line — peak of the mean AUC curve (NOT a plain
        # average of the per-decoder peaks). Text + a circular info badge; the
        # full multiline explanation is in the shared tooltip.
        sug_row = QHBoxLayout()
        sug_row.setContentsMargins(0, 0, 0, 0)
        sug_row.setSpacing(5)
        self._roster_suggested_lbl = QLabel("Suggested · peak of mean AUC")
        self._roster_suggested_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px;"
        )
        self._roster_suggested_lbl.setToolTip(_SUGGESTED_TOOLTIP)
        sug_row.addWidget(self._roster_suggested_lbl)
        sug_row.addWidget(self._build_info_badge(_SUGGESTED_TOOLTIP))
        sug_row.addStretch()
        body.addLayout(sug_row)

        body.addWidget(self._build_roster_header())

        # Rows are (re)built per eval run in ``_rebuild_roster``.
        self._roster_rows_layout = QVBoxLayout()
        self._roster_rows_layout.setSpacing(4)
        self._roster_rows_layout.setContentsMargins(0, 0, 0, 0)
        body.addLayout(self._roster_rows_layout)

        body.addStretch()

        # ─── Footer: status + reset-all ───────────────────────────────────
        self._roster_status_lbl = QLabel("0 / 0 confirmed")
        self._roster_status_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600;"
        )
        body.addWidget(self._roster_status_lbl)

        self._roster_reset_btn = QPushButton("Reset all to suggested")
        self._roster_reset_btn.setProperty("class", "secondary")
        self._roster_reset_btn.clicked.connect(self._reset_all_to_suggested)
        self._roster_reset_btn.setEnabled(False)
        body.addWidget(self._roster_reset_btn)

        return card

    @staticmethod
    def _build_info_badge(tooltip: str) -> QLabel:
        """A small circular 'i' badge that shows ``tooltip`` on hover.

        Rendered as a 14px outlined circle with an italic 'i', which reads as
        a proper info affordance instead of a raw ⓘ glyph stuck in the text.
        """
        badge = QLabel("i")
        badge.setFixedSize(14, 14)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setToolTip(tooltip)
        badge.setCursor(Qt.CursorShape.WhatsThisCursor)
        badge.setStyleSheet(
            "QLabel { "
            f"color: {TEXT_MUTED}; "
            f"border: 1px solid {TEXT_MUTED}; "
            "border-radius: 7px; "
            "font-size: 9px; font-weight: 700; font-style: italic; "
            "font-family: Georgia, 'Times New Roman', serif; "
            "} "
            f"QLabel:hover {{ color: {PRIMARY_BLUE}; border-color: {PRIMARY_BLUE}; }}"
        )
        return badge

    def _build_roster_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("roster_header")
        header.setStyleSheet(
            "QFrame#roster_header { background: transparent; "
            f"border-bottom: 1px solid {BORDER_GRAY}; }}"
            "QFrame#roster_header QLabel { background: transparent; }"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(6)
        cols = (
            ("Decoder", 1, Qt.AlignmentFlag.AlignLeft, 0),
            ("ms", 0, Qt.AlignmentFlag.AlignLeft, _ROSTER_SPIN_W),
            ("peak", 0, Qt.AlignmentFlag.AlignRight, _ROSTER_PEAK_W),
            ("AUC", 0, Qt.AlignmentFlag.AlignRight, _ROSTER_AUC_W),
            ("", 0, Qt.AlignmentFlag.AlignRight, _ROSTER_CONFIRM_W),
        )
        for text, stretch, align, width in cols:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 9px; font-weight: 700; "
                "letter-spacing: 0.4px;"
            )
            lbl.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            if width:
                lbl.setFixedWidth(width)
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
        """Per-decoder tab — two rows.

        Row 1: AUC chart + TGM heatmap side-by-side (equal width). The
        TGM aspect-locks to a square, so it occupies its half of the
        row as a square with whitespace below or above as needed.

        Row 2: horizontal stats card spanning the full width — SELECTED
        TIMEPOINT spinbox + AUC@selected + Peak AUC + Peak time, laid
        out as cells across the row.

        Picking a timepoint here (spinbox or chart click) sets ONLY this
        decoder via ``_set_decoder_timepoint`` and stays in sync with its
        Summary-roster row.
        """
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

        # ── Row 1: AUC chart + TGM heatmap side-by-side ──────────────
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        auc = AUCChart(show_legend=False)
        auc.setMinimumHeight(360)
        # Click sets ONLY this decoder's timepoint (capture task_name).
        auc.timepoint_clicked.connect(
            lambda t, _n=task_name: self._set_decoder_timepoint(_n, t)
        )
        self._decoder_auc_charts[task_name] = auc
        charts_row.addWidget(auc, 1)

        tgm_card = QFrame()
        tgm_card.setObjectName("tgm_card")
        tgm_card.setStyleSheet(
            f"QFrame#tgm_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
        )
        tgm_body = QVBoxLayout(tgm_card)
        tgm_body.setContentsMargins(10, 10, 10, 10)
        tgm_body.setSpacing(6)
        tgm_caption = QLabel("TEMPORAL GENERALIZATION (TRAIN × TEST)")
        tgm_caption.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        tgm_body.addWidget(tgm_caption)
        tgm = TGMChart()
        tgm.setMinimumHeight(360)
        tgm.timepoint_clicked.connect(
            lambda t, _n=task_name: self._set_decoder_timepoint(_n, t)
        )
        self._decoder_tgm_charts[task_name] = tgm
        tgm_body.addWidget(tgm, 1)
        charts_row.addWidget(tgm_card, 1)

        layout.addLayout(charts_row, 1)

        # ── Row 2: full-width horizontal stats strip ──────────────────
        layout.addWidget(self._build_decoder_stats_card(task_name))

        return tab

    def _build_decoder_stats_card(self, task_name: str) -> QWidget:
        """Horizontal stats strip for the per-decoder tab.

        Lays out four cells across the full-width card — SELECTED
        TIMEPOINT spinbox + AUC@selected + Peak AUC + Peak time. Each
        cell mirrors the same small-caps label / value pair used in the
        Summary roster. The spinbox edits ONLY this decoder's timepoint
        and stays in sync with its Summary-roster row.
        """
        card = QFrame()
        card.setObjectName("decoder_stats_card")
        card.setStyleSheet(
            f"QFrame#decoder_stats_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
            "QFrame#decoder_stats_card QLabel { background: transparent; }"
        )

        row = QHBoxLayout(card)
        row.setContentsMargins(16, 12, 16, 12)
        row.setSpacing(24)

        # ── SELECTED TIMEPOINT cell (with the spinbox) ───────────────
        tp_cell = QVBoxLayout()
        tp_cell.setSpacing(4)
        tp_cap = QLabel("SELECTED TIMEPOINT")
        tp_cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        tp_cell.addWidget(tp_cap)

        spin = QSpinBox()
        f = spin.font()
        f.setPointSize(14)
        f.setWeight(QFont.Weight.Bold)
        spin.setFont(f)
        spin.setRange(-9999, 9999)
        spin.setSingleStep(10)
        spin.setKeyboardTracking(False)
        spin.setAlignment(Qt.AlignmentFlag.AlignLeft)
        spin.setCursor(Qt.CursorShape.IBeamCursor)
        self._apply_spinbox_input_style(spin, PRIMARY_BLUE)
        spin.editingFinished.connect(
            lambda _n=task_name: self._on_decoder_input_committed(_n)
        )
        spin.valueChanged.connect(
            lambda _v, _n=task_name: self._on_decoder_input_committed(_n)
        )
        spin_row = QHBoxLayout()
        spin_row.setContentsMargins(0, 0, 0, 0)
        spin_row.setSpacing(6)
        spin_row.addWidget(spin)
        ms_lbl = QLabel("ms")
        ms_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 600;")
        spin_row.addWidget(ms_lbl)
        spin_row.addStretch()
        tp_cell.addLayout(spin_row)
        row.addLayout(tp_cell, 0)

        # ── stat cells (filled on result arrival) ─────────────────────
        auc_at_t = self._build_stat_cell(row, "AUC @ SELECTED")
        peak_auc = self._build_stat_cell(row, "PEAK AUC")
        peak_ms = self._build_stat_cell(row, "PEAK TIME", suffix=" ms")

        row.addStretch()

        self._decoder_stats[task_name] = {
            "timepoint": spin,
            "auc_at_t": auc_at_t,
            "peak_auc": peak_auc,
            "peak_ms": peak_ms,
        }
        return card

    def _build_stat_cell(
        self, row: QHBoxLayout, caption_text: str, *, suffix: str = ""
    ) -> QLabel:
        """A two-line stat cell: small-caps caption above a bold value."""
        cell = QVBoxLayout()
        cell.setSpacing(4)
        cap = QLabel(caption_text)
        cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        cell.addWidget(cap)
        value = QLabel("—")
        f = value.font()
        f.setPointSize(14)
        f.setWeight(QFont.Weight.Bold)
        value.setFont(f)
        value.setStyleSheet(f"color: {TEXT_PRIMARY};")
        value.setProperty("suffix", suffix)
        cell.addWidget(value)
        cell.addStretch()
        row.addLayout(cell, 0)
        return value

    # ── populate helpers (called by _populate_results) ───────────────────────

    def _rebuild_roster(self, tasks_result: dict[str, dict]) -> None:
        """(Re)build the Summary-tab per-decoder roster rows.

        Each row owns that decoder's timepoint spinbox, AUC@t label and
        Confirm button. Cleared and rebuilt on every fresh eval run.
        """
        layout = self._roster_rows_layout
        assert layout is not None
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._roster_spins.clear()
        self._roster_auc_lbls.clear()
        self._roster_confirm_btns.clear()

        for i, name in enumerate(tasks_result):
            wrapper = QFrame()
            wrapper.setObjectName("roster_row")
            wrapper.setStyleSheet(
                "QFrame#roster_row { background: transparent; }"
                "QFrame#roster_row QLabel { background: transparent; }"
            )
            row = QHBoxLayout(wrapper)
            row.setContentsMargins(0, 1, 0, 1)
            row.setSpacing(6)

            # Colour dot + name.
            name_lbl = QLabel(f"● {name}")
            name_lbl.setStyleSheet(
                f"color: {chart_line_color(i)}; font-size: 11px; font-weight: 600;"
            )
            name_lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            name_lbl.setToolTip(name)
            row.addWidget(name_lbl, 1)

            # Timepoint spinbox (this decoder only).
            spin = QSpinBox()
            sf = spin.font()
            sf.setPointSize(11)
            sf.setWeight(QFont.Weight.Bold)
            spin.setFont(sf)
            spin.setFixedWidth(_ROSTER_SPIN_W)
            spin.setRange(-9999, 9999)  # widened on result arrival
            spin.setSingleStep(10)
            spin.setKeyboardTracking(False)
            spin.setAlignment(Qt.AlignmentFlag.AlignLeft)
            spin.setCursor(Qt.CursorShape.IBeamCursor)
            self._apply_spinbox_input_style(spin, PRIMARY_BLUE)
            spin.editingFinished.connect(
                lambda _n=name: self._on_roster_input_committed(_n)
            )
            spin.valueChanged.connect(
                lambda _v, _n=name: self._on_roster_input_committed(_n)
            )
            self._roster_spins[name] = spin
            row.addWidget(spin, 0)

            # This decoder's own suggested peak (read-only, static per run).
            peak_s = self._suggested_timepoints.get(name)
            peak_lbl = QLabel(f"{peak_s * 1000.0:.0f}" if peak_s is not None else "—")
            peak_lbl.setFixedWidth(_ROSTER_PEAK_W)
            peak_lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 11px;"
            )
            peak_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            peak_lbl.setToolTip("This decoder's own peak-AUC timepoint")
            row.addWidget(peak_lbl, 0)

            # AUC @ this decoder's timepoint.
            auc_lbl = QLabel("—")
            auc_lbl.setFixedWidth(_ROSTER_AUC_W)
            auc_lbl.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 11px; font-weight: 600;"
            )
            auc_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._roster_auc_lbls[name] = auc_lbl
            row.addWidget(auc_lbl, 0)

            # Per-decoder Confirm button (fixed width → no reflow on toggle).
            confirm_btn = QPushButton("Confirm")
            confirm_btn.setFixedWidth(_ROSTER_CONFIRM_W)
            confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            confirm_btn.clicked.connect(
                lambda _checked=False, _n=name: self._toggle_confirm(_n)
            )
            self._roster_confirm_btns[name] = confirm_btn
            row.addWidget(confirm_btn, 0)

            layout.addWidget(wrapper)

        if self._roster_reset_btn is not None:
            self._roster_reset_btn.setEnabled(bool(tasks_result))

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

        Caller stashes the AUC label so ``_update_decoder_auc`` can rewrite
        it whenever that decoder's timepoint changes.
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

        # AUC text + colour rewritten in _update_decoder_auc.
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

    def _set_decoder_timepoint(self, task_name: str, t_seconds: float) -> None:
        """Set ONE decoder's selected timepoint and sync just that decoder.

        Snaps to the nearest sample, writes ``_selected_timepoints[name]``,
        and updates only its spinboxes (roster + tab), AUC labels, chart
        markers, amber styling and confirm button. A *genuine* change
        unconfirms that decoder; re-selecting the same sample keeps the
        lock. Other decoders are untouched.
        """
        t_seconds = self._snap_to_sample(t_seconds)
        prev = self._selected_timepoints.get(task_name)
        self._selected_timepoints[task_name] = t_seconds
        # Only a genuine change unlocks — a no-op re-select keeps the lock.
        if prev is None or abs(prev - t_seconds) > 1e-9:
            self._confirmed[task_name] = False

        suggested = self._suggested_timepoints.get(task_name, t_seconds)
        sel_ms = t_seconds * 1000.0
        dev_ms = abs(sel_ms - suggested * 1000.0)
        warn = dev_ms > _DEVIATION_WARN_MS
        color = AMBER if warn else PRIMARY_BLUE

        # Mirror into both of this decoder's spinboxes (roster + tab card).
        # ``blockSignals`` prevents setValue from re-entering the committed
        # handlers → ``_set_decoder_timepoint``.
        for spin in self._decoder_spinboxes(task_name):
            spin.blockSignals(True)
            spin.setValue(int(round(sel_ms)))
            spin.blockSignals(False)
            self._apply_spinbox_input_style(spin, color)
            spin.setToolTip(
                f"±{dev_ms:.0f} ms from suggested ({suggested * 1000.0:.0f} ms)"
                if warn else ""
            )

        self._update_decoder_auc(task_name, t_seconds)
        # This decoder's chart markers only.
        auc = self._decoder_auc_charts.get(task_name)
        if auc is not None:
            auc.set_selected_timepoint(t_seconds)
        tgm = self._decoder_tgm_charts.get(task_name)
        if tgm is not None:
            tgm.set_selected_timepoint(t_seconds)

        self._refresh_confirm_button(task_name)
        self._refresh_roster_status()
        self._update_ready_state()

    def _update_decoder_auc(self, task_name: str, t_seconds: float) -> None:
        """Rewrite one decoder's AUC@t labels — roster row, summary table
        row, and per-decoder tab card — to show AUC at ``t_seconds``."""
        if self._result is None:
            return
        times = self._result.get("times")
        task = self._result.get("tasks", {}).get(task_name)
        if times is None or task is None:
            return
        import numpy as np
        idx = int(np.argmin(np.abs(np.asarray(times) - t_seconds)))
        diag = task.get("diagonal_auc")
        if diag is None or len(diag) <= idx:
            return
        v = float(diag[idx])
        text = f"{v:.2f}"
        colour = "#16A34A" if v >= 0.70 else "#DC2626"

        roster_lbl = self._roster_auc_lbls.get(task_name)
        if roster_lbl is not None:
            roster_lbl.setText(text)
            roster_lbl.setStyleSheet(
                f"color: {colour}; font-size: 11px; font-weight: 600;"
            )
        tbl_lbl = self._table_auc_lbls.get(task_name)
        if tbl_lbl is not None:
            tbl_lbl.setText(text)
            tbl_lbl.setStyleSheet(
                "font-family: monospace; font-size: 11px; "
                f"font-weight: 600; color: {colour};"
            )
        decoder_stats = self._decoder_stats.get(task_name)
        if decoder_stats is not None:
            self._set_stat_value(decoder_stats["auc_at_t"], text)

    def _on_roster_input_committed(self, task_name: str) -> None:
        """Summary-roster spinbox edit handler — sets this decoder only."""
        spin = self._roster_spins.get(task_name)
        if not isinstance(spin, QSpinBox):
            return
        self._set_decoder_timepoint(task_name, spin.value() / 1000.0)

    def _on_decoder_input_committed(self, task_name: str) -> None:
        """Per-decoder tab spinbox edit handler — sets this decoder only,
        synced with its Summary-roster row via ``_set_decoder_timepoint``."""
        stats = self._decoder_stats.get(task_name)
        if stats is None:
            return
        spin = stats.get("timepoint")
        if not isinstance(spin, QSpinBox):
            return
        self._set_decoder_timepoint(task_name, spin.value() / 1000.0)

    def _toggle_confirm(self, task_name: str) -> None:
        """Lock / unlock one decoder's timepoint.

        Doesn't advance to Node 5 by itself — that's the journey-panel
        "Approve && Continue" button, gated by ``_all_confirmed()``.
        """
        if not self._done or task_name not in self._selected_timepoints:
            return
        self._confirmed[task_name] = not self._confirmed.get(task_name, False)
        self._refresh_confirm_button(task_name)
        self._refresh_roster_status()
        self._update_ready_state()

    def _refresh_confirm_button(self, task_name: str) -> None:
        """Flip one decoder's Confirm button between blue and success-green.

        Inline stylesheet for both states (Qt doesn't re-polish class-based
        QSS on its own). Fixed width keeps the row from reflowing on toggle.
        """
        btn = self._roster_confirm_btns.get(task_name)
        if btn is None:
            return
        confirmed = self._confirmed.get(task_name, False)
        btn.setEnabled(self._done)
        if confirmed:
            btn.setText("Locked ✓")
            btn.setStyleSheet(
                f"QPushButton {{ background: {SUCCESS_GREEN}; color: white; "
                "border: none; border-radius: 2px; padding: 4px 8px; "
                "font-size: 11px; font-weight: 600; }"
            )
        else:
            btn.setText("Confirm")
            btn.setStyleSheet(
                f"QPushButton {{ background: {PRIMARY_BLUE}; color: white; "
                "border: none; border-radius: 2px; padding: 4px 8px; "
                "font-size: 11px; font-weight: 600; }"
                f"QPushButton:hover {{ background: {PRIMARY_BLUE_HOVER}; }}"
                f"QPushButton:disabled {{ background: #D1D5DB; "
                f"color: {TEXT_MUTED}; }}"
            )

    def _refresh_roster_status(self) -> None:
        """Update the ``N / M confirmed`` footer; green once all confirmed."""
        if self._roster_status_lbl is None:
            return
        names = list(self._selected_timepoints.keys())
        n_done = sum(1 for x in names if self._confirmed.get(x, False))
        self._roster_status_lbl.setText(f"{n_done} / {len(names)} confirmed")
        all_done = bool(names) and n_done == len(names)
        colour = SUCCESS_GREEN if all_done else TEXT_MUTED
        self._roster_status_lbl.setStyleSheet(
            f"color: {colour}; font-size: 11px; font-weight: 600;"
        )

    def _reset_all_to_suggested(self) -> None:
        """Reset every decoder to its own suggested peak — a clean slate.

        Explicitly unconfirms all decoders (unlike the per-edit guard in
        ``_set_decoder_timepoint``), so a decoder already sitting at its
        suggested value is also unlocked.
        """
        for name, t in self._suggested_timepoints.items():
            self._confirmed[name] = False
            self._set_decoder_timepoint(name, t)
        self._refresh_roster_status()

    @staticmethod
    def _apply_spinbox_input_style(spin: QSpinBox, text_color: str) -> None:
        """Style any timepoint spinbox as a visible input field.

        Light border + padded background so the operator can tell the
        value is editable, plus a focused-state highlight + visible
        up/down arrows. ``text_color`` flips between PRIMARY_BLUE
        (within suggested) and AMBER (deviating > _DEVIATION_WARN_MS).
        """
        spin.setStyleSheet(
            "QSpinBox { "
            f"color: {text_color}; "
            f"background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; "
            "border-radius: 4px; "
            "padding: 4px 6px; "
            "} "
            "QSpinBox:focus { "
            f"border: 1.5px solid {PRIMARY_BLUE}; "
            "} "
            "QSpinBox::up-button, QSpinBox::down-button { "
            "subcontrol-origin: border; "
            "width: 16px; "
            "background: transparent; "
            "border: none; "
            "} "
            "QSpinBox::up-button { subcontrol-position: top right; } "
            "QSpinBox::down-button { subcontrol-position: bottom right; } "
            "QSpinBox::up-arrow { "
            f"image: none; width: 0; height: 0; "
            f"border-left: 4px solid transparent; "
            f"border-right: 4px solid transparent; "
            f"border-bottom: 5px solid {TEXT_MUTED}; "
            "} "
            "QSpinBox::down-arrow { "
            f"image: none; width: 0; height: 0; "
            f"border-left: 4px solid transparent; "
            f"border-right: 4px solid transparent; "
            f"border-top: 5px solid {TEXT_MUTED}; "
            "} "
        )

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
        tasks = self._session.settings["decoders"]["tasks"]
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
        # EVERY decoder's timepoint has been confirmed in the roster — the
        # journey-panel "Approve && Continue" stays disabled until then).
        page0_ready = (
            self._session is not None
            and self._preproc_done
            and not self._running
            and not self._done
        )
        page1_ready = self._done and self._all_confirmed()
        ready = page0_ready or page1_ready
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
