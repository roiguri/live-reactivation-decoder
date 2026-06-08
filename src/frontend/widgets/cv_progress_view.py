from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, SUCCESS_GREEN, TEXT_MUTED,
    TEXT_PRIMARY, chart_line_color, progress_bar_qss,
)

# Cadence of the ETA countdown tick. It only refreshes the remaining-time
# *text* (after the first decoder lands a real duration) — never the bar,
# which jumps in discrete sections at real completion events.
_ETA_TICK_MS = 1000

# Per-decoder grid: 3 columns mirrors the design mock (knowledge_base/
# 02_reference/ui_demo/Phase1Screen.jsx, WorkspaceNode3CVProgress).
_GRID_COLS = 3


class CVProgressView(QWidget):
    """Per-decoder evaluation progress screen.

    Replaces the generic indeterminate overlay for Node 4 with the mock's
    decoder-card grid + overall bar. The backend reports real progress at
    decoder granularity (``ModelEvaluator.run_evaluation``'s ``on_progress``
    hook → ``update_progress``).

    Honest-by-construction, with no between-event guessing:

    * The overall bar **jumps in discrete sections** — it advances to
      ``decoders done / total`` only when a real completion event lands, and
      reaches 100 % only via ``mark_all_complete``. It never creeps
      gradually (we have no within-decoder signal, so a smooth fill would be
      invented).
    * The remaining-time estimate appears **only after the first decoder
      finishes** — that's the first real duration sample. Before then there
      is nothing to estimate from, so no ETA is shown. The running decoder's
      card carries an indeterminate shimmer so the screen still reads as
      live while the bar sits between section jumps.

    Decoders run serially in the backend, so a completion event for decoder
    *i* implies decoder *i+1* is now the one running; the card grid is
    advanced on that assumption.

    Lifecycle::

        set_decoders([...])      # build the grid (all Pending)
        start()                  # first card → Running, begin animation
        update_progress(c, n, name)   # one call per real completion event
        mark_all_complete()      # all → Complete, bar → 100 %
        reset()                  # stop timer, clear grid

    Public methods are no-ops when called out of order (e.g.
    ``mark_all_complete`` before ``start``) so the owning view can drive
    them defensively.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._names: list[str] = []
        # Per-decoder widget refs, keyed by decoder name.
        self._cards: dict[str, QFrame] = {}
        self._card_bars: dict[str, QProgressBar] = {}
        self._card_status: dict[str, QLabel] = {}
        self._card_glyph: dict[str, QLabel] = {}
        self._status: dict[str, str] = {}  # name → "pending"|"running"|"done"

        self._total: int = 0
        self._completed: int = 0
        # Measured wall-clock durations of finished decoders. Their mean is
        # the per-decoder estimate that drives the ETA — available only once
        # the first decoder has finished (≥1 sample).
        self._durations: list[float] = []
        self._segment_clock = QElapsedTimer()  # time in the current decoder
        # Ticks the remaining-time text only; the bar is event-driven.
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(_ETA_TICK_MS)
        self._eta_timer.timeout.connect(self._tick_eta)

        self._build_ui()

    # ── public API ─────────────────────────────────────────────────────────

    def set_decoders(self, names: list[str]) -> None:
        """(Re)build the card grid, one card per decoder, all Pending."""
        self.reset()
        self._names = list(names)
        self._total = len(self._names)
        for i, name in enumerate(self._names):
            card = self._build_card(i, name)
            self._cards[name] = card
            self._status[name] = "pending"
            self._grid.addWidget(card, i // _GRID_COLS, i % _GRID_COLS)
        self._refresh_overall(0.0)
        self._eta_lbl.setText("")

    def start(self) -> None:
        """Begin the run: first decoder → Running, bar at 0 %.

        No ETA yet — there's no duration sample to estimate from until the
        first decoder finishes. The ETA-tick timer runs so the estimate can
        appear the moment that first sample lands.
        """
        if not self._names:
            return
        self._completed = 0
        self._durations.clear()
        self._set_status(self._names[0], "running")
        self._segment_clock.restart()
        self._eta_timer.start()
        self._refresh_overall(0.0)
        self._eta_lbl.setText("")

    def update_progress(self, completed: int, total: int, name: str) -> None:
        """Handle one real backend completion event.

        ``name`` (the ``completed``-th decoder) just finished; mark it Complete,
        jump the overall bar to the new section, and — since decoders run
        serially — advance the next one to Running.
        """
        if name not in self._status:
            return
        self._set_status(name, "done")
        # Record this decoder's wall-clock duration → feeds the ETA estimate.
        self._durations.append(self._segment_clock.elapsed() / 1000.0)
        self._segment_clock.restart()
        self._completed = completed
        idx = self._names.index(name)
        if idx + 1 < len(self._names):
            self._set_status(self._names[idx + 1], "running")
        # Discrete section jump — the bar only advances on real events.
        self._refresh_overall(completed / max(1, total))
        self._tick_eta()

    def mark_all_complete(self) -> None:
        """Snap every card to Complete and the bar to 100 % — the only path
        that reaches 100 %. Safe to call even if ``start`` never ran."""
        self._eta_timer.stop()
        for name in self._names:
            self._set_status(name, "done")
        self._completed = self._total
        self._refresh_overall(1.0)
        self._eta_lbl.setText("Done")

    def reset(self) -> None:
        """Stop the ETA timer and tear down the grid."""
        self._eta_timer.stop()
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._names = []
        self._cards.clear()
        self._card_bars.clear()
        self._card_status.clear()
        self._card_glyph.clear()
        self._status.clear()
        self._total = 0
        self._completed = 0
        self._durations.clear()
        self._refresh_overall(0.0)
        self._eta_lbl.setText("")

    # ── ETA ──────────────────────────────────────────────────────────────────

    def _tick_eta(self) -> None:
        """Refresh the remaining-time text (the bar is event-driven).

        Shown only after the first decoder finishes — until then there's no
        duration sample, so the estimate stays blank. Estimate = (decoders
        left, including the running one) × mean measured duration, minus the
        time already spent in the running decoder.
        """
        if self._total == 0 or not self._durations:
            return  # no sample yet → no estimate (label stays blank)
        avg = sum(self._durations) / len(self._durations)
        segment_elapsed = self._segment_clock.elapsed() / 1000.0
        remaining_decoders = self._total - self._completed
        eta = max(0.0, remaining_decoders * avg - segment_elapsed)
        self._eta_lbl.setText(
            "Finishing up…" if eta < 1.0 else f"~{round(eta)} s remaining"
        )

    # ── rendering helpers ────────────────────────────────────────────────────

    def _refresh_overall(self, fraction: float) -> None:
        pct = int(round(max(0.0, min(1.0, fraction)) * 100))
        self._overall_bar.setValue(pct)
        self._pct_lbl.setText(f"{pct}%")

    def _set_status(self, name: str, status: str) -> None:
        """Flip one card between pending / running / done states."""
        if name not in self._cards:
            return
        self._status[name] = status
        bar = self._card_bars[name]
        glyph = self._card_glyph[name]
        text = self._card_status[name]
        if status == "running":
            bar.setRange(0, 0)  # indeterminate shimmer — we don't fake folds
            glyph.setText("▶")
            glyph.setStyleSheet(f"color: {PRIMARY_BLUE}; font-size: 12px;")
            text.setText("Running…")
            text.setStyleSheet(f"color: {PRIMARY_BLUE}; font-size: 10px;")
            self._style_card(name, active=True, done=False)
        elif status == "done":
            bar.setRange(0, 1)
            bar.setValue(1)
            glyph.setText("✓")
            glyph.setStyleSheet(f"color: {SUCCESS_GREEN}; font-size: 12px;")
            text.setText("Complete")
            text.setStyleSheet(f"color: {SUCCESS_GREEN}; font-size: 10px;")
            self._style_card(name, active=False, done=True)
        else:  # pending
            bar.setRange(0, 1)
            bar.setValue(0)
            glyph.setText("•")
            glyph.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
            text.setText("Pending")
            text.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
            self._style_card(name, active=False, done=False)

    def _style_card(self, name: str, *, active: bool, done: bool) -> None:
        card = self._cards[name]
        if done:
            border, bg = "#BBF7D0", "#F0FDF4"
        elif active:
            border, bg = PRIMARY_BLUE, "#EFF6FF"
        else:
            border, bg = BORDER_GRAY, CARD_WHITE
        card.setStyleSheet(
            f"QFrame#decoder_card {{ background: {bg}; "
            f"border: 1px solid {border}; border-radius: 4px; }}"
            "QFrame#decoder_card QLabel { background: transparent; border: none; }"
        )

    # ── construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(0)
        outer.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Caption mirrors LoadingOverlay's message (12 pt Medium) so the two
        # loading idioms read as one design system.
        caption = QLabel("Running cross-validation across decoders…")
        cf = caption.font()
        cf.setPointSize(12)
        cf.setWeight(QFont.Weight.Medium)
        caption.setFont(cf)
        caption.setStyleSheet(f"color: {TEXT_PRIMARY};")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(caption)
        center.addSpacing(6)

        # Percent + ETA line.
        meta_row = QHBoxLayout()
        meta_row.setSpacing(10)
        meta_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-family: monospace; font-size: 11px;"
        )
        self._eta_lbl = QLabel("")
        self._eta_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px;"
        )
        meta_row.addWidget(self._pct_lbl)
        meta_row.addWidget(self._eta_lbl)
        center.addLayout(meta_row)
        center.addSpacing(14)

        # Overall bar.
        self._overall_bar = QProgressBar()
        self._overall_bar.setObjectName("overall_bar")
        self._overall_bar.setRange(0, 100)
        self._overall_bar.setValue(0)
        self._overall_bar.setTextVisible(False)
        self._overall_bar.setFixedHeight(6)
        self._overall_bar.setFixedWidth(560)
        self._overall_bar.setStyleSheet(progress_bar_qss("overall_bar"))
        center.addWidget(self._overall_bar, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(26)

        # Decoder-card grid.
        grid_host = QWidget()
        grid_host.setFixedWidth(560)
        self._grid = QGridLayout(grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        center.addWidget(grid_host, 0, Qt.AlignmentFlag.AlignHCenter)

        outer.addLayout(center)
        outer.addStretch()

    def _build_card(self, index: int, name: str) -> QFrame:
        card = QFrame()
        card.setObjectName("decoder_card")
        card.setFixedHeight(96)

        body = QVBoxLayout(card)
        body.setContentsMargins(12, 10, 12, 10)
        body.setSpacing(8)

        # Top row: colour dot + name + status glyph.
        top = QHBoxLayout()
        top.setSpacing(6)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {chart_line_color(index)}; font-size: 11px;")
        top.addWidget(dot)
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        name_lbl.setToolTip(name)
        top.addWidget(name_lbl, 1)
        glyph = QLabel("•")
        glyph.setFixedWidth(14)
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(glyph)
        body.addLayout(top)

        # Per-decoder bar (indeterminate while running; full when done).
        bar = QProgressBar()
        bar.setObjectName("card_bar")
        bar.setRange(0, 1)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        bar.setStyleSheet(progress_bar_qss("card_bar"))
        body.addWidget(bar)

        status = QLabel("Pending")
        status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        body.addWidget(status)

        self._card_bars[name] = bar
        self._card_status[name] = status
        self._card_glyph[name] = glyph
        self._style_card_initial(card)
        return card

    @staticmethod
    def _style_card_initial(card: QFrame) -> None:
        card.setStyleSheet(
            f"QFrame#decoder_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 4px; }}"
            "QFrame#decoder_card QLabel { background: transparent; border: none; }"
        )
