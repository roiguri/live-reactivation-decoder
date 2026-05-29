"""Rolling probability chart for Phase 2 live inference.

Streams one curve per decoder over a fixed-length time window
(default 10 s) at the inference target rate (default 100 Hz). The
widget is driven by ``append_predictions(...)``, called from the UI
slot connected to ``LiveStreamSession.prediction_ready``.

Performance notes
-----------------
``prediction_ready`` fires at ~25 Hz; the chart paints at 30 Hz on
its own ``QTimer`` so the signal-rate is decoupled from the repaint
rate. ``append_predictions`` is the hot path — it only writes into
preallocated numpy ring buffers, never allocates, and never touches
the pyqtgraph scene.

The ring buffer is **double-length** (``2 * capacity``): every write
lands at both ``idx`` and ``idx + capacity``, so a contiguous slice
``buf[idx+1 : idx+1+capacity]`` reads the most recent ``capacity``
samples without ``np.roll`` or concatenation. Costs ~2× memory
(~16 kB per task at 1000 samples × float64) — trivial vs. the
allocation cost on every repaint.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPen

from frontend.styles.theme import (
    ALERT_RED,
    BORDER_GRAY,
    CARD_WHITE,
    PRIMARY_BLUE,
    TEXT_MUTED,
    chart_line_color,
)

# Apply the same global look the Phase 1 charts use. Must run before the
# first ``pg.PlotWidget`` constructs so the viewbox defaults stick.
pg.setConfigOptions(background="w", foreground="k", antialias=True)

_Y_RANGE = (0.0, 1.05)
_CHANCE_LEVEL = 0.5
_REFRESH_HZ = 30
# Empty space to the right of x = 0 so the latest sample isn't pressed
# against the right edge. Makes the "this is now" cue obvious paired
# with the vertical NOW line.
_NOW_GAP_SECONDS = 0.5


class LiveProbabilityChart(pg.PlotWidget):
    """Rolling per-decoder probability chart with chance + threshold guides.

    Constructor args
    ----------------
    task_names
        Decoder names. One ``PlotDataItem`` is created per name, in order.
    window_seconds
        Width of the rolling window. The x-axis is fixed at
        ``[-window_seconds, 0]`` — samples scroll right-to-left.
    target_sfreq
        Sample rate of the prediction stream (after Phase 2 decimation).
        Used to size the ring buffer: ``capacity = window_seconds * target_sfreq``.
    threshold
        y-position of the dashed red threshold guide. Purely visual for
        now; operator-tunable threshold semantics are a follow-up.
    """

    def __init__(
        self,
        task_names: list[str],
        *,
        window_seconds: float = 10.0,
        target_sfreq: float = 100.0,
        threshold: float = 0.85,
        parent=None,
    ) -> None:
        super().__init__(parent, background=CARD_WHITE)

        if not task_names:
            raise ValueError("LiveProbabilityChart requires at least one task name.")

        self._task_names: list[str] = list(task_names)
        self._window_seconds = float(window_seconds)
        self._target_sfreq = float(target_sfreq)
        self._threshold = float(threshold)

        self._capacity: int = max(1, int(round(window_seconds * target_sfreq)))
        # Double-length ring: writes mirror into [idx] and [idx + capacity]
        # so the most-recent window is always a contiguous slice ending
        # at write_idx + capacity.
        self._timestamps = np.full(2 * self._capacity, np.nan, dtype=np.float64)
        self._buffers: dict[str, np.ndarray] = {
            name: np.full(2 * self._capacity, np.nan, dtype=np.float64)
            for name in self._task_names
        }
        self._write_idx: int = 0  # next slot to fill, in [0, capacity)
        self._latest_ts: float | None = None  # most recent timestamp seen

        self._configure_axes()
        self._add_h_line(y=_CHANCE_LEVEL, color=TEXT_MUTED, dashed=True, width=1.0)
        self._add_h_line(y=self._threshold, color=ALERT_RED, dashed=True, width=1.2)
        # Vertical line at x = 0 marking the present. Paired with the
        # right-side gap (``_NOW_GAP_SECONDS``) so the operator reads it
        # unambiguously as "this is now" rather than "this is the edge
        # of the data".
        self._add_v_line(x=0.0, color=PRIMARY_BLUE, dashed=False, width=1.4)

        # Canonical name → hex colour mapping, in insertion order. Exposed
        # via the ``task_colors`` property so the parent widget can build
        # an external legend (see Phase2Screen / scripts/test_live_chart.py)
        # — we deliberately don't render an in-plot legend so layout
        # decisions live with the screen, not the chart.
        self._task_colors: dict[str, str] = {
            name: chart_line_color(i) for i, name in enumerate(self._task_names)
        }
        self._curves: dict[str, pg.PlotDataItem] = {}
        for name, color in self._task_colors.items():
            curve = self.plot(pen=pg.mkPen(color=color, width=2), name=name)
            self._curves[name] = curve

        # 30 Hz repaint timer — decoupled from the ~25 Hz signal rate so
        # the inference loop is never blocked by chart work.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(int(round(1000 / _REFRESH_HZ)))
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def task_colors(self) -> dict[str, str]:
        """Decoder name → hex colour, in the order passed to the constructor.

        Parents use this to build an external legend or controls panel
        that stays in sync with the plotted curves.
        """
        return dict(self._task_colors)

    def append_predictions(
        self,
        predictions: dict[str, np.ndarray],
        timestamps: np.ndarray,
    ) -> None:
        """Hot path. Writes into ring buffers; does not touch the scene.

        ``predictions`` keys must be a subset of ``task_names`` passed
        to the constructor. Missing keys are skipped (their curve will
        keep whatever was last drawn until the next ``_refresh``).
        Extra keys are silently ignored.
        """
        if timestamps.size == 0:
            return
        n = timestamps.size
        cap = self._capacity
        idx = self._write_idx

        for i in range(n):
            ts = float(timestamps[i])
            self._timestamps[idx] = ts
            self._timestamps[idx + cap] = ts
            for name in self._task_names:
                vals = predictions.get(name)
                if vals is None or i >= len(vals):
                    continue
                v = float(vals[i])
                self._buffers[name][idx] = v
                self._buffers[name][idx + cap] = v
            idx = (idx + 1) % cap

        self._write_idx = idx
        self._latest_ts = float(timestamps[-1])

    def set_task_visible(self, name: str, visible: bool) -> None:
        """Show or hide a single decoder's curve at runtime.

        Data keeps flowing into the ring buffer regardless of visibility,
        so toggling a curve back on instantly restores the full visible
        history. Unknown ``name`` is a no-op (symmetric with
        :meth:`append_predictions`'s tolerance for missing keys).
        """
        curve = self._curves.get(name)
        if curve is not None:
            curve.setVisible(visible)

    def reset_buffers(self) -> None:
        """Reset the ring buffers. Curves go blank on the next refresh tick.

        Named ``reset_buffers`` rather than ``clear`` because pyqtgraph's
        ``PlotWidget.__init__`` installs an instance-attribute ``clear``
        that forwards to ``PlotItem.clear`` (which removes plot items —
        not what callers want here). The instance attribute would shadow
        any class-level ``clear`` method on a subclass.
        """
        self._timestamps.fill(np.nan)
        for buf in self._buffers.values():
            buf.fill(np.nan)
        self._write_idx = 0
        self._latest_ts = None

    # ── internals ─────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Called by the 30 Hz timer. Reads a contiguous window from each
        ring buffer and pushes it to the corresponding ``PlotDataItem``.

        Timestamps are rebased to seconds-relative-to-latest so the
        x-axis stays anchored at ``[-window_seconds, 0]`` and the curve
        appears to scroll right-to-left.
        """
        if self._latest_ts is None:
            return
        cap = self._capacity
        start = self._write_idx
        end = start + cap
        # ``ts`` is the contiguous most-recent window; NaN entries (the
        # cold-start tail before the buffer filled) render as line breaks
        # in pyqtgraph — exactly what we want, since we don't fabricate
        # fake leading samples.
        ts = self._timestamps[start:end]
        x = ts - self._latest_ts  # rebased: latest sample at 0, older negative
        for name, curve in self._curves.items():
            curve.setData(x, self._buffers[name][start:end])

    def _configure_axes(self) -> None:
        """Lock both axes to the analyst-facing units and hide chart-junk."""
        self.setYRange(*_Y_RANGE, padding=0)
        self.setXRange(-self._window_seconds, _NOW_GAP_SECONDS, padding=0)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)

        plot_item = self.getPlotItem()
        plot_item.setLabel("left", "P(class = 1)")
        plot_item.setLabel("bottom", "Time (s, relative to latest)")
        plot_item.showGrid(x=False, y=False)

        for ax_name in ("left", "bottom"):
            ax = plot_item.getAxis(ax_name)
            ax.setPen(QPen(QColor(BORDER_GRAY)))
            ax.setTextPen(QPen(QColor(TEXT_MUTED)))

        # Subtle horizontal grid lines at major Y positions — same trick
        # AUCChart uses to avoid pyqtgraph's heavy default grid.
        gridline_pen = QPen(QColor(BORDER_GRAY))
        gridline_pen.setWidthF(0.8)
        gridline_pen.setCosmetic(True)
        for y in (0.2, 0.4, 0.6, 0.8):
            self.addItem(
                pg.InfiniteLine(pos=y, angle=0, pen=gridline_pen, movable=False)
            )

    def _add_h_line(
        self, *, y: float, color: str, dashed: bool, width: float
    ) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            pos=y, angle=0, pen=self._pen(color, dashed, width), movable=False
        )
        self.addItem(line)
        return line

    def _add_v_line(
        self, *, x: float, color: str, dashed: bool, width: float
    ) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            pos=x, angle=90, pen=self._pen(color, dashed, width), movable=False
        )
        self.addItem(line)
        return line

    @staticmethod
    def _pen(color: str, dashed: bool, width: float) -> QPen:
        pen = QPen(QColor(color))
        pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setWidthF(width)
        # Cosmetic — dash + width stay in device pixels, not view units.
        # Same trap AUCChart documents at length: a non-cosmetic dashed
        # pen on an InfiniteLine paints in y-axis units (0..1) which then
        # get scaled to pixels and ruin the dash pattern.
        pen.setCosmetic(True)
        return pen
