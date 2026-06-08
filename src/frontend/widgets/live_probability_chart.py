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

# Probabilities live in [0, 1]; the view extends to _Y_RANGE[1] to leave an
# empty band above the data where event-marker labels are drawn (so they sit
# clear of the curves). _Y_TICK_MAX caps the labelled Y ticks at 1.0 so that
# headroom reads as blank margin, not as "probability > 1".
_Y_RANGE = (0.0, 1.18)
_Y_TICK_MAX = 1.0
_CHANCE_LEVEL = 0.5
_REFRESH_HZ = 30
# Empty space to the right of x = 0 so the latest sample isn't pressed
# against the right edge. Makes the "this is now" cue obvious paired
# with the vertical NOW line.
_NOW_GAP_SECONDS = 0.5

# All event markers share one colour — the line distinguishes itself by
# position + label (the configured event name), not by hue. Black for a
# strong, neutral contrast against the curves; deliberately decoupled
# from the event name.
_MARKER_COLOR = "#000000"
# Hard cap on retained marker records so a stalled stream (latest_ts not
# advancing) or a flood of unmapped codes can't grow the scene unbounded.
# Pruning by window in ``_refresh`` is the normal path; this is a backstop.
_MAX_MARKERS = 128


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
    event_names
        Trigger code → event name (from ``markers_mapping.events``). Used
        to label markers drawn by :meth:`append_markers` and to decide
        which codes to draw: a marker whose code is **not** in this map is
        dropped. All markers share one neutral colour (``_MARKER_COLOR``).
    """

    def __init__(
        self,
        task_names: list[str],
        *,
        window_seconds: float = 10.0,
        target_sfreq: float = 100.0,
        threshold: float = 0.85,
        event_names: dict[int, str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent, background=CARD_WHITE)

        if not task_names:
            raise ValueError("LiveProbabilityChart requires at least one task name.")

        self._task_names: list[str] = list(task_names)
        self._window_seconds = float(window_seconds)
        self._target_sfreq = float(target_sfreq)
        self._threshold = float(threshold)
        self._event_names: dict[int, str] = {
            int(code): str(name) for code, name in (event_names or {}).items()
        }
        # Cue markers, oldest first. Each: {"ts": float, "code": int,
        # "line": InfiniteLine | None}. The line is created lazily in
        # ``_refresh`` (the only place that touches the scene) the first
        # time a marker is inside the visible window, and removed when it
        # scrolls off the left edge.
        self._markers: list[dict] = []

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
            # ``antialias=False`` per-curve overrides the global
            # ``setConfigOptions(antialias=True)`` for the data curves only.
            # Antialiased curves cost ~2-3x more to paint, and with one
            # ~500-point curve per decoder repainted at 30 Hz that dominates
            # the frame budget. Static guide/marker InfiniteLines keep the
            # global setting, so they stay smooth. Tradeoff: slightly harder
            # curve edges / mild shimmer as the curve scrolls. See the "Chart
            # Rendering Performance" section in docs/plans/phase2_ui_plan_m2.md.
            curve = self.plot(
                pen=pg.mkPen(color=color, width=2), name=name, antialias=False
            )
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

    def append_markers(self, markers: list[tuple[float, int]]) -> None:
        """Buffer cue markers for the next repaint. Data-only — like
        :meth:`append_predictions`, this never touches the scene; the
        actual line items are created/moved/removed in ``_refresh``.

        ``markers`` is the ``(lsl_timestamp, code)`` list emitted by
        ``prediction_ready``. Timestamps share the prediction clock, so
        each marker rebases onto the same x-axis as the curves
        (``x = ts - latest_ts``). Codes absent from ``event_names`` (the
        ``markers_mapping.events`` set) are dropped — the receiver emits
        every non-zero trigger edge, but only configured events are drawn.
        """
        if not markers:
            return
        for ts, code in markers:
            code = int(code)
            if code not in self._event_names:
                continue
            self._markers.append({"ts": float(ts), "code": code, "line": None})
        # Backstop against unbounded growth (see _MAX_MARKERS). Drop the
        # oldest, removing any realised line from the scene.
        while len(self._markers) > _MAX_MARKERS:
            stale = self._markers.pop(0)
            if stale["line"] is not None:
                self.removeItem(stale["line"])

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
        for marker in self._markers:
            if marker["line"] is not None:
                self.removeItem(marker["line"])
        self._markers = []

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
        self._refresh_markers()

    def _refresh_markers(self) -> None:
        """Reposition cue markers onto the rebased x-axis, realising lines
        the first time they enter the window and dropping them once they
        scroll past the left edge. Called from ``_refresh`` so all marker
        scene mutation happens on the repaint tick, not the signal path.
        """
        if self._latest_ts is None:
            return
        left = -self._window_seconds
        kept: list[dict] = []
        for marker in self._markers:
            x = marker["ts"] - self._latest_ts
            if x < left:
                # Scrolled off the left edge — retire it.
                if marker["line"] is not None:
                    self.removeItem(marker["line"])
                continue
            if x <= _NOW_GAP_SECONDS:
                if marker["line"] is None:
                    marker["line"] = self._make_marker_line(marker["code"])
                    self.addItem(marker["line"])
                marker["line"].setPos(x)
            kept.append(marker)
        self._markers = kept

    def _make_marker_line(self, code: int) -> pg.InfiniteLine:
        """Build a black vertical line labelled with the configured event
        name for ``code`` (guaranteed present — unmapped codes are filtered
        out in :meth:`append_markers`). The label is bold black text in a
        solid white, black-bordered box for contrast against the curves."""
        line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=self._pen(_MARKER_COLOR, dashed=False, width=2.0),
            label=self._event_names.get(code, str(code)),
            labelOpts={
                "position": 0.95,
                "color": _MARKER_COLOR,
                "movable": False,
                # Solid white box with a black border — a crisp container
                # so the label stays legible over any curve behind it.
                "fill": (255, 255, 255, 235),
                "border": pg.mkPen(_MARKER_COLOR, width=1.0),
                "anchors": [(0.0, 0.5), (0.0, 0.5)],
            },
        )
        # Bold the label. InfLineLabel is a TextItem whose underlying
        # QGraphicsTextItem is ``.textItem``; the InfiniteLine reuses one
        # label, so styling it once here sticks for the marker's lifetime.
        font = line.label.textItem.font()
        font.setBold(True)
        line.label.textItem.setFont(font)
        return line

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

        # Pin labelled Y ticks to [0, 1.0] so the headroom above 1.0 (where
        # marker labels live) reads as blank margin, not a probability > 1.
        y_ticks = [
            (round(v, 1), f"{v:.1f}") for v in np.arange(0.0, _Y_TICK_MAX + 1e-9, 0.2)
        ]
        plot_item.getAxis("left").setTicks([y_ticks, []])

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
