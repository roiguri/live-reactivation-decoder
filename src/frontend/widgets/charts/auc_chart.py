"""Interactive AUC-over-time line chart for the Evaluation results screen.

Used in two places:

* The **Summary** tab plots all decoders together so the operator can
  see where they peak jointly vs. individually.
* Each **per-decoder** tab plots just that one decoder's curve, for
  closer inspection.

Caller-driven API — the widget knows nothing about the orchestrator or
the eval result schema; the view passes in raw arrays::

    chart = AUCChart()
    chart.set_curves(times, {"red decoder": diag_red, ...})
    chart.set_suggested_timepoint(0.130)   # dashed blue stationary marker
    chart.set_selected_timepoint(0.130)    # solid blue movable marker
    chart.timepoint_clicked.connect(handler)

Click handling snaps to the nearest sample in ``times`` so the emitted
value is always one the backend actually evaluated.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QFont, QPen

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, TEXT_MUTED, chart_line_color,
)

# Module-level config so every chart uses a white background + black axes.
# Must be set BEFORE the first ``pg.PlotWidget`` is constructed — the
# constructor's ``background`` arg only covers the widget surface; this
# also sets the viewbox + foreground defaults.
pg.setConfigOptions(background="w", foreground="k", antialias=True)

_Y_RANGE = (0.0, 1.0)  # full AUC range — values below 0.5 mean worse than chance
_CHANCE_LEVEL = 0.5


class AUCChart(pg.PlotWidget):
    """AUC-over-time chart with click-to-select-timepoint support.

    Three reference lines layered on top of the data:

    * Dashed grey horizontal at ``y=0.5`` — the chance line.
    * Dashed blue vertical at the **suggested** timepoint (stationary,
      shown once results land — set via ``set_suggested_timepoint``).
    * Solid blue vertical at the **selected** timepoint (movable, set via
      ``set_selected_timepoint`` or implicitly when the operator clicks).
    """

    timepoint_clicked = Signal(float)  # t in seconds, snapped to nearest sample

    def __init__(self, parent=None, *, show_legend: bool = True) -> None:
        super().__init__(parent, background=CARD_WHITE)

        self._times: Optional[np.ndarray] = None
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._legend: Optional[pg.LegendItem] = None
        self._show_legend = show_legend

        self._configure_axes()
        # Stimulus-onset reference at t = 0 ms — a dashed vertical line
        # in the same muted grey as the chance line. ``_add_v_line``
        # places the line at x=0 by default; the stimulus marker never
        # needs to move so we don't keep a reference to it.
        self._add_v_line(color=TEXT_MUTED, dashed=True, width=1.0)
        self._chance_line = self._add_h_line(
            y=_CHANCE_LEVEL, color=TEXT_MUTED, dashed=True, width=1.0
        )
        # Dashed reference at the algorithm-suggested timepoint. Hidden
        # until the view calls ``set_suggested_timepoint``.
        self._suggested_marker = self._add_v_line(
            color=PRIMARY_BLUE, dashed=True, width=1.4
        )
        self._suggested_marker.hide()
        # Solid line at the operator-selected timepoint. Same blue as
        # ``SELECTED TIMEPOINT`` in the stats panel.
        self._selected_marker = self._add_v_line(
            color=PRIMARY_BLUE, dashed=False, width=1.8
        )
        self._selected_marker.hide()

        self.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── public API ──────────────────────────────────────────────────────────

    def set_curves(
        self,
        times: np.ndarray,
        curves: Mapping[str, np.ndarray],
        *,
        colors: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Replace all plotted lines.

        ``colors`` optionally maps decoder name → hex colour. When
        omitted, each curve gets the i-th colour of
        :data:`CHART_LINE_COLORS` by insertion order. The caller passes
        ``colors`` when the same decoder must keep a stable colour across
        multiple charts (e.g. the per-decoder tab uses the same hue the
        Summary tab assigned).
        """
        self._times = np.asarray(times, dtype=float)
        for item in self._curves.values():
            self.removeItem(item)
        self._curves.clear()
        if self._legend is not None:
            scene = self._legend.scene()
            if scene is not None:
                scene.removeItem(self._legend)
            self._legend = None

        if self._show_legend and curves:
            self._legend = self.addLegend(
                offset=(-10, 10),
                pen=pg.mkPen(BORDER_GRAY),
                brush=pg.mkBrush(QColor(255, 255, 255, 230)),
                labelTextSize="9pt",
            )

        for i, (name, diag) in enumerate(curves.items()):
            color = colors[name] if colors and name in colors else chart_line_color(i)
            pen = pg.mkPen(color=color, width=2)
            item = self.plot(
                self._times * 1000.0,
                np.asarray(diag, dtype=float),
                pen=pen,
                name=name,
            )
            self._curves[name] = item

        # Lock x to the data range so the chart doesn't leave dead space
        # past the last sample.
        if len(self._times) > 0:
            self.setXRange(
                float(self._times[0]) * 1000.0,
                float(self._times[-1]) * 1000.0,
                padding=0,
            )

        # Keep the reference lines painted on top of the freshly added curves.
        for marker in (self._chance_line, self._suggested_marker, self._selected_marker):
            self.removeItem(marker)
            self.addItem(marker)

    def set_suggested_timepoint(self, t_seconds: float) -> None:
        """Place the dashed blue stationary marker. Called once per result."""
        self._suggested_marker.setPos(t_seconds * 1000.0)
        self._suggested_marker.show()

    def set_selected_timepoint(self, t_seconds: float) -> None:
        """Move the solid blue marker to ``t_seconds`` and show it."""
        self._selected_marker.setPos(t_seconds * 1000.0)
        self._selected_marker.show()

    # ── private ─────────────────────────────────────────────────────────────

    def _configure_axes(self) -> None:
        plot = self.getPlotItem()
        plot.getViewBox().setBackgroundColor(CARD_WHITE)

        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()

        # padding > 0 leaves room above 1.0 so pyqtgraph doesn't hide
        # the top tick label when it would clip the plot boundary.
        self.setYRange(*_Y_RANGE, padding=0.03)
        self.setLabel("left", "AUC", color=TEXT_MUTED)
        self.setLabel("bottom", "Time (ms)", color=TEXT_MUTED)

        # Kill all grids first. pyqtgraph treats ``setGrid(0)`` as
        # "enabled with alpha=0" (still drawn, somehow visible) — only
        # ``setGrid(False)`` truly disables. We then re-add subtle
        # horizontal grid lines by hand.
        plot.showGrid(x=False, y=False)
        plot.getAxis("bottom").setGrid(False)
        plot.getAxis("left").setGrid(False)

        font = QFont()
        font.setPointSize(9)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(BORDER_GRAY))
            axis.setTextPen(pg.mkPen(TEXT_MUTED))
            # Leave tickLength at the pyqtgraph default (-5, ticks point
            # OUTWARD from the axis line into the margin). An earlier
            # version flipped this to a positive value to "move ticks
            # outside" — but positive actually means *inward*, which
            # painted dense vertical bars across the plot area.
            axis.setStyle(tickFont=font)

        # Subtle horizontal reference lines at the major Y grid positions,
        # drawn as our own InfiniteLines (so pyqtgraph's grid system is
        # never involved). Cosmetic pen so the 0.8 px width stays in
        # device pixels rather than view units (same trap as the chance
        # line — see ``_pen``).
        gridline_pen = QPen(QColor(BORDER_GRAY))
        gridline_pen.setWidthF(0.8)
        gridline_pen.setCosmetic(True)
        for y in (0.2, 0.4, 0.6, 0.8):
            self.addItem(pg.InfiniteLine(pos=y, angle=0, pen=gridline_pen, movable=False))

    def _add_h_line(self, *, y: float, color: str, dashed: bool, width: float) -> pg.InfiniteLine:
        line = pg.InfiniteLine(pos=y, angle=0, pen=self._pen(color, dashed, width), movable=False)
        self.addItem(line)
        return line

    def _add_v_line(self, *, color: str, dashed: bool, width: float) -> pg.InfiniteLine:
        line = pg.InfiniteLine(pos=0.0, angle=90, pen=self._pen(color, dashed, width), movable=False)
        self.addItem(line)
        return line

    @staticmethod
    def _pen(color: str, dashed: bool, width: float) -> QPen:
        pen = QPen(QColor(color))
        pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setWidthF(width)
        # COSMETIC pens have width + dash pattern in device pixels rather
        # than view coordinates. Without this, a horizontal InfiniteLine
        # with a dashed pen renders its dash pattern in the y-axis units
        # (0..1 AUC) — which the view scales to ~400 px — and ends up
        # painting dense vertical bars across the entire plot area.
        pen.setCosmetic(True)
        return pen

    def _on_scene_clicked(self, ev) -> None:
        """Map the click position to a time, snap to nearest sample, emit."""
        if self._times is None or len(self._times) == 0:
            return
        vb = self.getPlotItem().getViewBox()
        if vb is None or not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        mouse_pt = vb.mapSceneToView(ev.scenePos())
        clicked_s = float(mouse_pt.x()) / 1000.0
        idx = int(np.argmin(np.abs(self._times - clicked_s)))
        self.timepoint_clicked.emit(float(self._times[idx]))
