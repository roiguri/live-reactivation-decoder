"""Temporal-generalization-matrix heatmap for the per-decoder tabs.

Shows ``tgm_matrix`` (shape ``(n_times, n_times)``) as a square image
with the x-axis = train time and y-axis = test time, both in ms. The
on-diagonal AUC at any time ``t`` is ``tgm[i, i]`` where ``times[i] = t``.

Colour map matches MNE's decoding convention: ``RdBu_r`` (red-blue
diverging, **red = high AUC**) with ``vmin=0, vmax=1`` so 0.5 (chance)
sits in the middle as near-white.

Caller-driven API mirrors :class:`AUCChart`::

    chart = TGMChart()
    chart.set_matrix(times, tgm)
    chart.set_selected_timepoint(0.130)      # moves crosshair
    chart.timepoint_clicked.connect(handler) # operator click → float seconds

Clicks snap to the nearest sample in ``times`` so the emitted second-
value is always one the backend evaluator actually scored.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QFont, QPen

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, TEXT_MUTED,
)

# ``pg.setConfigOptions`` is set globally in ``auc_chart`` — importing
# that module first (the package ``__init__`` does) is enough.


class TGMChart(pg.PlotWidget):
    """Heatmap of train-time × test-time AUC with a click-to-select crosshair."""

    timepoint_clicked = Signal(float)  # train time in seconds, snapped

    def __init__(self, parent=None) -> None:
        super().__init__(parent, background=CARD_WHITE)

        self._times: Optional[np.ndarray] = None
        self._image_item: pg.ImageItem = pg.ImageItem(axisOrder="row-major")
        self.addItem(self._image_item)

        # Cache the colour map (matplotlib's RdBu_r — MNE-style).
        self._set_colormap()

        self._configure_axes()
        self._suggested_v, self._suggested_h = self._add_crosshair(
            color=PRIMARY_BLUE, dashed=True, width=1.4
        )
        self._suggested_v.hide()
        self._suggested_h.hide()
        self._selected_v, self._selected_h = self._add_crosshair(
            color=PRIMARY_BLUE, dashed=False, width=1.8
        )
        self._selected_v.hide()
        self._selected_h.hide()

        self.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── public API ──────────────────────────────────────────────────────────

    def set_matrix(self, times: np.ndarray, tgm: np.ndarray) -> None:
        """Paint ``tgm`` (shape ``(n_times, n_times)``) onto the plot.

        ``tgm[i, j]`` = AUC obtained when training on ``times[i]`` and
        testing on ``times[j]``. The image is positioned so its bounds
        match the data range in ms (no separate axis-tick remapping
        needed).
        """
        self._times = np.asarray(times, dtype=float)
        arr = np.asarray(tgm, dtype=float)
        # pg.ImageItem with axisOrder='row-major' interprets ``arr[i,j]``
        # as "row i, column j" → row = train (y), col = test (x). For
        # the MNE-style "train on x-axis, test on y-axis" convention we
        # transpose so column = train, row = test.
        self._image_item.setImage(arr.T, autoLevels=False, levels=(0.0, 1.0))
        # Place the image so x/y axes read in ms.
        t_lo_ms = float(self._times[0]) * 1000.0
        t_hi_ms = float(self._times[-1]) * 1000.0
        self._image_item.setRect(t_lo_ms, t_lo_ms, t_hi_ms - t_lo_ms, t_hi_ms - t_lo_ms)
        self.setXRange(t_lo_ms, t_hi_ms, padding=0)
        self.setYRange(t_lo_ms, t_hi_ms, padding=0)
        # Aspect-lock keeps the matrix square; if the widget is taller
        # than wide (or vice versa), pyqtgraph "letterboxes" by
        # extending the shorter axis's *visible* range slightly past the
        # data — the heatmap stays square, only the axis tick labels go
        # a touch beyond the data range, which is harmless.

        # Reference markers stay above the image.
        for marker in (
            self._suggested_v, self._suggested_h,
            self._selected_v, self._selected_h,
        ):
            self.removeItem(marker)
            self.addItem(marker)

    def set_suggested_timepoint(self, t_seconds: float) -> None:
        """Dashed crosshair at ``(t, t)`` — the suggested point on the diagonal."""
        t_ms = t_seconds * 1000.0
        self._suggested_v.setPos(t_ms)
        self._suggested_h.setPos(t_ms)
        self._suggested_v.show()
        self._suggested_h.show()

    def set_selected_timepoint(self, t_seconds: float) -> None:
        """Solid crosshair at ``(t, t)`` — the operator's pick."""
        t_ms = t_seconds * 1000.0
        self._selected_v.setPos(t_ms)
        self._selected_h.setPos(t_ms)
        self._selected_v.show()
        self._selected_h.show()

    # ── private ─────────────────────────────────────────────────────────────

    def _set_colormap(self) -> None:
        """Use matplotlib's RdBu_r — same diverging palette MNE uses for
        decoding plots. Centred visually at 0.5 (chance) since vmin/vmax
        are 0/1.
        """
        try:
            cmap = pg.colormap.get("RdBu_r", source="matplotlib")
            self._image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:  # pragma: no cover — colormap unavailable
            # Fallback: pyqtgraph's built-in 'CET-D1' is a similar
            # diverging palette (blue → white → red).
            cmap = pg.colormap.get("CET-D1")
            if cmap is not None:
                self._image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

    def _configure_axes(self) -> None:
        plot = self.getPlotItem()
        plot.getViewBox().setBackgroundColor(CARD_WHITE)
        # Train and test time share the same range — lock the aspect
        # ratio at 1:1 so the rendered matrix is a square and the
        # diagonal genuinely looks 45°.
        plot.getViewBox().setAspectLocked(True, ratio=1.0)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.hideButtons()

        self.setLabel("left", "Test time (ms)", color=TEXT_MUTED)
        self.setLabel("bottom", "Train time (ms)", color=TEXT_MUTED)

        plot.showGrid(x=False, y=False)
        plot.getAxis("bottom").setGrid(False)
        plot.getAxis("left").setGrid(False)

        font = QFont()
        font.setPointSize(9)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(BORDER_GRAY))
            axis.setTextPen(pg.mkPen(TEXT_MUTED))
            axis.setStyle(tickFont=font)

    def _add_crosshair(
        self, *, color: str, dashed: bool, width: float
    ) -> tuple[pg.InfiniteLine, pg.InfiniteLine]:
        v = pg.InfiniteLine(pos=0.0, angle=90, pen=self._pen(color, dashed, width), movable=False)
        h = pg.InfiniteLine(pos=0.0, angle=0,  pen=self._pen(color, dashed, width), movable=False)
        self.addItem(v)
        self.addItem(h)
        return v, h

    @staticmethod
    def _pen(color: str, dashed: bool, width: float) -> QPen:
        pen = QPen(QColor(color))
        pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setWidthF(width)
        # Cosmetic pens stay in device pixels — same trap as AUCChart;
        # non-cosmetic dashed InfiniteLines paint stripes across the plot.
        pen.setCosmetic(True)
        return pen

    def _on_scene_clicked(self, ev) -> None:
        """Snap the click x to the nearest train-time sample and emit."""
        if self._times is None or len(self._times) == 0:
            return
        vb = self.getPlotItem().getViewBox()
        if vb is None or not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        mouse_pt = vb.mapSceneToView(ev.scenePos())
        clicked_s = float(mouse_pt.x()) / 1000.0
        idx = int(np.argmin(np.abs(self._times - clicked_s)))
        self.timepoint_clicked.emit(float(self._times[idx]))
