"""Single-decoder topomap embedded in a Qt widget.

Used by the Train view (Node 5) to show one spatial pattern per decoder
in a side-by-side grid. The topomap itself is drawn via
``mne.viz.plot_topomap`` — pyqtgraph has no equivalent renderer, so we
embed matplotlib via ``FigureCanvas``.

Caller-driven API::

    topo = TopomapWidget()
    topo.set_pattern(pattern, info, title="red decoder")
"""
from __future__ import annotations

from typing import Optional

import matplotlib
import mne
import numpy as np

# Force the Qt backend before importing FigureCanvas. Without this,
# matplotlib may have already initialised a different backend at import
# time and the embedded canvas won't render.
matplotlib.use("QtAgg", force=False)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtWidgets import QSizePolicy  # noqa: E402

from frontend.styles.theme import CARD_WHITE  # noqa: E402


class TopomapWidget(FigureCanvas):
    """Single-pattern topomap drawn via ``mne.viz.plot_topomap``."""

    def __init__(self, parent=None) -> None:
        # 1:1 figure aspect; the widget itself stays square via heightForWidth.
        fig = Figure(figsize=(3, 3), dpi=100, facecolor=CARD_WHITE)
        super().__init__(fig)
        self.setParent(parent)
        self._ax = fig.add_subplot(111)
        self._ax.set_axis_off()
        fig.tight_layout()

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Reasonable floor so a row of these doesn't collapse.
        self.setMinimumSize(140, 160)

    # ── public API ──────────────────────────────────────────────────────────

    def set_pattern(
        self,
        pattern: np.ndarray,
        info: mne.Info,
        *,
        title: Optional[str] = None,
    ) -> None:
        """Render the spatial pattern as a topomap.

        ``pattern`` is a 1-D array shaped ``(n_channels,)`` aligned with
        ``info["chs"]``. ``info`` carries the sensor positions
        ``mne.viz.plot_topomap`` needs.
        """
        self._ax.clear()
        self._ax.set_axis_off()
        # ``sphere='auto'`` lets MNE pick montage-appropriate head outline.
        # show=False because we manage the canvas refresh ourselves.
        mne.viz.plot_topomap(
            np.asarray(pattern, dtype=float),
            info,
            axes=self._ax,
            show=False,
            cmap="RdBu_r",
            sensors=True,
            contours=4,
            sphere="auto",
        )
        if title:
            self._ax.set_title(title, fontsize=10, pad=6)
        self.figure.tight_layout()
        self.draw_idle()
