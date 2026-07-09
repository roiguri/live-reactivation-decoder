"""Live decision panel: a single row of per-decoder tiles.

Decoders latch **independently**, so several tiles can be "active" at once (the
multi-active model made visible). Each tile shows the decoder's name; its state is
conveyed purely by the highlight — an active tile is bordered and tinted in the
decoder's own colour, an idle tile is a plain muted card.

The panel is driven by :class:`Phase2Screen` from the decision stream
(:meth:`update_decision`). It imports no backend types: the ``DecisionResult`` is
read duck-typed (only ``.active``), exactly as the chart reads the raw prediction
``dict``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from frontend.styles.theme import (
    BORDER_GRAY,
    CARD_WHITE,
    TEXT_MUTED,
)


class DecisionPanel(QWidget):
    """A row of decoder tiles; the active ones light up in their own colour.

    Constructed with ``task_colors`` (decoder name → hex), matching the chart's
    palette so a lit tile is the decoder's own colour.
    """

    def __init__(
        self,
        task_colors: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

        self._tiles: dict[str, _DecoderTile] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for name, color in task_colors.items():
            tile = _DecoderTile(name, color)
            self._tiles[name] = tile
            layout.addWidget(tile, 1)  # equal-width tiles across one row

    # ── public API ────────────────────────────────────────────────────────────

    def update_decision(self, result: Any) -> None:
        """Set each tile's state from the latest sample of ``result.active``."""
        active = result.active
        for name, tile in self._tiles.items():
            values = active.get(name)
            if values is None or len(values) == 0:
                continue
            tile.set_active(bool(np.asarray(values)[-1]))

    def reset(self) -> None:
        """Clear all tiles (a fresh run)."""
        for tile in self._tiles.values():
            tile.set_active(False)

    def is_active(self, name: str) -> bool:
        tile = self._tiles.get(name)
        return bool(tile.active) if tile is not None else False

    def active_decoders(self) -> set[str]:
        return {name for name, tile in self._tiles.items() if tile.active}


class _DecoderTile(QFrame):
    """A single decoder's tile: just the name.

    State is conveyed purely by the tile's highlight — an active tile is bordered
    and tinted in the decoder's colour; an idle tile is a plain muted card. Holds
    its own ``active`` state so the panel (and tests) can read it back without
    inspecting child widgets.
    """

    def __init__(self, name: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        rgb = QColor(color)
        self._tint = f"rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, 28)"
        self.active: bool = False

        self.setMinimumHeight(48)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)

        self._name_label = QLabel(name)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nf = self._name_label.font()
        nf.setPointSize(11)
        nf.setWeight(QFont.Weight.DemiBold)
        self._name_label.setFont(nf)
        layout.addWidget(self._name_label)

        self._restyle()

    def set_active(self, active: bool) -> None:
        self.active = active
        self._restyle()

    def _restyle(self) -> None:
        if self.active:
            self.setStyleSheet(
                f"QFrame {{ background: {self._tint};"
                f" border: 2px solid {self._color}; border-radius: 4px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame {{ background: {CARD_WHITE};"
                f" border: 1px solid {BORDER_GRAY}; border-radius: 4px; }}"
            )
        # Name colour tracks state — coloured when active, muted when idle.
        self._name_label.setStyleSheet(
            f"color: {self._color if self.active else TEXT_MUTED};"
            f" background: transparent; border: none;"
        )
