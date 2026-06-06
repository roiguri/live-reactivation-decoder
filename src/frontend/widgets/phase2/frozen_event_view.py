"""Browsable wrapper around :class:`FrozenEventChart` (Goals 9 + 11).

The chart epochs the live stream around each trigger event and keeps a
newest-first history of snapshots. This view adds the **browsing control**: an
event dropdown, older/newer step buttons, and a "Latest" button that doubles as
a **live toggle**. The Latest button is *active* (filled green) while
live-following; clicking it then **deactivates** follow, pinning the current
event in place so incoming events no longer advance the view. Clicking it while
inactive goes live again — jump to the newest event and resume following. The
step buttons' enabled/disabled state marks the ends of the history.

Auto-follow lives in the chart (``following``): while on, a new event replaces
the on-screen one; with it off (either by picking an older event or by pinning
the newest), incoming events lengthen the history without yanking the display
away. This view only mirrors that state into its controls — the chart is the
single source of truth for what's shown and whether it's live.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY,
    CARD_WHITE,
    SUCCESS_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from frontend.widgets.phase2.frozen_event_chart import FrozenEventChart

# Shared control height so the combo, step buttons, Latest button, and status
# pill all line up. Matches the secondary-button visual weight elsewhere.
_CONTROL_H = 30

_COMBO_QSS = f"""
QComboBox {{
    background: {CARD_WHITE};
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
    padding: 3px 8px;
    font-size: 12px;
    color: {TEXT_PRIMARY};
}}
QComboBox:disabled {{
    color: {TEXT_MUTED};
    background: #F3F4F6;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
"""

# Compact square step buttons (older / newer). A QToolButton with a native
# arrow type centres the glyph perfectly regardless of font metrics — the
# previous text chevrons sat slightly off-centre.
_STEP_QSS = f"""
QToolButton {{
    background: {CARD_WHITE};
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
}}
QToolButton:hover:enabled {{ background: #F3F4F6; }}
QToolButton:disabled {{ background: #F3F4F6; }}
"""

# The Latest button doubles as the current-vs-past indicator. "Active" =
# the newest event is on screen (filled green, matching the header's LIVE
# colour). "Normal" = reviewing an earlier event, so it's an outlined,
# actionable button that jumps back to the latest.
_LATEST_ACTIVE_QSS = f"""
QPushButton {{
    background: {SUCCESS_GREEN};
    color: white;
    border: 1px solid {SUCCESS_GREEN};
    border-radius: 2px;
    padding: 0px 14px;
    font-size: 12px;
    font-weight: 600;
}}
"""
_LATEST_NORMAL_QSS = f"""
QPushButton {{
    background: {CARD_WHITE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
    padding: 0px 14px;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton:hover {{ background: #F3F4F6; }}
"""


class FrozenEventView(QWidget):
    """Dropdown-browsable event-locked chart.

    Forwards the data hot paths (``append_predictions`` / ``append_markers``)
    and lifecycle (``set_task_visible`` / ``reset_buffers``) straight to the
    inner :class:`FrozenEventChart`; owns only the browsing UI.
    """

    def __init__(
        self,
        task_names: list[str],
        *,
        pre_seconds: float = 0.2,
        post_seconds: float = 1.0,
        target_sfreq: float = 100.0,
        threshold: float = 0.85,
        event_names: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._chart = FrozenEventChart(
            task_names,
            pre_seconds=pre_seconds,
            post_seconds=post_seconds,
            target_sfreq=target_sfreq,
            threshold=threshold,
            event_names=event_names,
        )

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.setFixedHeight(_CONTROL_H)
        self._combo.setStyleSheet(_COMBO_QSS)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        # Older = step toward earlier events (higher index); newer = toward 0.
        self._older_btn = self._make_step_button(
            Qt.ArrowType.LeftArrow, "Older event", self._show_older
        )
        self._newer_btn = self._make_step_button(
            Qt.ArrowType.RightArrow, "Newer event", self._show_newer
        )

        self._latest_btn = QPushButton("Latest")
        self._latest_btn.setFixedHeight(_CONTROL_H)
        self._latest_btn.setMinimumWidth(84)
        self._latest_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._latest_btn.clicked.connect(self._on_latest_clicked)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self._combo, 1)
        controls.addWidget(self._older_btn)
        controls.addWidget(self._newer_btn)
        controls.addWidget(self._latest_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addWidget(self._chart, 1)

        self._chart.event_captured.connect(self._on_event_captured)
        self._sync_controls()  # initial empty state

    # ── forwarded data / lifecycle API ──────────────────────────────────────────

    @property
    def task_colors(self) -> dict[str, str]:
        return self._chart.task_colors

    @property
    def chart(self) -> FrozenEventChart:
        return self._chart

    def append_predictions(self, predictions: dict, timestamps: np.ndarray) -> None:
        self._chart.append_predictions(predictions, timestamps)

    def append_markers(self, markers: list[tuple[float, int]]) -> None:
        self._chart.append_markers(markers)

    def set_task_visible(self, name: str, visible: bool) -> None:
        self._chart.set_task_visible(name, visible)

    def reset_buffers(self) -> None:
        self._chart.reset_buffers()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("No events yet")
        self._combo.blockSignals(False)
        self._sync_controls()

    # ── browsing wiring ─────────────────────────────────────────────────────────

    def _on_event_captured(self, current_index: int) -> None:
        """A new event was frozen. Rebuild the dropdown from the chart's
        history and reflect the index the chart is now showing — without
        re-triggering a render (the chart already drew the right snapshot)."""
        labels = self._chart.history_labels()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(labels)
        self._combo.setCurrentIndex(current_index)
        self._combo.blockSignals(False)
        self._sync_controls()

    def _on_combo_changed(self, index: int) -> None:
        if index < 0 or not self._chart._history:
            return
        self._chart.show_event(index)
        self._sync_controls()

    def _show_older(self) -> None:
        self._select(self._combo.currentIndex() + 1)

    def _show_newer(self) -> None:
        self._select(self._combo.currentIndex() - 1)

    def _on_latest_clicked(self) -> None:
        """Toggle live-follow. When active (following the newest event),
        clicking *deactivates* — the current event stays put and incoming
        events no longer advance the view. When inactive, clicking goes live:
        jump to the newest event and resume following."""
        if self._chart.following:
            self._chart.set_following(False)  # pin: stay on current event
            self._sync_controls()
        else:
            self._jump_to_latest()

    def _jump_to_latest(self) -> None:
        """Go live: show the newest event and resume auto-follow. Used by the
        Latest button (inactive state) and available as the public 'home'."""
        self._chart.follow_latest()
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(0)
        self._combo.blockSignals(False)
        self._sync_controls()

    def _select(self, index: int) -> None:
        if 0 <= index < self._combo.count():
            self._combo.setCurrentIndex(index)  # fires _on_combo_changed

    def _sync_controls(self) -> None:
        """Reflect the chart's current index into the controls: enable/disable
        the step buttons and put the Latest button into its active (on-newest)
        or normal (reviewing) state."""
        n = len(self._chart._history)
        has_events = n > 0
        index = self._combo.currentIndex() if has_events else -1
        # Active = live-following. This can differ from "on the newest event":
        # the operator can pin the newest event (follow off while index == 0).
        active = has_events and self._chart.following

        self._combo.setEnabled(has_events)
        self._older_btn.setEnabled(has_events and index < n - 1)
        self._newer_btn.setEnabled(has_events and index > 0)

        # Latest button = the live indicator + toggle. Active (filled green)
        # while following; normal + actionable otherwise (resume / go live).
        self._latest_btn.setEnabled(has_events)
        self._latest_btn.setProperty("active", active)
        self._latest_btn.setText("● Latest" if active else "⤓ Latest")
        self._latest_btn.setStyleSheet(
            _LATEST_ACTIVE_QSS if active else _LATEST_NORMAL_QSS
        )

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _make_step_button(self, arrow: Qt.ArrowType, tip: str, slot) -> QToolButton:
        btn = QToolButton()
        btn.setArrowType(arrow)
        btn.setToolTip(tip)
        btn.setFixedSize(_CONTROL_H, _CONTROL_H)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(_STEP_QSS)
        btn.clicked.connect(slot)
        return btn
