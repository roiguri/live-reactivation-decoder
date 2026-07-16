"""Browsable wrapper around :class:`FrozenEventChart` (Goals 9 + 11).

The chart epochs the live stream around each trigger event and keeps a
newest-first history of snapshots. This view adds the **browsing controls**:

* an event dropdown + older/newer step buttons to walk the history;
* a "Latest" button that doubles as a **live toggle** — active (filled green)
  while live-following; clicking it deactivates follow (pinning the current
  event so incoming events no longer advance the view), and clicking it while
  inactive goes live again;
* an **event filter** button: a small menu of the configured event types with
  Select-all / Clear-all, choosing which events the dropdown presents.

Filtering is **display-only** — every event is still captured into the chart's
history. The dropdown shows the filtered subset, and auto-follow tracks the
newest *visible* event, so toggling a type re-reveals or hides past events
without losing anything. The chart owns history + the filter + follow state;
this view maps each dropdown row back to its history index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY,
    CARD_WHITE,
    PRIMARY_BLUE,
    SUCCESS_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from frontend.widgets.phase2.frozen_event_chart import FrozenEventChart

# Shared control height so the combo, step buttons, filter, and Latest button
# all line up. Matches the secondary-button visual weight elsewhere.
_CONTROL_H = 30

# Funnel icons for the filter button (normal dark + active blue), as SVG assets
# alongside the app's other icons.
_ASSETS = Path(__file__).resolve().parents[2] / "styles" / "assets"
_FILTER_ICON = str(_ASSETS / "filter.svg")
_FILTER_ICON_ACTIVE = str(_ASSETS / "filter_active.svg")

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
# arrow type centres the glyph perfectly regardless of font metrics.
_STEP_QSS = f"""
QToolButton {{
    background: {CARD_WHITE};
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
}}
QToolButton:hover:enabled {{ background: #F3F4F6; }}
QToolButton:disabled {{ background: #F3F4F6; }}
"""

# Filter button (funnel icon). Outlined normally; tinted blue with a "k/n"
# count when a filter is active (some event types hidden) so the operator sees
# at a glance that not everything is shown. ``menu-indicator: none`` hides the
# default popup arrow — the funnel icon already signals it opens a menu.
_FILTER_BASE = (
    "border-radius: 2px; padding: 0px 8px; font-size: 12px; font-weight: 600;"
)
_FILTER_QSS = f"""
QToolButton {{
    background: {CARD_WHITE}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GRAY}; {_FILTER_BASE}
}}
QToolButton:hover {{ background: #F3F4F6; }}
QToolButton::menu-indicator {{ image: none; width: 0px; }}
"""
_FILTER_ACTIVE_QSS = f"""
QToolButton {{
    background: #EFF6FF; color: {PRIMARY_BLUE};
    border: 1px solid {PRIMARY_BLUE}; {_FILTER_BASE}
}}
QToolButton::menu-indicator {{ image: none; width: 0px; }}
"""

# The Latest button doubles as the live indicator + toggle. "Active" = the view
# is live-following (filled green, matching the header's LIVE colour). "Normal"
# = paused/reviewing — outlined and actionable to go live again.
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


class _StayOpenMenu(QMenu):
    """A QMenu that does not close when a "keep-open" action is clicked, so the
    operator can toggle several event filters (and Select/Clear all) in one
    visit. Actions opt in via ``action.setData("keepopen")``."""

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt override)
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.data() == "keepopen":
            action.trigger()
            return  # swallow the release so the menu stays open
        super().mouseReleaseEvent(event)


class FrozenEventView(QWidget):
    """Dropdown-browsable, filterable event-locked chart.

    Forwards the data hot paths (``append_predictions`` / ``append_markers``)
    and lifecycle (``set_task_visible`` / ``reset_buffers``) straight to the
    inner :class:`FrozenEventChart`; owns only the browsing + filter UI.
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
        # Distinct configured event names, for the filter menu.
        self._event_filter_names = sorted(set((event_names or {}).values()))
        # Combo rows for the current filter: (history_index, label), newest-first.
        self._visible: list[tuple[int, str]] = []
        self._event_actions: dict[str, QAction] = {}
        self._filter_btn: QToolButton | None = None

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
        # Filter sits just right of the dropdown; only useful with >1 type.
        if len(self._event_filter_names) >= 2:
            self._filter_btn = self._build_filter_button()
            controls.addWidget(self._filter_btn)
        controls.addWidget(self._older_btn)
        controls.addWidget(self._newer_btn)
        controls.addWidget(self._latest_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addWidget(self._chart, 1)

        self._chart.event_captured.connect(self._on_event_captured)
        self._rebuild_combo()  # initial empty state

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

    def set_threshold(self, value: float) -> None:
        self._chart.set_threshold(value)

    def set_task_visible(self, name: str, visible: bool) -> None:
        self._chart.set_task_visible(name, visible)

    def reset_buffers(self) -> None:
        # The event filter is an operator preference — keep it across Start.
        self._chart.reset_buffers()
        self._rebuild_combo()

    # ── browsing wiring ─────────────────────────────────────────────────────────

    def _on_event_captured(self, _current_index: int) -> None:
        """A new event was frozen (or the filter changed). Rebuild the dropdown
        from the chart's filtered history and reflect what the chart is showing
        — the chart already drew the right snapshot, so this only syncs the UI.
        """
        self._rebuild_combo()

    def _rebuild_combo(self) -> None:
        self._visible = self._chart.visible_history()
        self._combo.blockSignals(True)
        self._combo.clear()
        if self._visible:
            self._combo.addItems([label for _, label in self._visible])
            pos = self._position_of(self._chart.current_index)
            self._combo.setCurrentIndex(pos if pos is not None else 0)
            self._combo.setEnabled(True)
        else:
            # Distinguish "nothing captured yet" from "everything filtered out".
            empty = "No events yet" if not self._chart.history_labels() else "No matching events"
            self._combo.addItem(empty)
            self._combo.setEnabled(False)
        self._combo.blockSignals(False)
        self._sync_controls()

    def _on_combo_changed(self, pos: int) -> None:
        if pos < 0 or not self._visible:
            return
        self._chart.show_event(self._visible[pos][0])
        self._sync_controls()

    def _show_older(self) -> None:
        self._select(self._combo.currentIndex() + 1)

    def _show_newer(self) -> None:
        self._select(self._combo.currentIndex() - 1)

    def _on_latest_clicked(self) -> None:
        """Toggle live-follow. Active → deactivate (pin the current event so
        incoming events don't advance the view). Inactive → go live: jump to
        the newest visible event and resume following."""
        if self._chart.following:
            self._chart.set_following(False)  # pin
            self._sync_controls()
        else:
            self._jump_to_latest()

    def _jump_to_latest(self) -> None:
        """Go live: show the newest visible event and resume auto-follow."""
        self._chart.follow_latest()
        self._reflect_position()
        self._sync_controls()

    def _select(self, pos: int) -> None:
        if self._visible and 0 <= pos < len(self._visible):
            self._combo.setCurrentIndex(pos)  # fires _on_combo_changed

    def _reflect_position(self) -> None:
        """Set the combo to the row showing the chart's current event, without
        re-triggering a render."""
        pos = self._position_of(self._chart.current_index)
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(pos if pos is not None else 0)
        self._combo.blockSignals(False)

    def _position_of(self, history_index: int | None) -> int | None:
        if history_index is None:
            return None
        for pos, (hi, _) in enumerate(self._visible):
            if hi == history_index:
                return pos
        return None

    def _sync_controls(self) -> None:
        """Enable/disable the step buttons by position within the visible list
        and put the Latest button into its active (live) or normal state."""
        n = len(self._visible)
        has_events = n > 0
        pos = self._combo.currentIndex() if has_events else -1
        # Active = live-following (can differ from "newest event": the newest
        # can be pinned, follow off while on it).
        active = has_events and self._chart.following

        self._older_btn.setEnabled(has_events and pos < n - 1)
        self._newer_btn.setEnabled(has_events and pos > 0)

        self._latest_btn.setEnabled(has_events)
        self._latest_btn.setProperty("active", active)
        self._latest_btn.setText("● Latest" if active else "⤓ Latest")
        self._latest_btn.setStyleSheet(
            _LATEST_ACTIVE_QSS if active else _LATEST_NORMAL_QSS
        )

    # ── event filter ─────────────────────────────────────────────────────────────

    def _build_filter_button(self) -> QToolButton:
        menu = _StayOpenMenu(self)
        for name in self._event_filter_names:
            act = QAction(name, menu)
            act.setCheckable(True)
            act.setChecked(True)
            act.setData("keepopen")
            act.toggled.connect(self._on_filter_changed)
            menu.addAction(act)
            self._event_actions[name] = act
        menu.addSeparator()
        for label, slot in (
            ("Select all", self._select_all_events),
            ("Clear all", self._clear_all_events),
        ):
            act = QAction(label, menu)
            act.setData("keepopen")
            act.triggered.connect(slot)
            menu.addAction(act)

        btn = QToolButton()
        btn.setIcon(QIcon(_FILTER_ICON))
        btn.setIconSize(QSize(15, 15))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        btn.setToolTip("Filter which events are shown")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setMenu(menu)
        btn.setFixedHeight(_CONTROL_H)
        btn.setMinimumWidth(_CONTROL_H)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(_FILTER_QSS)
        btn.setProperty("active", False)  # all selected by default
        return btn

    def _on_filter_changed(self, _checked: bool = False) -> None:
        allowed = {n for n, a in self._event_actions.items() if a.isChecked()}
        # All selected → no filter (None) so the chart skips filtering work.
        full = len(allowed) == len(self._event_actions)
        self._chart.set_event_filter(None if full else allowed)
        self._rebuild_combo()
        self._update_filter_button()

    def _select_all_events(self, _checked: bool = False) -> None:
        self._set_all_events(True)

    def _clear_all_events(self, _checked: bool = False) -> None:
        self._set_all_events(False)

    def _set_all_events(self, checked: bool) -> None:
        for act in self._event_actions.values():
            act.blockSignals(True)
            act.setChecked(checked)
            act.blockSignals(False)
        self._on_filter_changed()

    def _update_filter_button(self) -> None:
        if self._filter_btn is None:
            return
        total = len(self._event_actions)
        selected = sum(1 for a in self._event_actions.values() if a.isChecked())
        filtering = selected != total
        btn = self._filter_btn
        if filtering:
            # Blue funnel + a compact "k/n" count beside it so the operator
            # sees a filter is active and how many types are shown.
            btn.setIcon(QIcon(_FILTER_ICON_ACTIVE))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setText(f"{selected}/{total}")
            btn.setStyleSheet(_FILTER_ACTIVE_QSS)
        else:
            btn.setIcon(QIcon(_FILTER_ICON))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            btn.setText("")
            btn.setStyleSheet(_FILTER_QSS)
        btn.setProperty("active", filtering)

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
