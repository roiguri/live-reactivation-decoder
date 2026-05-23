from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QFont

from backend.session import AppSession
from frontend.styles.theme import BG_LIGHT, BORDER_GRAY, CARD_WHITE, TEXT_MUTED
from frontend.widgets.live_probability_chart import LiveProbabilityChart
from frontend.widgets.phase2 import Phase2Header, Phase2SettingsPanel


# TODO: wire a Back button. Unresolved: which screen Back lands on
# (Node 5 results vs. journey reset) and what to do with a running
# stream (auto-halt vs. confirm). One-way for now — restart the app
# to leave Phase 2.
#
# TODO: threshold is hardcoded; the config schema has no
# ``decoders.threshold`` field yet. Once it does, read it from
# ``session.settings["decoders"]["threshold"]``.
_DEFAULT_THRESHOLD = 0.85
_CHART_MAX_HEIGHT = 420


class Phase2Screen(QWidget):
    """Live-inference screen. Layout glue only — each panel lives in its
    own module under ``frontend.widgets.phase2``.

    Body layout: ``[settings_panel | chart_panel]`` under
    :class:`Phase2Header`. The chart sits in a card with a max height
    so it reads as its intended size rather than stretching to fill
    the window.
    """

    def __init__(
        self,
        session: AppSession,
        decoder_pipeline_path: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.decoder_pipeline_path = Path(decoder_pipeline_path)
        self.setObjectName("phase2_screen")

        settings = session.settings
        task_names = [t["name"] for t in settings["decoders"]["tasks"]]
        target_sfreq = float(
            settings["preprocessing"]["final_resample"]["target_rate"]
        )

        self._chart = LiveProbabilityChart(
            task_names=task_names,
            target_sfreq=target_sfreq,
            threshold=_DEFAULT_THRESHOLD,
        )
        self._header = Phase2Header()
        self._settings_panel = Phase2SettingsPanel(
            task_colors=self._chart.task_colors
        )
        self._settings_panel.task_visibility_toggled.connect(
            self._chart.set_task_visible
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header)

        body = QWidget()
        body.setStyleSheet(f"background: {BG_LIGHT};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._settings_panel)
        body_layout.addWidget(self._build_chart_panel(), 1)
        root.addWidget(body, 1)

    # ── center panel ──────────────────────────────────────────────────────────

    def _build_chart_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG_LIGHT};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(8)

        title = QLabel("PROBABILITY ANALYSIS")
        tf = title.font()
        tf.setPointSize(9)
        tf.setWeight(QFont.Weight.DemiBold)
        title.setFont(tf)
        title.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent; letter-spacing: 1px;"
        )
        layout.addWidget(title)

        chart_card = QFrame()
        chart_card.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; }}"
        )
        chart_card.setMaximumHeight(_CHART_MAX_HEIGHT)
        card_layout = QVBoxLayout(chart_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.addWidget(self._chart)
        layout.addWidget(chart_card)
        # Stretch keeps the chart card pinned to the top at its
        # intended height; leaves room below for future content.
        layout.addStretch(1)
        return panel
