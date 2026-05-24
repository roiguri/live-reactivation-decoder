"""Phase 2 sidebar: stacked sections + a footer slot for the action button.

Sections:

* **Decoders** — one row per task: visibility checkbox + name + colour
  swatch. Toggling a checkbox emits :pyattr:`task_visibility_toggled`,
  which the screen wires to ``LiveProbabilityChart.set_task_visible``.
* **Decision Settings** — threshold + sustained-activation + conflict
  resolution. Inputs not yet exposed; section header reserves the slot.

Footer slot (see :pyattr:`footer_layout`) is where the Start/Halt
action button is dropped in once the live-stream wiring lands.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
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

_PANEL_WIDTH = 280
# Posix path keeps the QSS url() string portable across OSes.
_CHECKMARK_GREEN_URL = (
    Path(__file__).resolve().parents[2] / "styles" / "assets" / "checkmark_green.svg"
).as_posix()


class Phase2SettingsPanel(QWidget):
    """Phase 2 left panel: decoders + decision settings, with footer slot.

    Constructed with ``task_colors`` so the decoders section can render
    one row per task. The panel doesn't import the chart — it emits
    :pyattr:`task_visibility_toggled` and the screen wires it to
    ``chart.set_task_visible``.
    """

    task_visibility_toggled = Signal(str, bool)  # (task_name, visible)

    def __init__(
        self,
        task_colors: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedWidth(_PANEL_WIDTH)
        self.setObjectName("phase2_settings_panel")
        # WA_StyledBackground is required for a plain QWidget to actually
        # paint the border/background from its stylesheet (QFrame paints
        # by default; QWidget does not).
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # The decoder QCheckBoxes need an explicit ::indicator rule: once
        # any ancestor has a stylesheet, Qt switches QCheckBox from native
        # rendering to CSS rendering, and an empty CSS leaves the indicator
        # invisible on some platforms (notably Windows).
        self.setStyleSheet(
            f"QWidget#phase2_settings_panel {{"
            f"  background: #FAFAFA; border-right: 1px solid {BORDER_GRAY};"
            f"}}"
            f"QWidget#phase2_settings_panel QCheckBox {{"
            f"  background: transparent; spacing: 6px;"
            f"}}"
            f"QWidget#phase2_settings_panel QCheckBox::indicator {{"
            f"  width: 14px; height: 14px;"
            f"  border: 1px solid {BORDER_GRAY};"
            f"  border-radius: 2px;"
            f"  background: {CARD_WHITE};"
            f"}}"
            f"QWidget#phase2_settings_panel QCheckBox::indicator:hover {{"
            f"  border-color: {PRIMARY_BLUE};"
            f"}}"
            f"QWidget#phase2_settings_panel QCheckBox::indicator:checked {{"
            f"  background: {CARD_WHITE};"
            f"  border-color: {SUCCESS_GREEN};"
            f"  image: url({_CHECKMARK_GREEN_URL});"
            f"}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scrollable body so growing section content doesn't push the
        # footer's action button off-screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_decoders_section(task_colors))
        body_layout.addWidget(self._build_decision_settings_section())
        body_layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # Footer slot for the Start/Halt action button. Exposed via
        # ``footer_layout`` so callers can drop a button widget in
        # without touching the panel's structure.
        self._footer = QFrame()
        self._footer.setStyleSheet(
            f"background: {CARD_WHITE}; border-top: 1px solid {BORDER_GRAY};"
        )
        self._footer_layout = QVBoxLayout(self._footer)
        self._footer_layout.setContentsMargins(16, 16, 16, 16)
        self._footer_layout.setSpacing(0)
        root.addWidget(self._footer)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def footer_layout(self) -> QVBoxLayout:
        """Layout the caller should ``addWidget`` the action button to."""
        return self._footer_layout

    # ── sections ──────────────────────────────────────────────────────────────

    def _build_decoders_section(self, task_colors: dict[str, str]) -> QWidget:
        section = QWidget()
        section.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_section_header("Decoders"))

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 12, 16, 12)
        body_layout.setSpacing(8)
        for name, hex_color in task_colors.items():
            body_layout.addWidget(self._build_decoder_row(name, hex_color))
        layout.addWidget(body)
        return section

    def _build_decoder_row(self, name: str, hex_color: str) -> QWidget:
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY};"
            f" border-radius: 2px; }}"
        )
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.toggled.connect(
            lambda checked, n=name: self.task_visibility_toggled.emit(n, checked)
        )
        layout.addWidget(checkbox)

        label = QLabel(name)
        f = label.font()
        f.setPointSize(10)
        f.setWeight(QFont.Weight.Medium)
        label.setFont(f)
        label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent; border: none;"
        )
        layout.addWidget(label, 1)

        # Swatch is non-interactive for now; per-decoder colour picker
        # is a planned follow-up.
        swatch = QFrame()
        swatch.setFixedSize(20, 20)
        swatch.setStyleSheet(
            f"background: {hex_color}; border: 1px solid {BORDER_GRAY};"
            f" border-radius: 10px;"
        )
        layout.addWidget(swatch)
        return row

    def _build_decision_settings_section(self) -> QWidget:
        section = QWidget()
        section.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_section_header("Decision Settings"))

        placeholder = QLabel(
            "Threshold, sustained activation, and\nconflict-resolution rules will appear here."
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent; padding: 16px;"
        )
        layout.addWidget(placeholder)
        return section


# ── module helpers ────────────────────────────────────────────────────────────


def _section_header(title: str) -> QWidget:
    """Compact uppercase section divider matching the demo's section titles."""
    header = QFrame()
    header.setFixedHeight(36)
    header.setStyleSheet(
        f"background: {CARD_WHITE}; border-bottom: 1px solid {BORDER_GRAY};"
    )
    layout = QHBoxLayout(header)
    layout.setContentsMargins(16, 0, 16, 0)
    label = QLabel(title.upper())
    f = label.font()
    f.setPointSize(9)
    f.setWeight(QFont.Weight.DemiBold)
    label.setFont(f)
    label.setStyleSheet(
        f"color: {TEXT_MUTED}; background: transparent; letter-spacing: 1px;"
    )
    layout.addWidget(label)
    layout.addStretch(1)
    return header
