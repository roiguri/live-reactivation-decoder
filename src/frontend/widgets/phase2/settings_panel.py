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
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
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

# Apply/Reset button styling. Enabled and disabled states are visually distinct:
# Apply is a filled primary button that greys out when there's nothing to apply;
# Reset is an outline button that fades when clean.
_PRIMARY_BUTTON_QSS = (
    f"QPushButton {{ background: {PRIMARY_BLUE}; color: {CARD_WHITE};"
    f" border: none; border-radius: 4px; padding: 7px 14px; font-weight: 600; }}"
    f"QPushButton:hover {{ background: #2563EB; }}"
    f"QPushButton:disabled {{ background: #E5E7EB; color: {TEXT_MUTED}; }}"
)
_SECONDARY_BUTTON_QSS = (
    f"QPushButton {{ background: {CARD_WHITE}; color: {TEXT_PRIMARY};"
    f" border: 1px solid {BORDER_GRAY}; border-radius: 4px; padding: 7px 14px; }}"
    f"QPushButton:hover {{ border-color: {PRIMARY_BLUE}; color: {PRIMARY_BLUE}; }}"
    f"QPushButton:disabled {{ background: {CARD_WHITE}; color: #C7CCD1;"
    f" border-color: #EDEFF1; }}"
)


class Phase2SettingsPanel(QWidget):
    """Phase 2 left panel: decoders + decision settings, with footer slot.

    Constructed with ``task_colors`` so the decoders section can render
    one row per task. The panel doesn't import the chart — it emits
    :pyattr:`task_visibility_toggled` and the screen wires it to
    ``chart.set_task_visible``.
    """

    task_visibility_toggled = Signal(str, bool)  # (task_name, visible)
    decision_params_changed = Signal(dict)  # {"threshold", "sustain_timepoints"} on Apply

    def __init__(
        self,
        task_colors: dict[str, str],
        decision_defaults: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Applied decision settings; controls edit a draft, Apply commits it here.
        self._applied = dict(
            decision_defaults or {"threshold": 0.80, "sustain_timepoints": 3}
        )
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

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 12, 16, 12)
        body_layout.setSpacing(10)

        # Threshold: an editable spinbox for precise manual entry, plus a slider
        # for quick dragging — the two are kept in sync.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_control_label("Threshold"))
        header.addStretch(1)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(self._applied["threshold"])
        self._threshold_spin.valueChanged.connect(self._on_threshold_spin)
        header.addWidget(_horizontal_stepper(self._threshold_spin))
        body_layout.addLayout(header)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(round(self._applied["threshold"] * 100))
        self._threshold_slider.valueChanged.connect(self._on_threshold_slider)
        body_layout.addWidget(self._threshold_slider)

        # Sustain: integer timepoints (one prediction each) — how many consecutive
        # over-threshold predictions latch a decoder on. Step 1, no decimals.
        sustain_row = QHBoxLayout()
        sustain_row.setContentsMargins(0, 0, 0, 0)
        sustain_row.addWidget(_control_label("Sustain (timepoints)"))
        sustain_row.addStretch(1)
        self._sustain_spin = QSpinBox()
        self._sustain_spin.setRange(1, 500)
        self._sustain_spin.setSingleStep(1)
        self._sustain_spin.setValue(self._applied["sustain_timepoints"])
        self._sustain_spin.valueChanged.connect(self._on_control_changed)
        sustain_row.addWidget(_horizontal_stepper(self._sustain_spin))
        body_layout.addLayout(sustain_row)

        # Apply / Reset — enabled only while the draft differs from the applied
        # config; the styling makes the enabled/disabled states clearly distinct.
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 4, 0, 0)
        buttons.setSpacing(8)
        self._reset_button = QPushButton("Reset")
        self._reset_button.setStyleSheet(_SECONDARY_BUTTON_QSS)
        self._reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_button.clicked.connect(self._reset_decision_draft)
        buttons.addWidget(self._reset_button)
        self._apply_button = QPushButton("Apply")
        self._apply_button.setStyleSheet(_PRIMARY_BUTTON_QSS)
        self._apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_button.clicked.connect(self._apply_decision_draft)
        buttons.addWidget(self._apply_button)
        body_layout.addLayout(buttons)

        layout.addWidget(body)
        self._refresh_decision_controls()
        return section

    # ── decision settings API ───────────────────────────────────────────────────

    def draft_params(self) -> dict:
        """Current (possibly unapplied) control values."""
        return {
            "threshold": round(self._threshold_spin.value(), 2),
            "sustain_timepoints": self._sustain_spin.value(),
        }

    def applied_params(self) -> dict:
        """The last-applied decision settings."""
        return dict(self._applied)

    def is_dirty(self) -> bool:
        return self.draft_params() != self._applied

    def _on_control_changed(self, *_: object) -> None:
        self._refresh_decision_controls()

    def _on_threshold_slider(self, value: int) -> None:
        # Slider dragged → mirror into the spinbox (guard against a sync loop).
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(value / 100.0)
        self._threshold_spin.blockSignals(False)
        self._refresh_decision_controls()

    def _on_threshold_spin(self, value: float) -> None:
        # Manual spinbox entry → mirror into the slider.
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(round(value * 100))
        self._threshold_slider.blockSignals(False)
        self._refresh_decision_controls()

    def _refresh_decision_controls(self) -> None:
        dirty = self.is_dirty()
        self._apply_button.setEnabled(dirty)
        self._reset_button.setEnabled(dirty)

    def _apply_decision_draft(self) -> None:
        self._applied = self.draft_params()
        self._refresh_decision_controls()
        self.decision_params_changed.emit(dict(self._applied))

    def _reset_decision_draft(self) -> None:
        # Setting the spinbox re-syncs the slider and refreshes the dirty state;
        # no params signal is emitted (only Apply commits).
        self._threshold_spin.setValue(self._applied["threshold"])
        self._sustain_spin.setValue(self._applied["sustain_timepoints"])
        self._refresh_decision_controls()


# ── module helpers ────────────────────────────────────────────────────────────


_STEP_BUTTON_QSS = (
    f"QPushButton {{ background: {CARD_WHITE}; color: {TEXT_PRIMARY};"
    f" border: 1px solid {BORDER_GRAY}; border-radius: 3px; font-weight: 600; }}"
    f"QPushButton:hover {{ border-color: {PRIMARY_BLUE}; color: {PRIMARY_BLUE}; }}"
    f"QPushButton:pressed {{ background: #EFF3FB; }}"
)


def _horizontal_stepper(spin: QAbstractSpinBox) -> QWidget:
    """Wrap a spinbox with horizontal step buttons (down/− left, up/+ right),
    replacing the default stacked vertical arrows. The spinbox object is
    unchanged — callers keep their reference for value get/set."""
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setFixedWidth(52)
    spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    row = QWidget()
    row.setStyleSheet("background: transparent;")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(spin)
    for text, step in (("−", spin.stepDown), ("+", spin.stepUp)):
        button = QPushButton(text)
        button.setFixedSize(24, 24)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(_STEP_BUTTON_QSS)
        button.clicked.connect(step)
        layout.addWidget(button)
    return row


def _control_label(text: str) -> QLabel:
    """A compact left-aligned label for a decision-settings control."""
    label = QLabel(text)
    f = label.font()
    f.setPointSize(10)
    f.setWeight(QFont.Weight.Medium)
    label.setFont(f)
    label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
    return label


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
