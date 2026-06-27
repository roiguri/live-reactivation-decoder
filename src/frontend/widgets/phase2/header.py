"""Phase 2 status header: status indicator + target selector."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QWidget

from frontend.styles.theme import (
    AMBER,
    BG_LIGHT,
    BORDER_GRAY,
    PRIMARY_BLUE,
    SUCCESS_GREEN,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

# Buffer-health chip — a soft tinted chip (tint background + 1px border +
# coloured text, squared to the app's 2px radius), matching the active-filter
# idiom in frozen_event_view rather than a loud solid pill that would compete
# with the green LIVE status text. Green when the backlog is comfortably below
# the batch size; amber when it is building (the worker isn't keeping up).
_CHIP_BASE = (
    "border-radius: 2px; padding: 1px 7px;"
    "font-size: 11px; font-weight: 600; letter-spacing: 0.5px;"
)
_CHIP_OK_QSS = (
    f"{_CHIP_BASE} color: {SUCCESS_GREEN};"
    " background: #F0FDF4; border: 1px solid #BBF7D0;"
)
_CHIP_BUSY_QSS = (
    f"{_CHIP_BASE} color: {AMBER};"
    " background: #FFFBEB; border: 1px solid #FDE5B5;"
)


class Phase2Header(QWidget):
    """Header bar: ``[status] | [Choose target…]  …  [latency] [buffer]``.

    The target is a clickable control: it starts as *Choose target…* and,
    when clicked, emits :attr:`choose_target_clicked` so the screen can open
    the target-selection dialog. The right side carries live diagnostics
    (rolling latency + buffer-health chip), updated by the screen.
    """

    choose_target_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(
            f"background: {BG_LIGHT}; border-bottom: 1px solid {BORDER_GRAY};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("INFERENCE HALTED")
        sf = self._status_label.font()
        sf.setPointSize(12)
        sf.setWeight(QFont.Weight.DemiBold)
        self._status_label.setFont(sf)
        self._status_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; background: transparent;"
        )
        layout.addWidget(self._status_label)

        divider = QFrame()
        divider.setFixedSize(1, 16)
        divider.setStyleSheet(f"background: {BORDER_GRAY};")
        layout.addWidget(divider)

        self._target_button = QPushButton("Choose target…")
        self._target_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._target_button.setFlat(True)
        self._target_button.setStyleSheet(
            "QPushButton {"
            f"  color: {PRIMARY_BLUE}; background: transparent; border: none;"
            "  padding: 0; font-size: 10pt; text-align: left;"
            "}"
            "QPushButton:hover { text-decoration: underline; }"
        )
        self._target_button.clicked.connect(self.choose_target_clicked)
        layout.addWidget(self._target_button)

        layout.addStretch(1)

        # ── right side: live diagnostics ───────────────────────────────────
        self._latency_label = QLabel()
        lf = self._latency_label.font()
        lf.setPointSize(10)
        self._latency_label.setFont(lf)
        self._latency_label.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent;"
        )
        self._latency_label.setToolTip(
            "p50 — half of recent batches were processed faster than this "
            "(the typical case).\n"
            "p95 — 95% of recent batches were processed within this "
            "(the near-worst case)."
        )
        # Both diagnostics align to the row's vertical centre so the latency
        # text and the chip share a baseline.
        layout.addWidget(self._latency_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self._buffer_chip = QLabel()
        # Align centre so the layout sizes the chip to its text + padding
        # instead of stretching its tinted background to the full header
        # height. Centre the text within that box too.
        self._buffer_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._buffer_chip, 0, Qt.AlignmentFlag.AlignVCenter)

        # Start blank — diagnostics only have meaning while a stream runs.
        self.clear_diagnostics()

    def set_status(self, text: str, *, color: str = TEXT_PRIMARY) -> None:
        """Update the status label text and tint."""
        self._status_label.setText(text)
        self._status_label.setStyleSheet(
            f"color: {color}; background: transparent;"
        )

    def set_target_text(self, text: str) -> None:
        """Update the target selector's label (e.g. ``Target: X (LSL)``)."""
        self._target_button.setText(text)

    # ── diagnostics ──────────────────────────────────────────────────────────

    def set_latency(self, p50_ms: float, p95_ms: float) -> None:
        """Show the rolling per-batch latency percentiles."""
        self._latency_label.setText(f"Latency: {p50_ms:.0f} / {p95_ms:.0f} ms")

    def set_buffer_health(self, healthy: bool) -> None:
        """Set the buffer-health chip: green ``BUFFER OK`` or amber ``BACKLOG``."""
        self._buffer_chip.setText("BUFFER OK" if healthy else "BACKLOG")
        self._buffer_chip.setStyleSheet(_CHIP_OK_QSS if healthy else _CHIP_BUSY_QSS)

    def clear_diagnostics(self) -> None:
        """Blank the latency readout and hide the chip (stream not running)."""
        self._latency_label.setText("")
        self._buffer_chip.setText("")
        self._buffer_chip.setStyleSheet("background: transparent;")
