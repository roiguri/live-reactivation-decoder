from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from backend.online_phase.lsl_receiver import DEFAULT_STREAM_NAME
from backend.session import AppSession
from frontend.styles.theme import BG_LIGHT, BORDER_GRAY, TEXT_MUTED, TEXT_PRIMARY


# TODO(phase2-ui): wire a Back button once the back-flow semantics are
# settled. Open questions: should Back land on Node 5 results or restart
# the journey? What happens to a live stream that's running? For now the
# screen is one-way — operator restarts the app to leave Phase 2.
class Phase2Screen(QWidget):
    """Live-inference screen. Constructed fresh each time the operator
    clicks Go Live. Header-only shell for now; chart + Start/Halt land
    in later commits.
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

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        body = QWidget()
        body.setStyleSheet(f"background: {BG_LIGHT};")
        root.addWidget(body, 1)

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(56)
        header.setStyleSheet(
            f"background: {BG_LIGHT}; border-bottom: 1px solid {BORDER_GRAY};"
        )
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("INFERENCE HALTED")
        f = self._status_label.font()
        f.setPointSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        self._status_label.setFont(f)
        self._status_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(self._status_label)

        divider = QLabel()
        divider.setFixedSize(1, 16)
        divider.setStyleSheet(f"background: {BORDER_GRAY};")
        layout.addWidget(divider)

        self._target_label = QLabel(f"Target: {DEFAULT_STREAM_NAME} (LSL)")
        tf = self._target_label.font()
        tf.setPointSize(10)
        self._target_label.setFont(tf)
        self._target_label.setStyleSheet(
            f"color: {TEXT_MUTED}; background: transparent;"
        )
        layout.addWidget(self._target_label)

        layout.addStretch(1)
        return header
