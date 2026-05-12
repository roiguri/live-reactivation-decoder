from PyQt6.QtCore import (
    QPropertyAnimation, QEasingCurve, pyqtSignal as Signal, pyqtProperty,
    Qt, QRect, QPoint,
)
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient

from frontend.styles.theme import (
    PRIMARY_BLUE, SUCCESS_GREEN, BG_LIGHT, CARD_WHITE,
    TEXT_PRIMARY, TEXT_MUTED, BORDER_GRAY,
)

_CIRCLE_D  = 28
_CIRCLE_R  = _CIRCLE_D // 2
_CIRCLE_CX = 20   # circle center x in each node's local coords
_CIRCLE_CY = 20   # circle center y — enough room for full halo (halo_r=19, top=1)
_NODE_H    = 120  # fixed height of every JourneyNode
_DETAIL_H  = 72   # reserved height for desc + gap + button

_NODE_DATA = [
    ("Settings",      "Load experiment config YAML"),
    ("Load Data",     "Pick data directory, load .vhdr"),
    ("Preprocessing", "Run ICA, review components"),
    ("Evaluation",    "Run CV, inspect AUC results"),
    ("Train & Save",  "Train final decoders"),
]

_ACTION_LABELS = [
    "Continue",
    "Load && Continue",
    "Confirm && Continue",
    "Confirm Timepoint",
    "Go Live",
]


class JourneyNode(QWidget):
    action_clicked = Signal()

    def __init__(self, number: int, title: str, description: str,
                 action_label: str, parent=None):
        super().__init__(parent)
        self._number = number
        self._state = "inactive"
        self._filling = False
        self.__node_fill = 0.0

        self.setFixedHeight(_NODE_H)

        self._fill_anim = QPropertyAnimation(self, b"node_fill_progress")
        self._fill_anim.setDuration(600)
        self._fill_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._fill_anim.setStartValue(0.0)
        self._fill_anim.setEndValue(1.0)
        self._fill_anim.finished.connect(self._on_fill_finished)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_CIRCLE_D + 20, 12, 8, 4)
        layout.setSpacing(0)

        self._title_lbl = QLabel(title)
        f = self._title_lbl.font()
        f.setPointSize(11)
        f.setWeight(QFont.Weight.DemiBold)
        self._title_lbl.setFont(f)
        layout.addWidget(self._title_lbl)
        layout.addSpacing(8)

        self._detail = QWidget()
        self._detail.setFixedHeight(_DETAIL_H)
        self._detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        detail_layout = QVBoxLayout(self._detail)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        detail_layout.setSpacing(10)

        self._desc_lbl = QLabel(description)
        f2 = self._desc_lbl.font()
        f2.setPointSize(9)
        self._desc_lbl.setFont(f2)
        self._desc_lbl.setWordWrap(True)
        self._desc_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        detail_layout.addWidget(self._desc_lbl)

        self._btn = QPushButton(action_label)
        self._btn.setProperty("class", "primary")
        self._btn.setFixedHeight(26)
        self._btn.clicked.connect(self.action_clicked)
        detail_layout.addWidget(self._btn)

        layout.addWidget(self._detail)
        layout.addStretch(1)

        self._apply_state()

    # --- public API ---

    def set_state(self, state: str) -> None:
        if self._fill_anim.state() == QPropertyAnimation.State.Running:
            self._fill_anim.stop()
        self._filling = False
        self.__node_fill = 0.0
        self._state = state
        self._apply_state()
        self.update()

    def set_action_enabled(self, enabled: bool) -> None:
        """Enable/disable the node's action button independently of node state."""
        self._btn.setEnabled(enabled)

    def set_action_label(self, label: str) -> None:
        """Update the action button label (e.g. when the node's pending action changes)."""
        self._btn.setText(label)

    def start_fill_animation(self) -> None:
        """Play a top-to-bottom circle fill, then transition to active."""
        self._filling = True
        self.__node_fill = 0.0
        if self._fill_anim.state() == QPropertyAnimation.State.Running:
            self._fill_anim.stop()
        self._fill_anim.start()
        self.update()

    def circle_center(self) -> QPoint:
        return QPoint(_CIRCLE_CX, _CIRCLE_CY)

    # --- pyqtProperty for circle fill animation ---

    def _get_node_fill_progress(self) -> float:
        return self.__node_fill

    def _set_node_fill_progress(self, value: float) -> None:
        self.__node_fill = value
        self.update()

    node_fill_progress = pyqtProperty(
        float, fget=_get_node_fill_progress, fset=_set_node_fill_progress
    )

    # --- private ---

    def _on_fill_finished(self) -> None:
        self._filling = False
        self.__node_fill = 0.0
        self._state = "active"
        self._apply_state()
        self.update()

    def _apply_state(self) -> None:
        active   = self._state == "active"
        complete = self._state == "complete"
        self._desc_lbl.setVisible(active or complete)
        self._btn.setVisible(active)
        if active:
            self._title_lbl.setStyleSheet(f"color: {PRIMARY_BLUE}; font-weight: 600;")
            self._desc_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        elif complete:
            self._title_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
            self._desc_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        else:
            self._title_lbl.setStyleSheet(f"color: {TEXT_MUTED};")

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = _CIRCLE_CX, _CIRCLE_CY
        rect = QRect(cx - _CIRCLE_R, cy - _CIRCLE_R, _CIRCLE_D, _CIRCLE_D)

        # Card background behind detail widget
        if self._state in ("active", "complete"):
            p.setBrush(QBrush(QColor(BG_LIGHT)))
            p.setPen(QPen(QColor(BORDER_GRAY), 1))
            p.drawRoundedRect(self._detail.geometry(), 4, 4)

        # Halo ring (active or mid-fill)
        if self._state == "active" or self._filling:
            halo_r = _CIRCLE_R + 5
            halo_color = QColor(PRIMARY_BLUE)
            halo_color.setAlpha(38)
            p.setBrush(QBrush(halo_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QRect(cx - halo_r, cy - halo_r, halo_r * 2, halo_r * 2))

        if self._filling:
            # Empty circle shell
            p.setBrush(QBrush(QColor(CARD_WHITE)))
            p.setPen(QPen(QColor(PRIMARY_BLUE), 2))
            p.drawEllipse(rect)

            # Clip a growing rect from the top of the circle downward
            fill_h = int(_CIRCLE_D * self.__node_fill)
            if fill_h > 0:
                p.save()
                p.setClipRect(QRect(cx - _CIRCLE_R, cy - _CIRCLE_R, _CIRCLE_D, fill_h))
                p.setBrush(QBrush(QColor(PRIMARY_BLUE)))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(rect)
                p.restore()

            # Number — white once fill passes centre, blue before
            num_color = QColor("white") if self.__node_fill >= 0.5 else QColor(PRIMARY_BLUE)
            p.setPen(num_color)
            f = QFont()
            f.setPointSize(9)
            f.setWeight(QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._number))

        elif self._state == "complete":
            p.setBrush(QBrush(QColor(SUCCESS_GREEN)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(rect)
            pen = QPen(QColor("white"), 2, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawLine(cx - 5, cy + 1, cx - 1, cy + 5)
            p.drawLine(cx - 1, cy + 5, cx + 5, cy - 3)

        elif self._state == "active":
            p.setBrush(QBrush(QColor(PRIMARY_BLUE)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(rect)
            p.setPen(QColor("white"))
            f = QFont()
            f.setPointSize(9)
            f.setWeight(QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._number))

        else:
            p.setBrush(QBrush(QColor(CARD_WHITE)))
            p.setPen(QPen(QColor(BORDER_GRAY), 2))
            p.drawEllipse(rect)
            p.setPen(QColor(TEXT_MUTED))
            f = QFont()
            f.setPointSize(9)
            f.setWeight(QFont.Weight.Bold)
            p.setFont(f)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(self._number))


class JourneyPanel(QWidget):
    node_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setObjectName("journey_panel")

        self._nodes: list[JourneyNode] = []
        self._completed_segments = 0
        self._animating_segment = -1
        self.__fill_progress = 0.0

        self._anim = QPropertyAnimation(self, b"fill_progress")
        self._anim.setDuration(800)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.finished.connect(self._on_anim_finished)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 20, 16, 20)
        outer.setSpacing(0)

        header = QLabel("TRAINING PIPELINE")
        hf = header.font()
        hf.setPointSize(10)
        hf.setWeight(QFont.Weight.DemiBold)
        header.setFont(hf)
        header.setStyleSheet(f"color: {TEXT_MUTED}; letter-spacing: 1px;")
        outer.addWidget(header)
        outer.addSpacing(24)

        for i, (title, desc) in enumerate(_NODE_DATA):
            node = JourneyNode(i + 1, title, desc, _ACTION_LABELS[i])
            n = i + 1
            node.action_clicked.connect(lambda _=False, n=n: self.advance(n))
            self._nodes.append(node)
            outer.addWidget(node)

        outer.addStretch()

        self._nodes[0].set_state("active")

    # --- pyqtProperty for trail animation ---

    def _get_fill_progress(self) -> float:
        return self.__fill_progress

    def _set_fill_progress(self, value: float) -> None:
        self.__fill_progress = value
        self.update()

    fill_progress = pyqtProperty(float, fget=_get_fill_progress, fset=_set_fill_progress)

    # --- public API ---

    def set_node_action(self, node_index: int, handler) -> None:
        """Override the action button handler for the node at node_index (0-based).

        Disconnects the default advance() connection and wires handler instead.
        """
        node = self._nodes[node_index]
        try:
            node.action_clicked.disconnect()
        except TypeError:
            pass
        node.action_clicked.connect(handler)

    def set_node_action_label(self, node_index: int, label: str) -> None:
        """Update the action button label for the node at node_index (0-based)."""
        if 0 <= node_index < len(self._nodes):
            self._nodes[node_index].set_action_label(label)

    def set_node_ready(self, node_index: int, ready: bool) -> None:
        """Gate the action button on node_index (0-based).

        Views declare their prerequisites by emitting a `ready_changed(bool)`
        signal; `Phase1Screen` forwards it here so the panel reflects the gate.
        """
        if 0 <= node_index < len(self._nodes):
            self._nodes[node_index].set_action_enabled(ready)

    def advance(self, completed_node: int) -> None:
        """Complete node `completed_node` (1-indexed); trail animates then next node fills in."""
        idx = completed_node - 1
        if idx < 0 or idx >= len(self._nodes):
            return

        self._nodes[idx].set_state("complete")
        # next node activation is deferred: trail runs first, then circle fill

        if self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()
            self._completed_segments = self._animating_segment + 1

        self._animating_segment = idx
        self.__fill_progress = 0.0
        self._anim.start()

        self.node_changed.emit(idx + 1)

    # --- private slots ---

    def _on_anim_finished(self) -> None:
        self._completed_segments = self._animating_segment + 1
        next_idx = self._animating_segment + 1
        self._animating_segment = -1
        self.update()
        if next_idx < len(self._nodes):
            self._nodes[next_idx].start_fill_animation()

    # --- painting ---

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor("#FFFFFF"))
        grad.setColorAt(1.0, QColor("#FAFAFA"))
        p.fillRect(self.rect(), grad)

        if len(self._nodes) < 2:
            return

        centers = [n.mapTo(self, n.circle_center()) for n in self._nodes]
        trail_x = centers[0].x()

        pen_gray  = QPen(QColor(BORDER_GRAY),   1)
        pen_green = QPen(QColor(SUCCESS_GREEN), 2)

        for i in range(len(centers) - 1):
            y0 = centers[i].y() + _CIRCLE_R
            y1 = centers[i + 1].y() - _CIRCLE_R
            if y1 <= y0:
                continue

            if i < self._completed_segments:
                p.setPen(pen_green)
                p.drawLine(trail_x, y0, trail_x, y1)
            elif i == self._animating_segment:
                p.setPen(pen_gray)
                p.drawLine(trail_x, y0, trail_x, y1)
                y_curr = int(y0 + (y1 - y0) * self.__fill_progress)
                p.setPen(pen_green)
                p.drawLine(trail_x, y0, trail_x, y_curr)
            else:
                p.setPen(pen_gray)
                p.drawLine(trail_x, y0, trail_x, y1)
