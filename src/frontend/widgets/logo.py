"""Shared app-logo widget.

Single source for loading ``styles/assets/app_icon.png`` so every placement
(headers, launch hero) uses the same asset and scaling. The pixmap is rendered
at 2x and tagged with a device-pixel-ratio so it stays crisp on HiDPI displays
while occupying ``size`` logical pixels.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QWidget

_APP_ICON = Path(__file__).resolve().parents[1] / "styles" / "assets" / "app_icon.png"


def logo_label(size: int, parent: QWidget | None = None) -> QLabel:
    """Return a fixed ``size``×``size`` :class:`QLabel` showing the app logo.

    Loading is defensive: a missing asset yields an empty (but correctly sized)
    label rather than raising, mirroring the guard in ``MainWindow``.
    """
    label = QLabel(parent)
    label.setFixedSize(size, size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("background: transparent;")

    pixmap = QPixmap(str(_APP_ICON))
    if not pixmap.isNull():
        scaled = pixmap.scaled(
            size * 2,
            size * 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(2.0)
        label.setPixmap(scaled)
    return label
