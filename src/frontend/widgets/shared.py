from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget,
)

from frontend.styles.theme import BORDER_GRAY, TEXT_MUTED, TEXT_PRIMARY


_FIELD_QSS = f"""
QLineEdit {{
    background: #F9FAFB;
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
    font-family: monospace;
    font-size: 12px;
    color: #4B5563;
    padding: 3px 8px;
}}
QLineEdit:read-only {{
    background: #F9FAFB;
}}
"""


class ReadOnlyField(QWidget):
    """Inline label + bordered read-only QLineEdit + optional unit label.

    Usage:
        field = ReadOnlyField("l_freq", unit="Hz", field_width=80)
        field.set_value(1.0)
    """

    def __init__(self, label: str, unit: str = "",
                 field_width: int = 90, parent=None):
        super().__init__(parent)
        self._label_text = label

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        if label:
            lbl = QLabel(label.upper())
            lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 600;"
            )
            lbl.setFixedWidth(10 * len(label) + 20)  # rough proportional width
            row.addWidget(lbl)

        self._edit = QLineEdit("—")
        self._edit.setReadOnly(True)
        self._edit.setFixedWidth(field_width)
        self._edit.setStyleSheet(_FIELD_QSS)
        self._edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._edit)

        if unit:
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
            row.addWidget(unit_lbl)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_value(self, value) -> None:
        """Set the displayed value. None or empty string → shows '—'."""
        if value is None or value == "":
            self._edit.setText("—")
            self._edit.setStyleSheet(_FIELD_QSS + "QLineEdit { color: #9CA3AF; }")
        else:
            self._edit.setText(str(value))
            self._edit.setStyleSheet(_FIELD_QSS)

    @property
    def display_text(self) -> str:
        return self._edit.text()


class FilePicker(QWidget):
    """Button + path display row. Emits path_selected(str) when a path is chosen.

    Usage:
        picker = FilePicker("Load Config File", mode="file",
                            file_filter="Config (*.yaml *.yml)")
        picker.path_selected.connect(my_slot)
        picker.path   # → str | None
    """

    path_selected = Signal(str)

    def __init__(self, button_text: str, mode: str = "file",
                 file_filter: str = "", parent=None):
        super().__init__(parent)
        if mode not in ("file", "dir"):
            raise ValueError(f"mode must be 'file' or 'dir', got {mode!r}")
        self._mode = mode
        self._filter = file_filter
        self._path: str | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        self._btn = QPushButton(button_text)
        self._btn.setProperty("class", "secondary")
        self._btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._btn.clicked.connect(self._open_dialog)
        row.addWidget(self._btn)

        self._path_lbl = QLabel(self._placeholder())
        self._path_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        row.addWidget(self._path_lbl)
        row.addStretch()

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def path(self) -> str | None:
        return self._path

    def clear(self) -> None:
        """Reset to unselected state."""
        self._path = None
        self._path_lbl.setText(self._placeholder())
        self._path_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

    # ── private ───────────────────────────────────────────────────────────────

    def _placeholder(self) -> str:
        return "No file selected" if self._mode == "file" else "No directory selected"

    def _open_dialog(self) -> None:
        if self._mode == "file":
            path, _ = QFileDialog.getOpenFileName(
                self, self._btn.text(), "", self._filter
            )
        else:
            path = QFileDialog.getExistingDirectory(self, self._btn.text())

        if not path:
            return

        self._path = path
        self._path_lbl.setText(path)
        self._path_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 11px; font-family: monospace;"
        )
        self.path_selected.emit(path)
