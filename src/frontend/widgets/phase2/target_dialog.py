"""Target-selection dialog for the Phase 2 live screen.

Step 1 surfaces LSL stream discovery only (a combo populated by a manual
Refresh). Step 2 will extend this with a Live/Recording source choice and a
recording-folder picker. On accept, :meth:`selected_target` returns a target
descriptor consumed by ``Phase2Screen``::

    {"source": "lsl", "stream_name": "NeuroneStream"}
"""
from __future__ import annotations

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY,
    CARD_WHITE,
    TEXT_MUTED,
    TEXT_PRIMARY,
)
from frontend.workers.stream_discovery_worker import StreamDiscoveryWorker


class TargetSelectionDialog(QDialog):
    """Pick the LSL stream the live run should consume."""

    def __init__(self, session, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Target")
        self.setMinimumWidth(360)
        self._session = session
        self._result: dict | None = None
        self._thread: QThread | None = None
        self._worker: StreamDiscoveryWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        heading = QLabel("Live stream (LSL)")
        hf = heading.font()
        hf.setPointSize(10)
        hf.setBold(True)
        heading.setFont(hf)
        layout.addWidget(heading)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._combo = QComboBox()
        self._combo.setStyleSheet(
            "QComboBox {"
            f"  background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY};"
            f"  border-radius: 2px; padding: 3px 8px; font-size: 12px;"
            f"  color: {TEXT_PRIMARY};"
            "}"
        )
        self._combo.currentIndexChanged.connect(self._update_ok_enabled)
        row.addWidget(self._combo, 1)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setProperty("class", "secondary")
        self._refresh_btn.clicked.connect(self._on_refresh)
        row.addWidget(self._refresh_btn)
        layout.addLayout(row)

        self._status = QLabel("Click Refresh to discover streams.")
        self._status.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._update_ok_enabled()

    # ── public ─────────────────────────────────────────────────────────────────

    def selected_target(self) -> dict | None:
        """The chosen target descriptor, or None if cancelled."""
        return self._result

    # ── discovery ───────────────────────────────────────────────────────────────

    def _on_refresh(self) -> None:
        if self._thread is not None:
            return  # a scan is already in flight
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Scanning…")
        self._status.setText("Discovering streams…")

        thread = QThread(self)
        worker = StreamDiscoveryWorker(self._session)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(self._on_streams_found)
        worker.error_occurred.connect(self._on_discovery_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_scan_done)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_streams_found(self, names: list[str]) -> None:
        previous = self._combo.currentText()
        self._combo.clear()
        self._combo.addItems(names)
        if previous in names:
            self._combo.setCurrentText(previous)
        self._status.setText(
            f"{len(names)} stream(s) found." if names else "No streams found."
        )
        self._update_ok_enabled()

    def _on_discovery_error(self, message: str) -> None:
        self._status.setText(f"Discovery failed: {message}")

    def _on_scan_done(self) -> None:
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("Refresh")
        self._thread = None
        self._worker = None

    # ── internals ───────────────────────────────────────────────────────────────

    def _update_ok_enabled(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(self._combo.count() > 0)

    def _on_accept(self) -> None:
        if self._combo.count() == 0:
            return
        self._result = {"source": "lsl", "stream_name": self._combo.currentText()}
        self.accept()
