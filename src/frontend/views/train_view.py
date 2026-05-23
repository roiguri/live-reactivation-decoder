"""Node 5 — Train & Save workspace.

Two pages in a stack:

* Page 0 (Ready)    — a big ▶ button that triggers training. Mirrors
  the Eval Ready page so the visual rhythm matches across nodes.
* Page 1 (Complete) — read-only path showing where the trained pipeline
  was saved, plus a side-by-side grid of :class:`TopomapWidget` (one
  per decoder, fed from ``spatial_patterns`` + ``mne_info``).

Public surface mirrors the Eval view so ``Phase1Screen`` can wire the
same loading-overlay / ready-changed protocols:

* ``set_session(session)``   — provided when the operator continues
  past Settings.
* ``set_timepoint(t)``       — the operator's confirmed timepoint from
  the Eval view; the worker passes it straight to
  ``orchestrator.run_training``.
* ``trigger_run()``          — the journey-panel Node 5 button is
  rebound to this once the Ready page is showing.
* ``ready_changed(bool)``    — gates the journey-panel button.
* ``loading_requested(str)`` / ``loading_done()`` — Phase1Screen
  forwards these to its overlay.
* ``results_displayed``      — Phase1Screen rebinds the journey-panel
  button to ``trigger_go_live`` once results render (todo: hook this
  to Phase 2 once it ships).
* ``training_complete(dict)``— emitted on operator-side completion.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, SUCCESS_GREEN,
    TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.charts import TopomapWidget
from frontend.workers.training_worker import TrainingWorker

logger = logging.getLogger(__name__)


class TrainView(QWidget):
    """Node 5 workspace: Ready → Training → Complete (topomap grid)."""

    # Loading-overlay protocol — handled by Phase1Screen.
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 5 action button.
    ready_changed = Signal(bool)
    # Emitted once topomaps render so Phase1Screen can rebind the
    # journey-panel button (e.g. for "Go Live" in Phase 2).
    results_displayed = Signal()
    # Emitted when training finishes successfully.
    training_complete = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session = None
        self._timepoint: Optional[float] = None
        self._running: bool = False
        self._done: bool = False
        self._was_ready: bool = False
        self._result: Optional[dict[str, Any]] = None
        self._thread: QThread | None = None
        self._worker: Optional[TrainingWorker] = None

        # Topomap grid (populated on completion) lives inside its own
        # frame so the spacing/border stay consistent with the rest of
        # the workspace.
        self._topomap_row: Optional[QHBoxLayout] = None
        self._topomaps: dict[str, TopomapWidget] = {}
        self._save_path_field: Optional[QLineEdit] = None
        # Header field that reports the timepoint the decoders were
        # trained at — taken straight from the timepoint the operator
        # confirmed on the Evaluation screen.
        self._trained_at_lbl: Optional[QLabel] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_ready_page())
        self._pages.addWidget(self._build_complete_page())
        outer.addWidget(self._pages)

    # ── public ────────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        """Provide the AppSession built by Node 1."""
        self._session = session
        self._update_ready_state()

    def set_timepoint(self, t_seconds: float) -> None:
        """Stash the timepoint the operator confirmed on the Eval screen."""
        self._timepoint = float(t_seconds)
        self._update_ready_state()

    def trigger_run(self) -> None:
        """Start the training worker. Wired to the journey-panel Node 5 button."""
        if (
            self._session is None
            or self._session.offline is None
            or self._timepoint is None
            or self._running
            or self._done
        ):
            return
        self._running = True
        self._update_ready_state()
        worker = TrainingWorker(self._session.offline, self._timepoint)
        self.loading_requested.emit("Training decoders…")
        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.result_ready.connect(self._on_train_done)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    # ── worker plumbing ──────────────────────────────────────────────────────

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _on_error(self, message: str) -> None:
        self._running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Training Error", message)
        self._update_ready_state()

    def _on_train_done(self, result: dict) -> None:
        """Worker success callback. ``result`` is the orchestrator's
        ``run_training`` return — ``{model_filepath, spatial_patterns,
        mne_info}``.
        """
        self.loading_done.emit()
        self._result = result
        self._running = False
        self._done = True

        # Save-path field.
        path = result.get("model_filepath")
        if self._save_path_field is not None and path is not None:
            self._save_path_field.setText(str(path))

        # "Trained at" — read from ``self._timepoint`` (set by
        # Phase1Screen on evaluation_complete; ``run_training`` doesn't
        # echo the timepoint back in its result).
        if self._trained_at_lbl is not None and self._timepoint is not None:
            self._trained_at_lbl.setText(f"{self._timepoint * 1000.0:.0f} ms")

        # Build the topomap row.
        self._populate_topomaps(
            result.get("spatial_patterns", {}),
            result.get("mne_info"),
        )

        self._pages.setCurrentIndex(1)
        self._update_ready_state()
        self.training_complete.emit(result)
        self.results_displayed.emit()

    def _populate_topomaps(self, patterns: dict, info) -> None:
        if self._topomap_row is None:
            return
        # Drop any topomaps left from a previous run.
        for w in list(self._topomaps.values()):
            self._topomap_row.removeWidget(w)
            w.deleteLater()
        self._topomaps.clear()
        if info is None or not patterns:
            return
        for name, pattern in patterns.items():
            topo = TopomapWidget()
            topo.set_pattern(pattern, info, title=name)
            self._topomap_row.addWidget(topo)
            self._topomaps[name] = topo

    # ── page builders ────────────────────────────────────────────────────────

    def _build_ready_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._start_btn = QPushButton("▶")
        self._start_btn.setFixedSize(96, 96)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(
            # Same play-button treatment used on the Eval Ready page.
            f"QPushButton {{ background: #EFF6FF; color: {PRIMARY_BLUE}; "
            f"border: none; border-radius: 48px; font-size: 36px; "
            f"padding: 0 0 0 6px; }}"
            f"QPushButton:hover {{ background: #DBEAFE; }}"
            f"QPushButton:pressed {{ background: #BFDBFE; }}"
        )
        self._start_btn.clicked.connect(self.trigger_run)
        center.addWidget(self._start_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        title = QLabel("Ready to Train")
        f = title.font()
        f.setPointSize(16)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel(
            "Click play to train one classifier per decoder at the "
            "selected timepoint, bundle the spatial patterns + ICA "
            "unmixing into the online state, and save "
            "``decoder_pipeline.joblib`` to the output directory."
        )
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(440)
        desc.setMinimumHeight(60)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _build_complete_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # ── Header: ✓ Training complete + path field ─────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        ok = QLabel("✓")
        ok.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-size: 18px; font-weight: 700;"
        )
        header_row.addWidget(ok)
        ok_lbl = QLabel("Training complete")
        f = ok_lbl.font()
        f.setPointSize(13)
        f.setWeight(QFont.Weight.DemiBold)
        ok_lbl.setFont(f)
        ok_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        header_row.addWidget(ok_lbl)
        header_row.addStretch()
        layout.addLayout(header_row)

        trained_row = QHBoxLayout()
        trained_row.setSpacing(8)
        trained_cap = QLabel("Trained at:")
        trained_cap.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        trained_row.addWidget(trained_cap)
        self._trained_at_lbl = QLabel("—")
        self._trained_at_lbl.setStyleSheet(
            f"color: {PRIMARY_BLUE}; font-size: 12px; font-weight: 700;"
        )
        trained_row.addWidget(self._trained_at_lbl)
        trained_row.addStretch()
        layout.addLayout(trained_row)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        path_cap = QLabel("Saved to:")
        path_cap.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        path_row.addWidget(path_cap)
        self._save_path_field = QLineEdit()
        self._save_path_field.setReadOnly(True)
        self._save_path_field.setCursor(Qt.CursorShape.IBeamCursor)
        self._save_path_field.setStyleSheet(
            f"QLineEdit {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; "
            "padding: 4px 8px; font-family: monospace; font-size: 11px; "
            f"color: {TEXT_PRIMARY}; }}"
        )
        path_row.addWidget(self._save_path_field, 1)
        layout.addLayout(path_row)

        # ── Spatial-patterns grid ─────────────────────────────────────
        topo_card = QFrame()
        topo_card.setObjectName("topo_card")
        topo_card.setStyleSheet(
            f"QFrame#topo_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
            "QFrame#topo_card QLabel { background: transparent; }"
        )
        topo_body = QVBoxLayout(topo_card)
        topo_body.setContentsMargins(12, 12, 12, 12)
        topo_body.setSpacing(8)

        topo_cap = QLabel("SPATIAL PATTERNS")
        topo_cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700; "
            "letter-spacing: 0.6px;"
        )
        topo_body.addWidget(topo_cap)

        row_holder = QWidget()
        self._topomap_row = QHBoxLayout(row_holder)
        self._topomap_row.setContentsMargins(0, 0, 0, 0)
        self._topomap_row.setSpacing(12)
        topo_body.addWidget(row_holder, 1)
        layout.addWidget(topo_card, 1)

        return page

    # ── ready gating ─────────────────────────────────────────────────────────

    def _update_ready_state(self) -> None:
        # Page 0 ready: we can launch training as long as the session
        # is built and a timepoint has been confirmed.
        page0_ready = (
            self._session is not None
            and self._timepoint is not None
            and not self._running
            and not self._done
        )
        # Page 1 ready: training succeeded and the journey-panel button is
        # now the "Go Live" entry into Phase 2 (rebound by Phase1Screen
        # on results_displayed).
        page1_ready = self._done
        ready = page0_ready or page1_ready
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
