"""Qt binding that turns the pure :class:`DecisionEngine` into a live consumer.

The engine is deliberately Qt-free and unit-testable in isolation. This thin
:class:`QObject` adapter is the Phase-B seam: it consumes ``prediction_ready``
batches (a *sibling* of the logger — ``StreamWorker`` is never modified) and
re-emits each batch's :class:`DecisionResult` on ``decision_ready``, so decisions
fan out to the logger (persist) and the UI (display) exactly like predictions do.

Threading mirrors ``prediction_ready``: ``on_predictions`` runs on the worker
thread (direct connection), and ``decision_ready`` is delivered to the logger via
a direct connection and to the UI via a queued connection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from backend.online_phase.decision_engine import DecisionConfig, DecisionEngine

logger = logging.getLogger(__name__)


class DecisionBinding(QObject):
    """Adapter: ``prediction_ready`` batch → ``DecisionResult`` on ``decision_ready``."""

    decision_ready = pyqtSignal(object)  # carries a DecisionResult

    def __init__(self, engine: DecisionEngine, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine

    @property
    def engine(self) -> DecisionEngine:
        return self._engine

    def on_predictions(
        self,
        predictions: dict[str, np.ndarray],
        timestamps: np.ndarray,
        markers: Any = None,
    ) -> None:
        """Advance the engine over one batch and emit the result.

        ``markers`` is accepted to match ``prediction_ready`` but unused — decisions
        are free-running, not event-locked.
        """
        result = self._engine.process_batch(predictions, timestamps)
        self.decision_ready.emit(result)

    def reset(self) -> None:
        """Clear the engine's latch state for a fresh run."""
        self._engine.reset()

    def set_pending_config(self, config: DecisionConfig) -> None:
        """Stage new decision settings; applied at the next batch boundary."""
        self._engine.set_pending_config(config)
