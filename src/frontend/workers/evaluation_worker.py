from __future__ import annotations

from PyQt6.QtCore import pyqtSignal as Signal

from frontend.workers.base_worker import BaseWorker


class EvaluationWorker(BaseWorker):
    """Runs orchestrator.run_evaluation() off the GUI thread.

    Temporal-generalization CV is heavy (~tens of seconds on full data);
    running it on the GUI thread would freeze the app. Emits
    ``result_ready(dict)`` with the evaluator's full result dict on
    success — see ``ModelEvaluator.run_evaluation`` for the schema —
    so the main thread can build the results view from it.

    Forwards the evaluator's per-decoder ``on_progress`` hook as the
    ``decoder_progress(completed, total, name)`` signal. The hook fires on
    this worker thread; emitting a Qt signal hands it safely to the GUI
    thread (queued connection) where the progress screen consumes it.
    """

    decoder_progress = Signal(int, int, str)

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def execute(self):
        return self._orchestrator.run_evaluation(on_progress=self._on_progress)

    def _on_progress(self, completed: int, total: int, name: str) -> None:
        self.decoder_progress.emit(completed, total, name)
