from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class EvaluationWorker(BaseWorker):
    """Runs orchestrator.run_evaluation() off the GUI thread.

    Temporal-generalization CV is heavy (~tens of seconds on full data);
    running it on the GUI thread would freeze the app. Emits
    ``result_ready(dict)`` with the evaluator's full result dict on
    success — see ``ModelEvaluator.run_evaluation`` for the schema —
    so the main thread can build the results view from it.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def execute(self):
        return self._orchestrator.run_evaluation()
