from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class TrainingWorker(BaseWorker):
    """Runs orchestrator.run_training(timepoint) off the GUI thread.

    Training fits one classifier per task at the operator-selected
    timepoint and writes ``decoder_pipeline.joblib`` to the configured
    output directory. Emits ``result_ready(dict)`` with the orchestrator's
    return value on success — ``{model_filepath, spatial_patterns,
    mne_info}`` (see ``OfflineOrchestrator.run_training``).
    """

    def __init__(self, orchestrator, timepoint: float, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._timepoint = float(timepoint)

    def run(self) -> None:
        try:
            result = self._orchestrator.run_training(self._timepoint)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
