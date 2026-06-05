from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class TrainingWorker(BaseWorker):
    """Runs orchestrator.run_training(timepoints) off the GUI thread.

    Training fits one classifier per task, each at its operator-selected
    timepoint (a ``{task_name: seconds}`` dict), and writes
    ``decoder_pipeline.joblib`` to the configured output directory. Emits
    ``result_ready(dict)`` with the orchestrator's return value on success —
    ``{model_filepath, spatial_patterns, mne_info}`` (see
    ``OfflineOrchestrator.run_training``).
    """

    def __init__(self, orchestrator, timepoints: dict[str, float], parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._timepoints = {k: float(v) for k, v in timepoints.items()}

    def run(self) -> None:
        try:
            result = self._orchestrator.run_training(self._timepoints)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
