from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class PreprocessingStep1Worker(BaseWorker):
    """Runs orchestrator.run_step1_prepare_ica() off the GUI thread.

    Emits ``result_ready((ica, suggested_components))`` on success,
    ``error_occurred(str)`` on failure, and always emits ``finished`` so the
    owning thread can quit.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def run(self) -> None:
        try:
            ica, suggested = self._orchestrator.run_step1_prepare_ica()
            self.result_ready.emit((ica, suggested))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()


class PreprocessingStep2Worker(BaseWorker):
    """Runs orchestrator.run_step2_finish_pipeline(excluded) off the GUI thread.

    Emits ``result_ready({"n_epochs": int})`` on success, ``error_occurred(str)``
    on failure, and always emits ``finished`` so the owning thread can quit.
    """

    def __init__(self, orchestrator, excluded: list[int], parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._excluded = excluded

    def run(self) -> None:
        try:
            result = self._orchestrator.run_step2_finish_pipeline(self._excluded)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
