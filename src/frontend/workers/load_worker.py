from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class LoadWorker(BaseWorker):
    """Loads the raw EEG file off the GUI thread.

    Caller is expected to have already invoked ``orchestrator.set_file_path(...)``;
    this worker only runs the blocking ``load_raw_data()`` call. Emits
    ``result_ready(None)`` on success or ``error_occurred(str)`` on failure, and
    always emits ``finished`` so the owning thread can quit.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def run(self) -> None:
        try:
            self._orchestrator.load_raw_data()
            self.result_ready.emit(None)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
