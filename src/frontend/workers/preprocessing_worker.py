from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class PreprocessingStep1AWorker(BaseWorker):
    """Runs orchestrator.run_step1a_filter() off the GUI thread.

    Emits ``result_ready(raw)`` (the filtered ``mne.io.Raw``) on success so
    the main thread can pop MNE's interactive bad-channel window.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def execute(self):
        return self._orchestrator.run_step1a_filter()


class PreprocessingStep1BWorker(BaseWorker):
    """Runs orchestrator.run_step1b_fit_ica() off the GUI thread.

    Emits ``result_ready((ica, epochs, suggested))`` on success so the main
    thread can pop MNE's interactive ICA component window.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator

    def execute(self):
        return self._orchestrator.run_step1b_fit_ica()


class PreprocessingStep2Worker(BaseWorker):
    """Runs orchestrator.run_step2_apply_and_save(excluded) off the GUI thread.

    Emits ``result_ready({"n_epochs": int, "n_excluded": int})`` on success.
    """

    def __init__(self, orchestrator, excluded: list[int], parent=None):
        super().__init__(parent)
        self._orchestrator = orchestrator
        self._excluded = excluded

    def execute(self):
        return self._orchestrator.run_step2_apply_and_save(self._excluded)
