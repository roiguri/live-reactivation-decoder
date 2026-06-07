from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class ConfigLoaderWorker(BaseWorker):
    """Constructs an ``AppSession`` from a YAML config path off the GUI thread.

    Emits ``result_ready(session)`` with the constructed ``AppSession`` on
    success, or ``error_occurred(str)`` on any exception. Always emits
    ``finished`` so the owning thread can quit.
    """

    def __init__(self, config_path: str, parent=None):
        super().__init__(parent)
        self._config_path = config_path

    def execute(self):
        from backend.session import AppSession
        return AppSession(self._config_path)
