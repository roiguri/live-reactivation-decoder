from __future__ import annotations

from frontend.workers.base_worker import BaseWorker


class StreamDiscoveryWorker(BaseWorker):
    """Resolves available LSL stream names off the GUI thread.

    Calls ``AppSession.discover_streams`` (which launches the proxy and blocks
    for the resolve timeout) and emits ``result_ready(list[str])`` on success
    or ``error_occurred(str)`` on failure. Always emits ``finished`` so the
    owning thread can quit.
    """

    def __init__(self, session, timeout_sec: float = 3.0, parent=None):
        super().__init__(parent)
        self._session = session
        self._timeout_sec = timeout_sec

    def execute(self):
        return self._session.discover_streams(timeout_sec=self._timeout_sec)
