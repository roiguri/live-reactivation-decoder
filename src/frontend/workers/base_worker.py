from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal as Signal


class BaseWorker(QObject):
    """Base class for background workers using the worker-object pattern.

    Subclasses override ``run()`` and emit ``result_ready`` on success or
    ``error_occurred`` on failure. Always emit ``finished`` in a ``finally``
    block so the owning ``QThread`` can be quit and cleaned up.

    Standard lifecycle wiring on the GUI thread::

        worker = SubclassWorker(...)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result_ready.connect(on_result)
        worker.error_occurred.connect(on_error)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()
    """

    started = Signal()
    progress = Signal(str)
    result_ready = Signal(object)
    error_occurred = Signal(str)
    finished = Signal()

    def run(self) -> None:
        raise NotImplementedError
