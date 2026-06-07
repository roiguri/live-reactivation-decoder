from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal as Signal


class BaseWorker(QObject):
    """Base class for background workers using the worker-object pattern.

    Subclasses override :meth:`execute` and **return** their result; the base
    :meth:`run` handles the rest uniformly: it emits ``result_ready`` with that
    return value on success, or — on any exception — logs a full traceback
    (``logger.exception``) and emits ``error_occurred(str(exc))``. ``finished``
    is always emitted last so the owning ``QThread`` can quit.

    Logging the traceback here is the whole point: the UI only ever sees
    ``str(exc)`` in a dialog, so without this the stack trace of a backend
    failure would be lost. The traceback is logged under the subclass's own
    module logger.

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

    def execute(self):
        """Do the work and return the result to emit via ``result_ready``.

        Runs on the worker thread. Raise on failure — :meth:`run` logs the
        traceback and forwards the message to ``error_occurred``.
        """
        raise NotImplementedError

    def run(self) -> None:
        try:
            self.result_ready.emit(self.execute())
        except Exception as exc:
            logging.getLogger(type(self).__module__).exception(
                "%s failed", type(self).__name__
            )
            self.error_occurred.emit(str(exc))
        finally:
            self.finished.emit()
