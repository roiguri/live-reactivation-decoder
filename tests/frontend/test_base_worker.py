"""Contract tests for BaseWorker's template-method ``run()``.

``execute()`` is a plain method, so ``run()`` can be called synchronously here
(no ``QThread``/event loop). These tests pin the worker contract that the rest
of the app relies on:

* success → ``result_ready(<execute return>)`` then ``finished`` (no error)
* failure → a full traceback is logged AND ``error_occurred(str(exc))`` is
  emitted, then ``finished``

The traceback-logging assertion guards the whole point of the template-method
refactor: the UI only sees ``str(exc)``, so the stack trace must land in the log.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.workers.base_worker import BaseWorker  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


class _OkWorker(BaseWorker):
    def execute(self):
        return {"value": 42}


class _BoomWorker(BaseWorker):
    def execute(self):
        raise ValueError("kaboom")


def _capture(worker: BaseWorker) -> dict:
    events: dict = {"result": [], "error": [], "finished": 0}
    worker.result_ready.connect(lambda r: events["result"].append(r))
    worker.error_occurred.connect(lambda m: events["error"].append(m))
    worker.finished.connect(
        lambda: events.__setitem__("finished", events["finished"] + 1)
    )
    return events


def test_success_emits_result_then_finished(qapp):
    worker = _OkWorker()
    events = _capture(worker)

    worker.run()

    assert events["result"] == [{"value": 42}]
    assert events["error"] == []
    assert events["finished"] == 1


def test_failure_emits_error_str_and_finished(qapp):
    worker = _BoomWorker()
    events = _capture(worker)

    worker.run()

    assert events["result"] == []
    assert events["error"] == ["kaboom"]  # str(exc) forwarded to the UI
    assert events["finished"] == 1


def test_failure_logs_full_traceback(qapp, caplog):
    worker = _BoomWorker()

    with caplog.at_level(logging.ERROR):
        worker.run()

    rec = next(r for r in caplog.records if "_BoomWorker failed" in r.getMessage())
    assert rec.levelno == logging.ERROR
    assert rec.exc_info is not None  # logger.exception captured the traceback
