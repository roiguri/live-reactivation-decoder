"""Headless tests for EvaluationView per-decoder timepoint selection.

Drives ``_on_eval_done`` with a synthetic evaluator result (two decoders
with distinct AUC peaks) and asserts the per-decoder model: each decoder
pre-fills its own suggested peak, confirm is per-decoder, and the journey
"Approve && Continue" gate (``_all_confirmed`` → ``ready_changed``) only
fires once every decoder is confirmed. ``evaluation_complete`` carries the
per-decoder ``{task: seconds}`` dict.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.views.evaluation_view import EvaluationView  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _make_result() -> dict:
    """Two decoders with peaks at different timepoints."""
    times = np.linspace(-0.2, 1.0, 13)
    n = times.size

    def curve(peak_idx: int) -> np.ndarray:
        d = np.full(n, 0.5)
        d[peak_idx] = 0.9
        return d

    tasks = {}
    for name, peak_idx in (("red", 5), ("green", 9)):
        diag = curve(peak_idx)
        tasks[name] = {
            "diagonal_auc": diag,
            "tgm_matrix": np.outer(diag, diag),
            "peak_auc": float(np.max(diag)),
            "peak_timepoint": float(times[int(np.argmax(diag))]),
            "chance_level": 0.5,
        }
    return {
        "times": times,
        "suggested_timepoint": float(times[7]),
        "average_peak_auc": 0.7,
        "tasks": tasks,
    }


@pytest.fixture
def view(qapp) -> EvaluationView:
    v = EvaluationView()
    v._on_eval_done(_make_result())  # populate Page 1 directly
    return v


def test_prefills_per_decoder_suggested_peaks(view: EvaluationView) -> None:
    result = _make_result()
    expected = {
        name: task["peak_timepoint"] for name, task in result["tasks"].items()
    }
    assert view._selected_timepoints == pytest.approx(expected)
    # Distinct peaks → the two decoders really differ.
    assert view._selected_timepoints["red"] != view._selected_timepoints["green"]
    # Nothing confirmed yet.
    assert view._confirmed == {"red": False, "green": False}
    assert not view._all_confirmed()


def test_continue_gate_requires_all_confirmed(view: EvaluationView) -> None:
    ready_events: list[bool] = []
    view.ready_changed.connect(ready_events.append)

    view._toggle_confirm("red")
    assert not view._all_confirmed()
    assert ready_events[-1] is False if ready_events else True

    view._toggle_confirm("green")
    assert view._all_confirmed()
    assert ready_events[-1] is True


def test_evaluation_complete_emits_per_decoder_dict(view: EvaluationView) -> None:
    payloads: list[dict] = []
    view.evaluation_complete.connect(payloads.append)

    view._toggle_confirm("red")
    view._toggle_confirm("green")
    view.trigger_confirm()

    assert payloads, "evaluation_complete did not fire"
    assert payloads[-1] == pytest.approx(view._selected_timepoints)
    assert set(payloads[-1]) == {"red", "green"}


def test_changing_timepoint_unconfirms_only_that_decoder(view: EvaluationView) -> None:
    view._toggle_confirm("red")
    view._toggle_confirm("green")
    assert view._all_confirmed()

    # Move red to a different sample → red unconfirms, green stays.
    other = view._result["times"][2]
    view._set_decoder_timepoint("red", float(other))

    assert view._confirmed["red"] is False
    assert view._confirmed["green"] is True
    assert not view._all_confirmed()
    assert view._selected_timepoints["red"] == pytest.approx(float(other))


def test_suggested_caption_shows_cross_task_timepoint(view: EvaluationView) -> None:
    # _make_result sets suggested_timepoint = times[7] = 0.5 s → 500 ms.
    text = view._roster_suggested_lbl.text()
    assert "500" in text and "mean AUC" in text
    assert view._roster_suggested_lbl.toolTip()  # explanatory tooltip present


def test_reselecting_same_timepoint_keeps_lock(view: EvaluationView) -> None:
    view._toggle_confirm("red")
    view._toggle_confirm("green")
    assert view._all_confirmed()

    # Re-select red's current timepoint (a no-op change) → stays locked.
    same = view._selected_timepoints["red"]
    view._set_decoder_timepoint("red", same)

    assert view._confirmed["red"] is True
    assert view._all_confirmed()


def test_reset_all_returns_to_suggested(view: EvaluationView) -> None:
    view._set_decoder_timepoint("red", float(view._result["times"][1]))
    view._toggle_confirm("green")

    view._reset_all_to_suggested()

    assert view._selected_timepoints == pytest.approx(view._suggested_timepoints)
    # Reset unconfirms everything.
    assert not any(view._confirmed.values())
