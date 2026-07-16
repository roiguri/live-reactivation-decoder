from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from frontend.widgets.phase2.decision_panel import DecisionPanel

_COLORS = {"animate decoder": "#e6194b", "inanimate decoder": "#3cb44b"}


@pytest.fixture
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def panel(qapp) -> DecisionPanel:
    return DecisionPanel(task_colors=_COLORS)


def _result(active: dict[str, list[bool]]) -> SimpleNamespace:
    return SimpleNamespace(active={k: np.asarray(v, dtype=bool) for k, v in active.items()})


def test_one_row_per_decoder_starts_idle(panel):
    assert panel.active_decoders() == set()
    assert not panel.is_active("animate decoder")


def test_update_decision_uses_last_sample(panel):
    # Latched high mid-batch then still on at the last sample → active.
    panel.update_decision(_result({"animate decoder": [False, True, True],
                                   "inanimate decoder": [False, False, False]}))
    assert panel.is_active("animate decoder")
    assert not panel.is_active("inanimate decoder")


def test_multiple_decoders_active_at_once(panel):
    panel.update_decision(_result({"animate decoder": [True],
                                   "inanimate decoder": [True]}))
    assert panel.active_decoders() == {"animate decoder", "inanimate decoder"}


def test_last_sample_off_clears_active(panel):
    panel.update_decision(_result({"animate decoder": [True]}))
    panel.update_decision(_result({"animate decoder": [True, False]}))
    assert not panel.is_active("animate decoder")


def test_empty_batch_is_ignored(panel):
    panel.update_decision(_result({"animate decoder": [True]}))
    # An empty batch (n=0) must not flip the indicator off.
    panel.update_decision(_result({"animate decoder": []}))
    assert panel.is_active("animate decoder")


def test_reset_clears_state(panel):
    panel.update_decision(_result({"animate decoder": [True]}))
    panel.reset()
    assert panel.active_decoders() == set()
    assert not panel.is_active("animate decoder")
