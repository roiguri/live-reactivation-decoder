from __future__ import annotations

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.widgets.phase2.settings_panel import Phase2SettingsPanel

_COLORS = {"animate decoder": "#e6194b", "inanimate decoder": "#3cb44b"}
_DEFAULTS = {"threshold": 0.85, "sustain_timepoints": 10}


@pytest.fixture
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def panel(qapp) -> Phase2SettingsPanel:
    return Phase2SettingsPanel(task_colors=_COLORS, decision_defaults=dict(_DEFAULTS))


def test_seeds_controls_from_defaults(panel):
    assert panel.applied_params() == _DEFAULTS
    assert panel.draft_params() == _DEFAULTS
    assert not panel.is_dirty()
    # Clean state → Apply/Reset disabled.
    assert not panel._apply_button.isEnabled()
    assert not panel._reset_button.isEnabled()


def test_editing_marks_dirty_but_does_not_emit(panel):
    received = []
    panel.decision_params_changed.connect(received.append)

    panel._threshold_slider.setValue(70)  # 0.70

    assert panel.is_dirty()
    assert panel._apply_button.isEnabled()
    assert panel.applied_params() == _DEFAULTS  # unchanged until Apply
    assert received == []  # editing alone emits nothing


def test_apply_commits_and_emits(panel):
    received = []
    panel.decision_params_changed.connect(received.append)

    panel._threshold_slider.setValue(70)
    panel._sustain_spin.setValue(25)
    panel._apply_decision_draft()

    assert received == [{"threshold": 0.70, "sustain_timepoints": 25}]
    assert panel.applied_params() == {"threshold": 0.70, "sustain_timepoints": 25}
    # After Apply, draft == applied → clean, buttons disabled.
    assert not panel.is_dirty()
    assert not panel._apply_button.isEnabled()


def test_reset_reverts_draft_without_emitting(panel):
    received = []
    panel.decision_params_changed.connect(received.append)

    panel._threshold_slider.setValue(30)
    assert panel.is_dirty()

    panel._reset_decision_draft()

    assert panel.draft_params() == _DEFAULTS
    assert not panel.is_dirty()
    assert received == []  # Reset never emits


def test_threshold_slider_and_spinbox_stay_in_sync(panel):
    panel._threshold_slider.setValue(42)
    assert panel._threshold_spin.value() == pytest.approx(0.42)
    # Manual spinbox entry mirrors back into the slider.
    panel._threshold_spin.setValue(0.63)
    assert panel._threshold_slider.value() == 63
    assert panel.draft_params()["threshold"] == pytest.approx(0.63)


def test_manual_threshold_entry_marks_dirty(panel):
    received = []
    panel.decision_params_changed.connect(received.append)
    panel._threshold_spin.setValue(0.55)  # typed directly, no slider drag
    assert panel.is_dirty()
    assert panel._apply_button.isEnabled()
    assert received == []
