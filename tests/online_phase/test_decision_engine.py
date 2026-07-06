from __future__ import annotations

import pytest

from backend.online_phase.decision_engine import (
    DEFAULT_THRESHOLD,
    DecisionConfig,
    SustainGate,
    ThresholdCriterion,
    seconds_to_samples,
)


# ── DecisionConfig ────────────────────────────────────────────────────────────


def test_threshold_for_falls_back_to_global():
    config = DecisionConfig(threshold=0.8, thresholds={"animate decoder": 0.6})
    assert config.threshold_for("animate decoder") == 0.6
    assert config.threshold_for("inanimate decoder") == 0.8  # fallback


def test_defaults_come_from_constants():
    config = DecisionConfig()
    assert config.threshold == DEFAULT_THRESHOLD
    assert config.threshold_for("anything") == DEFAULT_THRESHOLD


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1.5},
        {"thresholds": {"a": -0.1}},
        {"sustain_seconds": -1.0},
        {"release_seconds": -0.5},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionConfig(**kwargs)


# ── ThresholdCriterion ────────────────────────────────────────────────────────


def test_threshold_criterion_is_per_decoder_and_inclusive():
    config = DecisionConfig(threshold=0.85, thresholds={"b": 0.5})
    criterion = ThresholdCriterion(["a", "b"])
    # a: 0.85 exactly passes (>=); b uses its 0.5 override.
    assert criterion.evaluate({"a": 0.85, "b": 0.49}, config) == {"a": True, "b": False}
    assert criterion.evaluate({"a": 0.84, "b": 0.5}, config) == {"a": False, "b": True}


# ── seconds_to_samples ────────────────────────────────────────────────────────


def test_seconds_to_samples_rounds_up_and_floors_at_one():
    assert seconds_to_samples(0.1, 250.0) == 25
    assert seconds_to_samples(0.0, 250.0) == 1  # 0 s → drop/latch on first sample
    assert seconds_to_samples(0.001, 250.0) == 1  # ceil, never 0


# ── SustainGate ───────────────────────────────────────────────────────────────


def _run(gate, decoder, passes):
    """Feed a sequence of pass/miss booleans; return the latched state per step."""
    return [gate.step({decoder: p})[decoder] for p in passes]


def test_latch_fires_exactly_sustain_samples_after_crossing():
    gate = SustainGate(["d"], sustain_samples=3, release_samples=1)
    # Off for the first two passing samples; latches on the 3rd.
    assert _run(gate, "d", [True, True, True, True]) == [False, False, True, True]


def test_sub_sustain_blip_never_latches():
    gate = SustainGate(["d"], sustain_samples=3, release_samples=1)
    assert _run(gate, "d", [True, True, False, True, True]) == [
        False, False, False, False, False
    ]  # the miss resets the run before it reaches 3


def test_release_after_release_samples_of_misses():
    gate = SustainGate(["d"], sustain_samples=1, release_samples=3)
    # Latches immediately (sustain=1); releases only after 3 consecutive misses.
    assert _run(gate, "d", [True, False, False, False, False]) == [
        True, True, True, False, False
    ]


def test_release_zero_drops_on_first_miss():
    gate = SustainGate(["d"], sustain_samples=1, release_samples=1)
    assert _run(gate, "d", [True, True, False, True]) == [True, True, False, True]


def test_momentary_dip_does_not_release_when_release_window_generous():
    gate = SustainGate(["d"], sustain_samples=1, release_samples=3)
    # A single-sample dip inside the release window doesn't unlatch.
    assert _run(gate, "d", [True, True, False, True, True]) == [
        True, True, True, True, True
    ]


def test_decoders_are_independent():
    gate = SustainGate(["a", "b"], sustain_samples=2, release_samples=1)
    # a passes continuously and latches; b never passes and stays off.
    out = [gate.step({"a": True, "b": False}) for _ in range(3)]
    assert [o["a"] for o in out] == [False, True, True]
    assert [o["b"] for o in out] == [False, False, False]


def test_reset_counters_keeps_latches_but_zeros_runs():
    gate = SustainGate(["d"], sustain_samples=3, release_samples=3)
    _run(gate, "d", [True, True, True])  # latched
    gate.reset_counters()
    # Still latched; a single miss must not release (needs 3 now from a clean run).
    assert _run(gate, "d", [False, False]) == [True, True]
    assert _run(gate, "d", [False]) == [False]  # 3rd consecutive miss releases


def test_set_windows_takes_effect_going_forward():
    gate = SustainGate(["d"], sustain_samples=5, release_samples=1)
    gate.set_windows(sustain_samples=2, release_samples=1)
    gate.reset_counters()
    assert _run(gate, "d", [True, True]) == [False, True]  # now latches after 2
