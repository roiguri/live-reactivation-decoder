from __future__ import annotations

import numpy as np
import pytest

from backend.online_phase.decision_engine import (
    DEFAULT_THRESHOLD,
    DecisionConfig,
    DecisionEngine,
    SustainGate,
    ThresholdCriterion,
)


# ── DecisionConfig ────────────────────────────────────────────────────────────


def test_defaults_come_from_constants():
    config = DecisionConfig()
    assert config.threshold == DEFAULT_THRESHOLD


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1.5},
        {"sustain_timepoints": 0},
        {"release_timepoints": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        DecisionConfig(**kwargs)


# ── ThresholdCriterion ────────────────────────────────────────────────────────


def test_threshold_criterion_is_global_and_inclusive():
    config = DecisionConfig(threshold=0.85)
    criterion = ThresholdCriterion(["a", "b"])
    # 0.85 exactly passes (>=); the same global threshold gates every decoder.
    assert criterion.evaluate({"a": 0.85, "b": 0.84}, config) == {"a": True, "b": False}
    assert criterion.evaluate({"a": 0.20, "b": 0.90}, config) == {"a": False, "b": True}


# ── SustainGate ───────────────────────────────────────────────────────────────


def _run(gate, decoder, passes):
    """Feed a sequence of pass/miss booleans; return the latched state per step."""
    return [gate.step({decoder: p})[decoder] for p in passes]


def test_latch_fires_exactly_sustain_timepoints_after_crossing():
    gate = SustainGate(["d"], sustain_timepoints=3, release_timepoints=1)
    # Off for the first two passing timepoints; latches on the 3rd.
    assert _run(gate, "d", [True, True, True, True]) == [False, False, True, True]


def test_sub_sustain_blip_never_latches():
    gate = SustainGate(["d"], sustain_timepoints=3, release_timepoints=1)
    assert _run(gate, "d", [True, True, False, True, True]) == [
        False, False, False, False, False
    ]  # the miss resets the run before it reaches 3


def test_release_after_release_timepoints_of_misses():
    gate = SustainGate(["d"], sustain_timepoints=1, release_timepoints=3)
    # Latches immediately (sustain=1); releases only after 3 consecutive misses.
    assert _run(gate, "d", [True, False, False, False, False]) == [
        True, True, True, False, False
    ]


def test_release_one_drops_on_first_miss():
    gate = SustainGate(["d"], sustain_timepoints=1, release_timepoints=1)
    assert _run(gate, "d", [True, True, False, True]) == [True, True, False, True]


def test_momentary_dip_does_not_release_when_release_window_generous():
    gate = SustainGate(["d"], sustain_timepoints=1, release_timepoints=3)
    # A single-timepoint dip inside the release window doesn't unlatch.
    assert _run(gate, "d", [True, True, False, True, True]) == [
        True, True, True, True, True
    ]


def test_decoders_are_independent():
    gate = SustainGate(["a", "b"], sustain_timepoints=2, release_timepoints=1)
    # a passes continuously and latches; b never passes and stays off.
    out = [gate.step({"a": True, "b": False}) for _ in range(3)]
    assert [o["a"] for o in out] == [False, True, True]
    assert [o["b"] for o in out] == [False, False, False]


def test_reset_counters_keeps_latches_but_zeros_runs():
    gate = SustainGate(["d"], sustain_timepoints=3, release_timepoints=3)
    _run(gate, "d", [True, True, True])  # latched
    gate.reset_counters()
    # Still latched; a single miss must not release (needs 3 now from a clean run).
    assert _run(gate, "d", [False, False]) == [True, True]
    assert _run(gate, "d", [False]) == [False]  # 3rd consecutive miss releases


def test_set_windows_takes_effect_going_forward():
    gate = SustainGate(["d"], sustain_timepoints=5, release_timepoints=1)
    gate.set_windows(sustain_timepoints=2, release_timepoints=1)
    gate.reset_counters()
    assert _run(gate, "d", [True, True]) == [False, True]  # now latches after 2


# ── DecisionEngine ────────────────────────────────────────────────────────────


def _engine(decoders=("a", "b"), sustain_timepoints=1, release_timepoints=1, **cfg):
    # sustain 1 timepoint: latch on the first pass (easy to reason about).
    config = DecisionConfig(
        sustain_timepoints=sustain_timepoints,
        release_timepoints=release_timepoints,
        **cfg,
    )
    return DecisionEngine(list(decoders), config)


def test_process_batch_returns_per_decoder_active_arrays():
    engine = _engine(threshold=0.5)
    result = engine.process_batch(
        {"a": np.array([0.9, 0.1]), "b": np.array([0.2, 0.8])},
        np.array([10.0, 11.0]),
    )
    assert result.config_version == 0
    assert result.config_change is None
    np.testing.assert_array_equal(result.active["a"], [True, False])
    np.testing.assert_array_equal(result.active["b"], [False, True])


def test_sustain_window_spans_batch_boundary():
    engine = _engine(decoders=("a",), sustain_timepoints=3, threshold=0.5)
    r1 = engine.process_batch({"a": np.array([0.9, 0.9])}, np.array([0.0, 1.0]))
    r2 = engine.process_batch({"a": np.array([0.9, 0.9])}, np.array([2.0, 3.0]))
    # Latches on the 3rd consecutive pass — the first timepoint of the second batch.
    np.testing.assert_array_equal(r1.active["a"], [False, False])
    np.testing.assert_array_equal(r2.active["a"], [True, True])


def test_pending_config_applies_at_next_batch_boundary():
    engine = _engine(decoders=("a",), threshold=0.9)
    # 0.8 does not pass at threshold 0.9.
    r1 = engine.process_batch({"a": np.array([0.8])}, np.array([100.0]))
    assert r1.config_version == 0 and not r1.active["a"][0]

    engine.set_pending_config(DecisionConfig(threshold=0.5, sustain_timepoints=1))
    # Not applied yet — only stashed.
    assert engine.config_version == 0

    r2 = engine.process_batch({"a": np.array([0.8])}, np.array([101.0]))
    assert r2.config_version == 1
    assert r2.active["a"][0]  # 0.8 now passes at threshold 0.5
    change = r2.config_change
    assert change is not None
    assert change.version == 1
    assert change.lsl_timestamp == 101.0  # stamped at the batch's first timepoint
    assert change.config["threshold"] == 0.5
    assert change.config["sustain_timepoints"] == 1


def test_config_change_resets_counters_but_keeps_latches():
    engine = _engine(decoders=("a",), threshold=0.5, sustain_timepoints=3)
    engine.process_batch({"a": np.array([0.9, 0.9, 0.9])}, np.array([0.0, 1.0, 2.0]))
    # Latched. Now change only the threshold; latch must survive, counters reset.
    engine.set_pending_config(
        DecisionConfig(threshold=0.6, sustain_timepoints=3, release_timepoints=3)
    )
    r = engine.process_batch({"a": np.array([0.1])}, np.array([3.0]))
    # A single miss must not release (release is 3 timepoints from a clean run).
    assert r.active["a"][0]


def test_empty_batch_leaves_pending_untouched():
    engine = _engine(decoders=("a",), threshold=0.5)
    engine.set_pending_config(DecisionConfig(threshold=0.1))
    empty = engine.process_batch({"a": np.array([])}, np.array([]))
    assert empty.config_version == 0 and empty.config_change is None
    assert empty.active["a"].shape == (0,)
    # Pending survives to the next real batch.
    r = engine.process_batch({"a": np.array([0.2])}, np.array([5.0]))
    assert r.config_version == 1


def test_reset_clears_latches():
    engine = _engine(decoders=("a",), threshold=0.5)
    engine.process_batch({"a": np.array([0.9])}, np.array([0.0]))
    engine.reset()
    r = engine.process_batch({"a": np.array([0.1])}, np.array([1.0]))
    assert not r.active["a"][0]


def test_missing_decoder_and_bad_shape_rejected():
    engine = _engine(decoders=("a", "b"))
    with pytest.raises(ValueError, match="missing configured decoder"):
        engine.process_batch({"a": np.array([0.9])}, np.array([0.0]))
    with pytest.raises(ValueError, match="expected"):
        engine.process_batch(
            {"a": np.array([0.9, 0.9]), "b": np.array([0.9])}, np.array([0.0, 1.0])
        )


def test_construction_requires_a_decoder():
    with pytest.raises(ValueError, match="at least one decoder"):
        DecisionEngine([], DecisionConfig())
