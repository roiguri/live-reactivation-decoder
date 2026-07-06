"""Turn the live per-decoder probability stream into latched on/off decisions.

The decision layer is **per-decoder and independent** — several decoders can be
active at once. For each decoder, two composable criteria collapse to one latched
boolean per sample:

    proba[decoder] ─► [threshold] ─► [sustain gate] ─► active[decoder]

- :class:`ThresholdCriterion` is instantaneous: ``proba >= per-decoder threshold``.
- :class:`SustainGate` is the temporal, stateful part: it latches ``on`` only after
  the threshold has held continuously for ``sustain`` samples, and ``off`` only after
  it has missed for ``release`` samples — debouncing single-sample noise on both edges.

This module is **pure Python — no Qt, no I/O**. Settings default to the hardcoded
module constants below; a YAML override is layered on later (plan Phase D). The Qt
signal that carries results to the UI/logger is added by a thin binding in Phase B,
so the logic here stays unit-testable without an event loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

# Hardcoded defaults — the sole source of decision settings until the optional
# YAML override lands (plan Phase D). Iterate on these numbers here / via the UI
# before freezing them into a config schema.
DEFAULT_THRESHOLD = 0.85
DEFAULT_SUSTAIN_SECONDS = 0.1
DEFAULT_RELEASE_SECONDS = 0.0


@dataclass(frozen=True)
class DecisionConfig:
    """Immutable snapshot of the tunable decision rule.

    ``threshold`` is the global positive-class gate; ``thresholds`` holds optional
    per-decoder overrides (a decoder absent from the map falls back to the global).
    ``sustain_seconds`` / ``release_seconds`` are in human units and converted to
    samples by the engine against ``target_sfreq``.
    """

    threshold: float = DEFAULT_THRESHOLD
    thresholds: Mapping[str, float] = field(default_factory=dict)
    sustain_seconds: float = DEFAULT_SUSTAIN_SECONDS
    release_seconds: float = DEFAULT_RELEASE_SECONDS

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")
        for name, value in self.thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"threshold for '{name}' must be in [0, 1], got {value}."
                )
        if self.sustain_seconds < 0:
            raise ValueError(
                f"sustain_seconds must be non-negative, got {self.sustain_seconds}."
            )
        if self.release_seconds < 0:
            raise ValueError(
                f"release_seconds must be non-negative, got {self.release_seconds}."
            )

    def threshold_for(self, decoder: str) -> float:
        """Per-decoder threshold, falling back to the global ``threshold``."""
        return float(self.thresholds.get(decoder, self.threshold))


class ThresholdCriterion:
    """Instantaneous per-decoder gate: ``proba >= per-decoder threshold``.

    Stateless — the config is passed at evaluation time so a live config change
    takes effect on the next sample without rebuilding the criterion.
    """

    def __init__(self, decoders: list[str]) -> None:
        self._decoders = tuple(decoders)

    def evaluate(
        self, probs: Mapping[str, float], config: DecisionConfig
    ) -> dict[str, bool]:
        return {
            name: float(probs[name]) >= config.threshold_for(name)
            for name in self._decoders
        }


class SustainGate:
    """Per-decoder, independent latch with both-edge debounce.

    Each decoder tracks its own latched state plus two run counters. It latches
    ``on`` once the threshold has passed for ``sustain_samples`` consecutive samples,
    and ``off`` once it has missed for ``release_samples`` consecutive samples. A
    ``sustain``/``release`` of 0 seconds resolves to 1 sample (latch on the first
    pass / drop on the first miss).
    """

    def __init__(
        self, decoders: list[str], sustain_samples: int, release_samples: int
    ) -> None:
        self._decoders = list(decoders)
        self._sustain = max(1, int(sustain_samples))
        self._release = max(1, int(release_samples))
        self.reset()

    def reset(self) -> None:
        """Clear latches and counters — a fresh run (all decoders off)."""
        self._latched = {d: False for d in self._decoders}
        self._on = {d: 0 for d in self._decoders}
        self._off = {d: 0 for d in self._decoders}

    def reset_counters(self) -> None:
        """Zero the debounce run counters but keep latched state.

        Used on a live config change: a run accrued under a different threshold is
        meaningless, but an already-active decoder shouldn't blink off on a knob move.
        """
        self._on = {d: 0 for d in self._decoders}
        self._off = {d: 0 for d in self._decoders}

    def set_windows(self, sustain_samples: int, release_samples: int) -> None:
        """Update the debounce windows (on a config change). Latches unaffected."""
        self._sustain = max(1, int(sustain_samples))
        self._release = max(1, int(release_samples))

    def step(self, passed: Mapping[str, bool]) -> dict[str, bool]:
        """Advance every decoder one sample; return the new latched booleans."""
        for decoder in self._decoders:
            if passed[decoder]:
                self._on_pass(decoder)
            else:
                self._on_miss(decoder)
        return dict(self._latched)

    def _on_pass(self, decoder: str) -> None:
        if self._latched[decoder]:
            self._off[decoder] = 0
            return
        self._on[decoder] += 1
        if self._on[decoder] >= self._sustain:
            self._latched[decoder] = True
            self._off[decoder] = 0

    def _on_miss(self, decoder: str) -> None:
        if not self._latched[decoder]:
            self._on[decoder] = 0
            return
        self._off[decoder] += 1
        if self._off[decoder] >= self._release:
            self._latched[decoder] = False
            self._on[decoder] = 0


def seconds_to_samples(seconds: float, target_sfreq: float) -> int:
    """Convert a debounce window in seconds to whole samples (min 1)."""
    return max(1, math.ceil(float(seconds) * float(target_sfreq)))
