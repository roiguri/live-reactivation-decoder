"""Turn the live per-decoder probability stream into latched on/off decisions.

The decision layer is **per-decoder and independent** — several decoders can be
active at once. For each decoder, two composable criteria collapse to one latched
boolean per sample:

    proba[decoder] ─► [threshold] ─► [sustain gate] ─► active[decoder]

- :class:`ThresholdCriterion` is instantaneous: ``proba >= threshold``.
- :class:`SustainGate` is the temporal, stateful part: it latches ``on`` only after
  the threshold has held continuously for ``sustain`` timepoints, and ``off`` only
  after it has missed for ``release`` timepoints — debouncing noise on both edges.

Everything is counted in **timepoints** — one timepoint is one prediction (the unit
``prediction_ready`` already delivers). Because predictions/decisions are inherently
per-timepoint, no sampling frequency is needed: the gate simply counts the
predictions it receives.

This module is **pure Python — no Qt, no I/O**. Settings default to the hardcoded
module constants below; a YAML override is layered on later (plan Phase D). The Qt
signal that carries results to the UI/logger is added by a thin binding in Phase B,
so the logic here stays unit-testable without an event loop.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping

import numpy as np

# Hardcoded defaults — the sole source of decision settings until the optional
# YAML override lands (plan Phase D). Counted in timepoints (one prediction each).
DEFAULT_THRESHOLD = 0.85
DEFAULT_SUSTAIN_TIMEPOINTS = 10
DEFAULT_RELEASE_TIMEPOINTS = 1


@dataclass(frozen=True)
class DecisionConfig:
    """Immutable snapshot of the tunable decision rule.

    ``threshold`` is the single global positive-class gate, applied to every
    decoder (decoders still latch independently — they just share this value).
    ``sustain_timepoints`` / ``release_timepoints`` are counts of predictions: how
    many consecutive over-threshold predictions latch a decoder on, and how many
    consecutive misses latch it off. No frequency is involved — a timepoint is one
    prediction.
    """

    threshold: float = DEFAULT_THRESHOLD
    sustain_timepoints: int = DEFAULT_SUSTAIN_TIMEPOINTS
    release_timepoints: int = DEFAULT_RELEASE_TIMEPOINTS

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")
        if self.sustain_timepoints < 1:
            raise ValueError(
                f"sustain_timepoints must be >= 1, got {self.sustain_timepoints}."
            )
        if self.release_timepoints < 1:
            raise ValueError(
                f"release_timepoints must be >= 1, got {self.release_timepoints}."
            )


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
            name: float(probs[name]) >= config.threshold
            for name in self._decoders
        }


class SustainGate:
    """Per-decoder, independent latch with both-edge debounce.

    Each decoder tracks its own latched state plus two run counters. It latches
    ``on`` once the threshold has passed for ``sustain_timepoints`` consecutive
    timepoints, and ``off`` once it has missed for ``release_timepoints`` consecutive
    timepoints. Both windows are at least 1 (latch on the first pass / drop on the
    first miss).
    """

    def __init__(
        self, decoders: list[str], sustain_timepoints: int, release_timepoints: int
    ) -> None:
        self._decoders = list(decoders)
        self._sustain = max(1, int(sustain_timepoints))
        self._release = max(1, int(release_timepoints))
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

    def set_windows(self, sustain_timepoints: int, release_timepoints: int) -> None:
        """Update the debounce windows (on a config change). Latches unaffected."""
        self._sustain = max(1, int(sustain_timepoints))
        self._release = max(1, int(release_timepoints))

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


@dataclass(frozen=True)
class ConfigChange:
    """A committed config version, stamped at the sample it took effect.

    ``config`` is the public, decoder-expanded snapshot (human units) the timeline
    (`decision_config.jsonl`) records. ``lsl_timestamp`` is ``None`` only for the
    implicit version 0 (in effect from the start).
    """

    version: int
    lsl_timestamp: float | None
    config: dict


@dataclass(frozen=True)
class DecisionResult:
    """One batch of latched decisions, mirroring one ``prediction_ready`` emission.

    ``active`` is per-decoder boolean, row-aligned with ``timestamps``.
    ``config_version`` is constant across the batch (a staged config is only ever
    applied at the batch boundary). ``config_change`` is set only on the batch that
    applied a new config, carrying the snapshot the logger appends to the timeline.
    """

    timestamps: np.ndarray
    active: dict[str, np.ndarray]
    config_version: int
    config_change: ConfigChange | None = None


class DecisionEngine:
    """Per-decoder latched decisions over the live probability stream.

    ``process_batch`` consumes exactly what ``prediction_ready`` carries — a dict of
    per-decoder probability vectors plus their timestamps — and threads the
    :class:`SustainGate` state across batch boundaries. It is the single owner of the
    decision state and mutates it only on the worker thread; the sole cross-thread
    entry point is :meth:`set_pending_config`, which stashes a config the next
    ``process_batch`` applies at its boundary (see the plan's "Live config changes").
    """

    def __init__(
        self,
        decoder_names: list[str],
        config: DecisionConfig,
    ) -> None:
        decoders = list(decoder_names)
        if not decoders:
            raise ValueError("DecisionEngine requires at least one decoder.")

        self._decoders = decoders
        self._config = config
        self._version = 0
        self._criterion = ThresholdCriterion(decoders)
        self._gate = SustainGate(
            decoders, config.sustain_timepoints, config.release_timepoints
        )

        self._pending: DecisionConfig | None = None
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────────

    @property
    def config_version(self) -> int:
        return self._version

    def public_config(self) -> dict:
        """The current config as the decoder-expanded public snapshot (for version 0)."""
        return self._public_config(self._config)

    def set_pending_config(self, config: DecisionConfig) -> None:
        """Stage a config for the next batch to apply. Thread-safe (UI thread)."""
        with self._lock:
            self._pending = config

    def reset(self) -> None:
        """Clear all latch state for a fresh run (called on session start)."""
        self._gate.reset()

    def process_batch(
        self, predictions: Mapping[str, np.ndarray], timestamps: np.ndarray
    ) -> DecisionResult:
        """Advance the decision state over one batch; return per-sample latches."""
        timestamps = np.asarray(timestamps, dtype=float)
        n = timestamps.shape[0]
        if n == 0:
            # Nothing to decide and no timestamp to stamp a config change against;
            # leave any pending config for the next non-empty batch.
            empty = {d: np.empty(0, dtype=bool) for d in self._decoders}
            return DecisionResult(timestamps, empty, self._version, None)

        change = self._maybe_apply_pending(timestamps)
        columns = self._batch_columns(predictions, n)
        active = {d: np.empty(n, dtype=bool) for d in self._decoders}

        for i in range(n):
            probs = {d: columns[d][i] for d in self._decoders}
            latched = self._gate.step(self._criterion.evaluate(probs, self._config))
            for decoder in self._decoders:
                active[decoder][i] = latched[decoder]

        return DecisionResult(timestamps, active, self._version, change)

    # ── internals ────────────────────────────────────────────────────────────────

    def _maybe_apply_pending(self, timestamps: np.ndarray) -> ConfigChange | None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return None

        self._config = pending
        self._version += 1
        self._gate.set_windows(pending.sustain_timepoints, pending.release_timepoints)
        self._gate.reset_counters()  # keep latches; drop stale runs
        return ConfigChange(
            version=self._version,
            lsl_timestamp=float(timestamps[0]),
            config=self._public_config(pending),
        )

    def _public_config(self, config: DecisionConfig) -> dict:
        return {
            "threshold": config.threshold,
            "sustain_timepoints": config.sustain_timepoints,
            "release_timepoints": config.release_timepoints,
        }

    def _batch_columns(
        self, predictions: Mapping[str, np.ndarray], n_rows: int
    ) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {}
        for name in self._decoders:
            if name not in predictions:
                raise ValueError(
                    f"Predictions missing configured decoder '{name}'. "
                    f"Got {sorted(predictions)}."
                )
            col = np.asarray(predictions[name], dtype=float)
            if col.shape != (n_rows,):
                raise ValueError(
                    f"Prediction vector for '{name}' has shape {col.shape}, "
                    f"expected ({n_rows},)."
                )
            columns[name] = col
        return columns
