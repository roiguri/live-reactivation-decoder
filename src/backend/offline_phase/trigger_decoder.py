"""Decode parallel-port trigger pulses recorded as analog voltage on the
trigger channel into MNE Annotations.

In the current recording setup, the trigger channel is named "EMG" but carries
hardware-smoothed parallel-port byte values rather than muscle activity. Each
trigger byte appears as a brief voltage plateau whose height equals
``code / VOLTAGE_TO_CODE_SCALE`` volts (e.g. code 11 → 1.1 mV).

Descriptions are emitted in BrainVision "Stimulus/S{code}" form so that
``mne.events_from_annotations`` extracts the integer code unchanged.
"""

from __future__ import annotations

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)

TRIGGER_CHANNEL_NAME = "EMG"
VOLTAGE_TO_CODE_SCALE = 1.0e4
NOISE_THRESHOLD_CODE_UNITS = 0.5
PULSE_MIN_DURATION_MS = 5.0
PLATEAU_WINDOW_MS = 10.0


def decode_parallel_port_channel(raw: mne.io.Raw) -> mne.Annotations:
    """Extract parallel-port trigger events from the analog trigger channel.

    Returns Annotations with one entry per detected pulse. The trigger channel
    itself is left on ``raw``; the caller is expected to drop it after
    annotations are set.

    Raises:
        ValueError: if the trigger channel is missing from ``raw``.
    """
    if TRIGGER_CHANNEL_NAME not in raw.ch_names:
        raise ValueError(
            f"Trigger channel '{TRIGGER_CHANNEL_NAME}' not found in raw "
            f"(channels: {raw.ch_names!r})"
        )

    sfreq = raw.info["sfreq"]
    plateau_win_samples = max(1, int(PLATEAU_WINDOW_MS * sfreq / 1000.0))
    min_duration_samples = max(1, int(PULSE_MIN_DURATION_MS * sfreq / 1000.0))

    trigger_volts = raw.get_data(picks=[TRIGGER_CHANNEL_NAME])[0]
    scaled = trigger_volts * VOLTAGE_TO_CODE_SCALE

    above_noise = scaled > NOISE_THRESHOLD_CODE_UNITS
    edges = np.flatnonzero(np.diff(above_noise.astype(np.int8)) == 1) + 1
    n_samples = len(scaled)

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []

    for edge in edges:
        end = edge
        while end < n_samples and above_noise[end]:
            end += 1
        pulse_samples = end - edge
        if pulse_samples < min_duration_samples:
            continue

        window_end = min(edge + plateau_win_samples, n_samples)
        code = int(round(scaled[edge:window_end].max()))
        if code <= 0:
            continue

        onsets.append(float(edge) / sfreq)
        durations.append(float(pulse_samples) / sfreq)
        descriptions.append(f"Stimulus/S{code:3d}")

    logger.info(
        "Decoded %d parallel-port pulses from channel %r",
        len(onsets),
        TRIGGER_CHANNEL_NAME,
    )

    return mne.Annotations(
        onset=np.asarray(onsets, dtype=float),
        duration=np.asarray(durations, dtype=float),
        description=descriptions,
        orig_time=raw.info.get("meas_date"),
    )
