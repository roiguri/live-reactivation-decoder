from __future__ import annotations

import mne
import numpy as np
import pytest

from backend.offline_phase.trigger_decoder import (
    NOISE_THRESHOLD_CODE_UNITS,
    PLATEAU_WINDOW_MS,
    PULSE_MIN_DURATION_MS,
    TRIGGER_CHANNEL_NAME,
    VOLTAGE_TO_CODE_SCALE,
    decode_parallel_port_channel,
)

SFREQ = 5000.0  # matches production NeurOne setup
DURATION_S = 10.0
N_SAMPLES = int(SFREQ * DURATION_S)


def _build_raw(trigger_signal: np.ndarray) -> mne.io.RawArray:
    """Wrap a 1-D trigger waveform plus a couple of dummy EEG channels into a Raw."""
    rng = np.random.default_rng(0)
    eeg = rng.standard_normal((2, len(trigger_signal))) * 5e-6
    data = np.vstack([eeg, trigger_signal[np.newaxis, :]])
    info = mne.create_info(
        ch_names=["Fp1", "Fp2", TRIGGER_CHANNEL_NAME],
        sfreq=SFREQ,
        ch_types=["eeg", "eeg", "eeg"],
    )
    return mne.io.RawArray(data, info, verbose=False)


def _stamp_pulse(signal: np.ndarray, onset_sample: int, code: int,
                 duration_ms: float) -> None:
    """Write a flat plateau of `code` (in voltage units) into the signal."""
    n_samples = int(duration_ms * SFREQ / 1000.0)
    voltage = code / VOLTAGE_TO_CODE_SCALE
    signal[onset_sample : onset_sample + n_samples] = voltage


class TestDecodeParallelPortChannel:
    def test_raises_when_trigger_channel_missing(self) -> None:
        info = mne.create_info(ch_names=["Fp1", "Fp2"], sfreq=SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose=False)
        with pytest.raises(ValueError, match=TRIGGER_CHANNEL_NAME):
            decode_parallel_port_channel(raw)

    def test_returns_empty_annotations_when_no_pulses(self) -> None:
        signal = np.zeros(N_SAMPLES)
        raw = _build_raw(signal)
        ann = decode_parallel_port_channel(raw)
        assert len(ann) == 0

    def test_decodes_single_clean_pulse(self) -> None:
        signal = np.zeros(N_SAMPLES)
        onset = int(1.0 * SFREQ)
        _stamp_pulse(signal, onset, code=11, duration_ms=25)

        ann = decode_parallel_port_channel(_build_raw(signal))

        assert len(ann) == 1
        assert ann.description[0] == "Stimulus/S 11"
        assert ann.onset[0] == pytest.approx(1.0, abs=1 / SFREQ)
        assert ann.duration[0] == pytest.approx(0.025, abs=2 / SFREQ)

    def test_decodes_multiple_distinct_codes(self) -> None:
        signal = np.zeros(N_SAMPLES)
        spec = [(1.0, 11, 20), (2.5, 12, 20), (4.0, 13, 20), (5.5, 18, 25)]
        for t, code, dur_ms in spec:
            _stamp_pulse(signal, int(t * SFREQ), code, dur_ms)

        ann = decode_parallel_port_channel(_build_raw(signal))

        assert len(ann) == len(spec)
        expected_descs = {f"Stimulus/S{c:3d}" for _, c, _ in spec}
        assert set(ann.description) == expected_descs

    def test_drops_pulses_below_min_duration(self) -> None:
        # 1 ms blip — under the PULSE_MIN_DURATION_MS = 5 ms threshold.
        signal = np.zeros(N_SAMPLES)
        _stamp_pulse(signal, int(1.0 * SFREQ), code=11,
                     duration_ms=PULSE_MIN_DURATION_MS / 2)
        # Plus one real pulse to confirm the rest still decodes.
        _stamp_pulse(signal, int(3.0 * SFREQ), code=12,
                     duration_ms=PULSE_MIN_DURATION_MS * 4)

        ann = decode_parallel_port_channel(_build_raw(signal))

        assert len(ann) == 1
        assert ann.description[0] == "Stimulus/S 12"

    def test_uses_plateau_peak_not_leading_edge(self) -> None:
        # Real recordings ramp up smoothly to the plateau. The decoder must
        # read the plateau (peak in the post-edge window), not the first
        # above-noise sample.
        signal = np.zeros(N_SAMPLES)
        onset = int(2.0 * SFREQ)
        ramp_samples = int(2 * SFREQ / 1000.0)        # 2 ms ramp
        plateau_code = 17
        # Linear ramp from 1 → plateau_code, then hold for 30 ms total
        ramp = np.linspace(1, plateau_code, ramp_samples) / VOLTAGE_TO_CODE_SCALE
        plateau_samples = int(28 * SFREQ / 1000.0)
        signal[onset : onset + ramp_samples] = ramp
        signal[onset + ramp_samples : onset + ramp_samples + plateau_samples] = (
            plateau_code / VOLTAGE_TO_CODE_SCALE
        )

        ann = decode_parallel_port_channel(_build_raw(signal))

        assert len(ann) == 1
        assert ann.description[0] == f"Stimulus/S{plateau_code:3d}"

    def test_noise_below_threshold_does_not_trigger(self) -> None:
        # Background noise just under the noise floor (0.5 code units).
        rng = np.random.default_rng(0)
        noise_amp = (NOISE_THRESHOLD_CODE_UNITS - 0.1) / VOLTAGE_TO_CODE_SCALE
        signal = rng.uniform(-noise_amp, noise_amp, size=N_SAMPLES)

        ann = decode_parallel_port_channel(_build_raw(signal))
        assert len(ann) == 0

    def test_annotations_round_trip_through_events_from_annotations(self) -> None:
        # Downstream pipeline uses mne.events_from_annotations to extract
        # integer codes. Confirm the "Stimulus/S<n>" descriptions parse back
        # to the original integers.
        signal = np.zeros(N_SAMPLES)
        for t, code in [(1.0, 11), (2.0, 12), (3.0, 13)]:
            _stamp_pulse(signal, int(t * SFREQ), code, duration_ms=25)
        raw = _build_raw(signal)
        raw.set_annotations(decode_parallel_port_channel(raw))

        events, found_event_id = mne.events_from_annotations(raw, verbose="ERROR")

        assert set(found_event_id.values()) == {11, 12, 13}
        assert events.shape == (3, 3)


class TestModuleConstants:
    def test_plateau_window_at_least_one_sample(self) -> None:
        # If sfreq somehow makes PLATEAU_WINDOW_MS round to zero samples,
        # the decoder uses max(1, ...) — sanity-check the constants are sane.
        assert PLATEAU_WINDOW_MS > 0
        assert PULSE_MIN_DURATION_MS > 0
        assert VOLTAGE_TO_CODE_SCALE > 0
        assert NOISE_THRESHOLD_CODE_UNITS > 0
