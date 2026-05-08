from __future__ import annotations

import numpy as np
import pytest

from backend.online_phase.lsl_receiver import (
    LSLReceiver,
    decode_trigger_value,
    extract_markers_from_trigger_channel,
    split_eeg_and_markers,
)


def _chunk_from_trigger_values(trigger_values: list[int]) -> np.ndarray:
    chunk = np.zeros((len(trigger_values), 65), dtype=float)
    chunk[:, 64] = trigger_values
    chunk[:, 0] = np.arange(len(trigger_values), dtype=float)
    return chunk


class FakeInlet:
    def __init__(self, chunks: list[tuple[np.ndarray, list[float]]]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def pull_chunk(self, timeout: float = 0.0):
        if not self._chunks:
            return [], []
        return self._chunks.pop(0)

    def close_stream(self) -> None:
        self.closed = True


def test_decode_trigger_value_extracts_parallel_port_bits():
    assert decode_trigger_value(0) == 0
    assert decode_trigger_value(1 << 8) == 1
    assert decode_trigger_value(98 << 8) == 98
    assert decode_trigger_value((99 << 8) + 3) == 99


def test_extract_markers_from_trigger_channel_detects_edges_only():
    trigger_values = np.asarray([0, 1 << 8, 1 << 8, 0, 2 << 8, 2 << 8, 0], dtype=float)

    markers, last_code = extract_markers_from_trigger_channel(trigger_values)

    assert markers == [1, 2]
    assert last_code == 0


def test_split_eeg_and_markers_removes_trigger_channel():
    chunk = _chunk_from_trigger_values([0, 1 << 8, 0])

    eeg_chunk, markers, last_code = split_eeg_and_markers(chunk)

    assert eeg_chunk.shape == (3, 64)
    assert np.array_equal(eeg_chunk[:, 0], np.array([0.0, 1.0, 2.0]))
    assert markers == [1]
    assert last_code == 0


def test_split_eeg_and_markers_raises_when_trigger_channel_missing():
    with pytest.raises(ValueError, match="out of bounds"):
        split_eeg_and_markers(np.zeros((5, 64), dtype=float))


def test_pull_new_data_drains_available_chunks_and_decodes_markers():
    chunk1 = _chunk_from_trigger_values([0, 1 << 8])
    chunk2 = _chunk_from_trigger_values([1 << 8, 0, 2 << 8])
    inlet = FakeInlet(
        [
            (chunk1, [1.0, 1.001]),
            (chunk2, [1.002, 1.003, 1.004]),
        ]
    )
    receiver = LSLReceiver(launch_proxy=False)
    receiver.inlet = inlet

    timestamps, eeg_chunk, markers = receiver.pull_new_data()

    assert np.array_equal(timestamps, np.array([1.0, 1.001, 1.002, 1.003, 1.004]))
    assert eeg_chunk.shape == (5, 64)
    assert markers == [1, 2]


def test_pull_new_data_preserves_trigger_state_across_calls():
    receiver = LSLReceiver(launch_proxy=False)
    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([0, 1 << 8, 1 << 8]), [1.0, 1.001, 1.002])])

    _, _, markers_first = receiver.pull_new_data()

    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([1 << 8, 0, 1 << 8]), [1.003, 1.004, 1.005])])
    _, _, markers_second = receiver.pull_new_data()

    assert markers_first == [1]
    assert markers_second == [1]


def test_stop_closes_inlet_when_present():
    receiver = LSLReceiver(launch_proxy=False)
    inlet = FakeInlet([])
    receiver.inlet = inlet

    receiver.stop()

    assert inlet.closed is True
    assert receiver.inlet is None
