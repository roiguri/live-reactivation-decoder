from __future__ import annotations

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from backend.online_phase import lsl_receiver
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


class FakePylslModule:
    def __init__(self) -> None:
        self.created_inlets: list[FakeInlet] = []

    def StreamInlet(self, stream, recover: bool = True):
        inlet = FakeInlet([])
        self.created_inlets.append(inlet)
        return inlet


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
    assert markers == [(1, 1)]
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
    receiver = LSLReceiver()
    receiver.inlet = inlet

    timestamps, eeg_chunk, markers = receiver.pull_new_data()

    assert np.array_equal(timestamps, np.array([1.0, 1.001, 1.002, 1.003, 1.004]))
    assert eeg_chunk.shape == (5, 64)
    assert markers == [(1.001, 1), (1.004, 2)]


def test_pull_new_data_marker_timestamp_matches_trigger_sample():
    chunk = _chunk_from_trigger_values([0, 0, 7 << 8, 0])
    timestamps_in = [10.0, 10.001, 10.002, 10.003]
    receiver = LSLReceiver()
    receiver.inlet = FakeInlet([(chunk, timestamps_in)])

    _, _, markers = receiver.pull_new_data()

    assert markers == [(timestamps_in[2], 7)]


def test_pull_new_data_preserves_trigger_state_across_calls():
    receiver = LSLReceiver()
    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([0, 1 << 8, 1 << 8]), [1.0, 1.001, 1.002])])

    _, _, markers_first = receiver.pull_new_data()

    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([1 << 8, 0, 1 << 8]), [1.003, 1.004, 1.005])])
    _, _, markers_second = receiver.pull_new_data()

    assert markers_first == [(1.001, 1)]
    assert markers_second == [(1.005, 1)]


def test_pull_new_data_held_trigger_across_calls_emits_once():
    receiver = LSLReceiver()
    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([0, 1 << 8, 1 << 8]), [1.0, 1.001, 1.002])])

    _, _, markers_first = receiver.pull_new_data()

    receiver.inlet = FakeInlet([(_chunk_from_trigger_values([1 << 8, 1 << 8]), [1.003, 1.004])])
    _, _, markers_second = receiver.pull_new_data()

    assert markers_first == [(1.001, 1)]
    assert markers_second == []


def test_stop_closes_inlet_when_present():
    receiver = LSLReceiver()
    inlet = FakeInlet([])
    receiver.inlet = inlet

    receiver.stop()

    assert inlet.closed is True
    assert receiver.inlet is None


def test_pull_new_data_skips_malformed_chunks_gracefully(caplog):
    """Test that malformed chunks are logged and skipped, allowing data reception to continue."""
    # Good chunk followed by malformed chunk (wrong shape) followed by another good chunk
    good_chunk1 = _chunk_from_trigger_values([0, 1 << 8])
    malformed_chunk = np.zeros((2, 63), dtype=float)  # Wrong channel count!
    good_chunk2 = _chunk_from_trigger_values([0, 2 << 8])

    inlet = FakeInlet(
        [
            (good_chunk1, [1.0, 1.001]),
            (malformed_chunk, [1.002, 1.003]),  # This will be skipped
            (good_chunk2, [1.004, 1.005]),
        ]
    )
    receiver = LSLReceiver()
    receiver.inlet = inlet

    timestamps, eeg_chunk, markers = receiver.pull_new_data()

    # Should have data from good chunks only (2 + 2 samples)
    assert timestamps.shape[0] == 4
    assert eeg_chunk.shape == (4, 64)
    assert markers == [(1.001, 1), (1.005, 2)]

    # Should have logged a warning about the malformed chunk
    assert "Malformed chunk received" in caplog.text
    assert "Skipping" in caplog.text


def test_start_success_returns_none_and_opens_inlet(monkeypatch):
    receiver = LSLReceiver(stream_name="TestStream")

    mock_stream = MagicMock()
    mock_stream.nominal_srate.return_value = 1000
    mock_stream.channel_count.return_value = 65
    mock_stream.name.return_value = "TestStream"
    mock_stream.type.return_value = "EEG"
    mock_stream.source_id.return_value = "test_source"

    fake_pylsl = FakePylslModule()
    receiver._resolve_stream = Mock(return_value=mock_stream)
    monkeypatch.setattr(lsl_receiver, "pylsl", fake_pylsl)

    assert receiver.start() is None
    assert receiver.inlet is fake_pylsl.created_inlets[0]
    assert receiver._last_trigger_code == 0


def test_start_raises_runtime_error_when_stream_not_found():
    """Test that start() raises RuntimeError with helpful message when stream cannot be resolved."""
    receiver = LSLReceiver(stream_name="NonExistentStream")

    # Mock _resolve_stream to always return None (stream not found)
    receiver._resolve_stream = Mock(return_value=None)

    with pytest.raises(RuntimeError, match="Stream 'NonExistentStream' not found"):
        receiver.start()


def test_start_raises_value_error_when_sample_rate_wrong():
    """Test that start() raises ValueError when stream has wrong sample rate."""
    receiver = LSLReceiver(stream_name="TestStream")

    # Mock stream with wrong sample rate
    mock_stream = MagicMock()
    mock_stream.nominal_srate.return_value = 500  # Wrong! Should be 1000
    mock_stream.channel_count.return_value = 65
    mock_stream.name.return_value = "TestStream"
    mock_stream.type.return_value = "EEG"
    mock_stream.source_id.return_value = "test_source"

    receiver._resolve_stream = Mock(return_value=mock_stream)

    with pytest.raises(ValueError, match="Expected 1000 Hz stream, got 500 Hz"):
        receiver.start()


def test_start_raises_value_error_when_channel_count_wrong():
    """Test that start() raises ValueError when stream has wrong channel count."""
    receiver = LSLReceiver(stream_name="TestStream")

    # Mock stream with wrong channel count
    mock_stream = MagicMock()
    mock_stream.nominal_srate.return_value = 1000
    mock_stream.channel_count.return_value = 32  # Wrong! Should be 65
    mock_stream.name.return_value = "TestStream"
    mock_stream.type.return_value = "EEG"
    mock_stream.source_id.return_value = "test_source"

    receiver._resolve_stream = Mock(return_value=mock_stream)

    with pytest.raises(ValueError, match="Expected 65 channels"):
        receiver.start()
