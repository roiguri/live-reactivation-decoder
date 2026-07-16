from __future__ import annotations

import threading

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSlot

from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.stream_worker import StreamWorker


N_CHANNELS = 4
INPUT_SFREQ = 1000.0


class FakeReceiver:
    def __init__(
        self,
        chunks: list[tuple[np.ndarray, np.ndarray, list[tuple[float, int]]]] | None = None,
    ) -> None:
        self._chunks = list(chunks or [])
        self._lock = threading.Lock()

    def add_chunk(
        self,
        timestamps: np.ndarray,
        eeg_chunk: np.ndarray,
        markers: list[tuple[float, int]] | None = None,
    ) -> None:
        with self._lock:
            self._chunks.append((timestamps, eeg_chunk, list(markers or [])))

    def pull_new_data(self):
        with self._lock:
            if self._chunks:
                return self._chunks.pop(0)
        return np.empty((0,)), np.empty((0, N_CHANNELS)), []


class FakeInferenceEngine:
    def predict(self, features):
        features = np.asarray(features)
        n_rows = features.shape[0]
        return {
            "object": np.linspace(0.1, 0.9, n_rows),
            "scene": np.linspace(0.9, 0.1, n_rows),
        }


class RaisingReceiver:
    def pull_new_data(self):
        raise RuntimeError("receiver unavailable")


class PassThroughPreprocessor:
    def process_batch(self, eeg_batch, timestamps):
        return np.asarray(eeg_batch), np.asarray(timestamps)


class RaisingPreprocessor:
    def process_batch(self, eeg_batch, timestamps):
        raise RuntimeError("preprocessor failed")


class RaisingInferenceEngine:
    def predict(self, features):
        raise RuntimeError("inference failed")


class FakeLatencyPanel(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.payloads = []

    @pyqtSlot(dict)
    def display_latency(self, payload: dict) -> None:
        self.payloads.append(dict(payload))
        # TODO(open): A real UI should throttle or aggregate this batch-cadence
        # signal before rendering, for example once per second with rolling
        # mean/p95 latency. This fake panel prints every payload only to prove
        # the consumer contract in a small test.
        print(
            f"latency total={payload['total_ms']:.3f}ms "
            f"preprocessing={payload['preprocessing_ms']:.3f}ms "
            f"inference={payload['inference_ms']:.3f}ms "
            f"rows={payload['emitted_rows']} "
            f"pending={payload['pending_samples']}",
            flush=True,
        )


def _make_data(n_samples: int, *, start_sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = np.arange(start_sample, start_sample + n_samples)
    timestamps = sample_indices.astype(float) / INPUT_SFREQ
    eeg = np.column_stack([
        timestamps,
        timestamps + 0.1,
        timestamps + 0.2,
        timestamps + 0.3,
    ])
    return timestamps, eeg


def _make_online_state() -> dict:
    rng = np.random.default_rng(0)
    n_components = 2
    unmixing = rng.standard_normal((n_components, n_components))
    return {
        "eeg_chunk_indices": list(range(N_CHANNELS)),
        "bad_indices": [],
        "interp_weights": None,
        "ica_unmixing": unmixing,
        "ica_mixing": np.linalg.pinv(unmixing),
        "ica_pca_components": rng.standard_normal((n_components, N_CHANNELS)),
        "ica_pca_mean": np.zeros(N_CHANNELS),
        "ica_exclude": [],
        "pre_whitener": np.ones((N_CHANNELS, 1)),
    }


def _make_preprocessor() -> OnlinePreprocessor:
    return OnlinePreprocessor(_make_online_state(), INPUT_SFREQ)


def _make_worker(receiver: FakeReceiver, poll_interval_sec: float = 0.001) -> StreamWorker:
    return StreamWorker(
        receiver=receiver,
        preprocessor=_make_preprocessor(),
        inference_engine=FakeInferenceEngine(),
        batch_size_samples=40,
        poll_interval_sec=poll_interval_sec,
    )


def _stop_worker(worker: StreamWorker) -> None:
    worker.stop()
    assert worker.wait(2000)


def _make_simple_worker(
    receiver,
    preprocessor,
    inference_engine,
) -> StreamWorker:
    return StreamWorker(
        receiver=receiver,
        preprocessor=preprocessor,
        inference_engine=inference_engine,
        batch_size_samples=40,
        poll_interval_sec=0.001,
    )


def test_emits_prediction_payload_for_ready_batch(qtbot):
    timestamps, eeg = _make_data(40)
    receiver = FakeReceiver([(timestamps, eeg, [(float(timestamps[10]), 5)])])
    worker = _make_worker(receiver)

    with qtbot.waitSignal(worker.prediction_ready, timeout=3000) as blocker:
        worker.start()

    _stop_worker(worker)
    predictions, out_ts, markers = blocker.args
    assert set(predictions) == {"object", "scene"}
    assert isinstance(out_ts, np.ndarray)
    assert markers == [(float(timestamps[10]), 5)]
    assert predictions["object"].shape == out_ts.shape
    assert predictions["scene"].shape == out_ts.shape


def test_below_threshold_accumulation_waits_until_batch_ready(qtbot):
    first_ts, first_eeg = _make_data(25)
    second_ts, second_eeg = _make_data(15, start_sample=25)
    receiver = FakeReceiver([(first_ts, first_eeg, [])])
    worker = _make_worker(receiver)
    worker.start()

    with qtbot.waitSignal(worker.prediction_ready, timeout=50, raising=False) as no_signal:
        pass
    assert not no_signal.signal_triggered

    with qtbot.waitSignal(worker.prediction_ready, timeout=3000) as blocker:
        receiver.add_chunk(second_ts, second_eeg)

    _stop_worker(worker)
    _, out_ts, _ = blocker.args
    assert out_ts.shape[0] > 0


def test_markers_after_batch_end_are_deferred_to_next_emission(qtbot):
    first_ts, first_eeg = _make_data(60)
    second_ts, second_eeg = _make_data(20, start_sample=60)
    deferred_marker = (float(first_ts[50]), 11)
    receiver = FakeReceiver([
        (first_ts, first_eeg, [deferred_marker]),
        (second_ts, second_eeg, []),
    ])
    worker = _make_worker(receiver)
    received = []
    worker.prediction_ready.connect(lambda predictions, out_ts, markers: received.append(markers))

    worker.start()
    qtbot.waitUntil(lambda: len(received) >= 2, timeout=3000)
    _stop_worker(worker)

    assert received[0] == []
    assert received[1] == [deferred_marker]


def test_stop_wait_joins_worker_thread(qtbot):
    receiver = FakeReceiver([])
    worker = _make_worker(receiver)

    worker.start()
    qtbot.waitUntil(worker.isRunning, timeout=1000)
    worker.stop()

    assert worker.wait(2000)
    assert not worker.isRunning()


def test_receiver_error_emits_and_stops_worker(qtbot):
    worker = _make_simple_worker(
        receiver=RaisingReceiver(),
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.error_occurred, timeout=3000) as blocker:
        worker.start()

    assert blocker.args == [
        "receiver pull failed: RuntimeError: receiver unavailable"
    ]
    assert worker.wait(2000)
    assert not worker.isRunning()


def test_batch_accumulation_error_emits_and_stops_worker(qtbot):
    worker = _make_simple_worker(
        receiver=FakeReceiver([(np.array([0.0, 0.001]), np.zeros((1, 4)), [])]),
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.error_occurred, timeout=3000) as blocker:
        worker.start()

    assert blocker.args[0].startswith(
        "batch accumulation failed: ValueError: timestamps length 2"
    )
    assert worker.wait(2000)
    assert not worker.isRunning()


def test_preprocessor_error_emits_and_stops_worker(qtbot):
    timestamps, eeg = _make_data(40)
    worker = _make_simple_worker(
        receiver=FakeReceiver([(timestamps, eeg, [])]),
        preprocessor=RaisingPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.error_occurred, timeout=3000) as blocker:
        worker.start()

    assert blocker.args == [
        "preprocessing failed: RuntimeError: preprocessor failed"
    ]
    assert worker.wait(2000)
    assert not worker.isRunning()


def test_inference_error_emits_and_stops_worker(qtbot):
    timestamps, eeg = _make_data(40)
    worker = _make_simple_worker(
        receiver=FakeReceiver([(timestamps, eeg, [])]),
        preprocessor=PassThroughPreprocessor(),
        inference_engine=RaisingInferenceEngine(),
    )

    with qtbot.waitSignal(worker.error_occurred, timeout=3000) as blocker:
        worker.start()

    assert blocker.args == [
        "inference failed: RuntimeError: inference failed"
    ]
    assert worker.wait(2000)
    assert not worker.isRunning()


def test_latency_diagnostics_emit_for_processed_batch(qtbot):
    timestamps, eeg = _make_data(40)
    receiver = FakeReceiver([(timestamps, eeg, [(float(timestamps[10]), 5)])])
    worker = _make_simple_worker(
        receiver=receiver,
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.latency_ready, timeout=3000) as blocker:
        worker.start()

    _stop_worker(worker)
    (payload,) = blocker.args
    assert payload["input_samples"] == 40
    assert payload["emitted_rows"] == 40
    assert payload["marker_count"] == 1
    assert payload["pending_samples"] == 0

    timing_keys = {
        "pull_ms",
        "accumulation_ms",
        "preprocessing_ms",
        "inference_ms",
        "emit_ms",
        "total_ms",
    }
    assert timing_keys.issubset(payload)
    for key in timing_keys:
        assert isinstance(payload[key], float)
        assert payload[key] >= 0.0

    # total_ms folds in pull_ms + accumulation_ms (once per outer-loop pass,
    # attributed to every batch that pass produced) on top of this batch's own
    # preprocess/infer/emit time, so it must be at least their sum.
    assert payload["total_ms"] >= payload["pull_ms"] + payload["accumulation_ms"]

    # FakeReceiver has no time_correction()/local_clock() — sample_to_decision_ms
    # must degrade to None rather than crash the batch.
    assert payload["sample_to_decision_ms"] is None


class ClockAwareReceiver(FakeReceiver):
    """A FakeReceiver that also implements the clock-correction contract, so
    sample_to_decision_ms can be computed on the happy path."""

    def __init__(self, chunks, *, correction: float = 0.25, now: float = 100.0) -> None:
        super().__init__(chunks)
        self._correction = correction
        self._now = now

    def time_correction(self):
        return self._correction

    def local_clock(self):
        return self._now


def test_sample_to_decision_latency_computed_when_receiver_supports_clock_sync(qtbot):
    timestamps, eeg = _make_data(40)  # last timestamp = 39/1000 = 0.039s
    receiver = ClockAwareReceiver([(timestamps, eeg, [])], correction=0.25, now=100.0)
    worker = _make_simple_worker(
        receiver=receiver,
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.latency_ready, timeout=3000) as blocker:
        worker.start()
    _stop_worker(worker)

    (payload,) = blocker.args
    expected_ms = (100.0 - (float(timestamps[-1]) + 0.25)) * 1000.0
    assert payload["sample_to_decision_ms"] == pytest.approx(expected_ms)
    assert payload["total_ms"] >= payload["pull_ms"] + payload["accumulation_ms"]


class RaisingTimeCorrectionReceiver(FakeReceiver):
    """time_correction() raises (e.g. TimeoutError/LostError from pylsl) —
    the batch must still be processed and emitted normally."""

    def time_correction(self):
        raise TimeoutError("clock sync unavailable")

    def local_clock(self):
        return 100.0


def test_sample_to_decision_latency_none_when_time_correction_raises(qtbot):
    timestamps, eeg = _make_data(40)
    receiver = RaisingTimeCorrectionReceiver([(timestamps, eeg, [])])
    worker = _make_simple_worker(
        receiver=receiver,
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )

    with qtbot.waitSignal(worker.latency_ready, timeout=3000) as blocker:
        worker.start()
    _stop_worker(worker)

    (payload,) = blocker.args
    assert payload["sample_to_decision_ms"] is None
    assert payload["emitted_rows"] == 40


def test_latency_diagnostics_can_drive_fake_ui_panel(qtbot, capsys):
    timestamps, eeg = _make_data(40)
    receiver = FakeReceiver([(timestamps, eeg, [])])
    worker = _make_simple_worker(
        receiver=receiver,
        preprocessor=PassThroughPreprocessor(),
        inference_engine=FakeInferenceEngine(),
    )
    panel = FakeLatencyPanel()
    worker.latency_ready.connect(panel.display_latency)

    worker.start()
    qtbot.waitUntil(lambda: len(panel.payloads) == 1, timeout=3000)
    _stop_worker(worker)

    payload = panel.payloads[0]
    assert payload["emitted_rows"] == 40
    assert payload["pending_samples"] == 0

    printed = capsys.readouterr().out.strip()
    assert "latency total=" in printed
    assert "preprocessing=" in printed
    assert "inference=" in printed
    assert "ms" in printed
    assert "rows=40" in printed
    assert "pending=0" in printed


def test_importable_from_online_phase_package():
    from backend.online_phase import StreamWorker as ExportedStreamWorker

    assert ExportedStreamWorker is StreamWorker
