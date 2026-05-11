from __future__ import annotations

import threading

import numpy as np

from backend.online_phase.online_preprocessor import OnlinePreprocessor
from backend.online_phase.stream_worker import StreamWorker


N_CHANNELS = 4
INPUT_SFREQ = 1000.0
TARGET_SFREQ = 250


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
        "bad_channels": [],
        "interp_weights": None,
        "ch_names": [f"EEG{i}" for i in range(N_CHANNELS)],
        "ica_unmixing": unmixing,
        "ica_mixing": np.linalg.pinv(unmixing),
        "ica_pca_components": rng.standard_normal((n_components, N_CHANNELS)),
        "ica_pca_mean": np.zeros(N_CHANNELS),
        "ica_exclude": [],
        "pre_whitener": np.ones((N_CHANNELS, 1)),
        "sfreq_offline": float(TARGET_SFREQ),
    }


def _make_settings() -> dict:
    return {
        "bandpass": {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "method": "iir",
            "notch": None,
        },
        "resample": {"target_rate": TARGET_SFREQ},
    }


def _make_preprocessor() -> OnlinePreprocessor:
    return OnlinePreprocessor(_make_settings(), _make_online_state(), INPUT_SFREQ)


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


def test_importable_from_online_phase_package():
    from backend.online_phase import StreamWorker as ExportedStreamWorker

    assert ExportedStreamWorker is StreamWorker
