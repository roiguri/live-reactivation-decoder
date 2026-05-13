from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.online_phase.artifact_loader import DecoderPipelineArtifact
from backend.session import AppSession, LiveStreamSession


class FakeReceiver:
    instances = []

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        FakeReceiver.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class FakePreprocessor:
    instances = []

    def __init__(self, preprocessing_settings, online_state) -> None:
        self.preprocessing_settings = preprocessing_settings
        self.online_state = online_state
        self.target_sfreq = preprocessing_settings["resample"]["target_rate"]
        FakePreprocessor.instances.append(self)


class FakeInferenceEngine:
    instances = []

    def __init__(self, models, metadata) -> None:
        self.models = models
        self.metadata = metadata
        FakeInferenceEngine.instances.append(self)


class FakeSignal:
    def __init__(self) -> None:
        self.slots = []

    def connect(self, slot, *args) -> None:
        self.slots.append(slot)

    def emit(self, *args) -> None:
        for slot in self.slots:
            slot(*args)


class FakeWorker:
    instances = []

    def __init__(
        self,
        receiver,
        preprocessor,
        inference_engine,
        batch_size_samples,
    ) -> None:
        self.receiver = receiver
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.batch_size_samples = batch_size_samples
        self.prediction_ready = FakeSignal()
        self.error_occurred = FakeSignal()
        self.latency_ready = FakeSignal()
        self.start_calls = 0
        self.stop_calls = 0
        self.wait_calls = 0
        FakeWorker.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def wait(self) -> bool:
        self.wait_calls += 1
        return True


class OrderedReceiver:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.order.append("receiver.start")

    def stop(self) -> None:
        self.stop_calls += 1
        self.order.append("receiver.stop")


class OrderedWorker:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.prediction_ready = FakeSignal()
        self.error_occurred = FakeSignal()
        self.latency_ready = FakeSignal()
        self.start_calls = 0
        self.stop_calls = 0
        self.wait_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.order.append("worker.start")

    def stop(self) -> None:
        self.stop_calls += 1
        self.order.append("worker.stop")

    def wait(self) -> bool:
        self.wait_calls += 1
        self.order.append("worker.wait")
        return True


class OrderedLogger:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.order.append("logger.close")


def _artifact() -> DecoderPipelineArtifact:
    return DecoderPipelineArtifact(
        models={"object": object(), "scene": object()},
        online_state={"opaque": True},
        metadata={"feature_width": 64},
    )


def _patch_runtime(monkeypatch):
    FakeReceiver.instances = []
    FakePreprocessor.instances = []
    FakeInferenceEngine.instances = []
    FakeWorker.instances = []
    monkeypatch.setattr(
        "backend.session.load_decoder_pipeline_artifact",
        lambda path: _artifact(),
    )
    monkeypatch.setattr("backend.session.LSLReceiver", FakeReceiver)
    monkeypatch.setattr("backend.session.OnlinePreprocessor", FakePreprocessor)
    monkeypatch.setattr("backend.session.LiveInferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr("backend.session.StreamWorker", FakeWorker)


def test_live_stream_session_exposes_prediction_signal_and_lifecycle_order():
    order: list[str] = []
    receiver = OrderedReceiver(order)
    worker = OrderedWorker(order)
    logger = OrderedLogger(order)
    live = LiveStreamSession(
        _receiver=receiver,
        _worker=worker,
        _logger=logger,
    )

    assert live.prediction_ready is worker.prediction_ready
    assert live.error_occurred is worker.error_occurred
    assert live.latency_ready is worker.latency_ready

    live.start()
    live.start()
    live.stop()
    live.stop()

    assert order == [
        "receiver.start",
        "worker.start",
        "worker.stop",
        "worker.wait",
        "logger.close",
        "receiver.stop",
    ]
    assert receiver.start_calls == 1
    assert receiver.stop_calls == 1
    assert worker.start_calls == 1
    assert worker.stop_calls == 1
    assert worker.wait_calls == 1
    assert logger.close_calls == 1


def test_live_stream_session_cannot_restart_after_stop():
    live = LiveStreamSession(
        _receiver=OrderedReceiver([]),
        _worker=OrderedWorker([]),
    )

    live.start()
    live.stop()

    with pytest.raises(RuntimeError, match="Cannot restart"):
        live.start()


def test_build_live_stream_session_returns_live_session(sample_config_path, monkeypatch):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    live = session.build_live_stream_session(
        Path("decoder_pipeline.joblib"),
        batch_size_samples=12,
    )
    worker = FakeWorker.instances[0]

    assert isinstance(live, LiveStreamSession)
    assert live.prediction_ready is worker.prediction_ready
    assert live.error_occurred is worker.error_occurred
    assert live.latency_ready is worker.latency_ready
    assert not hasattr(session, "online")
    assert not hasattr(live, "worker")
    assert not hasattr(live, "receiver")
    assert worker.batch_size_samples == 12
    assert worker.receiver is FakeReceiver.instances[0]
    assert worker.preprocessor is FakePreprocessor.instances[0]
    assert worker.inference_engine is FakeInferenceEngine.instances[0]
    assert FakePreprocessor.instances[0].online_state == {"opaque": True}
    assert FakeInferenceEngine.instances[0].metadata == {"feature_width": 64}


def test_build_live_stream_session_without_log_has_no_logger_slot(
    sample_config_path,
    monkeypatch,
):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    live = session.build_live_stream_session(Path("decoder_pipeline.joblib"))

    assert live.prediction_ready.slots == []


def test_build_live_stream_session_with_log_connects_logger(
    sample_config_path,
    tmp_path,
    monkeypatch,
):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)
    log_path = tmp_path / "predictions.csv"

    live = session.build_live_stream_session(
        Path("decoder_pipeline.joblib"),
        log_path=log_path,
    )
    live.prediction_ready.emit(
        {
            "object": np.array([0.2]),
            "scene": np.array([0.8]),
        },
        np.array([1.0]),
        [],
    )
    live.stop()

    assert len(live.prediction_ready.slots) == 1
    assert log_path.read_text().splitlines() == [
        "timestamp,marker_code,object,scene",
        "1.0,,0.2,0.8",
    ]
