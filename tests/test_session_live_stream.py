from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.core.preprocessing_constants import FINAL_RESAMPLE_RATE
from backend.core.session_paths import SessionPaths
from backend.online_phase.artifact_loader import DecoderPipelineArtifact
from backend.session import AppSession, LiveStreamSession


class FakeReceiver:
    instances = []

    def __init__(self, stream_name=None, *, resolve_timeout_sec=5.0, **kwargs) -> None:
        self.stream_name = stream_name
        self.resolve_timeout_sec = resolve_timeout_sec
        self.start_calls = 0
        self.stop_calls = 0
        self.discover_calls = []
        FakeReceiver.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def discover_streams(self, timeout_sec=3.0) -> list[str]:
        self.discover_calls.append(timeout_sec)
        return ["NeuroneStream", "OtherStream"]


class FakeProxySource:
    instances = []

    def __init__(self, proxy_path=None) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self._running = False
        FakeProxySource.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1
        self._running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


class FakePreprocessor:
    instances = []

    def __init__(self, preprocessing_settings, online_state) -> None:
        self.preprocessing_settings = preprocessing_settings
        self.online_state = online_state
        self.target_sfreq = float(FINAL_RESAMPLE_RATE)
        self.input_sfreq = 1000.0
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
    FakeProxySource.instances = []
    FakePreprocessor.instances = []
    FakeInferenceEngine.instances = []
    FakeWorker.instances = []
    monkeypatch.setattr(
        "backend.session.load_decoder_pipeline_artifact",
        lambda path: _artifact(),
    )
    monkeypatch.setattr("backend.session.LSLReceiver", FakeReceiver)
    monkeypatch.setattr("backend.session.LslProxySource", FakeProxySource)
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


def test_build_live_stream_session_forwards_stream_name(
    sample_config_path,
    monkeypatch,
):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    session.build_live_stream_session(
        Path("decoder_pipeline.joblib"),
        stream_name="CustomStream",
    )

    receiver = FakeReceiver.instances[0]
    assert receiver.stream_name == "CustomStream"
    # No active source / proxy source → short resolve timeout.
    assert receiver.resolve_timeout_sec == 5.0


def test_build_live_stream_session_receiver_defaults(sample_config_path, monkeypatch):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    session.build_live_stream_session(Path("decoder_pipeline.joblib"))

    receiver = FakeReceiver.instances[0]
    assert receiver.stream_name is None


def test_discover_streams_starts_proxy_and_leaves_it_running(
    sample_config_path,
    monkeypatch,
):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    names = session.discover_streams(timeout_sec=1.5)

    assert names == ["NeuroneStream", "OtherStream"]
    # A single proxy source was started and NOT stopped (reused by the run).
    assert len(FakeProxySource.instances) == 1
    proxy = FakeProxySource.instances[0]
    assert proxy.start_calls == 1
    assert proxy.stop_calls == 0
    assert proxy.is_running is True
    # The consumer receiver did the resolve and was not stopped.
    assert FakeReceiver.instances[0].discover_calls == [1.5]


def test_start_stream_source_reuses_discovery_proxy(sample_config_path, monkeypatch):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    session.discover_streams()
    session.start_stream_source()

    # Same proxy instance reused; start() is idempotent on an already-running proxy.
    assert len(FakeProxySource.instances) == 1
    assert FakeProxySource.instances[0].start_calls == 2


def test_stop_stream_source_stops_and_clears(sample_config_path, monkeypatch):
    _patch_runtime(monkeypatch)
    session = AppSession(sample_config_path)

    session.start_stream_source()
    session.stop_stream_source()

    proxy = FakeProxySource.instances[0]
    assert proxy.stop_calls == 1
    assert proxy.is_running is False
    # Idempotent when already idle.
    session.stop_stream_source()
    assert proxy.stop_calls == 1


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
    log_dir = tmp_path / "run"

    live = session.build_live_stream_session(
        Path("decoder_pipeline.joblib"),
        log_dir=log_dir,
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
    predictions = (log_dir / "predictions.csv").read_text().splitlines()
    assert predictions == [
        "lsl_timestamp,t_sec,object,scene",
        "1.0,0.0,0.2,0.8",
    ]
    # Sidecar markers + manifest + npz are created alongside.
    assert (log_dir / "markers.csv").exists()
    assert (log_dir / "manifest.json").exists()
    assert (log_dir / "predictions.npz").exists()


def test_new_phase2_log_dir_uses_workspace(sample_config_path, tmp_path):
    session = AppSession(sample_config_path)
    session.paths = SessionPaths(tmp_path / "subject_root")

    run_dir = session.new_phase2_log_dir()

    # <root>/phase2_live/<timestamp>/, resolved from the owned workspace
    # (not inferred from any artifact path), created on demand.
    assert run_dir.parent == tmp_path / "subject_root" / "phase2_live"
    assert run_dir.is_dir()


def test_new_phase2_log_dir_none_without_workspace(sample_config_path):
    session = AppSession(sample_config_path)
    # No workspace configured → live inference can still run, just unlogged.
    assert session.new_phase2_log_dir() is None
