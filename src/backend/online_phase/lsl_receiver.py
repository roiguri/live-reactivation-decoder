from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pylsl
except ImportError:  # pragma: no cover - handled explicitly at runtime
    pylsl = None


logger = logging.getLogger(__name__)


DEFAULT_STREAM_NAME = "NeuroneStream"
DEFAULT_STREAM_TYPE = "EEG"
DEFAULT_EEG_CHANNEL_COUNT = 64
DEFAULT_TRIGGER_CHANNEL_INDEX = 64


def default_proxy_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "lslproxy" / "LSLProxy.exe"


def decode_trigger_value(raw_value: float | int) -> int:
    """Decode the PsychoPy trigger code from NeurOne's packed trigger word."""

    return (int(raw_value) >> 8) & 0xFF


def extract_markers_from_trigger_channel(
    raw_trigger_values: np.ndarray | list[float] | list[int],
    *,
    previous_trigger_code: int = 0,
) -> tuple[list[int], int]:
    """Return non-zero trigger edges and the last observed code."""

    markers: list[int] = []
    last_code = previous_trigger_code

    for raw_value in np.asarray(raw_trigger_values).reshape(-1):
        trigger_code = decode_trigger_value(raw_value)
        if trigger_code != 0 and trigger_code != last_code:
            markers.append(trigger_code)
        last_code = trigger_code

    return markers, last_code


def split_eeg_and_markers(
    samples: np.ndarray | list[list[float]],
    *,
    eeg_channel_count: int = DEFAULT_EEG_CHANNEL_COUNT,
    trigger_channel_index: int = DEFAULT_TRIGGER_CHANNEL_INDEX,
    previous_trigger_code: int = 0,
) -> tuple[np.ndarray, list[int], int]:
    """Split the trigger channel from EEG samples and decode marker edges."""

    chunk = np.asarray(samples, dtype=float)
    if chunk.size == 0:
        return np.empty((0, eeg_channel_count), dtype=float), [], previous_trigger_code

    if chunk.ndim == 1:
        chunk = chunk[np.newaxis, :]

    if chunk.ndim != 2:
        raise ValueError(f"Expected 2D chunk, got shape {chunk.shape}.")
    if chunk.shape[1] <= trigger_channel_index:
        raise ValueError(
            f"Trigger channel index {trigger_channel_index} is out of bounds for chunk with "
            f"{chunk.shape[1]} channels."
        )

    raw_trigger_values = chunk[:, trigger_channel_index]
    eeg_chunk = np.delete(chunk, trigger_channel_index, axis=1)

    if eeg_chunk.shape[1] != eeg_channel_count:
        raise ValueError(
            f"Expected {eeg_channel_count} EEG channels after removing trigger channel, "
            f"got {eeg_chunk.shape[1]}."
        )

    markers, last_code = extract_markers_from_trigger_channel(
        raw_trigger_values,
        previous_trigger_code=previous_trigger_code,
    )
    return eeg_chunk, markers, last_code


class LSLReceiver:
    """Manage the LSL proxy process and pull EEG data from the LSL inlet."""

    def __init__(
        self,
        proxy_path: str | Path | None = None,
        stream_name: Optional[str] = None,
        *,
        stream_type: str = DEFAULT_STREAM_TYPE,
        eeg_channel_count: int = DEFAULT_EEG_CHANNEL_COUNT,
        trigger_channel_index: int = DEFAULT_TRIGGER_CHANNEL_INDEX,
        resolve_timeout_sec: float = 5.0,
        pull_timeout_sec: float = 0.0,
        launch_proxy: bool = True,
    ) -> None:
        self.proxy_path = Path(proxy_path) if proxy_path is not None else default_proxy_path()
        self.stream_name = stream_name or DEFAULT_STREAM_NAME
        self.stream_type = stream_type
        self.eeg_channel_count = eeg_channel_count
        self.trigger_channel_index = trigger_channel_index
        self.resolve_timeout_sec = resolve_timeout_sec
        self.pull_timeout_sec = pull_timeout_sec
        self.launch_proxy = launch_proxy

        self.proxy_process: Optional[subprocess.Popen] = None
        self.inlet = None
        self._last_trigger_code = 0

        logger.info(
            f"LSLReceiver initialized: stream='{self.stream_name}', type='{self.stream_type}', "
            f"proxy_path='{self.proxy_path}', launch_proxy={self.launch_proxy}"
        )

    def _require_pylsl(self):
        if pylsl is None:
            raise RuntimeError("pylsl is required for LSLReceiver.")
        return pylsl

    def _start_proxy_process(self) -> None:
        if self.proxy_process is not None and self.proxy_process.poll() is None:
            logger.debug("LSL proxy already running, skipping launch")
            return

        if not self.proxy_path.exists():
            raise FileNotFoundError(f"LSL proxy executable not found: {self.proxy_path}")

        if os.name != "nt" and self.proxy_path.suffix.lower() == ".exe":
            raise RuntimeError(
                f"Proxy executable {self.proxy_path.name} requires Windows. "
                "Run this on the decoding machine or set launch_proxy=False."
            )

        logger.info(f"Starting LSL proxy: {self.proxy_path}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proxy_process = subprocess.Popen(
            [str(self.proxy_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        logger.debug(f"LSL proxy spawned with PID {self.proxy_process.pid}")

        # Wait briefly for proxy to initialize and check if it died immediately
        time.sleep(0.5)
        if self.proxy_process.poll() is not None:
            # Proxy died immediately - capture diagnostics
            _, stderr = self.proxy_process.communicate(timeout=1.0)
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else "(no error output)"
            raise RuntimeError(
                f"LSL proxy executable failed to start. Exit code: {self.proxy_process.returncode}. "
                f"Error output: {stderr_text}"
            )

    def _resolve_stream(self, timeout_sec: float):
        pylsl_module = self._require_pylsl()
        if self.stream_type:
            streams = pylsl_module.resolve_byprop("type", self.stream_type, timeout=timeout_sec)
        else:
            streams = pylsl_module.resolve_streams(wait_time=timeout_sec)

        if self.stream_name:
            streams = [stream for stream in streams if stream.name() == self.stream_name]

        if not streams and self.stream_name:
            fallback_streams = pylsl_module.resolve_streams(wait_time=timeout_sec)
            streams = [stream for stream in fallback_streams if stream.name() == self.stream_name]

        return streams[0] if streams else None

    def discover_streams(self, timeout_sec: float = 3.0) -> list[str]:
        if self.launch_proxy:
            self._start_proxy_process()

        pylsl_module = self._require_pylsl()
        streams = pylsl_module.resolve_streams(wait_time=timeout_sec)
        return sorted(
            {
                stream.name()
                for stream in streams
                if not self.stream_type or stream.type() == self.stream_type
            }
        )

    def set_stream(self, stream_name: str) -> None:
        self.stream_name = stream_name

    def start(self) -> None:
        if self.launch_proxy:
            self._start_proxy_process()

        logger.info(
            f"Resolving LSL stream '{self.stream_name}' (type='{self.stream_type}', "
            f"timeout={self.resolve_timeout_sec}s)"
        )
        deadline = time.monotonic() + self.resolve_timeout_sec
        resolved_stream = None
        attempt_count = 0

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resolved_stream = self._resolve_stream(timeout_sec=min(0.5, remaining))
            attempt_count += 1
            if resolved_stream is not None:
                logger.debug(f"Stream resolved on attempt {attempt_count}")
                break
            logger.debug(f"Attempt {attempt_count} failed, retrying...")

        if resolved_stream is None:
            raise RuntimeError(
                f"Stream '{self.stream_name}' not found after {self.resolve_timeout_sec}s "
                f"({attempt_count} attempts). Check that NeurOne is streaming and LSLProxy is running."
            )

        # Validate stream properties before connecting
        stream_info = resolved_stream
        nominal_rate = stream_info.nominal_srate()
        channel_count = stream_info.channel_count()

        logger.debug(
            f"Stream properties: {channel_count} channels @ {nominal_rate} Hz, "
            f"type='{stream_info.type()}', source='{stream_info.source_id()}'"
        )

        expected_channels = self.eeg_channel_count + 1  # EEG + trigger channel
        if nominal_rate != 1000:
            raise ValueError(
                f"Expected 1000 Hz stream, got {nominal_rate} Hz. "
                f"Check NeurOne hardware configuration."
            )

        if channel_count != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels (64 EEG + 1 trigger), got {channel_count}. "
                f"Check NeurOne channel configuration."
            )

        logger.info(
            f"Connected to stream '{stream_info.name()}': "
            f"{channel_count} channels @ {nominal_rate} Hz"
        )

        pylsl_module = self._require_pylsl()
        self.inlet = pylsl_module.StreamInlet(resolved_stream, recover=True)
        self._last_trigger_code = 0
        logger.info("LSLReceiver started successfully")

    def pull_new_data(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if self.inlet is None:
            raise RuntimeError("LSLReceiver.start() must be called before pull_new_data().")

        timestamps_parts: list[np.ndarray] = []
        eeg_parts: list[np.ndarray] = []
        markers: list[int] = []

        while True:
            samples, timestamps = self.inlet.pull_chunk(timeout=self.pull_timeout_sec)
            if not timestamps:
                break

            timestamps_array = np.asarray(timestamps, dtype=float)

            try:
                eeg_chunk, chunk_markers, self._last_trigger_code = split_eeg_and_markers(
                    samples,
                    eeg_channel_count=self.eeg_channel_count,
                    trigger_channel_index=self.trigger_channel_index,
                    previous_trigger_code=self._last_trigger_code,
                )
            except ValueError as e:
                # Graceful degradation: skip malformed chunk and continue
                chunk_shape = np.asarray(samples).shape if len(samples) > 0 else "empty"
                logger.warning(f"Malformed chunk received: {e}. Chunk shape: {chunk_shape}. Skipping.")
                continue

            if len(chunk_markers) > 0:
                logger.debug(f"Detected markers: {chunk_markers}")

            timestamps_parts.append(timestamps_array)
            eeg_parts.append(eeg_chunk)
            markers.extend(chunk_markers)

        if not timestamps_parts:
            return (
                np.empty((0,), dtype=float),
                np.empty((0, self.eeg_channel_count), dtype=float),
                [],
            )

        total_samples = sum(part.shape[0] for part in eeg_parts)
        logger.debug(f"Pulled {total_samples} samples, {len(markers)} markers")

        return np.concatenate(timestamps_parts), np.vstack(eeg_parts), markers

    def stop(self) -> None:
        logger.info("Stopping LSLReceiver")

        if self.inlet is not None and hasattr(self.inlet, "close_stream"):
            self.inlet.close_stream()
            logger.debug("LSL inlet closed")
        self.inlet = None

        if self.proxy_process is not None:
            if self.proxy_process.poll() is None:
                logger.debug("Terminating LSL proxy process")
                self.proxy_process.terminate()
                try:
                    self.proxy_process.wait(timeout=2.0)
                    logger.debug("LSL proxy terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("LSL proxy did not terminate gracefully, killing process")
                    self.proxy_process.kill()
                    self.proxy_process.wait(timeout=2.0)
            self.proxy_process = None

        logger.info("LSLReceiver stopped")
