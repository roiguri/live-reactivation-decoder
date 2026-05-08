from __future__ import annotations

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

    def _require_pylsl(self):
        if pylsl is None:
            raise RuntimeError("pylsl is required for LSLReceiver.")
        return pylsl

    def _start_proxy_process(self) -> None:
        if self.proxy_process is not None and self.proxy_process.poll() is None:
            return

        if not self.proxy_path.exists():
            raise FileNotFoundError(f"LSL proxy executable not found: {self.proxy_path}")

        if os.name != "nt" and self.proxy_path.suffix.lower() == ".exe":
            raise RuntimeError(
                f"Proxy executable {self.proxy_path.name} requires Windows. "
                "Run this on the decoding machine or set launch_proxy=False."
            )

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proxy_process = subprocess.Popen(
            [str(self.proxy_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
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

    def start(self) -> bool:
        if self.launch_proxy:
            self._start_proxy_process()

        deadline = time.monotonic() + self.resolve_timeout_sec
        resolved_stream = None

        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            resolved_stream = self._resolve_stream(timeout_sec=min(0.5, remaining))
            if resolved_stream is not None:
                break

        if resolved_stream is None:
            return False

        pylsl_module = self._require_pylsl()
        self.inlet = pylsl_module.StreamInlet(resolved_stream, recover=True)
        self._last_trigger_code = 0
        return True

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
            eeg_chunk, chunk_markers, self._last_trigger_code = split_eeg_and_markers(
                samples,
                eeg_channel_count=self.eeg_channel_count,
                trigger_channel_index=self.trigger_channel_index,
                previous_trigger_code=self._last_trigger_code,
            )

            timestamps_parts.append(timestamps_array)
            eeg_parts.append(eeg_chunk)
            markers.extend(chunk_markers)

        if not timestamps_parts:
            return (
                np.empty((0,), dtype=float),
                np.empty((0, self.eeg_channel_count), dtype=float),
                [],
            )

        return np.concatenate(timestamps_parts), np.vstack(eeg_parts), markers

    def stop(self) -> None:
        if self.inlet is not None and hasattr(self.inlet, "close_stream"):
            self.inlet.close_stream()
        self.inlet = None

        if self.proxy_process is not None:
            if self.proxy_process.poll() is None:
                self.proxy_process.terminate()
                try:
                    self.proxy_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proxy_process.kill()
                    self.proxy_process.wait(timeout=2.0)
            self.proxy_process = None
