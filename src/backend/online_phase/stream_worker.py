from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class StreamWorker(QThread):
    """Background orchestrator for the online decoder micro-batch loop."""

    prediction_ready = pyqtSignal(dict, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    latency_ready = pyqtSignal(dict)

    def __init__(
        self,
        receiver: Any,
        preprocessor: Any,
        inference_engine: Any,
        batch_size_samples: int = 40,
        poll_interval_sec: float = 0.01,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if batch_size_samples <= 0:
            raise ValueError("batch_size_samples must be positive.")
        if poll_interval_sec < 0:
            raise ValueError("poll_interval_sec must be non-negative.")

        self.receiver = receiver
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.batch_size_samples = int(batch_size_samples)
        self.poll_interval_sec = float(poll_interval_sec)

        self._batch_ts: list[np.ndarray] = []
        self._batch_eeg: list[np.ndarray] = []
        self._pending_markers: list[tuple[float, int]] = []
        self._stop_requested = False

    def run(self) -> None:
        logger.debug("Stream worker loop started")
        while not self._stop_requested:
            batch_processed = False
            pull_started = time.perf_counter()
            try:
                timestamps, eeg_chunk, markers = self.receiver.pull_new_data()
            except Exception as exc:
                self._fail("receiver pull", exc)
                return
            pull_ms = (time.perf_counter() - pull_started) * 1000.0

            accumulation_started = time.perf_counter()
            try:
                self._append_to_batch(timestamps, eeg_chunk)
                self._pending_markers.extend(markers)
            except Exception as exc:
                self._fail("batch accumulation", exc)
                return
            accumulation_ms = (time.perf_counter() - accumulation_started) * 1000.0

            while self._accumulated_samples >= self.batch_size_samples:
                batch_started = time.perf_counter()
                batch_ts, batch_eeg = self._pop_batch(self.batch_size_samples)
                batch_end_ts = float(batch_ts[-1])

                preprocessing_started = time.perf_counter()
                try:
                    out_eeg, out_ts = self.preprocessor.process_batch(batch_eeg, batch_ts)
                except Exception as exc:
                    self._fail("preprocessing", exc)
                    return
                preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0

                inference_started = time.perf_counter()
                try:
                    predictions = self.inference_engine.predict(out_eeg)
                except Exception as exc:
                    self._fail("inference", exc)
                    return
                inference_ms = (time.perf_counter() - inference_started) * 1000.0

                batch_markers = self._pop_markers_through(batch_end_ts)

                emit_started = time.perf_counter()
                self.prediction_ready.emit(predictions, out_ts, batch_markers)
                emit_ms = (time.perf_counter() - emit_started) * 1000.0
                batch_processed = True
                total_ms = (time.perf_counter() - batch_started) * 1000.0

                # TODO(open): Consider moving diagnostics throttling/aggregation
                # to a dedicated consumer. At the default 40-sample batch size
                # on a 1000 Hz stream, this signal emits about 25 times/second.
                # UI/log consumers should usually display rolling summaries
                # such as mean/p95 latency and backlog instead of every batch.
                self.latency_ready.emit({
                    "pull_ms": pull_ms,
                    "accumulation_ms": accumulation_ms,
                    "preprocessing_ms": preprocessing_ms,
                    "inference_ms": inference_ms,
                    "emit_ms": emit_ms,
                    "total_ms": total_ms,
                    "input_samples": int(batch_eeg.shape[0]),
                    "emitted_rows": int(np.asarray(out_ts).shape[0]),
                    "marker_count": len(batch_markers),
                    "pending_samples": self._accumulated_samples,
                })

                if self._stop_requested:
                    break

            if not batch_processed and not self._stop_requested:
                time.sleep(self.poll_interval_sec)
        logger.debug("Stream worker loop exited")

    def stop(self) -> None:
        self._stop_requested = True

    def _fail(self, stage: str, exc: Exception) -> None:
        self._stop_requested = True
        # Called from within the run()-loop except blocks, so the active
        # exception's traceback is still live and is captured here.
        logger.exception("%s failed", stage)
        self.error_occurred.emit(f"{stage} failed: {type(exc).__name__}: {exc}")

    @property
    def _accumulated_samples(self) -> int:
        return sum(part.shape[0] for part in self._batch_ts)

    def _append_to_batch(self, timestamps: np.ndarray, eeg_chunk: np.ndarray) -> None:
        timestamps = np.asarray(timestamps, dtype=float)
        eeg_chunk = np.asarray(eeg_chunk, dtype=float)
        if timestamps.shape[0] == 0:
            return
        if eeg_chunk.ndim != 2:
            raise ValueError(f"eeg_chunk must be 2D, got shape {eeg_chunk.shape}.")
        if eeg_chunk.shape[0] != timestamps.shape[0]:
            raise ValueError(
                f"timestamps length {timestamps.shape[0]} != eeg_chunk rows {eeg_chunk.shape[0]}"
            )
        self._batch_ts.append(timestamps)
        self._batch_eeg.append(eeg_chunk)

    def _pop_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        ts_parts: list[np.ndarray] = []
        eeg_parts: list[np.ndarray] = []
        remaining = n_samples

        while remaining > 0:
            current_ts = self._batch_ts[0]
            current_eeg = self._batch_eeg[0]
            take = min(remaining, current_ts.shape[0])

            ts_parts.append(current_ts[:take])
            eeg_parts.append(current_eeg[:take])

            if take == current_ts.shape[0]:
                self._batch_ts.pop(0)
                self._batch_eeg.pop(0)
            else:
                self._batch_ts[0] = current_ts[take:]
                self._batch_eeg[0] = current_eeg[take:]

            remaining -= take

        return np.concatenate(ts_parts), np.vstack(eeg_parts)

    def _pop_markers_through(self, batch_end_ts: float) -> list[tuple[float, int]]:
        ready: list[tuple[float, int]] = []
        pending: list[tuple[float, int]] = []
        for marker_ts, code in self._pending_markers:
            if marker_ts <= batch_end_ts:
                ready.append((marker_ts, code))
            else:
                pending.append((marker_ts, code))
        self._pending_markers = pending
        return ready
