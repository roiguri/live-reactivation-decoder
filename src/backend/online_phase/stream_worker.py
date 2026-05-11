from __future__ import annotations

import time
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal


class StreamWorker(QThread):
    """Background orchestrator for the online decoder micro-batch loop."""

    prediction_ready = pyqtSignal(dict, np.ndarray, list)
    error_occurred = pyqtSignal(str)

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
        while not self._stop_requested:
            batch_processed = False
            try:
                timestamps, eeg_chunk, markers = self.receiver.pull_new_data()
            except Exception as exc:
                self._fail("receiver pull", exc)
                return

            try:
                self._append_to_batch(timestamps, eeg_chunk)
                self._pending_markers.extend(markers)
            except Exception as exc:
                self._fail("batch accumulation", exc)
                return

            while self._accumulated_samples >= self.batch_size_samples:
                batch_ts, batch_eeg = self._pop_batch(self.batch_size_samples)
                batch_end_ts = float(batch_ts[-1])

                try:
                    out_eeg, out_ts = self.preprocessor.process_batch(batch_eeg, batch_ts)
                except Exception as exc:
                    self._fail("preprocessing", exc)
                    return

                try:
                    predictions = self.inference_engine.predict(out_eeg)
                except Exception as exc:
                    self._fail("inference", exc)
                    return

                batch_markers = self._pop_markers_through(batch_end_ts)

                self.prediction_ready.emit(predictions, out_ts, batch_markers)
                batch_processed = True

                if self._stop_requested:
                    break

            if not batch_processed and not self._stop_requested:
                time.sleep(self.poll_interval_sec)

    def stop(self) -> None:
        self._stop_requested = True

    def _fail(self, stage: str, exc: Exception) -> None:
        self._stop_requested = True
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
