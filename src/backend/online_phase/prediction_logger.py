from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot


class PredictionLogger(QObject):
    """CSV sink for live decoder prediction batches."""

    def __init__(
        self,
        out_path: str | Path,
        task_names: list[str],
        target_sfreq: float,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._out_path = Path(out_path)
        self._task_names = list(task_names)
        self._target_sfreq = float(target_sfreq)
        if self._target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive.")

        # TODO(open): CSV is the first persistence sink for backend/frontend integration;
        # revisit whether live logging should use another storage format or sink.
        self._file = self._out_path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "marker_code", *self._task_names])
        self._closed = False

    @pyqtSlot(dict, np.ndarray, list)
    def on_predictions(
        self,
        predictions: dict[str, np.ndarray],
        timestamps: np.ndarray,
        markers: list[tuple[float, int]],
    ) -> None:
        timestamps = np.asarray(timestamps, dtype=float)
        marker_codes = self._marker_codes_by_row(timestamps, markers)

        prediction_arrays = {
            task_name: np.asarray(predictions[task_name])
            for task_name in self._task_names
        }
        for task_name, values in prediction_arrays.items():
            if values.shape != timestamps.shape:
                raise ValueError(
                    f"Prediction vector for '{task_name}' has shape {values.shape}, "
                    f"expected {timestamps.shape}."
                )

        for row_index, timestamp in enumerate(timestamps):
            self._writer.writerow([
                timestamp,
                marker_codes.get(row_index, ""),
                *[
                    prediction_arrays[task_name][row_index]
                    for task_name in self._task_names
                ],
            ])
        self._file.flush()

    def close(self) -> None:
        """Flush and close the CSV file. Safe to call more than once."""
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True

    def _marker_codes_by_row(
        self,
        timestamps: np.ndarray,
        markers: list[tuple[float, int]],
    ) -> dict[int, int]:
        # TODO(open): see docs/stream_worker_design.md Open §1 — wire tolerance from SettingsManager
        tolerance = 0.5 / self._target_sfreq
        marker_codes: dict[int, int] = {}
        if timestamps.size == 0:
            return marker_codes

        for marker_ts, code in markers:
            deltas = np.abs(timestamps - float(marker_ts))
            row_index = int(np.argmin(deltas))
            if deltas[row_index] <= tolerance:
                marker_codes[row_index] = int(code)
        return marker_codes
