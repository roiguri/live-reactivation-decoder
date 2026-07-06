"""Persistent logging for one Phase 2 live decoding run.

Two pieces, deliberately decoupled:

- :class:`LiveSessionLogger` — the **live sink**. A plain (non-Qt) callable
  wired to ``StreamWorker.prediction_ready`` via a direct connection. Its only
  job is to append each batch to two line-buffered CSV streams (the crash-safe
  source of truth) and to own the run manifest. It also keeps the raw batch
  arrays in memory so it can emit a numpy bundle at ``close()`` — at a few MB
  per run that is free, and it gives the ``.npz`` full ``float64`` precision
  independent of the CSV's rounding.

- :func:`export_session_npz` — a standalone projection of a run directory to a
  ``predictions.npz``. ``close()`` calls the shared writer directly from the
  in-memory arrays; this function rebuilds the bundle from the CSVs instead, so
  a session that crashed before ``close()`` can still be exported after the
  fact. The CSVs are always the source of truth; the ``.npz`` is derived.

Run directory layout::

    <run_dir>/
    ├── predictions.csv   lsl_timestamp, t_sec, <task1..taskN>
    ├── markers.csv       lsl_timestamp, t_sec, code, name
    ├── manifest.json     schema_version, wall-clock + lsl_t0, counts, metadata
    └── predictions.npz   (written at close) arrays + embedded manifest

``lsl_timestamp`` is the raw shared LSL clock both CSVs live on, so markers and
predictions align exactly (and a future group-delay offset stamps against it).
``t_sec`` is seconds since the first sample seen this run.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1

PREDICTIONS_CSV = "predictions.csv"
MARKERS_CSV = "markers.csv"
MANIFEST_JSON = "manifest.json"
PREDICTIONS_NPZ = "predictions.npz"
DECISIONS_CSV = "decisions.csv"
DECISION_CONFIG_JSONL = "decision_config.jsonl"

# Probabilities are rounded in the CSV for readability/size; the in-memory
# arrays (and thus the close()-time npz) keep full precision. Timestamps are
# never rounded — lsl_timestamp is the join key, t_sec is derived from it.
_PROB_DECIMALS = 5
_T_SEC_DECIMALS = 6

_MARKER_DTYPE = np.dtype(
    [("lsl_timestamp", "f8"), ("t_sec", "f8"), ("code", "i8"), ("name", "U64")]
)


class LiveSessionLogger:
    """Live CSV sink + in-memory accumulator for one decoding run.

    Args:
        run_dir: Directory for this run's files (created if missing). A fresh
            timestamped directory per Start keeps each run self-contained.
        task_names: Decoder names, in column order for both the CSV and the
            npz prediction matrix.
        event_names: ``{code: name}`` for resolving marker names (empty string
            for codes absent from the map). Every edge is logged regardless.
        metadata: Extra fields embedded verbatim in the manifest (e.g.
            ``target_sfreq``, ``config``) — not load-bearing for any logic.
    """

    def __init__(
        self,
        run_dir: str | Path,
        task_names: list[str],
        event_names: dict[int, str] | None = None,
        metadata: dict[str, Any] | None = None,
        decision_config: dict | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._task_names = list(task_names)
        self._event_names = {int(c): str(n) for c, n in (event_names or {}).items()}
        self._metadata = dict(metadata or {})

        self._wall_clock_start = datetime.now().isoformat(timespec="seconds")
        self._lsl_t0: float | None = None

        # Decision logging is opt-in: enabled only when the version-0 config is
        # supplied. When off, the sinks below stay None and on_decisions is a no-op.
        self._decision_config = dict(decision_config) if decision_config else None
        self._n_decision_samples = 0
        self._decisions_file = None
        self._decisions_writer = None
        self._decision_config_file = None

        # In-memory accumulation (raw, full precision) for the close()-time npz.
        # Per-batch chunks; vstacked once at close — never grown row-by-row.
        self._pred_chunks: list[np.ndarray] = []
        self._ts_chunks: list[np.ndarray] = []
        self._marker_rows: list[tuple[float, float, int, str]] = []

        self._predictions_file = (self._run_dir / PREDICTIONS_CSV).open(
            "w", newline="", buffering=1
        )
        self._predictions_writer = csv.writer(self._predictions_file)
        self._predictions_writer.writerow(["lsl_timestamp", "t_sec", *self._task_names])

        self._markers_file = (self._run_dir / MARKERS_CSV).open(
            "w", newline="", buffering=1
        )
        self._markers_writer = csv.writer(self._markers_file)
        self._markers_writer.writerow(["lsl_timestamp", "t_sec", "code", "name"])

        if self._decision_config is not None:
            self._decisions_file = (self._run_dir / DECISIONS_CSV).open(
                "w", newline="", buffering=1
            )
            self._decisions_writer = csv.writer(self._decisions_file)
            self._decisions_writer.writerow(
                ["lsl_timestamp", "t_sec", *self._task_names, "config_version"]
            )
            self._decision_config_file = (
                self._run_dir / DECISION_CONFIG_JSONL
            ).open("w", buffering=1)
            # Version 0: the settings in effect from the start (lsl_timestamp null).
            self._append_config_version(0, None, self._decision_config)

        self._closed = False
        # A preliminary manifest so a run that crashes before close() is still
        # interpretable; close() rewrites it with lsl_t0, end time, and counts.
        self._write_manifest()
        logger.info(
            "Live session logging to %s (%d decoder(s))",
            self._run_dir, len(self._task_names),
        )

    # ── live sink ──────────────────────────────────────────────────────────────

    def on_predictions(
        self,
        predictions: dict[str, np.ndarray],
        timestamps: np.ndarray,
        markers: list[tuple[float, int]],
    ) -> None:
        """Append one batch to both CSV streams and the in-memory buffers.

        Runs on the worker thread (direct connection); it is the only writer,
        so the buffers need no locking.
        """
        timestamps = np.asarray(timestamps, dtype=float)
        if timestamps.size and self._lsl_t0 is None:
            self._lsl_t0 = float(timestamps[0])

        if timestamps.size:
            matrix = self._stack_predictions(predictions, timestamps.shape)
            self._ts_chunks.append(timestamps)
            self._pred_chunks.append(matrix)

            t_sec = np.round(timestamps - self._t0(), _T_SEC_DECIMALS)
            rounded = np.round(matrix, _PROB_DECIMALS)
            self._predictions_writer.writerows(
                [ts, ts_rel, *row]
                for ts, ts_rel, row in zip(timestamps, t_sec, rounded.tolist())
            )
            self._predictions_file.flush()

        if markers:
            marker_rows = []
            for marker_ts, code in markers:
                marker_ts = float(marker_ts)
                code = int(code)
                name = self._event_names.get(code, "")
                t_sec = round(marker_ts - self._t0(), _T_SEC_DECIMALS)
                marker_rows.append((marker_ts, t_sec, code, name))
            self._marker_rows.extend(marker_rows)
            self._markers_writer.writerows(marker_rows)
            self._markers_file.flush()

    def on_decisions(self, result: Any) -> None:
        """Append one batch of latched decisions to ``decisions.csv``.

        ``result`` is duck-typed (no backend-type import): ``.timestamps`` (n,),
        ``.active`` ({decoder: bool (n,)}), ``.config_version`` (int), and an
        optional ``.config_change`` carrying a new timeline snapshot. Runs on the
        worker thread (direct connection); no-op when decision logging is off.
        """
        if self._decisions_writer is None:
            return
        timestamps = np.asarray(result.timestamps, dtype=float)
        if timestamps.size == 0:
            return
        if self._lsl_t0 is None:
            self._lsl_t0 = float(timestamps[0])

        change = getattr(result, "config_change", None)
        if change is not None:
            self._append_config_version(
                change.version, change.lsl_timestamp, change.config
            )

        version = int(result.config_version)
        t_sec = np.round(timestamps - self._t0(), _T_SEC_DECIMALS)
        columns = [np.asarray(result.active[name], dtype=bool) for name in self._task_names]
        self._decisions_writer.writerows(
            [timestamps[i], t_sec[i], *[bool(col[i]) for col in columns], version]
            for i in range(timestamps.size)
        )
        self._decisions_file.flush()
        self._n_decision_samples += int(timestamps.size)

    def _append_config_version(
        self, version: int, lsl_timestamp: float | None, config: dict
    ) -> None:
        """Append one full config snapshot to the timeline (crash-safe)."""
        self._decision_config_file.write(
            json.dumps(
                {
                    "config_version": int(version),
                    "lsl_timestamp": lsl_timestamp,
                    "config": config,
                }
            )
            + "\n"
        )
        self._decision_config_file.flush()

    def close(self) -> None:
        """Close all sinks, finalize the manifest, and write the npz. Idempotent."""
        if self._closed:
            return
        self._predictions_file.flush()
        self._predictions_file.close()
        self._markers_file.flush()
        self._markers_file.close()
        if self._decisions_file is not None:
            self._decisions_file.flush()
            self._decisions_file.close()
        if self._decision_config_file is not None:
            self._decision_config_file.flush()
            self._decision_config_file.close()

        timestamps, predictions, markers = self._collect_arrays()
        self._write_manifest(final=True, n_predictions=timestamps.size, n_markers=markers.size)
        _save_npz(
            self._run_dir / PREDICTIONS_NPZ,
            timestamps=timestamps,
            predictions=predictions,
            task_names=self._task_names,
            markers=markers,
            manifest=self._build_manifest(
                final=True, n_predictions=timestamps.size, n_markers=markers.size
            ),
        )
        self._closed = True
        logger.info(
            "Live run logged: %d prediction(s), %d marker(s) → %s",
            int(timestamps.size), int(markers.size), self._run_dir,
        )

    # ── internals ────────────────────────────────────────────────────────────────

    def _t0(self) -> float:
        return self._lsl_t0 if self._lsl_t0 is not None else 0.0

    def _stack_predictions(
        self, predictions: dict[str, np.ndarray], expected_shape: tuple[int, ...]
    ) -> np.ndarray:
        columns = []
        for name in self._task_names:
            values = np.asarray(predictions.get(name))
            if values.shape != expected_shape:
                raise ValueError(
                    f"Prediction vector for '{name}' has shape {values.shape}, "
                    f"expected {expected_shape}."
                )
            columns.append(values.astype(float))
        return np.column_stack(columns) if columns else np.empty((expected_shape[0], 0))

    def _collect_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_tasks = len(self._task_names)
        if self._ts_chunks:
            timestamps = np.concatenate(self._ts_chunks)
            predictions = np.vstack(self._pred_chunks)
        else:
            timestamps = np.empty(0, dtype=float)
            predictions = np.empty((0, n_tasks), dtype=float)
        markers = np.array(self._marker_rows, dtype=_MARKER_DTYPE)
        return timestamps, predictions, markers

    def _build_manifest(
        self,
        *,
        final: bool = False,
        n_predictions: int | None = None,
        n_markers: int | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "wall_clock_start": self._wall_clock_start,
            "wall_clock_end": datetime.now().isoformat(timespec="seconds") if final else None,
            "lsl_t0": self._lsl_t0,
            "task_names": self._task_names,
            "event_map": {str(code): name for code, name in self._event_names.items()},
            "n_predictions": n_predictions,
            "n_markers": n_markers,
            **self._metadata,
        }
        if self._decision_config is not None:
            manifest["decision_schema_version"] = DECISION_SCHEMA_VERSION
            manifest["decision_initial_config"] = self._decision_config
            manifest["n_decision_samples"] = self._n_decision_samples
        return manifest

    def _write_manifest(self, **kwargs: Any) -> None:
        (self._run_dir / MANIFEST_JSON).write_text(
            json.dumps(self._build_manifest(**kwargs), indent=2)
        )


# ── npz writer + recovery exporter ───────────────────────────────────────────────


def _save_npz(
    path: str | Path,
    *,
    timestamps: np.ndarray,
    predictions: np.ndarray,
    task_names: list[str],
    markers: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    """Write the prediction/marker arrays + manifest into a single ``.npz``.

    Shared by ``LiveSessionLogger.close()`` (full-precision in-memory arrays)
    and :func:`export_session_npz` (arrays rebuilt from the CSVs).
    """
    t0 = manifest.get("lsl_t0")
    t_sec = (
        np.round(timestamps - t0, _T_SEC_DECIMALS)
        if t0 is not None and timestamps.size
        else np.zeros_like(timestamps)
    )
    np.savez(
        path,
        predictions=predictions,
        task_names=np.array(task_names, dtype="U"),
        lsl_timestamp=timestamps,
        t_sec=t_sec,
        markers=markers,
        manifest_json=np.array(json.dumps(manifest)),
    )


def export_session_npz(run_dir: str | Path) -> Path:
    """Rebuild ``predictions.npz`` from a run directory's CSVs + manifest.

    Use for sessions that crashed before ``close()`` could write the npz, or to
    re-export. Reads the source-of-truth CSVs, so the result inherits the CSV's
    probability rounding (unlike the live close() path, which keeps full
    precision). Returns the npz path.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / MANIFEST_JSON).read_text())
    task_names = list(manifest["task_names"])

    timestamps, predictions = _read_predictions_csv(run_dir / PREDICTIONS_CSV, task_names)
    markers = _read_markers_csv(run_dir / MARKERS_CSV)

    npz_path = run_dir / PREDICTIONS_NPZ
    _save_npz(
        npz_path,
        timestamps=timestamps,
        predictions=predictions,
        task_names=task_names,
        markers=markers,
        manifest=manifest,
    )
    return npz_path


def _read_predictions_csv(
    path: Path, task_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    data = rows[1:]  # drop header
    if not data:
        return np.empty(0, dtype=float), np.empty((0, len(task_names)), dtype=float)
    arr = np.array(data, dtype=float)
    return arr[:, 0], arr[:, 2:]  # lsl_timestamp, <task cols> (skip t_sec)


def _read_markers_csv(path: Path) -> np.ndarray:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    data = rows[1:]  # drop header
    return np.array(
        [(float(r[0]), float(r[1]), int(r[2]), r[3]) for r in data],
        dtype=_MARKER_DTYPE,
    )


# ── decisions reader ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecisionEpisode:
    """One closed (or open) activation interval for a single decoder.

    ``offset_ts`` is the timestamp of the first sample the decoder was *off* again,
    or ``None`` if it was still active when the run ended (a trailing open episode).
    """

    decoder: str
    onset_ts: float
    offset_ts: float | None
    config_version_at_onset: int


def episodes_from_decisions(run_dir: str | Path) -> list[DecisionEpisode]:
    """Pair the dense ``decisions.csv`` booleans into per-decoder episodes.

    Edges are recovered by diffing each decoder's column: a ``False→True`` step
    opens an episode (onset at the ``True`` sample), a ``True→False`` step closes it
    (offset at the ``False`` sample). A column still ``True`` at the end yields an
    open episode (``offset_ts=None``).
    """
    run_dir = Path(run_dir)
    with (run_dir / DECISIONS_CSV).open(newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    decoders = header[2:-1]  # between t_sec and config_version
    version_col = len(header) - 1

    episodes: list[DecisionEpisode] = []
    for offset, decoder in enumerate(decoders):
        col = offset + 2
        active = False
        onset_ts = 0.0
        onset_version = 0
        for row in data:
            ts = float(row[0])
            is_on = row[col] == "True"
            if is_on and not active:
                onset_ts, onset_version = ts, int(row[version_col])
            elif not is_on and active:
                episodes.append(
                    DecisionEpisode(decoder, onset_ts, ts, onset_version)
                )
            active = is_on
        if active:
            episodes.append(DecisionEpisode(decoder, onset_ts, None, onset_version))
    return episodes
