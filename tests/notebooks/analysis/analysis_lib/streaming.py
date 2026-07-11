"""Recording load, marker extraction, and probability-stream epoching.

The exact plumbing the analysis notebooks share: read a BrainVision recording
the way the replay feed does (EEG channels, EMG dropped), pull stimulus markers
with their real trigger codes, replay the EEG through the stateful
``OnlinePreprocessor`` in StreamWorker-sized micro-batches, and epoch the
resulting probability stream onto a common time grid around markers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import mne
import numpy as np


def find_vhdr(directory: str | Path) -> Path:
    """Return the single ``.vhdr`` in ``directory`` (first if several)."""
    vhdrs = list(Path(directory).glob("*.vhdr"))
    if not vhdrs:
        raise FileNotFoundError(f"No .vhdr file found in {directory}")
    return vhdrs[0]


def load_recording(
    directory: str | Path, max_seconds: float | None = None
) -> tuple[mne.io.BaseRaw, np.ndarray, float]:
    """Load a recording and return ``(raw, eeg, sfreq)``.

    ``eeg`` is the ``(n_times, n_eeg)`` array fed to the online preprocessor:
    EEG-typed channels with EMG dropped, matching the 64-channel replay feed,
    in **microvolts** — ``OnlinePreprocessor.process_batch`` applies a fixed
    µV->SI-volt scale (``LSL_TO_SI_SCALE``) up front, mirroring the real LSL
    wire unit, so replay must hand it µV the same way ``replay_vhdr_to_lsl.py``
    does (MNE's default ``get_data()`` is SI volts, which this fix converts
    from). When ``max_seconds`` is set the file is lazily cropped first so the
    full multi-hour buffer never lives in RAM.
    """
    vhdr = find_vhdr(directory)
    if max_seconds is not None:
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        raw.crop(tmin=0.0, tmax=float(max_seconds))
        raw.load_data(verbose=False)
    else:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    if "EMG" in eeg_names:
        eeg_names.remove("EMG")
    eeg = raw.copy().pick(eeg_names).get_data(units="uV").T  # (n_times, n_eeg) µV
    return raw, eeg, float(raw.info["sfreq"])


def extract_markers(
    raw: mne.io.BaseRaw, event_mapping: dict[str, int], marker_names: list[str],
    n_times: int | None = None,
) -> list[tuple[int, int]]:
    """Return ``[(sample, code), ...]`` for the requested marker names.

    Parses the real trigger codes from annotations (``'Stimulus/S 11' -> 11``)
    and keeps only events whose code is one of ``marker_names``' codes.
    """
    desc_to_code = {}
    for desc in set(raw.annotations.description):
        m = re.search(r"(\d+)\s*$", desc)
        if m:
            desc_to_code[desc] = int(m.group(1))
    events, _ = mne.events_from_annotations(raw, event_id=desc_to_code, verbose=False)
    wanted = {event_mapping[n] for n in marker_names if n in event_mapping}
    limit = n_times if n_times is not None else raw.n_times
    return [(int(s), int(c)) for s, _, c in events if c in wanted and s < limit]


def run_online_stream(
    preproc, eeg: np.ndarray, batch_size: int = 40
) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay ``eeg`` through ``preproc`` in micro-batches (StreamWorker-style).

    Returns ``(features, out_samples, fs_out)`` — the stacked feature rows at
    the target rate, the original-sample index of each output row, and the
    output sampling rate.
    """
    preproc.reset_state()
    n_times = eeg.shape[0]
    sample_idx = np.arange(n_times)
    feat_chunks, out_idx_chunks = [], []
    for start in range(0, n_times, batch_size):
        sl = slice(start, start + batch_size)
        feats, out_ts = preproc.process_batch(eeg[sl], sample_idx[sl])
        if feats.shape[0]:
            feat_chunks.append(feats)
            out_idx_chunks.append(out_ts)
    features = np.vstack(feat_chunks)
    out_samples = np.concatenate(out_idx_chunks)
    return features, out_samples, float(preproc.target_sfreq)


def _epoch_grid(fs: float, tmin: float, tmax: float) -> np.ndarray:
    return np.arange(round(tmin * fs), round(tmax * fs) + 1) / fs


def _interp_epoch(
    rel_time: np.ndarray, values: np.ndarray, t_grid: np.ndarray, tmin: float, tmax: float
) -> np.ndarray | None:
    m = (rel_time >= tmin - 0.05) & (rel_time <= tmax + 0.05)
    if m.sum() < 2:
        return None
    return np.interp(t_grid, rel_time[m], values[m])


def make_epocher(
    out_samples: np.ndarray, sfreq: float, fs_out: float, tmin: float, tmax: float
) -> tuple[np.ndarray, Callable[[np.ndarray, list[int]], np.ndarray]]:
    """Return ``(t_grid, epoch_stream)`` for a raw-sample-indexed probability stream.

    ``epoch_stream(prob, marker_samples)`` interpolates each marker-locked
    window onto the shared ``t_grid``, returning an ``(n_epochs, n_grid)`` array.
    """
    t_grid = _epoch_grid(fs_out, tmin, tmax)
    rel_time_base = out_samples / sfreq

    def epoch_stream(prob: np.ndarray, marker_samples: list[int]) -> np.ndarray:
        rows = []
        for s in marker_samples:
            row = _interp_epoch(rel_time_base - s / sfreq, prob, t_grid, tmin, tmax)
            if row is not None:
                rows.append(row)
        return np.array(rows) if rows else np.empty((0, len(t_grid)))

    return t_grid, epoch_stream


def _interp_epoch_multichannel(
    rel_time: np.ndarray, values: np.ndarray, t_grid: np.ndarray, tmin: float, tmax: float
) -> np.ndarray | None:
    """Multi-channel sibling of :func:`_interp_epoch`.

    ``values`` is ``(n_samples, n_ch)``; interpolates each channel independently
    onto ``t_grid`` and returns ``(n_ch, n_grid)`` (or ``None`` if fewer than 2
    stream samples fall in the marker window — same drop rule as the 1-D case).
    """
    m = (rel_time >= tmin - 0.05) & (rel_time <= tmax + 0.05)
    if m.sum() < 2:
        return None
    rt = rel_time[m]
    vals = values[m]  # (n_sel, n_ch)
    return np.stack([np.interp(t_grid, rt, vals[:, c]) for c in range(vals.shape[1])])


def make_epocher_multichannel(
    out_samples: np.ndarray, sfreq: float, fs_out: float, tmin: float, tmax: float
) -> tuple[np.ndarray, Callable[[np.ndarray, list[int]], tuple[np.ndarray, list[int]]]]:
    """Return ``(t_grid, epoch_features)`` for a raw-sample-indexed feature stream.

    The multi-channel counterpart of :func:`make_epocher`: instead of epoching a
    1-D probability stream, it epochs the ``(n_out, n_ch)`` feature stream that
    :func:`run_online_stream` returns. ``epoch_features(features, marker_samples)``
    interpolates every channel of each marker-locked window onto the shared
    ``t_grid`` and returns ``(epochs, kept_samples)`` where ``epochs`` is
    ``(n_epochs, n_ch, n_grid)`` and ``kept_samples`` lists the marker samples
    that actually produced an epoch (those with >=2 stream samples in-window) —
    the caller needs the survivors to pair online epochs with their offline
    counterparts.
    """
    t_grid = _epoch_grid(fs_out, tmin, tmax)
    rel_time_base = out_samples / sfreq

    def epoch_features(
        features: np.ndarray, marker_samples: list[int]
    ) -> tuple[np.ndarray, list[int]]:
        rows, kept = [], []
        for s in marker_samples:
            row = _interp_epoch_multichannel(
                rel_time_base - s / sfreq, features, t_grid, tmin, tmax
            )
            if row is not None:
                rows.append(row)
                kept.append(s)
        arr = (np.array(rows) if rows
               else np.empty((0, features.shape[1], len(t_grid))))
        return arr, kept

    return t_grid, epoch_features
