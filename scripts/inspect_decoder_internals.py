"""Visualise spatial patterns + experiment timeline for the trained decoders.

Three artifacts:

1. **Spatial patterns** (topomaps) — Haufe-transformed weights for each decoder
   show what scalp pattern each model is picking up. Tightly localised vs
   diffuse patterns help explain visual differences between decoders.
2. **Marker timeline** — scatter of every stimulus marker by type over the
   full recording. Reveals block structure vs. interleaving and any drift.
3. **Per-trial P(positive) variance at peak** — a numeric companion to the
   live-inference overlay plots: low SD = tight trial cluster (e.g. red),
   high SD = noisy trial cloud (e.g. green/yellow).

Outputs (suffixed via ``--suffix`` to preserve previous runs):
  debug_snapshots/decoder_spatial_patterns<suffix>.png
  debug_snapshots/marker_timeline<suffix>.png
  + per-trial-variance numeric summary to stdout
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

mne.set_log_level("ERROR")

from backend.core.settings_manager import SettingsManager
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.online_preprocessor import OnlinePreprocessor

OUT_DIR = PROJECT_ROOT / "debug_snapshots"
TRAIN_DONE_PATH = OUT_DIR / "train_done.joblib"
ARTIFACT_PATH = OUT_DIR / "models" / "decoder_pipeline.joblib"
CONFIG_PATH = OUT_DIR / "experiment_config.yaml"
RECORDING_DIR = PROJECT_ROOT / "data" / "split" / "functional_localizer"
BATCH_SIZE_SAMPLES = 40

MARKER_COLORS = {
    "red": "crimson",
    "green": "green",
    "yellow": "goldenrod",
    "living_room": "purple",
    "bathroom": "teal",
    "kitchen": "saddlebrown",
}


def _png(stem: str, suffix: str) -> Path:
    return OUT_DIR / f"{stem}{suffix}.png"


# ── Spatial patterns ─────────────────────────────────────────────────────────


def plot_spatial_patterns(
    spatial_patterns: dict[str, np.ndarray],
    info: mne.Info,
    out_path: Path,
) -> None:
    """One topomap per decoder, Haufe-transformed weights projected to scalp."""
    task_names = list(spatial_patterns.keys())
    n_cols = 3
    n_rows = (len(task_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows), squeeze=False)
    # Common colour scale across decoders so visual amplitude is comparable.
    all_values = np.concatenate([np.asarray(p, dtype=float) for p in spatial_patterns.values()])
    vmax = float(np.percentile(np.abs(all_values), 99))
    for plot_index, task_name in enumerate(task_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        pattern = np.asarray(spatial_patterns[task_name], dtype=float)
        mne.viz.plot_topomap(
            pattern, info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-vmax, vmax), contours=4,
        )
        ax.set_title(task_name, fontsize=10)
    for unused_index in range(len(task_names), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.suptitle("Decoder spatial patterns (Haufe-transformed weights)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Marker timeline ──────────────────────────────────────────────────────────


def find_vhdr(directory: Path) -> Path:
    candidates = list(directory.glob("*.vhdr"))
    if not candidates:
        raise FileNotFoundError(f"No .vhdr in {directory}")
    return candidates[0]


def extract_stimulus_events(
    vhdr_path: Path, event_mapping: dict[str, int]
) -> tuple[dict[str, np.ndarray], float, float]:
    """Return (marker_times_by_name in seconds, sfreq, recording_duration_s)."""
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    description_to_code: dict[str, int] = {}
    for description in set(raw.annotations.description):
        match = re.search(r"(\d+)\s*$", description)
        if match:
            description_to_code[description] = int(match.group(1))
    events, _ = mne.events_from_annotations(raw, event_id=description_to_code, verbose=False)
    marker_times_by_name: dict[str, np.ndarray] = {}
    for marker_name, code in event_mapping.items():
        times = np.array(
            [sample / sfreq for sample, _, evt in events if evt == code],
            dtype=float,
        )
        marker_times_by_name[marker_name] = times
    return marker_times_by_name, sfreq, duration


def plot_marker_timeline(
    marker_times_by_name: dict[str, np.ndarray],
    duration_s: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ordered = [
        ("red", marker_times_by_name.get("red", np.array([]))),
        ("green", marker_times_by_name.get("green", np.array([]))),
        ("yellow", marker_times_by_name.get("yellow", np.array([]))),
        ("living_room", marker_times_by_name.get("living_room", np.array([]))),
        ("bathroom", marker_times_by_name.get("bathroom", np.array([]))),
        ("kitchen", marker_times_by_name.get("kitchen", np.array([]))),
    ]
    for name, times in ordered:
        ax.scatter(times, [name] * len(times),
                   s=16, color=MARKER_COLORS.get(name), alpha=0.8, edgecolors="none")
    ax.set_xlim(0, duration_s)
    ax.set_xlabel("time in recording (s)")
    ax.set_title(f"Stimulus-marker timeline (full {duration_s:.0f}s recording)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_marker_stats(marker_times_by_name: dict[str, np.ndarray]) -> None:
    print("\n=== Marker timing stats ===")
    column_header = (
        f"{'marker':<14}{'first (s)':>11}{'last (s)':>11}{'span (s)':>11}"
        f"{'median ISI':>13}{'min ISI':>10}{'max ISI':>10}{'n':>5}"
    )
    print(column_header)
    print("-" * len(column_header))
    for marker_name, times in marker_times_by_name.items():
        if times.size == 0:
            continue
        if times.size == 1:
            median_isi = min_isi = max_isi = float("nan")
        else:
            diffs = np.diff(times)
            median_isi = float(np.median(diffs))
            min_isi = float(diffs.min())
            max_isi = float(diffs.max())
        print(
            f"{marker_name:<14}{times.min():>11.1f}{times.max():>11.1f}"
            f"{(times.max() - times.min()):>11.1f}"
            f"{median_isi:>13.2f}{min_isi:>10.2f}{max_isi:>10.2f}{times.size:>5d}"
        )


def print_block_structure(marker_times_by_name: dict[str, np.ndarray]) -> None:
    """Are markers blocked (each type clustered) or interleaved?

    For each marker, compute the fraction of the recording's total span over
    which it occurs. A "blocked" marker covers a small fraction of total time;
    an "interleaved" marker spans the whole recording.
    """
    all_times = np.concatenate(
        [t for t in marker_times_by_name.values() if t.size > 0]
    )
    if all_times.size == 0:
        return
    total_span = all_times.max() - all_times.min()
    print("\n=== Block-vs-interleaved test ===")
    print(f"{'marker':<14}{'span / total':>14}{'verdict':>20}")
    print("-" * 48)
    for marker_name, times in marker_times_by_name.items():
        if times.size < 2:
            continue
        span = times.max() - times.min()
        fraction = span / total_span if total_span > 0 else 0.0
        verdict = "blocked" if fraction < 0.5 else "interleaved" if fraction > 0.8 else "mixed"
        print(f"{marker_name:<14}{fraction:>14.2%}{verdict:>20s}")


# ── Per-trial variance at peak ────────────────────────────────────────────────


def stream_and_compute_per_trial_variance(
    vhdr_path: Path,
    settings: SettingsManager,
    artifact_path: Path,
) -> None:
    """Stream the full recording, epoch around markers, compute per-trial SD
    of P(positive) at each decoder's trained timepoint.

    Memory-conscious load (preload=False + pick + load_data).
    """
    artifact = load_decoder_pipeline_artifact(artifact_path)
    preproc_settings = settings.get_preprocessing_params()
    decoder_settings = settings.get_decoder_settings()
    event_mapping = settings.get_event_mapping()
    representative_tp = float(artifact.metadata.get("decoding_timepoint"))
    per_task_tps_raw = artifact.metadata.get("decoding_timepoints") or {}
    task_to_tp = {
        task["name"]: float(per_task_tps_raw.get(task["name"], representative_tp))
        for task in decoder_settings["tasks"]
    }
    task_to_marker = {task["name"]: task["pos_labels"][0] for task in decoder_settings["tasks"]}

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    if "EMG" in eeg_names:
        eeg_names.remove("EMG")
    raw.pick(eeg_names)
    raw.load_data(verbose=False)
    n_times = raw.n_times
    eeg = raw.get_data().T.astype(np.float64, copy=False)

    description_to_code: dict[str, int] = {}
    for description in set(raw.annotations.description):
        match = re.search(r"(\d+)\s*$", description)
        if match:
            description_to_code[description] = int(match.group(1))
    events, _ = mne.events_from_annotations(raw, event_id=description_to_code, verbose=False)
    del raw

    online_preprocessor = OnlinePreprocessor(preproc_settings, artifact.online_state)
    engine = LiveInferenceEngine(artifact.models, artifact.metadata)
    feature_chunks: list[np.ndarray] = []
    output_index_chunks: list[np.ndarray] = []
    online_preprocessor.reset_state()
    sample_idx = np.arange(n_times)
    print("\n=== Per-trial variance at trained timepoint ===")
    print("(streaming online preprocessor; ~60-90s)")
    for start in range(0, n_times, BATCH_SIZE_SAMPLES):
        sl = slice(start, start + BATCH_SIZE_SAMPLES)
        feats, out_ts = online_preprocessor.process_batch(eeg[sl], sample_idx[sl])
        if feats.shape[0]:
            feature_chunks.append(feats)
            output_index_chunks.append(out_ts)
    features = np.vstack(feature_chunks)
    output_samples = np.concatenate(output_index_chunks)
    predictions = engine.predict(features)
    fs_out = online_preprocessor.target_sfreq

    tmin = preproc_settings["epochs"]["tmin"]
    tmax = preproc_settings["epochs"]["tmax"]
    t_grid = np.arange(round(tmin * fs_out), round(tmax * fs_out) + 1) / fs_out
    relative_time_per_output = output_samples / sfreq

    column_header = (
        f"{'decoder':<22}{'tp(s)':>8}{'pos mean':>12}{'pos SD':>10}"
        f"{'neg mean':>12}{'neg SD':>10}{'n_pos':>8}{'n_neg':>8}"
    )
    print(column_header)
    print("-" * len(column_header))
    for task_name, probability_stream in predictions.items():
        marker_name = task_to_marker.get(task_name)
        target_code = event_mapping[marker_name]
        task_tp = task_to_tp[task_name]
        tp_index = int(np.argmin(np.abs(t_grid - task_tp)))
        pos_values: list[float] = []
        neg_values: list[float] = []
        for sample, _prev, code in events:
            if code not in event_mapping.values():
                continue
            relative = relative_time_per_output - sample / sfreq
            window = (relative >= tmin - 0.05) & (relative <= tmax + 0.05)
            if window.sum() < 2:
                continue
            epoch_curve = np.interp(t_grid, relative[window], probability_stream[window])
            value = float(epoch_curve[tp_index])
            if int(code) == int(target_code):
                pos_values.append(value)
            else:
                neg_values.append(value)
        pos_arr = np.array(pos_values, dtype=float)
        neg_arr = np.array(neg_values, dtype=float)
        print(
            f"{task_name:<22}{task_tp:>8.3f}"
            f"{pos_arr.mean():>12.3f}{pos_arr.std(ddof=1):>10.3f}"
            f"{neg_arr.mean():>12.3f}{neg_arr.std(ddof=1):>10.3f}"
            f"{pos_arr.size:>8d}{neg_arr.size:>8d}"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix", default="",
        help="Suffix appended to PNG filenames to preserve previous-run outputs.",
    )
    parser.add_argument(
        "--skip-variance", action="store_true",
        help="Skip the per-trial-variance computation (saves ~60-90s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not TRAIN_DONE_PATH.exists():
        print(f"train_done.joblib not found: {TRAIN_DONE_PATH}", file=sys.stderr)
        return 2
    state = joblib.load(TRAIN_DONE_PATH)
    ui = state.get("_ui_state") or {}
    spatial_patterns = ui.get("spatial_patterns")
    mne_info = ui.get("mne_info")
    if not spatial_patterns or mne_info is None:
        print("train_done.joblib has no spatial_patterns / mne_info.", file=sys.stderr)
        return 2

    # 1. Spatial-pattern topomaps
    spatial_png = _png("decoder_spatial_patterns", args.suffix)
    plot_spatial_patterns(spatial_patterns, mne_info, spatial_png)
    print(f"Spatial-pattern figure: {spatial_png}")

    # 2. Marker timeline
    settings = SettingsManager(CONFIG_PATH)
    event_mapping = settings.get_event_mapping()
    vhdr_path = find_vhdr(RECORDING_DIR)
    marker_times_by_name, _sfreq, duration_s = extract_stimulus_events(
        vhdr_path, event_mapping
    )
    timeline_png = _png("marker_timeline", args.suffix)
    plot_marker_timeline(marker_times_by_name, duration_s, timeline_png)
    print(f"Marker-timeline figure: {timeline_png}")
    print_marker_stats(marker_times_by_name)
    print_block_structure(marker_times_by_name)

    # 3. Per-trial variance at peak
    if not args.skip_variance:
        stream_and_compute_per_trial_variance(vhdr_path, settings, ARTIFACT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
