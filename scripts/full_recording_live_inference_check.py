"""Standalone full-recording version of validate_live_inference_epoched cell 19.

Runs the production online path (OnlinePreprocessor + LiveInferenceEngine) over
the entire FL recording and prints the decoder × marker probability table at
the trained timepoint. This is the load-bearing diagonal-dominance test
after the causal-filter fix.

Memory-conscious so it fits in ~4 GB:
  - VHDR loaded with preload=False (only header parsed)
  - EEG channels picked in place before load_data (drops trigger + EMG)
  - Raw is dropped as soon as we have the EEG array

Run from the project root:
    conda activate reactivation-decoder
    python scripts/full_recording_live_inference_check.py
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")  # headless-safe; we save PNGs, no display needed
import matplotlib.pyplot as plt
import mne
import numpy as np
from tqdm import tqdm

mne.set_log_level("ERROR")

MARKER_COLORS = {
    "red": "crimson",
    "green": "green",
    "yellow": "goldenrod",
    "living_room": "purple",
    "bathroom": "teal",
    "kitchen": "saddlebrown",
}

from backend.core.settings_manager import SettingsManager
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.online_preprocessor import OnlinePreprocessor

CONFIG_PATH = PROJECT_ROOT / "debug_snapshots" / "experiment_config.yaml"
ARTIFACT_PATH = PROJECT_ROOT / "debug_snapshots" / "models" / "decoder_pipeline.joblib"
RECORDING_DIR = PROJECT_ROOT / "data" / "split" / "functional_localizer"
BATCH_SIZE_SAMPLES = 40
OUT_DIR = PROJECT_ROOT / "debug_snapshots"


def _png_path(stem: str, suffix: str) -> Path:
    return OUT_DIR / f"{stem}{suffix}.png"


def find_vhdr(directory: Path) -> Path:
    candidates = list(directory.glob("*.vhdr"))
    if not candidates:
        raise FileNotFoundError(f"No .vhdr file in {directory}")
    return candidates[0]


def _trained_tp_for(task_name: str, task_to_tp: dict[str, float] | None,
                    fallback: float | None) -> float | None:
    if task_to_tp and task_name in task_to_tp:
        return task_to_tp[task_name]
    return fallback


def plot_individual_epochs(
    epoched: dict[str, dict[str, np.ndarray]],
    task_to_marker: dict[str, str],
    t_grid: np.ndarray,
    trained_timepoint: float | None,
    out_path: Path,
    task_to_trained_tp: dict[str, float] | None = None,
) -> None:
    """Per-decoder: faint per-trial P(t) for the decoder's own positive marker,
    plus the across-trial mean.

    Mirrors cell 15 of validate_live_inference_epoched.ipynb.
    """
    task_names = list(epoched.keys())
    n_cols = 3
    n_rows = (len(task_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for plot_index, task_name in enumerate(task_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        marker_name = task_to_marker.get(task_name)
        epochs_for_marker = epoched.get(task_name, {}).get(marker_name, np.empty((0, t_grid.size)))
        for trial_row in epochs_for_marker:
            ax.plot(t_grid, trial_row, color="steelblue", alpha=0.25, lw=0.8)
        if epochs_for_marker.shape[0]:
            ax.plot(t_grid, epochs_for_marker.mean(axis=0), color="navy", lw=2.5,
                    label=f"mean (n={epochs_for_marker.shape[0]})")
        ax.axvline(0, color="k", ls=":", lw=1)
        task_tp = _trained_tp_for(task_name, task_to_trained_tp, trained_timepoint)
        if task_tp is not None:
            ax.axvline(task_tp, color="crimson", ls="--", lw=1,
                       label=f"trained tp ({task_tp:.2f}s)")
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.set(title=f"{task_name} — '{marker_name}'", ylim=(0, 1),
               xlabel="time from marker (s)", ylabel="P(positive)")
        ax.legend(fontsize=7, loc="upper right")
    for unused_index in range(len(task_names), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_decoder_baselines(
    predictions: dict[str, np.ndarray],
    output_samples: np.ndarray,
    sfreq: float,
    marker_sample_indices: list[int],
    exclude_window_s: float = 2.0,
) -> dict[str, tuple[float, float]]:
    """Per-decoder (mean, SD) of P(positive) on samples NOT within ±exclude_window_s
    of any stimulus marker. These are the "inter-trial baseline" statistics
    used for z-score normalisation.
    """
    output_times = output_samples / sfreq
    baseline_mask = np.ones(output_times.size, dtype=bool)
    for marker_sample in marker_sample_indices:
        marker_time = marker_sample / sfreq
        near_marker = np.abs(output_times - marker_time) < exclude_window_s
        baseline_mask &= ~near_marker
    baselines: dict[str, tuple[float, float]] = {}
    for task_name, probability_stream in predictions.items():
        baseline_values = probability_stream[baseline_mask]
        mu = float(baseline_values.mean())
        sigma = float(baseline_values.std(ddof=1))
        if sigma < 1e-9:
            sigma = 1e-9
        baselines[task_name] = (mu, sigma)
    return baselines


def zscore_epoched(
    epoched: dict[str, dict[str, np.ndarray]],
    baselines: dict[str, tuple[float, float]],
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for task_name, marker_trials in epoched.items():
        mu, sigma = baselines[task_name]
        out[task_name] = {
            marker_name: (trials - mu) / sigma
            for marker_name, trials in marker_trials.items()
        }
    return out


def plot_per_marker_decoder_overlay_zscore(
    epoched_z: dict[str, dict[str, np.ndarray]],
    markers_of_interest: list[str],
    task_to_marker: dict[str, str],
    t_grid: np.ndarray,
    out_path: Path,
    task_to_trained_tp: dict[str, float] | None = None,
) -> None:
    """Z-scored 'decoder competition' view. Y axis is SDs above each decoder's
    own inter-trial baseline. Z=0 is baseline; large positive z = decoder fired
    well above its rest level. This makes decoder outputs comparable across
    decoders despite different absolute baselines.
    """
    task_names = list(epoched_z.keys())
    decoder_to_color = {
        task_name: MARKER_COLORS.get(task_to_marker.get(task_name))
        for task_name in task_names
    }

    # Auto-scale y across all panels so they're visually comparable.
    all_means = []
    for task_name in task_names:
        for marker_name in markers_of_interest:
            trials = epoched_z[task_name].get(marker_name)
            if trials is not None and trials.shape[0]:
                all_means.append(trials.mean(axis=0))
    if all_means:
        global_max = max(float(np.max(m)) for m in all_means)
        global_min = min(float(np.min(m)) for m in all_means)
    else:
        global_max, global_min = 5.0, -2.0
    ylim = (min(global_min - 0.5, -1.0), max(global_max + 0.5, 5.0))

    n_cols = 3
    n_rows = (len(markers_of_interest) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for plot_index, marker_name in enumerate(markers_of_interest):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        n_trials_here = 0
        for task_name in task_names:
            trials = epoched_z[task_name].get(marker_name)
            if trials is None or trials.shape[0] == 0:
                continue
            n_trials_here = trials.shape[0]
            mean_curve = trials.mean(axis=0)
            sem_curve = trials.std(axis=0) / np.sqrt(trials.shape[0])
            color = decoder_to_color.get(task_name)
            ax.plot(t_grid, mean_curve, color=color, lw=1.8, label=task_name)
            ax.fill_between(t_grid, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.15)
            task_tp = _trained_tp_for(task_name, task_to_trained_tp, None)
            if task_tp is not None and t_grid[0] <= task_tp <= t_grid[-1]:
                ax.axvline(task_tp, color=color, ls=":", lw=0.6, alpha=0.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.8, label="baseline (z=0)")
        ax.axhline(3, color="gray", lw=0.4, ls="--", alpha=0.5)  # ~"significant fire"
        ax.set(title=f"'{marker_name}' marker  (n={n_trials_here} trials)",
               ylim=ylim, xlabel="time from marker (s)",
               ylabel="z-score per decoder")
        ax.legend(fontsize=6, loc="upper right")
    for unused_index in range(len(markers_of_interest), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_marker_decoder_overlay(
    epoched: dict[str, dict[str, np.ndarray]],
    markers_of_interest: list[str],
    task_to_marker: dict[str, str],
    t_grid: np.ndarray,
    out_path: Path,
    task_to_trained_tp: dict[str, float] | None = None,
) -> None:
    """Transposed view: one panel per MARKER, each line = one DECODER.

    Around each stimulus type, shows how every decoder responds. The decoder
    whose target IS this marker should clearly dominate; others should stay
    near their respective baselines.
    """
    task_names = list(epoched.keys())
    # Color each decoder by its target-marker color (1:1 mapping).
    decoder_to_color = {
        task_name: MARKER_COLORS.get(task_to_marker.get(task_name))
        for task_name in task_names
    }

    n_cols = 3
    n_rows = (len(markers_of_interest) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for plot_index, marker_name in enumerate(markers_of_interest):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        n_trials_here = 0
        for task_name in task_names:
            trials = epoched[task_name].get(marker_name)
            if trials is None or trials.shape[0] == 0:
                continue
            n_trials_here = trials.shape[0]
            mean_curve = trials.mean(axis=0)
            sem_curve = trials.std(axis=0) / np.sqrt(trials.shape[0])
            color = decoder_to_color.get(task_name)
            ax.plot(t_grid, mean_curve, color=color, lw=1.8,
                    label=f"{task_name}")
            ax.fill_between(t_grid, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.15)
            # Tick mark at this decoder's own trained timepoint.
            task_tp = _trained_tp_for(task_name, task_to_trained_tp, None)
            if task_tp is not None and t_grid[0] <= task_tp <= t_grid[-1]:
                ax.axvline(task_tp, color=color, ls=":", lw=0.6, alpha=0.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.set(title=f"'{marker_name}' marker  (n={n_trials_here} trials)",
               ylim=(0, 1), xlabel="time from marker (s)",
               ylabel="P(positive) per decoder")
        ax.legend(fontsize=6, loc="upper right")
    for unused_index in range(len(markers_of_interest), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_marker_overlay(
    epoched: dict[str, dict[str, np.ndarray]],
    markers_of_interest: list[str],
    t_grid: np.ndarray,
    trained_timepoint: float | None,
    out_path: Path,
    task_to_trained_tp: dict[str, float] | None = None,
) -> None:
    """Per-decoder: mean ± SEM P(t) for EVERY marker overlaid.

    Mirrors cell 17 of validate_live_inference_epoched.ipynb.
    """
    task_names = list(epoched.keys())
    n_cols = 3
    n_rows = (len(task_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for plot_index, task_name in enumerate(task_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        for marker_name in markers_of_interest:
            trials = epoched[task_name][marker_name]
            if trials.shape[0] == 0:
                continue
            mean_curve = trials.mean(axis=0)
            sem_curve = trials.std(axis=0) / np.sqrt(trials.shape[0])
            color = MARKER_COLORS.get(marker_name)
            ax.plot(t_grid, mean_curve, color=color, lw=1.8,
                    label=f"{marker_name} ({trials.shape[0]})")
            ax.fill_between(t_grid, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.15)
        ax.axvline(0, color="k", ls=":", lw=1)
        task_tp = _trained_tp_for(task_name, task_to_trained_tp, trained_timepoint)
        if task_tp is not None:
            ax.axvline(task_tp, color="black", ls="--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6)
        title = task_name if task_tp is None else f"{task_name}  (trained tp: {task_tp:.2f}s)"
        ax.set(title=title, ylim=(0, 1),
               xlabel="time from marker (s)", ylabel="P(positive)")
        ax.legend(fontsize=6, loc="upper right")
    for unused_index in range(len(task_names), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to all PNG filenames (e.g. '_stepC') to preserve "
             "previous-run outputs side by side.",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Epoch window start in seconds (overrides config). Negative = "
             "before marker. Use a wider window (e.g. -1.0) to see baseline.",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Epoch window end in seconds (overrides config). Default ~1.0s. "
             "Use 5-10s to see inter-trial transients spanning subsequent "
             "markers (median ISI ~5-7s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suffix = args.suffix
    started = time.perf_counter()
    settings = SettingsManager(CONFIG_PATH)
    preprocessing_settings = settings.get_preprocessing_params()
    decoder_settings = settings.get_decoder_settings()
    event_mapping = settings.get_event_mapping()
    name_by_code = {code: name for name, code in event_mapping.items()}
    artifact = load_decoder_pipeline_artifact(ARTIFACT_PATH)

    markers_of_interest = list(dict.fromkeys(
        label for task in decoder_settings["tasks"] for label in task["pos_labels"]
    ))
    codes_of_interest = {event_mapping[name] for name in markers_of_interest}

    print(f"resample_filter_stage : {preprocessing_settings.get('resample_filter_stage')}")
    print(f"lowpass.method        : {preprocessing_settings['lowpass'].get('method')}")
    print(f"tasks                 : {list(artifact.models.keys())}")
    print(f"markers of interest   : {markers_of_interest}")
    print(f"trained timepoint     : {artifact.metadata.get('decoding_timepoint')}")
    print()

    # ── Load the recording, memory-conscious ────────────────────────────────
    vhdr_path = find_vhdr(RECORDING_DIR)
    print(f"loading {vhdr_path.name} (lazy)…")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose=False)
    sfreq = float(raw.info["sfreq"])

    # Pick EEG channels in place BEFORE materialising the buffer, so trigger +
    # EMG never live in RAM. This keeps the peak load at 64ch × n_samples × 8B
    # (~1.67 GB for a 3267 s recording).
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    if "EMG" in eeg_names:
        eeg_names.remove("EMG")
    raw.pick(eeg_names)
    raw.load_data(verbose=False)
    n_times = raw.n_times
    print(f"  sfreq={sfreq:g} Hz | channels={len(raw.ch_names)} | "
          f"samples={n_times} ({n_times / sfreq:.1f} s)")

    # ── Extract markers ─────────────────────────────────────────────────────
    description_to_code: dict[str, int] = {}
    for description in set(raw.annotations.description):
        match = re.search(r"(\d+)\s*$", description)
        if match:
            description_to_code[description] = int(match.group(1))
    events, _ = mne.events_from_annotations(raw, event_id=description_to_code, verbose=False)
    markers: list[tuple[int, int]] = [
        (int(sample), int(code))
        for sample, _prev, code in events
        if code in codes_of_interest and sample < n_times
    ]
    print(f"  epochable markers: "
          f"{ {name_by_code[code]: count for code, count in Counter(c for _, c in markers).items()} }")
    print()

    eeg = raw.get_data().T.astype(np.float64, copy=False)  # (n_times, 64)
    raw.close()  # drop the underlying Raw to free memory
    del raw
    gc.collect()

    # ── Build the live engine ───────────────────────────────────────────────
    preproc = OnlinePreprocessor(preprocessing_settings, artifact.online_state)
    engine = LiveInferenceEngine(artifact.models, artifact.metadata)
    if engine.feature_width != preproc.n_channels:
        raise RuntimeError(
            f"feature_width {engine.feature_width} != preproc.n_channels {preproc.n_channels}"
        )

    # ── Stream through the online preprocessor in micro-batches ─────────────
    sample_idx = np.arange(n_times)
    feature_chunks: list[np.ndarray] = []
    output_index_chunks: list[np.ndarray] = []
    preproc.reset_state()
    n_batches = (n_times + BATCH_SIZE_SAMPLES - 1) // BATCH_SIZE_SAMPLES
    for start in tqdm(
        range(0, n_times, BATCH_SIZE_SAMPLES),
        total=n_batches,
        desc="online streaming",
        unit="batch",
    ):
        slice_ = slice(start, start + BATCH_SIZE_SAMPLES)
        features_batch, output_index_batch = preproc.process_batch(eeg[slice_], sample_idx[slice_])
        if features_batch.shape[0]:
            feature_chunks.append(features_batch)
            output_index_chunks.append(output_index_batch)

    features = np.vstack(feature_chunks)
    output_samples = np.concatenate(output_index_chunks)
    del feature_chunks, output_index_chunks, eeg
    gc.collect()
    print(f"  output rows={features.shape[0]} @ {preproc.target_sfreq:g} Hz")
    print()

    # ── Predict across the whole stream ─────────────────────────────────────
    predictions = engine.predict(features)  # dict[task_name -> (n_out,)]
    target_sfreq = preproc.target_sfreq

    # ── Epoch the prediction stream at marker times ─────────────────────────
    tmin = args.tmin if args.tmin is not None else preprocessing_settings["epochs"]["tmin"]
    tmax = args.tmax if args.tmax is not None else preprocessing_settings["epochs"]["tmax"]
    if args.tmin is not None or args.tmax is not None:
        print(f"  epoch window override: tmin={tmin:.2f}s, tmax={tmax:.2f}s "
              f"(config: {preprocessing_settings['epochs']['tmin']:.2f} to "
              f"{preprocessing_settings['epochs']['tmax']:.2f})")
    t_grid = np.arange(round(tmin * target_sfreq), round(tmax * target_sfreq) + 1) / target_sfreq
    relative_time = output_samples / sfreq

    epoched: dict[str, dict[str, np.ndarray]] = {task: {} for task in predictions}
    for task, probability_stream in predictions.items():
        for marker_name in markers_of_interest:
            target_code = event_mapping[marker_name]
            target_samples = [sample for sample, code in markers if code == target_code]
            rows: list[np.ndarray] = []
            for sample in target_samples:
                relative = relative_time - sample / sfreq
                window = (relative >= tmin - 0.05) & (relative <= tmax + 0.05)
                if window.sum() < 2:
                    continue
                rows.append(np.interp(t_grid, relative[window], probability_stream[window]))
            epoched[task][marker_name] = (
                np.asarray(rows) if rows else np.empty((0, t_grid.size))
            )

    # ── Resolve per-decoder timepoints from artifact metadata ───────────────
    # Step C: each decoder may have its own trained timepoint. Fall back to the
    # representative single timepoint for legacy artifacts that lack the dict.
    representative_timepoint = artifact.metadata.get("decoding_timepoint")
    if representative_timepoint is None:
        print("artifact has no decoding_timepoint; cannot summarise.")
        return 1
    per_task_timepoints_raw = artifact.metadata.get("decoding_timepoints") or {}
    task_to_timepoint: dict[str, float] = {
        task: float(per_task_timepoints_raw.get(task, representative_timepoint))
        for task in predictions
    }

    column_width = max(9, max(len(name) for name in markers_of_interest) + 1)
    if per_task_timepoints_raw:
        spread = (
            max(task_to_timepoint.values()) - min(task_to_timepoint.values())
        )
        print(f"Mean P(positive) at each decoder's own trained timepoint "
              f"(spread {spread * 1000:.0f}ms):\n")
        tp_header_label = "tp(s)"
    else:
        print(f"Mean P(positive) at trained tp={representative_timepoint:.3f}s "
              f"(legacy single-timepoint artifact):\n")
        tp_header_label = "tp(s)"
    header = (
        "task".ljust(22)
        + tp_header_label.rjust(8)
        + "".join(name.rjust(column_width) for name in markers_of_interest)
    )
    print(header)
    print("-" * len(header))
    for task in predictions:
        task_tp = task_to_timepoint[task]
        timepoint_index = int(np.argmin(np.abs(t_grid - task_tp)))
        row = task.ljust(22) + f"{task_tp:.3f}".rjust(8)
        for marker_name in markers_of_interest:
            trials = epoched[task][marker_name]
            cell = (
                f"{trials[:, timepoint_index].mean():.3f}"
                if trials.shape[0]
                else "n/a"
            )
            row += cell.rjust(column_width)
        print(row)

    # Diagonal-dominance summary — each decoder sampled at its own timepoint.
    task_to_marker = {task["name"]: task["pos_labels"][0] for task in decoder_settings["tasks"]}
    dominant = 0
    print("\nDiagonal-dominance check (each decoder at its own trained tp):")
    for task_name in predictions:
        marker_name = task_to_marker.get(task_name)
        if marker_name is None:
            continue
        task_tp = task_to_timepoint[task_name]
        timepoint_index = int(np.argmin(np.abs(t_grid - task_tp)))
        diagonal_value = epoched[task_name][marker_name][:, timepoint_index].mean()
        row_values = [
            (other, epoched[task_name][other][:, timepoint_index].mean())
            for other in markers_of_interest
        ]
        winning_marker, winning_value = max(row_values, key=lambda pair: pair[1])
        ok = winning_marker == marker_name
        if ok:
            dominant += 1
        flag = "✓" if ok else "✗"
        print(f"  {flag} {task_name:>22s} @ {task_tp:.3f}s: "
              f"diagonal={diagonal_value:.3f}  "
              f"winner={winning_marker}({winning_value:.3f})")
    print(f"\nDiagonal-dominant decoders: {dominant}/{len(predictions)}")

    individual_epochs_png = _png_path("live_inference_individual_epochs", suffix)
    per_marker_overlay_png = _png_path("live_inference_per_marker_overlay", suffix)
    per_marker_decoder_overlay_png = _png_path(
        "live_inference_per_marker_decoder_overlay", suffix
    )

    plot_individual_epochs(
        epoched=epoched,
        task_to_marker=task_to_marker,
        t_grid=t_grid,
        trained_timepoint=representative_timepoint,
        out_path=individual_epochs_png,
        task_to_trained_tp=task_to_timepoint,
    )
    plot_per_marker_overlay(
        epoched=epoched,
        markers_of_interest=markers_of_interest,
        t_grid=t_grid,
        trained_timepoint=representative_timepoint,
        out_path=per_marker_overlay_png,
        task_to_trained_tp=task_to_timepoint,
    )
    plot_per_marker_decoder_overlay(
        epoched=epoched,
        markers_of_interest=markers_of_interest,
        task_to_marker=task_to_marker,
        t_grid=t_grid,
        out_path=per_marker_decoder_overlay_png,
        task_to_trained_tp=task_to_timepoint,
    )

    # ── Z-scored decoder competition view (handles per-decoder baseline drift)
    baselines = compute_decoder_baselines(
        predictions=predictions,
        output_samples=output_samples,
        sfreq=sfreq,
        marker_sample_indices=[s for s, _c in markers],
        exclude_window_s=2.0,
    )
    print("\n=== Per-decoder inter-trial baseline (P(positive)) ===")
    print(f"{'decoder':<22}{'baseline μ':>12}{'baseline σ':>12}")
    print("-" * 46)
    for task_name, (mu, sigma) in baselines.items():
        print(f"{task_name:<22}{mu:>12.3f}{sigma:>12.3f}")
    epoched_z = zscore_epoched(epoched, baselines)
    per_marker_decoder_overlay_zscore_png = _png_path(
        "live_inference_per_marker_decoder_overlay_zscore", suffix
    )
    plot_per_marker_decoder_overlay_zscore(
        epoched_z=epoched_z,
        markers_of_interest=markers_of_interest,
        task_to_marker=task_to_marker,
        t_grid=t_grid,
        out_path=per_marker_decoder_overlay_zscore_png,
        task_to_trained_tp=task_to_timepoint,
    )
    print(f"\nFigures written:")
    print(f"  {individual_epochs_png}")
    print(f"  {per_marker_overlay_png}")
    print(f"  {per_marker_decoder_overlay_png}")
    print(f"  {per_marker_decoder_overlay_zscore_png}")

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
