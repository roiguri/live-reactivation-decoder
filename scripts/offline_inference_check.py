"""Per-class offline-vs-online decoder overlay + CV evaluation graphs.

For each stimulus class (one of the decoder's positive markers), produces a
single PNG showing the corresponding decoder's mean P(positive) trajectory
on that class's trials with the OFFLINE preprocessing path overlaid on the
ONLINE streaming preprocessing path. Same trained decoders, same recording,
two preprocessing paths — if the curves overlap, streaming is faithful.

Also emits semester-A-style CV evaluation graphs (per-decoder AUC over time
+ temporal generalization matrices) from the saved CV results.

Outputs land in debug_snapshots/plots/offline_sanity_check/:
  - comparison_<class>.png (one per class: red, green, yellow, living_room, bathroom, kitchen)
  - cv_auc_curves.png      (per-decoder AUC over time)
  - cv_tgm_heatmaps.png    (per-decoder temporal generalization matrices)

Run from the project root:
    conda activate reactivation-decoder
    python scripts/offline_inference_check.py
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

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from tqdm import tqdm

mne.set_log_level("ERROR")

from backend.core.settings_manager import SettingsManager
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.online_preprocessor import OnlinePreprocessor

CONFIG_PATH = PROJECT_ROOT / "debug_snapshots" / "experiment_config.yaml"
ARTIFACT_PATH = PROJECT_ROOT / "debug_snapshots" / "models" / "decoder_pipeline.joblib"
SNAPSHOT_PATH = PROJECT_ROOT / "debug_snapshots" / "train_done.joblib"
EVAL_PATH = PROJECT_ROOT / "debug_snapshots" / "eval_done.joblib"
RECORDING_DIR = PROJECT_ROOT / "data" / "split" / "functional_localizer"
OUT_DIR_DEFAULT = PROJECT_ROOT / "debug_snapshots" / "plots" / "offline_sanity_check"
BATCH_SIZE_SAMPLES = 40

OFFLINE_COLOR = "navy"
ONLINE_COLOR = "crimson"


# ── small utilities ─────────────────────────────────────────────────────────


def find_vhdr(directory: Path) -> Path:
    candidates = list(directory.glob("*.vhdr"))
    if not candidates:
        raise FileNotFoundError(f"No .vhdr file in {directory}")
    return candidates[0]


def _positive_class_index(model, metadata: dict) -> int:
    positive_class = metadata.get("positive_class", 1)
    matches = np.where(np.asarray(model.classes_) == positive_class)[0]
    if matches.size == 0:
        raise ValueError(
            f"Model classes_={list(model.classes_)} does not contain positive_class={positive_class}"
        )
    return int(matches[0])


# ── trajectory builders ─────────────────────────────────────────────────────


def build_offline_trajectories(
    epochs: mne.Epochs,
    models: dict,
    metadata: dict,
    markers_of_interest: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Per-task, per-timepoint predict_proba over offline-preprocessed epochs.

    Returns:
        epoched: dict[task_name][marker_name] -> (n_trials_of_marker, n_times)
        t_grid:  the epoch time axis (seconds relative to marker, length n_times)
    """
    data = epochs.get_data()
    n_epochs, _, n_times = data.shape
    t_grid = np.asarray(epochs.times, dtype=float)

    trajectories: dict[str, np.ndarray] = {}
    for task_name, model in tqdm(models.items(), desc="offline predict", unit="task"):
        positive_idx = _positive_class_index(model, metadata)
        per_epoch_trajectory = np.empty((n_epochs, n_times), dtype=float)
        for t_idx in range(n_times):
            X_t = data[:, :, t_idx]
            per_epoch_trajectory[:, t_idx] = model.predict_proba(X_t)[:, positive_idx]
        trajectories[task_name] = per_epoch_trajectory

    inverse_event_id = {code: name for name, code in epochs.event_id.items()}
    per_epoch_marker_name = [inverse_event_id.get(int(code)) for code in epochs.events[:, 2]]

    epoched: dict[str, dict[str, list[np.ndarray]]] = {
        task_name: {marker_name: [] for marker_name in markers_of_interest}
        for task_name in models
    }
    for epoch_index, marker_name in enumerate(per_epoch_marker_name):
        if marker_name not in markers_of_interest:
            continue
        for task_name in models:
            epoched[task_name][marker_name].append(trajectories[task_name][epoch_index])

    return (
        {
            task_name: {
                marker_name: (
                    np.asarray(rows) if rows else np.empty((0, n_times))
                )
                for marker_name, rows in marker_rows.items()
            }
            for task_name, marker_rows in epoched.items()
        },
        t_grid,
    )


def build_online_trajectories(
    raw_path: Path,
    artifact,
    preprocessing_settings: dict,
    event_mapping: dict[str, int],
    markers_of_interest: list[str],
    epoch_tmin: float,
    epoch_tmax: float,
    chunk_samples: int,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Drive OnlinePreprocessor.process_batch over the raw VHDR (no LSL),
    then LiveInferenceEngine.predict, then epoch around marker times.

    Returns:
        epoched: dict[task_name][marker_name] -> (n_trials, n_times)
        t_grid:  time axis (s) relative to marker, length n_times
    """
    raw = mne.io.read_raw_brainvision(str(raw_path), preload=False, verbose=False)
    input_sfreq = float(raw.info["sfreq"])

    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    if "EMG" in eeg_names:
        eeg_names.remove("EMG")
    raw.pick(eeg_names)
    raw.load_data(verbose=False)
    n_times = raw.n_times

    description_to_code: dict[str, int] = {}
    for description in set(raw.annotations.description):
        match = re.search(r"(\d+)\s*$", description)
        if match:
            description_to_code[description] = int(match.group(1))
    events, _ = mne.events_from_annotations(
        raw, event_id=description_to_code, verbose=False
    )
    codes_of_interest = {event_mapping[name] for name in markers_of_interest}
    markers: list[tuple[int, int]] = [
        (int(sample), int(code))
        for sample, _prev, code in events
        if code in codes_of_interest and sample < n_times
    ]

    eeg_array = raw.get_data().T.astype(np.float64, copy=False)
    raw.close()
    del raw
    gc.collect()

    preproc = OnlinePreprocessor(
        preprocessing_settings, artifact.online_state, input_sfreq=input_sfreq
    )
    engine = LiveInferenceEngine(artifact.models, artifact.metadata)

    sample_indices = np.arange(n_times)
    feature_chunks: list[np.ndarray] = []
    output_index_chunks: list[np.ndarray] = []
    preproc.reset_state()
    n_batches = (n_times + chunk_samples - 1) // chunk_samples
    for start in tqdm(
        range(0, n_times, chunk_samples),
        total=n_batches,
        desc="online streaming",
        unit="batch",
    ):
        end = start + chunk_samples
        features_batch, output_index_batch = preproc.process_batch(
            eeg_array[start:end], sample_indices[start:end]
        )
        if features_batch.shape[0]:
            feature_chunks.append(features_batch)
            output_index_chunks.append(output_index_batch)

    features = np.vstack(feature_chunks)
    output_samples = np.concatenate(output_index_chunks)
    del feature_chunks, output_index_chunks, eeg_array
    gc.collect()

    predictions = engine.predict(features)
    target_sfreq = preproc.target_sfreq

    t_grid = (
        np.arange(round(epoch_tmin * target_sfreq), round(epoch_tmax * target_sfreq) + 1)
        / target_sfreq
    )
    relative_time = output_samples / input_sfreq

    epoched: dict[str, dict[str, np.ndarray]] = {task: {} for task in predictions}
    for task_name, probability_stream in predictions.items():
        for marker_name in markers_of_interest:
            target_code = event_mapping[marker_name]
            target_samples = [sample for sample, code in markers if code == target_code]
            rows: list[np.ndarray] = []
            for marker_sample in target_samples:
                relative = relative_time - marker_sample / input_sfreq
                window = (relative >= epoch_tmin - 0.05) & (relative <= epoch_tmax + 0.05)
                if window.sum() < 2:
                    continue
                rows.append(np.interp(t_grid, relative[window], probability_stream[window]))
            epoched[task_name][marker_name] = (
                np.asarray(rows) if rows else np.empty((0, t_grid.size))
            )

    return epoched, t_grid


# ── per-class overlay renderer ──────────────────────────────────────────────


def render_per_class_overlay(
    class_name: str,
    decoder_task_name: str,
    offline_trials: np.ndarray,
    online_trials: np.ndarray | None,
    t_grid: np.ndarray,
    trained_tp: float | None,
    out_path: Path,
) -> None:
    """One PNG per class: offline (navy) and online (crimson) mean ±SEM,
    both overlaid on the same axes.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    n_off = offline_trials.shape[0]
    if n_off:
        off_mean = offline_trials.mean(axis=0)
        off_sem = offline_trials.std(axis=0) / np.sqrt(n_off)
        ax.plot(t_grid, off_mean, color=OFFLINE_COLOR, lw=2.4, label=f"offline (n={n_off})")
        ax.fill_between(
            t_grid, off_mean - off_sem, off_mean + off_sem,
            color=OFFLINE_COLOR, alpha=0.20, linewidth=0,
        )

    n_on = 0
    if online_trials is not None and online_trials.shape[0]:
        n_on = online_trials.shape[0]
        on_mean = online_trials.mean(axis=0)
        on_sem = online_trials.std(axis=0) / np.sqrt(n_on)
        ax.plot(t_grid, on_mean, color=ONLINE_COLOR, lw=2.4, label=f"online (n={n_on})")
        ax.fill_between(
            t_grid, on_mean - on_sem, on_mean + on_sem,
            color=ONLINE_COLOR, alpha=0.20, linewidth=0,
        )

    ax.axvline(0, color="black", ls=":", lw=1)
    if trained_tp is not None:
        ax.axvline(
            trained_tp, color="black", ls="--", lw=1.2,
            label=f"trained tp ({trained_tp:.2f}s)",
        )
    ax.axhline(0.5, color="gray", lw=0.6)

    if online_trials is None:
        trial_label = f"n={n_off} trials"
    elif n_on == n_off:
        trial_label = f"n={n_off} trials"
    else:
        trial_label = f"n={n_off} offline / {n_on} online"
    tp_label = (
        f"trained tp {trained_tp:.2f}s" if trained_tp is not None else "no trained tp"
    )
    ax.set_title(
        f"{decoder_task_name} on '{class_name}' trials  ({trial_label} | {tp_label})",
        fontsize=12,
    )
    ax.set(xlabel="time from marker (s)", ylabel="P(positive)", ylim=(0.0, 1.0))
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── CV evaluation plots ─────────────────────────────────────────────────────


def render_cv_auc_curves(
    eval_results: dict,
    task_to_tp: dict[str, float],
    out_path: Path,
    peak_window_ms: float = 50.0,
    above_margin: float = 0.05,
) -> None:
    times = np.asarray(eval_results["times"], dtype=float)
    task_results = eval_results["tasks"]
    decoder_names = list(task_results.keys())
    n_cols = 3
    n_rows = (len(decoder_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    sfreq = 1.0 / float(np.median(np.diff(times)))
    peak_window_samples = int(round(peak_window_ms * 1e-3 * sfreq))
    for plot_index, decoder_name in enumerate(decoder_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        diagonal = np.asarray(task_results[decoder_name]["diagonal_auc"], dtype=float)
        chance = float(task_results[decoder_name]["chance_level"])
        peak_index = int(np.argmax(diagonal))
        trained_tp = task_to_tp.get(decoder_name)

        ax.plot(times, diagonal, color="navy", lw=1.8)
        ax.axhline(chance, color="gray", lw=0.8, ls=":")
        ax.axhline(chance + above_margin, color="lightgray", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.6, ls=":")
        if trained_tp is not None:
            ax.axvline(
                trained_tp, color="black", lw=0.8, ls="--",
                label=f"trained tp ({trained_tp:.2f}s)",
            )
        ax.axvline(
            times[peak_index], color="crimson", lw=0.8, ls="--",
            label=f"peak ({times[peak_index]:.2f}s, {diagonal[peak_index]:.2f})",
        )
        start = max(0, peak_index - peak_window_samples)
        end = min(diagonal.size, peak_index + peak_window_samples + 1)
        ax.axvspan(times[start], times[end - 1], color="crimson", alpha=0.10)
        ax.set(
            title=decoder_name, xlabel="time (s)", ylabel="CV AUC",
            ylim=(
                min(0.40, float(diagonal.min()) - 0.02),
                max(0.85, float(diagonal.max()) + 0.02),
            ),
        )
        ax.legend(fontsize=7, loc="upper right")
    for unused_index in range(len(decoder_names), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_cv_tgm_heatmaps(
    eval_results: dict,
    task_to_tp: dict[str, float],
    out_path: Path,
) -> None:
    times = np.asarray(eval_results["times"], dtype=float)
    t0, t1 = float(times[0]), float(times[-1])
    task_results = eval_results["tasks"]
    decoder_names = list(task_results.keys())
    n_cols = 3
    n_rows = (len(decoder_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4.6 * n_rows), squeeze=False
    )
    last_im = None
    for plot_index, decoder_name in enumerate(decoder_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        tgm = np.asarray(task_results[decoder_name]["tgm_matrix"], dtype=float)
        trained_tp = task_to_tp.get(decoder_name)
        im = ax.imshow(
            tgm, origin="lower", extent=[t0, t1, t0, t1], aspect="auto",
            cmap="RdBu_r", vmin=0.3, vmax=0.7,
        )
        last_im = im
        ax.plot([t0, t1], [t0, t1], color="black", lw=0.6, alpha=0.5)
        if trained_tp is not None:
            ax.axvline(trained_tp, color="black", lw=0.7, ls="--", alpha=0.7)
            ax.axhline(trained_tp, color="black", lw=0.7, ls="--", alpha=0.7)
        ax.set(
            title=(
                f"{decoder_name}"
                + (f"  (trained tp {trained_tp:.2f}s)" if trained_tp else "")
            ),
            xlabel="test time (s)", ylabel="train time (s)",
        )
    for unused_index in range(len(decoder_names), n_rows * n_cols):
        axes[unused_index // n_cols][unused_index % n_cols].axis("off")
    fig.tight_layout(rect=(0, 0, 0.94, 1.0))
    if last_im is not None:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.012, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label="AUC (chance=0.5)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── CLI + main ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=None,
                        help="Path to the raw VHDR (defaults to first VHDR under data/split/functional_localizer/).")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--eval", dest="eval_path", type=Path, default=EVAL_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--chunk-samples", type=int, default=BATCH_SIZE_SAMPLES,
                        help="Online-path chunk size in input samples (~50ms at 1000Hz with default).")
    parser.add_argument("--tmin", type=float, default=None,
                        help="Epoch window start in seconds (default from config).")
    parser.add_argument("--tmax", type=float, default=None,
                        help="Epoch window end in seconds (default from config).")
    parser.add_argument("--skip-online", action="store_true",
                        help="Skip the online preprocessing path (offline curve only on the per-class plots).")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip the CV evaluation figures.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    settings = SettingsManager(args.config)
    preprocessing_settings = settings.get_preprocessing_params()
    decoder_settings = settings.get_decoder_settings()
    event_mapping = settings.get_event_mapping()
    artifact = load_decoder_pipeline_artifact(args.artifact)

    task_to_marker: dict[str, str] = {
        task["name"]: task["pos_labels"][0] for task in decoder_settings["tasks"]
    }
    markers_of_interest: list[str] = list(
        dict.fromkeys(label for task in decoder_settings["tasks"] for label in task["pos_labels"])
    )
    marker_to_task = {marker: task for task, marker in task_to_marker.items()}

    representative_tp = artifact.metadata.get("decoding_timepoint")
    per_task_tp_raw = artifact.metadata.get("decoding_timepoints") or {}
    task_to_tp: dict[str, float] = {
        task_name: float(per_task_tp_raw.get(task_name, representative_tp))
        for task_name in artifact.models
    }

    epoch_tmin = args.tmin if args.tmin is not None else preprocessing_settings["epochs"]["tmin"]
    epoch_tmax = args.tmax if args.tmax is not None else preprocessing_settings["epochs"]["tmax"]

    print(f"tasks               : {list(artifact.models.keys())}")
    print(f"markers of interest : {markers_of_interest}")
    print(f"per-task timepoints : {task_to_tp}")
    print(f"epoch window        : [{epoch_tmin:.2f}, {epoch_tmax:.2f}] s")
    print(f"output dir          : {out_dir}")
    print()

    print("Loading offline-preprocessed epochs from train_done snapshot…")
    snapshot = joblib.load(args.snapshot)
    offline_epochs = snapshot["_epochs"]
    print(f"  epochs: {len(offline_epochs)} | sfreq: {offline_epochs.info['sfreq']:g} Hz "
          f"| n_channels: {len(offline_epochs.ch_names)} | tmin/tmax: "
          f"{offline_epochs.tmin:.2f}/{offline_epochs.tmax:.2f}")
    inv_event_id = {code: name for name, code in offline_epochs.event_id.items()}
    counts_by_marker = Counter(
        inv_event_id.get(int(code)) for code in offline_epochs.events[:, 2]
    )
    print(f"  per-marker trial counts: "
          f"{ {m: counts_by_marker.get(m, 0) for m in markers_of_interest} }")
    print()

    offline_epoched, offline_t_grid = build_offline_trajectories(
        offline_epochs, artifact.models, artifact.metadata, markers_of_interest
    )
    del snapshot
    gc.collect()

    online_epoched: dict[str, dict[str, np.ndarray]] | None = None
    t_grid_for_render = offline_t_grid
    if not args.skip_online:
        raw_path = args.raw if args.raw is not None else find_vhdr(RECORDING_DIR)
        print(f"Streaming online path from {raw_path.name}…")
        online_epoched, online_t_grid = build_online_trajectories(
            raw_path, artifact, preprocessing_settings, event_mapping,
            markers_of_interest, epoch_tmin, epoch_tmax, args.chunk_samples,
        )
        if not np.allclose(online_t_grid, offline_t_grid):
            print(f"WARNING: t_grid mismatch — offline n={offline_t_grid.size} vs "
                  f"online n={online_t_grid.size}. Re-interpolating offline onto online grid.")
            offline_epoched = {
                task_name: {
                    marker_name: (
                        np.array(
                            [np.interp(online_t_grid, offline_t_grid, row) for row in trials]
                        )
                        if trials.shape[0]
                        else np.empty((0, online_t_grid.size))
                    )
                    for marker_name, trials in marker_trials.items()
                }
                for task_name, marker_trials in offline_epoched.items()
            }
            t_grid_for_render = online_t_grid

    print("Rendering per-class overlay figures…")
    for class_name in markers_of_interest:
        decoder_task_name = marker_to_task.get(class_name)
        if decoder_task_name is None or decoder_task_name not in artifact.models:
            print(f"  skipping '{class_name}': no decoder maps to this marker")
            continue
        offline_trials = offline_epoched[decoder_task_name].get(
            class_name, np.empty((0, t_grid_for_render.size))
        )
        if online_epoched is not None:
            online_trials = online_epoched[decoder_task_name].get(
                class_name, np.empty((0, t_grid_for_render.size))
            )
        else:
            online_trials = None
        render_per_class_overlay(
            class_name=class_name,
            decoder_task_name=decoder_task_name,
            offline_trials=offline_trials,
            online_trials=online_trials,
            t_grid=t_grid_for_render,
            trained_tp=task_to_tp.get(decoder_task_name),
            out_path=out_dir / f"comparison_{class_name}.png",
        )

    if not args.skip_cv:
        print("Rendering CV evaluation figures…")
        eval_results = joblib.load(args.eval_path)["_eval_results"]
        render_cv_auc_curves(eval_results, task_to_tp, out_dir / "cv_auc_curves.png")
        render_cv_tgm_heatmaps(eval_results, task_to_tp, out_dir / "cv_tgm_heatmaps.png")

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s. Figures in {out_dir}")
    for png in sorted(out_dir.glob("*.png")):
        print(f"  {png.name}  ({png.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
