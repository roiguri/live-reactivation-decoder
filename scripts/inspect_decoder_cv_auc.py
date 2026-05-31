"""Inspect per-decoder cross-validated AUC curves from eval_done.joblib.

For each decoder, report not just the single-sample peak, but the SURROUNDING
area: mean AUC in ±50 ms and ±100 ms windows around the peak, plus the width
of the contiguous above-threshold region. This separates "stable signal
plateau" from "single-sample peak that's probably noise."

Also saves a 2×3 PNG with per-decoder AUC curves, the peak marked, the trained
timepoint marked, and the chance line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_PATH = PROJECT_ROOT / "debug_snapshots" / "eval_done.joblib"
OUT_DIR = PROJECT_ROOT / "debug_snapshots"


def _png_path(suffix: str) -> Path:
    return OUT_DIR / f"decoder_cv_auc_curves{suffix}.png"

PEAK_WINDOW_MS = 50.0  # half-window around the peak for "near-peak mean"
WIDER_WINDOW_MS = 100.0
ABOVE_MARGIN = 0.05  # "above-threshold" = chance + this margin


def summarise_decoder(
    times: np.ndarray,
    diagonal_auc: np.ndarray,
    chance: float,
    trained_timepoint: float,
) -> dict[str, float]:
    sfreq = 1.0 / float(np.median(np.diff(times)))
    peak_window_samples = int(round(PEAK_WINDOW_MS * 1e-3 * sfreq))
    wider_window_samples = int(round(WIDER_WINDOW_MS * 1e-3 * sfreq))

    peak_index = int(np.argmax(diagonal_auc))
    peak_auc = float(diagonal_auc[peak_index])
    peak_time = float(times[peak_index])

    def _window_mean(half_width_samples: int) -> float:
        start = max(0, peak_index - half_width_samples)
        end = min(diagonal_auc.size, peak_index + half_width_samples + 1)
        return float(np.mean(diagonal_auc[start:end]))

    near_peak_mean = _window_mean(peak_window_samples)
    wider_peak_mean = _window_mean(wider_window_samples)

    trained_index = int(np.argmin(np.abs(times - trained_timepoint)))
    auc_at_trained = float(diagonal_auc[trained_index])

    above_mask = diagonal_auc >= (chance + ABOVE_MARGIN)
    if not above_mask.any():
        above_width_ms = 0.0
        n_above_at_peak = 0
    else:
        above_width_ms = float(above_mask.sum()) / sfreq * 1000.0
        # Width of the contiguous above-threshold region containing the peak.
        n_above_at_peak = 0
        if above_mask[peak_index]:
            left = peak_index
            while left > 0 and above_mask[left - 1]:
                left -= 1
            right = peak_index
            while right < above_mask.size - 1 and above_mask[right + 1]:
                right += 1
            n_above_at_peak = right - left + 1
    contiguous_peak_width_ms = float(n_above_at_peak) / sfreq * 1000.0

    return {
        "peak_auc": peak_auc,
        "peak_time_s": peak_time,
        "near_peak_mean": near_peak_mean,
        "wider_peak_mean": wider_peak_mean,
        "auc_at_trained": auc_at_trained,
        "above_width_total_ms": above_width_ms,
        "contiguous_peak_width_ms": contiguous_peak_width_ms,
    }


def print_table(
    summaries: dict[str, dict[str, float]],
    trained_timepoint: float,
) -> None:
    columns = [
        ("peak_auc",                "peak"),
        ("peak_time_s",             "peak_t(s)"),
        ("near_peak_mean",          f"mean±{int(PEAK_WINDOW_MS)}ms"),
        ("wider_peak_mean",         f"mean±{int(WIDER_WINDOW_MS)}ms"),
        ("auc_at_trained",          f"AUC@{trained_timepoint:.2f}s"),
        ("contiguous_peak_width_ms", "peak-band(ms)"),
        ("above_width_total_ms",    "tot>chance(ms)"),
    ]
    header = "decoder".ljust(22) + "".join(c[1].rjust(15) for c in columns)
    print(header)
    print("-" * len(header))
    for decoder_name, summary in summaries.items():
        row = decoder_name.ljust(22)
        for key, _label in columns:
            value = summary[key]
            if "time" in key or "width" in key or "trained" in key.lower():
                formatted = (
                    f"{value:.3f}" if "time" in key
                    else f"{value:.0f}" if "width" in key
                    else f"{value:.3f}"
                )
            else:
                formatted = f"{value:.3f}"
            row += formatted.rjust(15)
        print(row)


def plot_curves(
    times: np.ndarray,
    task_results: dict[str, dict],
    trained_timepoint: float,
    out_path: Path,
) -> None:
    decoder_names = list(task_results.keys())
    n_cols = 3
    n_rows = (len(decoder_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)
    for plot_index, decoder_name in enumerate(decoder_names):
        ax = axes[plot_index // n_cols][plot_index % n_cols]
        diagonal = task_results[decoder_name]["diagonal_auc"]
        chance = task_results[decoder_name]["chance_level"]
        peak_index = int(np.argmax(diagonal))

        ax.plot(times, diagonal, color="navy", lw=1.8)
        ax.axhline(chance, color="gray", lw=0.8, ls=":")
        ax.axhline(chance + ABOVE_MARGIN, color="lightgray", lw=0.6, ls=":")
        ax.axvline(0, color="black", lw=0.6, ls=":")
        ax.axvline(trained_timepoint, color="black", lw=0.8, ls="--",
                   label=f"trained tp ({trained_timepoint:.2f}s)")
        ax.axvline(times[peak_index], color="crimson", lw=0.8, ls="--",
                   label=f"peak ({times[peak_index]:.2f}s, {diagonal[peak_index]:.2f})")

        # Shade ±50 ms band around the peak.
        sfreq = 1.0 / float(np.median(np.diff(times)))
        peak_window_samples = int(round(PEAK_WINDOW_MS * 1e-3 * sfreq))
        start = max(0, peak_index - peak_window_samples)
        end = min(diagonal.size, peak_index + peak_window_samples + 1)
        ax.axvspan(times[start], times[end - 1], color="crimson", alpha=0.10)

        ax.set(title=decoder_name, xlabel="time (s)", ylabel="CV AUC",
               ylim=(min(0.40, float(diagonal.min()) - 0.02),
                     max(0.85, float(diagonal.max()) + 0.02)))
        ax.legend(fontsize=7, loc="upper right")

    for unused_index in range(len(decoder_names), n_rows * n_cols):
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
        help="Suffix appended to the PNG filename (e.g. '_stepC') to preserve "
             "previous-run outputs side by side.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_png = _png_path(args.suffix)
    if not EVAL_PATH.exists():
        print(f"eval_done.joblib not found: {EVAL_PATH}", file=sys.stderr)
        return 2

    state = joblib.load(EVAL_PATH)
    eval_results = state["_eval_results"]
    times = np.asarray(eval_results["times"], dtype=float)
    trained_timepoint = float(eval_results["suggested_timepoint"])
    average_peak_auc = float(eval_results["average_peak_auc"])
    task_results = eval_results["tasks"]

    print(f"trained timepoint        : {trained_timepoint:.3f} s")
    print(f"avg peak AUC across tasks: {average_peak_auc:.3f}")
    print(f"chance per decoder       : "
          f"{ {n: round(float(t['chance_level']), 3) for n, t in task_results.items()} }")
    print()

    summaries: dict[str, dict[str, float]] = {}
    for decoder_name, task_dict in task_results.items():
        summaries[decoder_name] = summarise_decoder(
            times=times,
            diagonal_auc=np.asarray(task_dict["diagonal_auc"], dtype=float),
            chance=float(task_dict["chance_level"]),
            trained_timepoint=trained_timepoint,
        )

    print_table(summaries, trained_timepoint)

    plot_curves(times, task_results, trained_timepoint, out_png)
    print(f"\nFigure written: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
