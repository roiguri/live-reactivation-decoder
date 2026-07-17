"""Aggregate the per-recording latency CSVs into report-ready statistics.

Reads every ``*_latency.csv`` written by ``scripts/collect_latency.py``, groups
by subject (the CSV's parent directory name), and reports:

  - Per subject and pooled: n_batches, minutes captured, mean / median / p95 /
    max for the E2E metric (``sample_to_decision_ms``, the headline) and for the
    compute-only ``total_ms``.
  - A stage breakdown of the mean cost (preprocessing / inference / pull /
    accumulation / emit), plus the buffering floor implied by the batch size.

Writes ``<in-root>/latency_summary.csv`` and prints tables to stdout.

Example::

    python scripts/summarize_latency.py --in-root docs/project_docs/latency
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fields summarised as distributions (headline metrics).
DIST_FIELDS = ["sample_to_decision_ms", "total_ms"]
# Fields summarised by mean only (stage breakdown of total_ms).
STAGE_FIELDS = ["pull_ms", "accumulation_ms", "preprocessing_ms", "inference_ms", "emit_ms"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--in-root",
        type=Path,
        default=PROJECT_ROOT / "docs" / "project_docs" / "latency",
        help="Directory tree holding <subject>/<recording>_latency.csv files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Batch size used during capture (for the buffering-floor note).",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=1000.0,
        help="Raw sample rate (for the buffering-floor note).",
    )
    parser.add_argument(
        "--skip-seconds",
        type=float,
        default=0.0,
        help="Drop the first N seconds of each recording before aggregating, "
        "to exclude the connection warm-up spike on the opening batches.",
    )
    return parser


def load_csv(path: Path, skip_rows: int = 0) -> dict[str, np.ndarray]:
    """Load a latency CSV into per-column float arrays (blanks -> NaN).

    ``skip_rows`` drops that many leading rows (the connection warm-up).
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for name, value in row.items():
                cols[name].append(float(value) if value not in ("", None) else np.nan)
    return {
        name: np.asarray(values[skip_rows:], dtype=float)
        for name, values in cols.items()
    }


def dist_stats(values: np.ndarray) -> dict[str, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def print_group(label: str, cols: dict[str, np.ndarray], minutes: float) -> None:
    print(f"\n{label}  ({minutes:.1f} min captured)")
    for field in DIST_FIELDS:
        if field not in cols:
            continue
        s = dist_stats(cols[field])
        print(
            f"  {field:<22} n={s['n']:>7}  mean {s['mean']:6.1f}  "
            f"median {s['median']:6.1f}  p95 {s['p95']:6.1f}  max {s['max']:6.1f}  ms"
        )
    breakdown = [
        f"{field.replace('_ms', '')} {np.nanmean(cols[field]):.2f}"
        for field in STAGE_FIELDS
        if field in cols and np.any(~np.isnan(cols[field]))
    ]
    if breakdown:
        print("  mean stage cost (ms): " + ", ".join(breakdown))


def concat(group: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    fields = set().union(*(g.keys() for g in group)) if group else set()
    return {
        field: np.concatenate([g[field] for g in group if field in g])
        for field in fields
    }


def minutes_of(cols: dict[str, np.ndarray], batch_seconds: float) -> float:
    """Data-time captured = n_batches x batch duration.

    Derived from the batch count rather than the wall_time span: each batch
    covers a fixed slice of recording, so this equals the replayed recording
    length and is immune to wall-clock gaps (system sleep, GC pauses).
    """
    ref = cols.get("total_ms")
    n = 0 if ref is None else int(ref.size)
    return n * batch_seconds / 60.0


def main() -> int:
    args = build_arg_parser().parse_args()
    csv_paths = sorted(args.in_root.rglob("*_latency.csv"))
    if not csv_paths:
        print(f"No *_latency.csv files under {args.in_root}")
        return 1

    batch_seconds = args.batch_size / args.sfreq
    skip_rows = round(args.skip_seconds / batch_seconds) if args.skip_seconds else 0
    by_subject: dict[str, list[dict[str, np.ndarray]]] = {}
    for path in csv_paths:
        subject = path.parent.name
        by_subject.setdefault(subject, []).append(load_csv(path, skip_rows))

    floor_ms = args.batch_size / args.sfreq * 1000.0
    if skip_rows:
        print(f"(skipping first {args.skip_seconds:g}s = {skip_rows} rows per recording)")
    print("Live-pipeline latency summary")
    print(f"(buffering floor at batch={args.batch_size} @ {args.sfreq:g} Hz "
          f"= {floor_ms:.0f} ms; replay measures the software path only, "
          f"excluding amplifier acquisition)")

    summary_rows: list[dict[str, object]] = []
    all_groups: list[dict[str, np.ndarray]] = []
    for subject in sorted(by_subject):
        cols = concat(by_subject[subject])
        all_groups.extend(by_subject[subject])
        print_group(subject, cols, minutes_of(cols, batch_seconds))
        for field in DIST_FIELDS:
            if field in cols:
                summary_rows.append({"group": subject, "metric": field, **dist_stats(cols[field])})

    pooled = concat(all_groups)
    print_group("POOLED (all subjects)", pooled, minutes_of(pooled, batch_seconds))
    for field in DIST_FIELDS:
        if field in pooled:
            summary_rows.append({"group": "POOLED", "metric": field, **dist_stats(pooled[field])})

    out_csv = args.in_root / "latency_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "metric", "n", "mean", "median", "p95", "max"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
