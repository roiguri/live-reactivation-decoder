"""Benchmark OnlinePreprocessor.process_batch() latency.

Measures per-batch wall-clock time over many iterations and reports
mean / p50 / p95 / p99 / max, plus how much of the 40-ms real-time
budget each percentile consumes.

No real recording is required — synthetic state and random batches are used
so the benchmark can run anywhere without hardware.

Usage
-----
    python scripts/benchmark_preprocessor.py
    python scripts/benchmark_preprocessor.py --n-channels 64 --n-components 40
    python scripts/benchmark_preprocessor.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.online_phase.online_preprocessor import OnlinePreprocessor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_state(
    n_channels: int,
    n_components: int,
    target_sfreq: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    pca_components = rng.standard_normal((n_components, n_channels))
    unmixing = rng.standard_normal((n_components, n_components))
    mixing = np.linalg.pinv(unmixing)
    return {
        "bad_channels": [],
        "interp_weights": None,
        "ch_names": [f"CH{i:03d}" for i in range(n_channels)],
        "ica_unmixing": unmixing,
        "ica_mixing": mixing,
        "ica_pca_components": pca_components,
        "ica_pca_mean": np.zeros(n_channels),
        "ica_exclude": [],
        "pre_whitener": np.ones((n_channels, 1)),
        "sfreq_offline": float(target_sfreq),
    }


def _make_settings(target_sfreq: int) -> dict[str, Any]:
    return {
        "bandpass": {"l_freq": 1.0, "h_freq": 40.0, "method": "iir", "notch": None},
        "resample": {"target_rate": target_sfreq},
    }


@dataclass(frozen=True)
class BenchmarkResult:
    n_batches: int
    warmup_batches: int
    batch_size: int
    n_channels: int
    n_components: int
    input_sfreq: float
    target_sfreq: int
    budget_ms: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @property
    def mean_pct(self) -> float:
        return 100.0 * self.mean_ms / self.budget_ms

    @property
    def p99_pct(self) -> float:
        return 100.0 * self.p99_ms / self.budget_ms


def run_benchmark(
    n_batches: int,
    warmup_batches: int,
    batch_size: int,
    n_channels: int,
    n_components: int,
    input_sfreq: float,
    target_sfreq: int,
) -> BenchmarkResult:
    state = _make_synthetic_state(n_channels, n_components, target_sfreq)
    settings = _make_settings(target_sfreq)
    preprocessor = OnlinePreprocessor(settings, state, input_sfreq=input_sfreq)

    rng = np.random.default_rng(42)
    times_ms: list[float] = []

    total = warmup_batches + n_batches
    for i in range(total):
        batch = rng.standard_normal((batch_size, n_channels)) * 1e-5
        ts = np.arange(batch_size, dtype=float) / input_sfreq + i * batch_size / input_sfreq

        t0 = time.perf_counter()
        preprocessor.process_batch(batch, ts)
        t1 = time.perf_counter()

        if i >= warmup_batches:
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    budget_ms = batch_size / input_sfreq * 1000.0

    return BenchmarkResult(
        n_batches=n_batches,
        warmup_batches=warmup_batches,
        batch_size=batch_size,
        n_channels=n_channels,
        n_components=n_components,
        input_sfreq=input_sfreq,
        target_sfreq=target_sfreq,
        budget_ms=budget_ms,
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(arr.max()),
    )


def format_result(r: BenchmarkResult) -> str:
    lines = [
        "=== OnlinePreprocessor Benchmark ===",
        f"Config: {r.batch_size} samples/batch, {r.n_channels} ch, "
        f"{r.n_components} ICA components",
        f"        {r.input_sfreq:.0f} Hz → {r.target_sfreq} Hz  |  "
        f"{r.n_batches} batches ({r.warmup_batches} warmup)",
        f"Budget per batch: {r.budget_ms:.1f} ms",
        "",
        "Latency (ms)",
        f"  mean  {r.mean_ms:6.3f}   std  {r.std_ms:.3f}",
        f"  p50   {r.p50_ms:6.3f}",
        f"  p95   {r.p95_ms:6.3f}",
        f"  p99   {r.p99_ms:6.3f}",
        f"  max   {r.max_ms:6.3f}",
        "",
        "Budget usage",
        f"  mean  {r.mean_pct:5.1f} %",
        f"  p99   {r.p99_pct:5.1f} %",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OnlinePreprocessor.process_batch() latency.",
    )
    parser.add_argument("--n-batches", type=int, default=1000,
                        help="Number of timed batches (default: 1000).")
    parser.add_argument("--warmup-batches", type=int, default=20,
                        help="Discarded warmup batches (default: 20).")
    parser.add_argument("--batch-size", type=int, default=40,
                        help="Samples per batch (default: 40).")
    parser.add_argument("--n-channels", type=int, default=20,
                        help="EEG channel count (default: 20).")
    parser.add_argument("--n-components", type=int, default=4,
                        help="ICA component count (default: 4).")
    parser.add_argument("--input-sfreq", type=float, default=1000.0,
                        help="Input sampling rate in Hz (default: 1000.0).")
    parser.add_argument("--target-sfreq", type=int, default=256,
                        help="Target sampling rate in Hz (default: 256).")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    result = run_benchmark(
        n_batches=args.n_batches,
        warmup_batches=args.warmup_batches,
        batch_size=args.batch_size,
        n_channels=args.n_channels,
        n_components=args.n_components,
        input_sfreq=args.input_sfreq,
        target_sfreq=args.target_sfreq,
    )

    if args.json:
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    else:
        print(format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
