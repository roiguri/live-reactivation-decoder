"""Smoke test for :class:`LiveProbabilityChart`.

Feeds the chart synthetic predictions at the same cadence the real
:class:`StreamWorker` does (~25 Hz emit rate, ~4 prediction rows per
emit at the default 40-sample batch / 1000 Hz source / 100 Hz target).
After ``--duration`` seconds the QApplication quits cleanly.

Verifies (without any LSL or backend involvement):

* The chart can sustain the production data rate without stutter.
* The 30 Hz refresh timer fires and updates the curves.
* Window closes cleanly with no exceptions on exit.

Usage::

    cd online_decoder
    PYTHONPATH=src python scripts/test_live_chart.py
    PYTHONPATH=src python scripts/test_live_chart.py --duration 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend.widgets.live_probability_chart import LiveProbabilityChart


_TASK_NAMES = ["decoder_A", "decoder_B", "decoder_C"]

# Real pipeline shape — see Phase2_Implementation_Plan.md / stream_worker_design.md.
# StreamWorker accumulates 40 raw samples at 1000 Hz, then OnlinePreprocessor
# decimates to 100 Hz, so each `prediction_ready` carries ~4 rows. Worker emits
# at 1000/40 = 25 Hz.
_SAMPLES_PER_EMIT = 4
_EMIT_HZ = 25
_TARGET_SFREQ = 100.0
_DT = 1.0 / _TARGET_SFREQ  # seconds between consecutive prediction rows


class SyntheticFeed:
    """Generator that produces one batch of fake predictions per call.

    The signal is deliberately interesting so the visual smoke test
    exercises the chart's full y-range:

    * ``decoder_A`` — slow sine around 0.5; never crosses threshold.
    * ``decoder_B`` — faster sine that periodically pushes above 0.85.
    * ``decoder_C`` — bounded random walk; tests irregular updates.
    """

    def __init__(self, t0: float, rng: np.random.Generator) -> None:
        self._t0 = t0
        self._rng = rng
        self._sample_idx = 0
        self._walk = 0.5

    def next_batch(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        n = _SAMPLES_PER_EMIT
        start = self._sample_idx
        idx = np.arange(start, start + n, dtype=np.float64)
        self._sample_idx += n
        t_rel = idx * _DT
        timestamps = self._t0 + t_rel

        a = 0.5 + 0.18 * np.sin(2 * np.pi * 0.15 * t_rel + start * _DT * 0.15 * 2 * np.pi)
        b = 0.5 + 0.42 * np.sin(2 * np.pi * 0.7 * t_rel + start * _DT * 0.7 * 2 * np.pi)
        steps = self._rng.normal(0.0, 0.04, size=n)
        c = np.empty(n, dtype=np.float64)
        v = self._walk
        for i in range(n):
            v = float(np.clip(v + steps[i] - 0.05 * (v - 0.5), 0.0, 1.0))
            c[i] = v
        self._walk = c[-1]

        return {
            "decoder_A": a,
            "decoder_B": b,
            "decoder_C": c,
        }, timestamps


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="How many seconds of synthetic data to feed before quitting.",
    )
    p.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Chart rolling window in seconds.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="y-position of the dashed threshold guide.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for repeatability.")
    return p.parse_args(argv)


def _build_legend(chart: LiveProbabilityChart) -> QWidget:
    """Build a horizontal swatch+name+checkbox row from the chart's
    ``task_colors``. Each checkbox drives ``chart.set_task_visible``.

    Mirrors the pattern Phase2Screen will follow in Commit 5 — the chart
    deliberately omits an in-plot legend so the screen owns layout and
    can wire visibility (M1) and per-decoder colour pickers (M2).
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(16)
    for name, hex_color in chart.task_colors.items():
        item = QWidget()
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(6)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        # Default-arg trick captures ``name`` by value — without it every
        # checkbox would close over the last loop iteration's name.
        checkbox.toggled.connect(
            lambda checked, n=name: chart.set_task_visible(n, checked)
        )
        item_layout.addWidget(checkbox)

        swatch = QFrame()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(f"background: {hex_color}; border-radius: 2px;")
        item_layout.addWidget(swatch)

        label = QLabel(name)
        item_layout.addWidget(label)
        layout.addWidget(item)
    layout.addStretch(1)
    return container


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    app = QApplication(sys.argv)
    chart = LiveProbabilityChart(
        task_names=_TASK_NAMES,
        window_seconds=args.window,
        target_sfreq=_TARGET_SFREQ,
        threshold=args.threshold,
    )

    # External legend — chart exposes the canonical name → colour mapping
    # and the script (acting as the "screen" in this smoke test) decides
    # where to place a matching widget.
    container = QWidget()
    container.setWindowTitle("LiveProbabilityChart — smoke test")
    container.resize(960, 520)
    root = QVBoxLayout(container)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    root.addWidget(_build_legend(chart))
    root.addWidget(chart, 1)
    container.show()

    t0 = time.time()
    feed = SyntheticFeed(t0=t0, rng=np.random.default_rng(args.seed))

    feed_timer = QTimer()
    feed_timer.setInterval(int(round(1000 / _EMIT_HZ)))

    def _tick() -> None:
        if time.time() - t0 >= args.duration:
            app.quit()
            return
        predictions, timestamps = feed.next_batch()
        chart.append_predictions(predictions, timestamps)

    feed_timer.timeout.connect(_tick)
    feed_timer.start()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
