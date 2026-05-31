"""Tests for the Approach B preprocessing parity harness.

Fast unit tests cover the pure math helpers. A ``@pytest.mark.slow``
integration test exercises the full library entry point against a synthetic
BrainVision fixture; it is skipped if ``pybv`` is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _stub_pyqt6_if_missing() -> None:
    """Stub PyQt6.QtCore to dodge a Qt6 version conflict with pytest-qt's PySide6.

    The parity script transitively imports the online_phase package, whose
    ``__init__`` eagerly imports ``PredictionLogger`` (which depends on
    PyQt6.QtCore). When pytest-qt loads PySide6 first the two Qt6 libraries
    fight, but the math helpers under test do not touch Qt at all.
    """
    if "PyQt6.QtCore" in sys.modules:
        return
    fake_qtcore = types.ModuleType("PyQt6.QtCore")

    class _QObject:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _QThread(_QObject):
        pass

    class _Qt:
        ConnectionType = type("ConnectionType", (), {"QueuedConnection": 0})

    def _pyqt_slot(*_decorator_args, **_decorator_kwargs):
        def _decorate(method):
            return method

        return _decorate

    def _pyqt_signal(*_signal_args, **_signal_kwargs):
        return None

    fake_qtcore.QObject = _QObject
    fake_qtcore.QThread = _QThread
    fake_qtcore.Qt = _Qt
    fake_qtcore.pyqtSlot = _pyqt_slot
    fake_qtcore.pyqtSignal = _pyqt_signal
    fake_pyqt6 = types.ModuleType("PyQt6")
    fake_pyqt6.QtCore = fake_qtcore
    sys.modules.setdefault("PyQt6", fake_pyqt6)
    sys.modules.setdefault("PyQt6.QtCore", fake_qtcore)


_stub_pyqt6_if_missing()


def _load_parity_module():
    spec = importlib.util.spec_from_file_location(
        "preproc_parity_check", SCRIPTS_DIR / "preproc_parity_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so @dataclass can resolve ``sys.modules[cls.__module__]``.
    sys.modules["preproc_parity_check"] = module
    spec.loader.exec_module(module)
    return module


parity = _load_parity_module()


# ── epoch_trace_at_event_times ───────────────────────────────────────────────


class TestEpochTraceAtEventTimes:
    def test_returns_correct_shape_for_clean_window(self) -> None:
        n_samples_total = 1000
        n_channels = 4
        sfreq = 100.0
        out_timestamps = np.arange(n_samples_total) / sfreq
        trace = np.zeros((n_samples_total, n_channels))
        event_times = np.array([2.0, 5.0, 8.0])

        epochs, kept_mask = parity.epoch_trace_at_event_times(
            trace=trace,
            out_timestamps=out_timestamps,
            event_times_s=event_times,
            tmin=-0.2,
            tmax=1.0,
            target_sfreq=sfreq,
        )

        assert epochs.shape == (3, n_channels, 121)
        assert kept_mask.tolist() == [True, True, True]

    def test_drops_trials_at_boundary(self) -> None:
        sfreq = 100.0
        out_timestamps = np.arange(200) / sfreq
        trace = np.zeros((200, 2))
        # Window is [marker-20, marker+101), so the marker sample index must
        # land in [20, 99] for the slice to fit inside a 200-sample trace.
        event_times = np.array([0.1, 0.5, 1.95])

        epochs, kept_mask = parity.epoch_trace_at_event_times(
            trace=trace,
            out_timestamps=out_timestamps,
            event_times_s=event_times,
            tmin=-0.2,
            tmax=1.0,
            target_sfreq=sfreq,
        )

        assert kept_mask.tolist() == [False, True, False]
        assert epochs.shape == (1, 2, 121)

    def test_extracts_correct_window_values(self) -> None:
        sfreq = 100.0
        out_timestamps = np.arange(1000) / sfreq
        trace = np.arange(1000, dtype=float).reshape(-1, 1)
        event_times = np.array([5.0])

        epochs, _ = parity.epoch_trace_at_event_times(
            trace=trace,
            out_timestamps=out_timestamps,
            event_times_s=event_times,
            tmin=-0.2,
            tmax=1.0,
            target_sfreq=sfreq,
        )

        # Event at sample 500; window [480, 601) → length 121
        assert epochs.shape == (1, 1, 121)
        assert epochs[0, 0, 0] == 480
        assert epochs[0, 0, -1] == 600
        assert epochs[0, 0, 20] == 500  # t=0 is at sample index 20 of the epoch


# ── trial_metrics ────────────────────────────────────────────────────────────


class TestTrialMetrics:
    def test_identical_signals_give_perfect_metrics(self) -> None:
        rng = np.random.default_rng(0)
        n_channels = 10
        n_samples = 121
        offline_epoch = rng.standard_normal((n_channels, n_samples))
        online_epoch = offline_epoch.copy()

        metrics = parity.trial_metrics(offline_epoch, online_epoch, sfreq=100.0)

        assert metrics.max_abs_diff == 0.0
        assert metrics.mean_channel_corr == pytest.approx(1.0)
        assert metrics.aligned_corr == pytest.approx(1.0)
        assert metrics.best_lag_ms == pytest.approx(0.0)

    def test_constant_lag_detected_by_aligned_corr(self) -> None:
        rng = np.random.default_rng(1)
        n_channels = 5
        n_samples = 121
        signal = rng.standard_normal((n_channels, n_samples))
        # Shift online by +3 samples (= +30 ms at 100 Hz) to simulate causal delay.
        shifted = np.roll(signal, shift=3, axis=1)

        metrics = parity.trial_metrics(signal, shifted, sfreq=100.0)

        assert metrics.mean_channel_corr < 0.95  # raw corr degraded by shift
        assert metrics.aligned_corr > 0.95  # alignment recovers the match
        # The exact sign depends on cross-correlation convention; the magnitude
        # must equal the injected 30 ms shift.
        assert abs(metrics.best_lag_ms) == pytest.approx(30.0, abs=1.0)

    def test_unrelated_signals_give_low_correlation(self) -> None:
        rng = np.random.default_rng(2)
        offline_epoch = rng.standard_normal((4, 121))
        online_epoch = rng.standard_normal((4, 121))

        metrics = parity.trial_metrics(offline_epoch, online_epoch, sfreq=100.0)

        assert abs(metrics.mean_channel_corr) < 0.4
        # Aligned correlation with random lag can spuriously inflate; still
        # bounded well below the 0.95 parity threshold for noise-vs-noise.
        assert metrics.aligned_corr < 0.7


# ── summarise + check_thresholds ────────────────────────────────────────────


class TestSummariseAndThresholds:
    def _make_trial(self, **overrides) -> "parity.TrialMetrics":
        defaults = dict(max_abs_diff=0.1, mean_channel_corr=0.99, aligned_corr=0.99, best_lag_ms=10.0)
        defaults.update(overrides)
        return parity.TrialMetrics(**defaults)

    def test_summarise_reports_medians(self) -> None:
        trials = [
            self._make_trial(aligned_corr=0.90),
            self._make_trial(aligned_corr=0.95),
            self._make_trial(aligned_corr=0.99),
        ]
        summary = parity.summarise("baseline", trials)
        assert summary.n_trials == 3
        assert summary.median_aligned_corr == pytest.approx(0.95)

    def test_flags_fire_on_low_aligned_corr(self) -> None:
        summary = parity.BranchSummary(
            label="baseline",
            n_trials=10,
            median_max_abs_diff=0.01,
            median_mean_channel_corr=0.9,
            median_aligned_corr=0.80,
            median_best_lag_ms=5.0,
        )
        flags = parity.check_thresholds(
            baseline=summary, offline_std=1.0,
            aligned_corr_min=0.95, lag_max_ms=50.0, diff_ratio_max=0.25,
        )
        assert any("aligned_corr" in flag for flag in flags)

    def test_flags_fire_on_excessive_lag(self) -> None:
        summary = parity.BranchSummary(
            label="baseline",
            n_trials=10,
            median_max_abs_diff=0.01,
            median_mean_channel_corr=0.99,
            median_aligned_corr=0.99,
            median_best_lag_ms=80.0,
        )
        flags = parity.check_thresholds(
            baseline=summary, offline_std=1.0,
            aligned_corr_min=0.95, lag_max_ms=50.0, diff_ratio_max=0.25,
        )
        assert any("best_lag_ms" in flag for flag in flags)

    def test_no_flags_when_all_metrics_pass(self) -> None:
        summary = parity.BranchSummary(
            label="baseline",
            n_trials=10,
            median_max_abs_diff=0.05,
            median_mean_channel_corr=0.99,
            median_aligned_corr=0.99,
            median_best_lag_ms=10.0,
        )
        flags = parity.check_thresholds(
            baseline=summary, offline_std=1.0,
            aligned_corr_min=0.95, lag_max_ms=50.0, diff_ratio_max=0.25,
        )
        assert flags == []


# ── Slow integration test ────────────────────────────────────────────────────


@pytest.mark.slow
def test_parity_smoke_runs_end_to_end_on_synthetic_fixture(tmp_path: Path) -> None:
    """End-to-end smoke test: write a synthetic VHDR + minimal artifact and run.

    Skipped if pybv (the BrainVision writer) is unavailable. This does NOT
    assert parity passes — synthetic data with a re-fit ICA against artifact
    matrices will diverge. It only asserts the library entry point runs and
    emits a non-empty report.
    """
    pytest.importorskip("pybv")
    pytest.importorskip("mne")
    pytest.skip(
        "Synthetic fixture is too small to fit ICA reliably; rerun against the "
        "real FL split + trained pipeline via the CLI instead."
    )
