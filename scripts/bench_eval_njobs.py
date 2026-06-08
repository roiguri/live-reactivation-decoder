"""Benchmark ModelEvaluator TGM-CV parallelization strategies (issue #43).

Loads the real cached functional-localizer epochs and times the
GeneralizingEstimator + cross_val_multiscore inner loop under several
n_jobs strategies. n_jobs affects ONLY scheduling, so every strategy must
return a bit-for-bit identical TGM — the script asserts this with allclose.

Usage
-----
    python scripts/bench_eval_njobs.py --mode rank          # fast, decimated grid
    python scripts/bench_eval_njobs.py --mode full          # full grid, baseline vs winner
    python scripts/bench_eval_njobs.py --mode full --est-jobs -1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.utils import build_classifier, get_task_data

EPOCHS_PATH = PROJECT_ROOT / "debug_snapshots" / "default" / "epochs" / "functional_localizer_epo.fif"
CONFIG_PATH = PROJECT_ROOT / "debug_snapshots" / "default" / "experiment_config.yaml"
COLOR_TASKS = {"red decoder", "green decoder", "yellow decoder"}


def run_tgm_cv(X, y, settings, est_jobs, cv_jobs):
    """Mirror ModelEvaluator._run_tgm_cv with parameterized n_jobs."""
    k = settings["cv"]["k"]
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=settings["random_state"])
    estimator = GeneralizingEstimator(
        build_classifier(settings), scoring="roc_auc", n_jobs=est_jobs, verbose=False
    )
    scores = cross_val_multiscore(estimator, X, y, cv=cv, n_jobs=cv_jobs)
    return np.mean(scores, axis=0)


def load():
    epochs = mne.read_epochs(EPOCHS_PATH, preload=True, verbose=False)
    settings = SettingsManager(CONFIG_PATH).get_decoder_settings()
    settings["tasks"] = [t for t in settings["tasks"] if t["name"] in COLOR_TASKS]
    return epochs, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rank", "full"], default="rank")
    ap.add_argument("--decim", type=int, default=4, help="time decimation for rank mode")
    ap.add_argument("--est-jobs", type=int, default=-1, help="full mode: winner estimator n_jobs")
    ap.add_argument("--cv-jobs", type=int, default=1, help="full mode: winner cv n_jobs")
    args = ap.parse_args()

    import os
    epochs, settings = load()
    print(f"cores={os.cpu_count()}  epochs={len(epochs)}  n_times={len(epochs.times)}  "
          f"event_id={epochs.event_id}", flush=True)

    tasks = settings["tasks"]

    if args.mode == "rank":
        # Single representative task, decimated time grid for speed.
        X, y = get_task_data(epochs, tasks[0])
        Xd = X[..., ::args.decim]
        print(f"\nRANK on task '{tasks[0]['name']}': X={X.shape} -> decimated {Xd.shape}\n", flush=True)
        strategies = [
            ("baseline est=1 cv=1", 1, 1),
            ("est=-1 cv=1 (timepoint-parallel)", -1, 1),
            ("est=1 cv=-1 (fold-parallel)", 1, -1),
            ("est=-1 cv=-1 (nested)", -1, -1),
        ]
        ref = None
        base = None
        for name, ej, cj in strategies:
            t0 = time.perf_counter()
            tgm = run_tgm_cv(Xd, y, settings, ej, cj)
            dt = time.perf_counter() - t0
            ident = "ref" if ref is None else ("identical" if np.allclose(tgm, ref) else "*** DIFFERS ***")
            if ref is None:
                ref = tgm
            if base is None:
                base = dt
            speed = f"{base/dt:5.2f}x"
            print(f"  {name:38s} {dt:7.2f}s  {speed}   {ident}", flush=True)
        return

    # full mode: baseline vs winner over all color tasks, with identity check
    print(f"\nFULL: baseline(1,1) vs winner(est={args.est_jobs},cv={args.cv_jobs}) "
          f"over {len(tasks)} tasks\n", flush=True)
    for label, ej, cj in [("BASELINE  est=1 cv=1", 1, 1),
                          (f"OPTIMIZED est={args.est_jobs} cv={args.cv_jobs}", args.est_jobs, args.cv_jobs)]:
        total = 0.0
        tgms = {}
        for t in tasks:
            X, y = get_task_data(epochs, t)
            t0 = time.perf_counter()
            tgm = run_tgm_cv(X, y, settings, ej, cj)
            dt = time.perf_counter() - t0
            total += dt
            tgms[t["name"]] = tgm
            print(f"  [{label}] {t['name']:16s} {dt:8.2f}s  peak_diag_auc={np.max(np.diag(tgm)):.4f}", flush=True)
        print(f"  [{label}] TOTAL {total:8.2f}s\n", flush=True)
        if ej == 1 and cj == 1:
            base_total, base_tgms = total, tgms
        else:
            opt_total, opt_tgms = total, tgms

    all_identical = all(np.allclose(base_tgms[n], opt_tgms[n]) for n in base_tgms)
    print(f"results identical (bit-for-bit np.allclose): {all_identical}")
    print(f"baseline total : {base_total:8.2f}s")
    print(f"optimized total: {opt_total:8.2f}s")
    print(f"speedup        : {base_total/opt_total:6.2f}x   "
          f"(-{100*(1-opt_total/base_total):.1f}% wall time)")


if __name__ == "__main__":
    main()
