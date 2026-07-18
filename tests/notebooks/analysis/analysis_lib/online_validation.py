"""Step 2 — online-pipeline validation: does the causal/streaming pipeline
degrade decoding AUC relative to best-case offline processing?

Methodology (deployment-faithful, paired). The shipped system trains its
decoder on **offline** epochs and then infers on the **online** causal pipeline.
We reproduce exactly that: on the same trials and the same CV folds, fit on
offline-processed train trials, then score the held-out trials twice — once
offline-processed (best-case reference + sanity anchor), once online-processed
(as-deployed). Everything but the test-time processing is held identical, so the
AUC difference is attributable purely to the pipeline (causal filtering,
micro-batch boundary state, decimation phase). The frozen ICA/interp/bad-channel
matrices come from the same Phase-1 artifact on both sides, so they never enter
the comparison as a confound.

**Every offline trial gets an online counterpart**, so the CV partition matches
step 1's exactly (standard ``StratifiedKFold``, all classes in train *and* test):
- Stimulus trials pair to their online trigger marker by nearest onset (precise).
- ``intervals:`` classes (e.g. rest windows) are offline-only synthetic epochs
  with no trigger marker — but the online feature stream is continuous, so they
  are epoched from it at their own offline onset. Those really are
  online-pipeline-processed features for those baseline windows.

Because rest windows separate easily from stimulus, this contrast (and its AUC)
matches step 1's rest-inflated number rather than the harder image-vs-image one.
That is intentional: the work-plan bar is "no >10% degradation vs offline on the
**same data partition**", and this reproduces step 1's partition. The headline is
the offline-vs-online *delta* on identical trials, not the absolute AUC — read it
as "the causal pipeline preserves decodability on step 1's partition," not as a
category-discrimination claim.

Onset alignment is by seconds — never sample index: offline epochs live on the
100 Hz decimated grid while the online replay reads the native ~1000 Hz
recording, and ``first_samp`` differs.

Reuses ``streaming.run_online_stream`` / ``make_epocher_multichannel`` (online
features), the on-disk offline epochs, and the offline decoding primitives
(``build_classifier``, ``StratifiedKFold``, ``GeneralizingEstimator``).
"""
from __future__ import annotations

import glob

import numpy as np


def load_offline_epochs(ctx):
    """Read the subject's offline epochs (``epochs/*epo.fif``) as an MNE object.

    Same source ``plots.cv_auc`` / ``plots.parity`` read — the best-case,
    offline-processed signal the online branch is compared against.
    """
    import mne

    epo_fif = sorted(glob.glob(str(ctx.paths.epochs_dir / "*epo.fif")))
    if not epo_fif:
        raise FileNotFoundError(f"no epochs .fif under {ctx.paths.epochs_dir}")
    return mne.read_epochs(epo_fif[0], verbose=False)


def _pair_by_onset(
    off_onsets: list[float], on_onsets: list[float], *, tol: float
) -> list[tuple[int, int]]:
    """Greedy nearest-onset pairing within one class (each online epoch used once).

    ``off_onsets``/``on_onsets`` are onset times in seconds. Returns
    ``[(off_idx, on_idx), ...]`` for offline trials that found an online partner
    within ``tol`` seconds; unmatched offline trials are simply absent. O(n^2)
    but n is dozens per class.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, ot in enumerate(off_onsets):
        best_j, best_d = None, tol
        for j, nt in enumerate(on_onsets):
            if j in used:
                continue
            d = abs(ot - nt)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def build_paired_features(
    ctx, offline_epochs, task_cfg, raw, features, out_samples, sfreq, fs_out,
    *, tmin, tmax, n_times=None, tol: float = 0.02,
):
    """Build offline/online feature epochs paired 1:1 for one decoder task.

    ``features``/``out_samples``/``fs_out`` come from a single
    ``streaming.run_online_stream(ctx.preproc, eeg, ...)`` replay of ``raw``.
    Selects the task's ``pos_labels + neg_labels`` from ``offline_epochs`` with
    the same semantics as ``backend.offline_phase.utils.get_task_data``, then
    epochs the online feature stream once per offline trial:

    - stimulus trials use their online trigger marker's raw sample, matched to
      the offline trial by nearest onset (precise, so the filtering-degradation
      measurement isn't blurred by misalignment);
    - marker-less interval/rest trials use the offline onset directly
      (``round(onset_seconds * sfreq_raw)``) — no transient, so timing precision
      is immaterial.

    A trial is dropped from *both* sides only if its online window has too few
    stream samples (edge of recording). Returns a dict with ``X_off``/``X_on``
    (``(n, n_ch, n_t)``), binary ``y``, ``t_grid``/``times`` (asserted equal
    length), and ``n_trials``/``n_marker``/``n_intervalied`` counts.
    """
    from analysis_lib import streaming

    pos_labels: list[str] = task_cfg["pos_labels"]
    neg_labels: list[str] = task_cfg["neg_labels"]
    all_labels = pos_labels + neg_labels

    missing = [lbl for lbl in all_labels if lbl not in offline_epochs.event_id]
    if missing:
        raise ValueError(
            f"Task {task_cfg['name']!r}: labels not in offline epochs: {missing}"
        )

    # Offline selected subset — mirrors get_task_data's selection + labeling.
    sel = offline_epochs[all_labels]
    X_off_all = sel.get_data()
    sfreq_off = float(sel.info["sfreq"])
    inv_event_id = {code: name for name, code in sel.event_id.items()}
    off_names = [inv_event_id[c] for c in sel.events[:, 2]]
    off_onsets = (sel.events[:, 0] / sfreq_off).tolist()
    pos_codes = {sel.event_id[lbl] for lbl in pos_labels}
    y_all = np.where(np.isin(sel.events[:, 2], list(pos_codes)), 1, 0)

    # Online trigger markers for the same classes, at the native recording rate.
    marker_pairs = streaming.extract_markers(
        raw, ctx.event_mapping, all_labels, n_times=n_times
    )
    on_samples_by_name: dict[str, list[int]] = {}
    for s, c in marker_pairs:
        on_samples_by_name.setdefault(ctx.name_by_code[c], []).append(s)

    # Choose the online-epoching raw sample for every offline trial.
    off_idx_by_name: dict[str, list[int]] = {}
    for i, name in enumerate(off_names):
        off_idx_by_name.setdefault(name, []).append(i)

    raw_sample_by_off_idx: dict[int, int] = {}
    n_marker = 0
    for name, idxs in off_idx_by_name.items():
        markers = sorted(on_samples_by_name.get(name, []))
        if markers:  # stimulus class — match each offline trial to its marker
            local_pairs = _pair_by_onset(
                [off_onsets[i] for i in idxs], [m / sfreq for m in markers], tol=tol
            )
            for local_i, marker_j in local_pairs:
                raw_sample_by_off_idx[idxs[local_i]] = markers[marker_j]
                n_marker += 1
            # offline stimulus trials with no marker within tol fall back to onset
            matched = {local_i for local_i, _ in local_pairs}
            for local_i, i in enumerate(idxs):
                if local_i not in matched:
                    raw_sample_by_off_idx[i] = int(round(off_onsets[i] * sfreq))
        else:  # marker-less interval/rest class — use the offline onset
            for i in idxs:
                raw_sample_by_off_idx[i] = int(round(off_onsets[i] * sfreq))

    # Epoch the online feature stream at each trial's chosen sample.
    t_grid, epoch_features = streaming.make_epocher_multichannel(
        out_samples, sfreq, fs_out, tmin, tmax
    )
    if len(t_grid) != len(sel.times):
        raise ValueError(
            f"offline/online time grids differ ({len(sel.times)} vs {len(t_grid)} "
            "points) — check FINAL_RESAMPLE_RATE / epoch window."
        )

    X_off_rows, X_on_rows, y_rows = [], [], []
    n_intervalied = 0
    for i in range(len(off_names)):
        s = raw_sample_by_off_idx[i]
        arr, _ = epoch_features(features, [s])
        if arr.shape[0]:
            X_off_rows.append(X_off_all[i])
            X_on_rows.append(arr[0])
            y_rows.append(int(y_all[i]))
            if off_names[i] not in on_samples_by_name:
                n_intervalied += 1

    if not y_rows:
        raise ValueError(
            f"Task {task_cfg['name']!r}: no trial produced an online epoch — "
            "check the recording / marker onsets."
        )

    return {
        "X_off": np.array(X_off_rows),
        "X_on": np.array(X_on_rows),
        "y": np.array(y_rows),
        "t_grid": t_grid,
        "times": sel.times,
        "n_trials": len(y_rows),
        "n_marker": n_marker,
        "n_intervalied": n_intervalied,
    }


def _check_cv_feasible(y: np.ndarray, k: int) -> None:
    min_class = int(np.min(np.bincount(y)))
    if min_class < k:
        raise ValueError(
            f"Too few trials for {k}-fold CV: minority class has "
            f"{min_class} (need >= {k})."
        )


def cross_pipeline_tgm(X_off, X_on, y, decoder_settings):
    """Cross-pipeline temporal-generalization: train offline, test offline & online.

    Same ``StratifiedKFold`` folds for both branches (same ``y``, same order).
    Per fold fit a ``GeneralizingEstimator`` on the offline train trials, then
    score the held-out trials on both processings. Returns fold-averaged
    ``tgm_off``/``tgm_on`` (``(n_train_t, n_test_t)``, train-time × test-time)
    and their diagonals ``diag_off``/``diag_on`` (train-time == test-time).

    ``tgm_off`` is the internal sanity anchor (should track step 1's CV AUC on
    this trial set); the ``diag_off`` − ``diag_on`` gap is the pipeline
    degradation over time — including any latency shift (peak pushed later).
    """
    from mne.decoding import GeneralizingEstimator
    from sklearn.model_selection import StratifiedKFold

    from backend.offline_phase.utils import build_classifier

    k = decoder_settings["cv"]["k"]
    _check_cv_feasible(y, k)
    cv = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=decoder_settings["random_state"]
    )
    tgm_off_folds, tgm_on_folds = [], []
    for train_idx, test_idx in cv.split(X_off, y):
        ge = GeneralizingEstimator(
            build_classifier(decoder_settings), scoring="roc_auc",
            n_jobs=-1, verbose=False,
        )
        ge.fit(X_off[train_idx], y[train_idx])
        tgm_off_folds.append(ge.score(X_off[test_idx], y[test_idx]))
        tgm_on_folds.append(ge.score(X_on[test_idx], y[test_idx]))
    tgm_off = np.mean(tgm_off_folds, axis=0)
    tgm_on = np.mean(tgm_on_folds, axis=0)
    return {
        "tgm_off": tgm_off,
        "tgm_on": tgm_on,
        "diag_off": np.diag(tgm_off),
        "diag_on": np.diag(tgm_on),
    }


def operating_point_degradation(
    X_off, X_on, y, decoder_settings, timepoint_idx,
    *, n_boot: int = 2000, threshold: float = 0.10, rng=None,
):
    """Paired offline-vs-online AUC at a single readout timepoint, with a CI.

    The deployment-faithful headline: the live decoder reads out at one trained
    timepoint, so we fix ``timepoint_idx`` (nearest grid point to the decoder's
    trained tp) and, over the same CV folds, cache each held-out trial's
    predicted P(class=1) under both processings. Because every trial appears
    once as test under both, the offline/online comparison is paired trial-by-
    trial and a bootstrap over trials yields a CI on the degradation without any
    refitting.

    Returns observed ``auc_off``/``auc_on``, absolute ``delta`` and relative
    ``pct_drop``, the bootstrap ``ci`` on the *relative* drop, and ``passed``
    (relative drop within ``threshold``).
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from backend.offline_phase.utils import build_classifier

    rng = rng or np.random.default_rng(0)
    k = decoder_settings["cv"]["k"]
    _check_cv_feasible(y, k)
    ti = int(timepoint_idx)
    Xo, Xn = X_off[:, :, ti], X_on[:, :, ti]

    cv = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=decoder_settings["random_state"]
    )
    p_off = np.full(len(y), np.nan)
    p_on = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(Xo, y):
        model = build_classifier(decoder_settings)
        model.fit(Xo[train_idx], y[train_idx])
        pos = list(model.classes_).index(1)
        p_off[test_idx] = model.predict_proba(Xo[test_idx])[:, pos]
        p_on[test_idx] = model.predict_proba(Xn[test_idx])[:, pos]

    auc_off = float(roc_auc_score(y, p_off))
    auc_on = float(roc_auc_score(y, p_on))
    delta = auc_off - auc_on
    pct_drop = delta / auc_off if auc_off else float("nan")

    n = len(y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:  # degenerate resample — skip
            boot[b] = np.nan
            continue
        a_off = roc_auc_score(yb, p_off[idx])
        a_on = roc_auc_score(yb, p_on[idx])
        boot[b] = (a_off - a_on) / a_off if a_off else np.nan
    ci = tuple(np.nanpercentile(boot, [2.5, 97.5]))

    return {
        "timepoint_idx": ti,
        "auc_off": auc_off,
        "auc_on": auc_on,
        "delta": delta,
        "pct_drop": pct_drop,
        "ci": ci,
        "threshold": threshold,
        "passed": bool(pct_drop <= threshold),
        "n_trials": n,
    }
