"""Step 3 — cross-domain generalization: does an FL-trained animacy decoder
generalize to the real-time task's encoding/retrieval phases, i.e. does a
perception-trained decoder read out a *memory* during recall?

Method (King & Dehaene temporal generalization, across domains). Fit one
``GeneralizingEstimator`` on **all** functional-localizer (FL) timepoints, then
score it on the encoding/retrieval feature epochs → a square
``(n_fl_train_times, n_test_times)`` AUC map. No cross-validation: the test
recordings were never part of FL training, so there is nothing to hold out.
Encoding (an image really is on screen) is the positive control; retrieval (pure
recall, no image) is the reactivation result.

Deployment-faithful: the estimator trains on the offline FL epochs and is scored
on real-time-processed (online-pipeline) recall features — exactly what the
shipped system does. Step 2 already bounded the pipeline's own AUC cost.

Honest significance (the easy-to-get-wrong part). Reactivation may sit at some
latency after the cue, so the headline statistic is the **max AUC over the
recall (test-time) axis** along one fixed FL-train row (the deployed operating
point). The label-shuffle null takes the *same* max, so the latency search is
baked into the null — no cherry-picking a cell then testing it. This yields one
family-wise-corrected p-value **and** the reactivation latency (the argmax
test-time), plus a bootstrap CI at that located cell.

Efficiency: AUC == the Mann-Whitney statistic, which depends only on the *ranks*
of the scores, not the labels. So we score the fitted estimators once, rank each
map cell once, and then the observed map, the permutation null, and the per-cell
nulls are all just "sum the ranks at the positive positions" — no refitting.

Pure/numeric core (numpy + scipy); ``mne``/backend are imported lazily inside
the two helpers that actually fit/score an estimator, so the metric functions
unit-test on synthetic arrays without either.
"""
from __future__ import annotations

import numpy as np


# ── test-set construction ────────────────────────────────────────────────────
def _label_categories(labels: list[str]) -> set[str]:
    """The set of categories a decoder's labels resolve to (drops non-``<cat>_NN``).

    FL decoder labels are per-image (``animate_01``); ``category_of`` pools them
    to their category (``animate``). Rest/interval labels (``rest_fixation``)
    don't match the convention and resolve to ``None`` — dropped here — which is
    exactly right: rest never occurs during encoding/retrieval, so it can't (and
    shouldn't) map any test trial.
    """
    from analysis_lib import task_labels

    return {task_labels.category_of(lbl) for lbl in labels} - {None}


def binary_test_set(X_all, cat_labels, task_cfg):
    """Slice a category-labeled test set down to one decoder's binary contrast.

    ``X_all`` is ``(n_trials, n_ch, n_grid)`` and ``cat_labels`` the per-trial
    category (from ``sources.build_*_features``). Maps each trial to the decoder's
    polarity — ``1`` if its category is among ``pos_labels``' categories, ``0`` if
    among ``neg_labels``' — and drops trials whose category is in neither. Returns
    ``(X_test, y_test)`` with the same channel/time axes, ready for
    :func:`cell_scores`.
    """
    pos_cats = _label_categories(task_cfg["pos_labels"])
    neg_cats = _label_categories(task_cfg["neg_labels"])
    overlap = pos_cats & neg_cats
    if overlap:
        raise ValueError(
            f"Task {task_cfg['name']!r}: categories in both pos and neg: {sorted(overlap)}"
        )

    cat_labels = np.asarray(cat_labels)
    keep = np.array([c in pos_cats or c in neg_cats for c in cat_labels], dtype=bool)
    X_test = np.asarray(X_all)[keep]
    kept = cat_labels[keep]
    y_test = np.array([1 if c in pos_cats else 0 for c in kept], dtype=int)

    if len(np.unique(y_test)) < 2:
        raise ValueError(
            f"Task {task_cfg['name']!r}: test set has a single class after mapping "
            f"(pos categories {sorted(pos_cats)}, present {sorted(set(kept.tolist()))})."
        )
    return X_test, y_test


# ── fit / score ──────────────────────────────────────────────────────────────
def fit_generalizer(X_fl, y_fl, decoder_settings):
    """Fit a ``GeneralizingEstimator`` on all FL timepoints (one estimator per row).

    Wraps the same scaler+classifier the offline evaluator uses
    (``build_classifier``). No ``scoring`` is set — we read raw
    ``decision_function`` scores and compute AUC ourselves so the permutation can
    reuse them without refitting.
    """
    from mne.decoding import GeneralizingEstimator

    from backend.offline_phase.utils import build_classifier

    ge = GeneralizingEstimator(build_classifier(decoder_settings), n_jobs=-1, verbose=False)
    ge.fit(X_fl, y_fl)
    return ge


def cell_scores(ge, X_test) -> np.ndarray:
    """Signed decision scores per (trial, FL-train-time, test-time).

    Returns ``(n_trials, n_fl_t, n_test_t)``. Higher == more class-1 (the
    decoder's positive category), so it feeds AUC directly. Binary decoders give
    a 3-D array; anything else means a non-binary task snuck in.
    """
    scores = np.asarray(ge.decision_function(X_test))
    if scores.ndim != 3:
        raise ValueError(
            f"expected (n_trials, n_fl_t, n_test_t) decision_function, got {scores.shape} "
            "(cross-domain generalization assumes binary decoders)."
        )
    return scores


# ── rank-based AUC (Mann-Whitney) ────────────────────────────────────────────
def _auc_from_ranks(ranks: np.ndarray, pos_mask: np.ndarray, n_pos: int, n_neg: int) -> np.ndarray:
    """AUC per column from precomputed average ranks and a positive-row mask.

    ``ranks`` is ``(n_trials, n_cols)`` of per-column average ranks;
    ``AUC = (Σ ranks_pos − n_pos(n_pos+1)/2) / (n_pos·n_neg)`` equals
    ``roc_auc_score`` per column, ties included.
    """
    s_pos = ranks[pos_mask].sum(axis=0)
    u = s_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def tgm_auc(scores: np.ndarray, y) -> np.ndarray:
    """Observed AUC for every map cell → ``(n_fl_t, n_test_t)``.

    Vectorized rank-AUC: rank each cell's scores across trials once, then read off
    the positive-row rank sum. Identical to calling ``roc_auc_score`` per cell but
    orders of magnitude faster.
    """
    from scipy.stats import rankdata

    y = np.asarray(y)
    n_trials, n_fl, n_test = scores.shape
    flat = scores.reshape(n_trials, n_fl * n_test)
    ranks = rankdata(flat, axis=0)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = n_trials - n_pos
    auc = _auc_from_ranks(ranks, pos, n_pos, n_neg)
    return auc.reshape(n_fl, n_test)


def row_index_for_timepoint(fl_times, tp: float) -> int:
    """Nearest FL-train-time row index to ``tp`` seconds (the operating point)."""
    return int(np.argmin(np.abs(np.asarray(fl_times, dtype=float) - float(tp))))


def row_permutation(scores: np.ndarray, y, fl_row_idx: int, *, n_perm: int = 2000, rng=None) -> dict:
    """Latency-corrected significance for one FL-train row of the map.

    Fixes the FL-train timepoint (``fl_row_idx`` — the deployed operating point)
    and takes the observed statistic to be the **max AUC over the recall/test
    axis** along that row, at latency ``argmax_test_idx``. The null shuffles the
    test labels ``n_perm`` times and takes the same max each time, so searching
    for the best latency is paid for. Because ranks don't depend on labels, each
    shuffle is just a rank sum over a random size-``n_pos`` subset — no refitting.

    ``p_value`` uses the Efron-Tibshirani ``(k+1)/(n_perm+1)`` correction, so it
    is never exactly 0 (needed for the downstream Stouffer combination).
    """
    from scipy.stats import rankdata

    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    row = scores[:, int(fl_row_idx), :]  # (n_trials, n_test)
    n_trials = row.shape[0]
    ranks = rankdata(row, axis=0)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = n_trials - n_pos

    obs_row = _auc_from_ranks(ranks, pos, n_pos, n_neg)  # (n_test,)
    argmax = int(np.argmax(obs_row))
    max_auc = float(obs_row[argmax])

    const = n_pos * (n_pos + 1) / 2.0
    denom = n_pos * n_neg
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n_trials)[:n_pos]  # a random label-shuffle's positives
        aucs = (ranks[idx].sum(axis=0) - const) / denom
        null_max[i] = aucs.max()

    p_value = (int(np.sum(null_max >= max_auc)) + 1) / (n_perm + 1)
    return {
        "obs_row": obs_row,
        "max_auc": max_auc,
        "argmax_test_idx": argmax,
        "null_max": null_max,
        "p_value": p_value,
    }


def matrix_permutation(scores: np.ndarray, y, *, n_perm: int = 2000, rng=None) -> dict:
    """Latency-corrected significance searching the **whole** map (both axes).

    The headline for cross-domain generalization when we don't privilege the
    deployed FL timepoint: the observed statistic is the max AUC over *every*
    ``(FL-train-time, test-time)`` cell, located at ``(fl_idx, test_idx)``. The
    null shuffles the test labels ``n_perm`` times and takes the same **global
    max** each time, so searching the entire matrix for the best cell is paid
    for. Same rank trick as :func:`row_permutation`, just maxing over all cells
    instead of one row — no refitting. Also returns the full observed AUC map
    (``tgm``) so the caller needn't recompute it.

    ``p_value`` uses the Efron-Tibshirani ``(k+1)/(n_perm+1)`` correction (never
    exactly 0 — needed for Stouffer).
    """
    from scipy.stats import rankdata

    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    n_trials, n_fl, n_test = scores.shape
    flat = scores.reshape(n_trials, n_fl * n_test)
    ranks = rankdata(flat, axis=0)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = n_trials - n_pos

    obs = _auc_from_ranks(ranks, pos, n_pos, n_neg)  # (n_cells,)
    flat_argmax = int(np.argmax(obs))
    fl_idx, test_idx = divmod(flat_argmax, n_test)
    max_auc = float(obs[flat_argmax])

    const = n_pos * (n_pos + 1) / 2.0
    denom = n_pos * n_neg
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n_trials)[:n_pos]
        aucs = (ranks[idx].sum(axis=0) - const) / denom
        null_max[i] = aucs.max()

    p_value = (int(np.sum(null_max >= max_auc)) + 1) / (n_perm + 1)
    return {
        "tgm": obs.reshape(n_fl, n_test),
        "max_auc": max_auc,
        "fl_idx": int(fl_idx),
        "test_idx": int(test_idx),
        "null_max": null_max,
        "p_value": p_value,
    }


def cell_bootstrap(
    scores: np.ndarray, y, fl_row_idx: int, test_col_idx: int,
    *, n_boot: int = 2000, rng=None, ci=(2.5, 97.5),
) -> tuple[float, float]:
    """Bootstrap CI on the AUC at one located map cell (resample trials).

    Resamples trials with replacement and recomputes AUC at the fixed
    ``(fl_row_idx, test_col_idx)`` cell — one column, so ``roc_auc_score`` per
    resample is cheap. Degenerate single-class resamples are dropped. Returns the
    ``ci`` percentiles (default central 95%).
    """
    from sklearn.metrics import roc_auc_score

    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    col = scores[:, int(fl_row_idx), int(test_col_idx)]
    n = len(y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(yb, col[idx])
    lo, hi = np.nanpercentile(boot, list(ci))
    return float(lo), float(hi)


# ── cross-subject combination ────────────────────────────────────────────────
def stouffer(pvalues, *, weights=None) -> tuple[float, float]:
    """Stouffer-combine one-sided p-values into a single z and combined p.

    Converts each ``p`` to ``z = Φ⁻¹(1 − p)`` (one-sided, AUC > 0.5 direction),
    combines as ``Z = Σ wᵢ zᵢ / √Σ wᵢ²`` (equal weights by default), and reports
    ``p_combined = 1 − Φ(Z)``. Inputs must be > 0 (guaranteed by
    :func:`row_permutation`'s Efron-Tibshirani correction) so ``z`` stays finite.
    """
    from scipy.stats import norm

    p = np.asarray(pvalues, dtype=float)
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError(f"p-values must be in (0, 1]; got {p.tolist()}")
    z = norm.isf(p)  # Φ⁻¹(1 − p)
    if weights is None:
        Z = z.sum() / np.sqrt(len(z))
    else:
        w = np.asarray(weights, dtype=float)
        Z = (w * z).sum() / np.sqrt((w ** 2).sum())
    return float(Z), float(norm.sf(Z))


# ── one-call convenience ─────────────────────────────────────────────────────
def cross_domain_result(
    ge, X_test, y_test, *, deployed_row_idx: int | None = None,
    n_perm: int = 2000, n_boot: int = 2000, rng=None,
) -> dict:
    """Full result for one (subject, decoder, source): map + headline + p + CI.

    Scores the fitted ``ge`` once, then derives the observed AUC map and the
    **whole-map** latency-corrected headline (max AUC over every FL-train ×
    recall-test cell via :func:`matrix_permutation`, located at
    ``(fl_idx, test_idx)``), plus a bootstrap CI at that located cell. ``diag`` is
    the train==test diagonal — the cleanest single trajectory when the FL and
    recall windows share a time axis.

    If ``deployed_row_idx`` (the shipped decoder's FL timepoint) is given, a
    secondary **deployed-row** readout is added: that row's own max-over-recall
    AUC, latency, and permutation p-value (via :func:`row_permutation`), plus the
    deployed diagonal cell — the honest "at the operating point that actually
    ships" number alongside the "best anywhere" headline.
    """
    rng = rng or np.random.default_rng(0)
    y_test = np.asarray(y_test)
    scores = cell_scores(ge, X_test)
    perm = matrix_permutation(scores, y_test, n_perm=n_perm, rng=rng)
    tgm = perm["tgm"]
    lo, hi = cell_bootstrap(
        scores, y_test, perm["fl_idx"], perm["test_idx"], n_boot=n_boot, rng=rng
    )
    out = {
        "tgm": tgm,
        "diag": np.diag(tgm),
        "max_auc": perm["max_auc"],
        "fl_idx": perm["fl_idx"],
        "test_idx": perm["test_idx"],
        "p_value": perm["p_value"],
        "ci": (lo, hi),
        "null_max": perm["null_max"],
        "n_trials": int(len(y_test)),
        "n_pos": int(np.sum(y_test == 1)),
    }
    if deployed_row_idx is not None:
        di = int(deployed_row_idx)
        dep = row_permutation(scores, y_test, di, n_perm=n_perm, rng=rng)
        out["deployed"] = {
            "row_idx": di,
            "diag_auc": float(tgm[di, di]),
            "max_auc": dep["max_auc"],
            "argmax_test_idx": dep["argmax_test_idx"],
            "p_value": dep["p_value"],
        }
    return out
