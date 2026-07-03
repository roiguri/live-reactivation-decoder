"""Analysis metrics shared across notebooks.

Generalized from the color-decoder notebook over arbitrary marker/task sets:
a winner-take-all confusion matrix, a label-permutation significance band, and
a diagonal cross-validated AUC curve. Plotting helpers stay minimal so notebooks
keep control of layout. Metrics graduate here from notebook cells as they prove
out — keep signatures parametrized (no hardcoded class names).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def winner_confusion(
    epoched: dict[str, dict[str, np.ndarray]],
    markers: list[str],
    tasks: list[str],
    marker_of_task: dict[str, str],
    t_grid: np.ndarray,
    *,
    mode: str = "vote",
    vote_from: float = -0.1,
    task_tps: dict[str, float] | None = None,
    sigma: float = 0.05,
) -> np.ndarray:
    """Winner-take-all confusion matrix (``K x K``: true marker x winning decoder).

    The per-epoch winner is chosen by ``mode``:

    - ``"vote"`` (default): at each timepoint take the top decoder, then the
      decoder that leads most often over ``t >= vote_from`` wins. Rank-based —
      only *who* is on top each moment matters, not by how much.
    - ``"weighted_prob"``: winner = ``argmax_k`` of a time-weighted mean of each
      decoder's ``P_k(t)``, with Gaussian weights centered on that decoder's
      trained timepoint (``task_tps[task]``, width ``sigma`` seconds). Magnitude-
      based — weights the area around the trained timepoint. A task with no tp
      falls back to a flat ``t >= vote_from`` window. Narrow ``sigma`` → "value at
      own tp"; wide ``sigma`` → "mean over window".
    """
    K = len(markers)
    col_of = {m: i for i, m in enumerate(markers)}
    conf = np.zeros((K, K), int)

    if mode == "weighted_prob":
        task_tps = task_tps or {}
        W = np.empty((len(tasks), len(t_grid)))            # per-task time weights
        for i, t in enumerate(tasks):
            tp = task_tps.get(t)
            W[i] = (np.exp(-0.5 * ((t_grid - tp) / sigma) ** 2)
                    if tp is not None else (t_grid >= vote_from).astype(float))
            total = W[i].sum()
            if total > 0:
                W[i] /= total
        for true_m in markers:
            arr = np.stack([epoched[t][true_m] for t in tasks])   # (K, n_ep, n_time)
            if arr.shape[1] == 0:
                continue
            scores = np.einsum("ket,kt->ke", arr, W)              # (K, n_ep) weighted mean P
            for winner_idx in scores.argmax(0):
                conf[col_of[true_m], col_of[marker_of_task[tasks[winner_idx]]]] += 1
        return conf

    if mode != "vote":
        raise ValueError(f"unknown winner mode {mode!r} (use 'vote' or 'weighted_prob')")

    vote_mask = t_grid >= vote_from
    for true_m in markers:
        arr = np.stack([epoched[t][true_m] for t in tasks])  # (K, n_ep, n_time)
        if arr.shape[1] == 0:
            continue
        for row in arr.argmax(0)[:, vote_mask]:
            winner_task = tasks[np.bincount(row, minlength=K).argmax()]
            conf[col_of[true_m], col_of[marker_of_task[winner_task]]] += 1
    return conf


def modality_groups(
    decoder_tasks: list[dict], *, rest_label: str = "rest",
    marker_of_task: dict[str, str] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Group decoders that share a training label-set into competition blocks.

    Each decoder's competition set is its own training labels — ``pos_labels``
    plus ``neg_labels`` minus ``rest_label`` — so decoders that were trained
    against the same stimuli (e.g. the 3 colours, within-modality) form one
    group. Returns ``[(markers, tasks), ...]`` where ``markers`` are the group's
    positive stimuli (in task order) and ``tasks`` the decoders competing within
    it. Within-modality configs yield one group per modality; a cross-modal
    config (every decoder vs all others) yields a single group.

    ``marker_of_task`` maps each task to the epoched-data key that represents
    its positive class — pass ``{task: dc.target_group(task)}`` when the
    display groups are pooled (e.g. per-image markers pooled into a category),
    otherwise a raw marker and its display group are the same name and the
    default (each task's first ``pos_labels`` entry) is correct.
    """
    label_set: dict[str, frozenset[str]] = {}
    for t in decoder_tasks:
        label_set[t["name"]] = (
            frozenset(t["pos_labels"]) | (frozenset(t["neg_labels"]) - {rest_label})
        )
    groups: dict[frozenset[str], list[str]] = {}
    for t in decoder_tasks:
        groups.setdefault(label_set[t["name"]], []).append(t["name"])
    out = []
    for names in groups.values():
        if marker_of_task is not None:
            markers = [marker_of_task[t["name"]] for t in decoder_tasks if t["name"] in names]
        else:
            markers = [t["pos_labels"][0] for t in decoder_tasks if t["name"] in names]
        out.append((markers, names))
    return out


def baseline_correct(
    epoched: dict[str, dict[str, np.ndarray]],
    t_grid: np.ndarray,
    *,
    mode: str = "prestim",
    window: tuple[float | None, float | None] = (None, 0.0),
) -> dict[str, dict[str, np.ndarray]]:
    """Return a copy of ``epoched`` with each decoder/trial curve baseline-subtracted.

    ``mode="prestim"`` subtracts, per trial, that decoder's **mean P over the
    pre-stimulus window** ``window`` (default ``t < 0``). The result is ΔP — the
    *rise from each decoder's own baseline* — which removes per-decoder offset so
    decoders are comparable in a winner competition. Empty arrays pass through.
    """
    if mode != "prestim":
        raise ValueError(f"unknown baseline mode {mode!r} (use 'prestim')")
    lo, hi = window
    mask = np.ones(len(t_grid), bool)
    if lo is not None:
        mask &= t_grid >= lo
    if hi is not None:
        mask &= t_grid < hi
    out: dict[str, dict[str, np.ndarray]] = {}
    for task, by_marker in epoched.items():
        out[task] = {}
        for m, arr in by_marker.items():
            if arr.shape[0] and mask.any():
                out[task][m] = arr - arr[:, mask].mean(axis=1, keepdims=True)
            else:
                out[task][m] = arr
    return out


def confusion_scores(conf: np.ndarray) -> dict[str, np.ndarray | float]:
    """Per-class precision/recall and overall accuracy from a confusion matrix."""
    diag = np.diag(conf).astype(float)
    recall = np.divide(diag, conf.sum(1), out=np.zeros_like(diag), where=conf.sum(1) > 0)
    precision = np.divide(diag, conf.sum(0), out=np.zeros_like(diag), where=conf.sum(0) > 0)
    acc = diag.sum() / conf.sum() if conf.sum() else float("nan")
    return {"precision": precision, "recall": recall, "accuracy": acc}


def plot_confusion(conf: np.ndarray, ax, markers: list[str], title: str):
    """Row-normalized confusion heatmap with raw counts annotated."""
    K = len(markers)
    rs = conf.sum(1, keepdims=True)
    norm = np.divide(conf, rs, out=np.zeros(conf.shape), where=rs > 0)
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_xticklabels(markers, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(markers)
    ax.set_xlabel("winner"); ax.set_ylabel("true stimulus"); ax.set_title(title)
    for r in range(K):
        for c in range(K):
            ax.text(c, r, f"{conf[r, c]}", ha="center", va="center", fontsize=9,
                    color="white" if norm[r, c] > 0.5 else "black")
    return im


def perm_band(
    epoched: dict[str, dict[str, np.ndarray]],
    task: str,
    target_marker: str,
    markers: list[str],
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label-permutation 5/95 null band for one decoder's mean ``P(t)``.

    Returns ``(observed, lo, hi, null_mean)`` over the time grid: the real mean
    P(t) for ``target_marker`` epochs and the 5th/95th percentiles + mean of the
    label-shuffled null.
    """
    rng = rng or np.random.default_rng(0)
    all_p = np.concatenate([epoched[task][m] for m in markers], axis=0)
    labels = np.concatenate([[m] * epoched[task][m].shape[0] for m in markers])
    mask = labels == target_marker
    obs = all_p[mask].mean(0)
    n, total = int(mask.sum()), all_p.shape[0]
    null = np.empty((n_perm, all_p.shape[1]))
    for i in range(n_perm):
        null[i] = all_p[rng.choice(total, n, replace=False)].mean(0)
    lo, hi = np.percentile(null, [5, 95], axis=0)
    return obs, lo, hi, null.mean(0)


def diag_auc(epochs, settings: dict, task: str) -> np.ndarray:
    """Diagonal-only cross-validated AUC for one task (train/test per timepoint).

    The diagonal of the temporal-generalization matrix via ``SlidingEstimator``
    — ~100x cheaper than the full grid and identical to the diagonal we plot.
    Uses the same classifier + CV folds as the offline evaluator.
    """
    from mne.decoding import SlidingEstimator, cross_val_multiscore
    from sklearn.model_selection import StratifiedKFold

    from backend.offline_phase.utils import build_classifier, get_task_data

    cfg = next(t for t in settings["tasks"] if t["name"] == task)
    X, y = get_task_data(epochs, cfg)
    cv = StratifiedKFold(
        n_splits=settings["cv"]["k"], shuffle=True, random_state=settings["random_state"]
    )
    est = SlidingEstimator(build_classifier(settings), scoring="roc_auc", n_jobs=1)
    return cross_val_multiscore(est, X, y, cv=cv, n_jobs=1).mean(0)
