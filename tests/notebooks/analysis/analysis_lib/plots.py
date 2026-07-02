"""Figure helpers for the epoched-decoding notebook.

Moves the bulky matplotlib out of notebook cells so each cell is a thin call.
Functions take the :class:`~analysis_lib.context.AnalysisContext` (``ctx``), a
:class:`DisplayConfig` (``dc``), and the epoched probability stream, then render
and show. Layout choices (3-wide grids, the known per-feature colours, the
bold-★ target styling) live here once instead of being copy-pasted per cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from analysis_lib import metrics

# Stable per-feature colours; anything else falls back to the tab10 cycle.
_KNOWN_COLORS = {
    "red": "crimson", "green": "green", "yellow": "goldenrod",
    "living_room": "purple", "bathroom": "teal", "kitchen": "saddlebrown",
    "color": "darkorange", "scene": "slateblue",
}


@dataclass
class DisplayConfig:
    """What/how to plot: the display groups, their colours, and each decoder's targets."""

    display_markers: list[str]
    colors: dict
    task_pos_markers: dict[str, list[str]]
    marker_groups: dict[str, list[str]]
    code_to_group: dict[int, str]
    raw_markers: list[str]

    def is_target(self, task: str, group: str) -> bool:
        """True if any of ``group``'s raw markers is a positive label of ``task``."""
        pos = self.task_pos_markers.get(task, [])
        return any(m in pos for m in self.marker_groups.get(group, [group]))

    def target_group(self, task: str) -> str | None:
        g = [g for g in self.display_markers if self.is_target(task, g)]
        return g[0] if g else None


def display_config(ctx, *, marker_groups=None, markers_of_interest=None) -> DisplayConfig:
    """Build the :class:`DisplayConfig` for a profile's decoders.

    ``marker_groups`` pools raw markers into named display groups (e.g. colour vs
    scene); otherwise each positive label is its own group (identity).
    """
    tasks = ctx.settings.get_decoder_settings()["tasks"]
    if marker_groups:
        mg = {g: list(ms) for g, ms in marker_groups.items()}
    else:
        if markers_of_interest is None:
            markers_of_interest = list(dict.fromkeys(
                lbl for t in tasks for lbl in t["pos_labels"]))
        mg = {m: [m] for m in markers_of_interest}
    display = list(mg)
    return DisplayConfig(
        display_markers=display,
        colors={m: _KNOWN_COLORS.get(m, plt.cm.tab10(i % 10)) for i, m in enumerate(display)},
        task_pos_markers={t["name"]: list(t["pos_labels"]) for t in tasks},
        marker_groups=mg,
        code_to_group={ctx.event_mapping[m]: g for g, ms in mg.items() for m in ms},
        raw_markers=list(dict.fromkeys(m for ms in mg.values() for m in ms)),
    )


def _grid(n, ncols=3):
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False)
    return fig, axes, nrows, ncols


def _blank(axes, n, nrows, ncols):
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")


def cv_auc(ctx):
    """Cross-validated diagonal AUC per decoder (the honest metric). Returns eval_results."""
    import joblib

    ev = joblib.load(ctx.profile.snapshot_paths["eval"])["_eval_results"]
    items = list(ev["tasks"].items())
    fig, axes, nrows, ncols = _grid(len(items))
    for idx, (task, td) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        c = plt.cm.tab10(idx % 10)
        ax.plot(ev["times"], td["diagonal_auc"], color=c, lw=1.8)
        tp = ctx.task_tp(task)
        if tp is not None:
            ax.axvline(tp, color=c, ls=":", lw=1)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.axvline(0.0, color="k", ls=":", lw=0.8)
        ax.set(title=f"{task}  (peak {td['peak_auc']:.3f})", ylim=(0.4, 1.0),
               xlabel="time (s)", ylabel="CV AUC")
    _blank(axes, len(items), nrows, ncols)
    fig.suptitle(f"Cross-validated diagonal AUC per decoder — '{ctx.profile.name}'", y=1.02)
    plt.tight_layout(); plt.show()
    print(f"average peak AUC: {ev['average_peak_auc']:.3f} | "
          f"suggested timepoint: {ev['suggested_timepoint']:.3f}s")
    return ev


def per_decoder(ctx, dc, epoched, t_grid, preds):
    """Single-trial (faint) + mean (navy) P(t) over each decoder's own positive group(s)."""
    tasks = list(preds)
    fig, axes, nrows, ncols = _grid(len(tasks))
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        groups = [g for g in dc.display_markers if dc.is_target(task, g)]
        for group in groups:
            for row in epoched.get(task, {}).get(group, np.empty((0, len(t_grid)))):
                ax.plot(t_grid, row, color="steelblue", alpha=0.25, lw=0.8)
        pos = [epoched.get(task, {}).get(g, np.empty((0, len(t_grid)))) for g in groups]
        pos = np.vstack(pos) if pos else np.empty((0, len(t_grid)))
        if pos.shape[0]:
            ax.plot(t_grid, pos.mean(0), color="navy", lw=2.5, label=f"mean (n={pos.shape[0]})")
        ax.axvline(0, color="k", ls=":", lw=1)
        tp = ctx.task_tp(task)
        if tp is not None:
            ax.axvline(tp, color="crimson", ls="--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.set(title=f"{task} — '{'/'.join(groups)}'" + (f"  (tp {tp:.2f}s)" if tp is not None else ""),
               ylim=(0, 1), xlabel="time from marker (s)", ylabel="P(positive)")
        ax.legend(fontsize=7, loc="upper right")
    _blank(axes, len(tasks), nrows, ncols)
    plt.tight_layout(); plt.show()


def selectivity(ctx, dc, epoched, t_grid, preds):
    """Per decoder, mean ± SEM P(t) for every display group overlaid."""
    tasks = list(preds)
    fig, axes, nrows, ncols = _grid(len(tasks))
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        for name in dc.display_markers:
            ep = epoched[task][name]
            if ep.shape[0] == 0:
                continue
            mean, sem = ep.mean(0), ep.std(0) / np.sqrt(ep.shape[0])
            c = dc.colors.get(name)
            ax.plot(t_grid, mean, color=c, lw=1.8, label=f"{name} ({ep.shape[0]})")
            ax.fill_between(t_grid, mean - sem, mean + sem, color=c, alpha=0.15)
        ax.axvline(0, color="k", ls=":", lw=1)
        tp = ctx.task_tp(task)
        if tp is not None:
            ax.axvline(tp, color="black", ls="--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.set(title=task + (f"  (tp {tp:.2f}s)" if tp is not None else ""),
               ylim=(0, 1), xlabel="time from marker (s)", ylabel="P(positive)")
        ax.legend(fontsize=6, loc="upper right")
    _blank(axes, len(tasks), nrows, ncols)
    plt.tight_layout(); plt.show()


def competition(ctx, dc, epoched, t_grid, preds):
    """One panel per group; every decoder's mean overlaid, target bold + ★. Raw then ΔP."""
    tasks = list(preds)
    decoder_color = {t: dc.colors.get(dc.target_group(t)) for t in tasks}

    def _grid_plot(ep_dict, *, ylim, ref, ylabel, suffix):
        fig, axes, nrows, ncols = _grid(len(dc.display_markers))
        for idx, group in enumerate(dc.display_markers):
            ax = axes[idx // ncols][idx % ncols]
            for task in tasks:
                ep = ep_dict[task][group]
                if ep.shape[0] == 0:
                    continue
                mean = ep.mean(0)
                target = dc.is_target(task, group)
                c = decoder_color.get(task)
                ax.plot(t_grid, mean, color=c, lw=2.6 if target else 1.2,
                        alpha=1.0 if target else 0.5, label=(f"{task} ★" if target else task))
                if target:
                    sem = ep.std(0) / np.sqrt(ep.shape[0])
                    ax.fill_between(t_grid, mean - sem, mean + sem, color=c, alpha=0.15)
                    tp = ctx.task_tp(task)
                    if tp is not None:
                        ax.axvline(tp, color=c, ls="--", lw=1)
            ax.axvline(0, color="k", ls=":", lw=1)
            ax.axhline(ref, color="gray", lw=0.6)
            ax.set(title=f"group '{group}' — which decoder fires? {suffix}", ylim=ylim,
                   xlabel="time from marker (s)", ylabel=ylabel)
            ax.legend(fontsize=6, loc="upper right")
        _blank(axes, len(dc.display_markers), nrows, ncols)
        plt.tight_layout(); plt.show()

    _grid_plot(epoched, ylim=(0, 1), ref=0.5, ylabel="P(positive)", suffix="(raw P)")
    ep_bc = metrics.baseline_correct(epoched, t_grid)
    means = [ep_bc[t][g].mean(0) for t in tasks for g in dc.display_markers if ep_bc[t][g].shape[0]]
    m = float(np.abs(np.concatenate(means)).max()) if means else 0.4
    _grid_plot(ep_bc, ylim=(-1.15 * m, 1.15 * m), ref=0.0, ylabel="ΔP from baseline",
               suffix="(baseline ΔP)")


def confusion_and_perm(ctx, dc, epoched, t_grid, preds, decoder_tasks,
                       *, mode="weighted_prob", sigma=0.01, n_perm=1000):
    """Within-modality winner confusion (raw + baseline ΔP) + label-permutation bands."""
    tasks = list(preds)
    marker_of_task = {t: dc.task_pos_markers[t][0] for t in tasks}
    task_tps = {t: ctx.task_tp(t) for t in tasks}
    groups = metrics.modality_groups(decoder_tasks)
    group_markers_of_task = {tn: gm for gm, gt in groups for tn in gt}

    def _confusion_row(ep_dict, tag):
        fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.2), squeeze=False)
        for ax, (gmarkers, gtasks) in zip(axes[0], groups):
            conf = metrics.winner_confusion(ep_dict, gmarkers, gtasks, marker_of_task, t_grid,
                                            mode=mode, task_tps=task_tps, sigma=sigma)
            metrics.plot_confusion(conf, ax, gmarkers, f"{'/'.join(gmarkers)} — {tag}")
            sc = metrics.confusion_scores(conf)
            print(f"[{tag}] {'/'.join(gmarkers)}: accuracy={sc['accuracy']:.3f} "
                  f"(chance {1/len(gtasks):.2f}) | macro precision {sc['precision'].mean():.3f}, "
                  f"recall {sc['recall'].mean():.3f}")
        plt.tight_layout(); plt.show()

    _confusion_row(epoched, f"{mode}, raw")
    _confusion_row(metrics.baseline_correct(epoched, t_grid), f"{mode}, baseline ΔP")

    rng = np.random.default_rng(0)
    fig, axes, nrows, ncols = _grid(len(tasks))
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        obs, lo, hi, nmean = metrics.perm_band(epoched, task, marker_of_task[task],
                                               group_markers_of_task[task], n_perm=n_perm, rng=rng)
        ax.fill_between(t_grid, lo, hi, color="gray", alpha=0.35, label="Null 5-95 pct")
        ax.fill_between(t_grid, 0, 1, where=obs > hi, step="mid", color="green", alpha=0.18,
                        label="real > 95th")
        ax.fill_between(t_grid, 0, 1, where=obs < lo, step="mid", color="red", alpha=0.15,
                        label="real < 5th")
        ax.plot(t_grid, nmean, color="gray", ls="--", lw=1, label="null mean")
        ax.plot(t_grid, obs, color="steelblue", lw=1.8, label="real")
        ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.axvline(0, color="k", ls=":", lw=0.8)
        tp = ctx.task_tp(task)
        if tp is not None:
            ax.axvline(tp, color="crimson", ls="--", lw=1)
        ax.set(title=task, ylim=(0, 1), xlabel="time from marker (s)", ylabel="mean P(pos)")
        ax.legend(fontsize=6, loc="upper right")
    _blank(axes, len(tasks), nrows, ncols)
    plt.tight_layout(); plt.show()


def parity(ctx, dc, epoched, t_grid, preds):
    """Offline (saved epochs swept through the model) vs online P(t), per decoder. FL only."""
    import glob

    import mne

    epo_fif = sorted(glob.glob(str(ctx.profile.root_dir / "epochs" / "*epo.fif")))
    assert epo_fif, f"no offline epochs fif under {ctx.profile.root_dir / 'epochs'}"
    off = mne.read_epochs(epo_fif[0], verbose=False)
    assert len(off.times) == len(t_grid), "offline/online time grids differ — check final_resample"

    def offline_proba(task, raw_marker):
        if raw_marker not in off.event_id:
            return None
        X = off[raw_marker].get_data()
        n_ep, n_ch, n_t = X.shape
        p = ctx.artifact.models[task].predict_proba(X.transpose(0, 2, 1).reshape(-1, n_ch))[:, 1]
        return p.reshape(n_ep, n_t)

    tasks = list(preds)
    fig, axes, nrows, ncols = _grid(len(tasks))
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        off_rows = [r for r in (offline_proba(task, m) for m in dc.task_pos_markers.get(task, []))
                    if r is not None]
        on_groups = [g for g in dc.display_markers if dc.is_target(task, g)]
        on_rows = [epoched.get(task, {}).get(g, np.empty((0, len(t_grid)))) for g in on_groups]
        on_rows = np.vstack(on_rows) if on_rows else np.empty((0, len(t_grid)))
        if off_rows:
            off_all = np.vstack(off_rows)
            ax.plot(off.times, off_all.mean(0), color="darkgreen", lw=2.2,
                    label=f"offline (n={off_all.shape[0]})")
        if on_rows.shape[0]:
            ax.plot(t_grid, on_rows.mean(0), color="navy", lw=2.2, ls="--",
                    label=f"online (n={on_rows.shape[0]})")
        ax.axvline(0, color="k", ls=":", lw=1)
        tp = ctx.task_tp(task)
        if tp is not None:
            ax.axvline(tp, color="crimson", ls="--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6)
        ax.set(title=f"{task} — '{'/'.join(on_groups)}'", ylim=(0, 1),
               xlabel="time from marker (s)", ylabel="P(positive)")
        ax.legend(fontsize=7, loc="upper right")
    _blank(axes, len(tasks), nrows, ncols)
    fig.suptitle("Offline vs online P(t) on each decoder's positive trials", y=1.02)
    plt.tight_layout(); plt.show()


def fl_marker_diagnostic(ctx, raw, dc, *, n_perm=2000):
    """FL stimulus-order diagnostic: marker timeline + a transition-bias shuffle test."""
    import re

    desc_to_code = {}
    for d in set(raw.annotations.description):
        m = re.search(r"(\d+)\s*$", d)
        if m:
            desc_to_code[d] = int(m.group(1))
    import mne
    events, _ = mne.events_from_annotations(raw, event_id=desc_to_code, verbose=False)
    sfreq = float(raw.info["sfreq"])

    times_by_name = {n: np.array([s / sfreq for s, _, c in events if c == ctx.event_mapping[n]])
                     for n in dc.raw_markers}
    fig, ax = plt.subplots(figsize=(12, 3))
    for name, ts in times_by_name.items():
        ax.scatter(ts, [name] * len(ts), s=12)
    ax.set(xlabel="time in recording (s)", title="Marker timeline — blocked or interleaved?")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()

    codes = {ctx.event_mapping[n] for n in dc.raw_markers}
    order = [ctx.name_by_code[c] for _, _, c in events if c in codes]
    TYPES = sorted(set(order))
    idx = {t: k for k, t in enumerate(TYPES)}
    K = len(TYPES)

    def counts(seq):
        M = np.zeros((K, K))
        for a, b in zip(seq[:-1], seq[1:]):
            M[idx[a], idx[b]] += 1
        return M

    obs = counts(order)
    row_tot = obs.sum(1, keepdims=True)
    obs_prob = np.divide(obs, row_tot, out=np.zeros_like(obs), where=row_tot > 0)
    rng = np.random.default_rng(0)
    perm = np.empty((n_perm, K, K))
    shuf = list(order)
    for s in range(n_perm):
        rng.shuffle(shuf)
        perm[s] = counts(shuf)
    mean, std = perm.mean(0), perm.std(0)
    z = np.divide(obs - mean, std, out=np.zeros_like(obs), where=std > 0)

    def chi2(M):
        e = M.sum(1, keepdims=True) * M.sum(0, keepdims=True) / M.sum()
        return np.divide((M - e) ** 2, e, out=np.zeros_like(M), where=e > 0).sum()
    global_p = (np.array([chi2(perm[s]) for s in range(n_perm)]) >= chi2(obs)).mean()

    fig, (axp, axz) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, vmax, cmap, ttl, is_z in (
            (axp, obs_prob, max(0.01, obs_prob.max()), "viridis", "P(next | current)", False),
            (axz, z, 4, "coolwarm", "z vs shuffle null (|z|>~3 biased)", True)):
        im = ax.imshow(M, vmin=(-4 if is_z else 0), vmax=vmax, cmap=cmap)
        ax.set_xticks(range(K)); ax.set_xticklabels(TYPES, rotation=45, ha="right")
        ax.set_yticks(range(K)); ax.set_yticklabels(TYPES)
        ax.set(xlabel="next type", ylabel="current type", title=ttl)
        for r in range(K):
            for c in range(K):
                col = "white" if (is_z and abs(M[r, c]) > 2.5) else "black"
                ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=7, color=col)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.show()
    print(f"global transition test: shuffle p={global_p:.3f} "
          + ("⚠ ordering bias" if global_p < 0.05 else "✓ consistent with random order"))


def timepoint_table(ctx, dc, epoched, t_grid, preds):
    """Print mean P(positive) for each decoder × group at the decoder's own trained tp."""
    w = max(9, max(len(n) for n in dc.display_markers) + 1)
    header = "task".ljust(22) + "tp(s)".rjust(8) + "".join(n.rjust(w) for n in dc.display_markers)
    print(header); print("-" * len(header))
    for task in preds:
        tp = ctx.task_tp(task)
        if tp is None:
            continue
        ti = int(np.argmin(np.abs(t_grid - tp)))
        row = task.ljust(22) + f"{t_grid[ti]:.2f}".rjust(8)
        for name in dc.display_markers:
            ep = epoched[task][name]
            row += (f"{ep[:, ti].mean():.3f}" if ep.shape[0] else "n/a").rjust(w)
        print(row)
