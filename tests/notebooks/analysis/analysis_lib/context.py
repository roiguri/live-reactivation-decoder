"""Analysis context shared by every analysis notebook.

Collapses the per-notebook boilerplate (repo-root walk, ``sys.path`` setup,
output root -> settings/artifact/preproc/engine) into :func:`load_context`.

An analysis root is any directory laid out per
``backend.core.session_paths.SessionPaths`` (``experiment_config.yaml``,
``models/decoder_pipeline.joblib``, ``epochs/``, optionally ``phase2_live/``) —
a real subject's output directory (e.g. ``data/sub_001``) qualifies directly,
with no separate profile/manifest layer.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_KNOWN_SUBDIRS = {"epochs", "evaluation", "models", "phase2_live"}


def bootstrap() -> Path:
    """Locate the repo root and put ``src/`` and the analysis dir on ``sys.path``.

    Walks up from this module until it finds ``src/backend`` so it works no
    matter where Jupyter was launched. Returns the repo root.
    """
    search = Path(__file__).resolve()
    repo_root: Optional[Path] = None
    for parent in search.parents:
        if (parent / "src" / "backend").is_dir():
            repo_root = parent
            break
    if repo_root is None:
        raise RuntimeError("Could not locate repo root (no src/backend above this file).")

    for path in (repo_root / "src", repo_root / "tests" / "notebooks" / "analysis"):
        sp = str(path)
        if path.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)
    return repo_root


def find_raw_dirs(root: Path) -> dict[str, Path]:
    """``{subdir_name: path}`` for every child of ``root`` holding a ``.vhdr``.

    Raw recording directories (the functional localizer, a held-out task
    recording, ...) aren't part of ``SessionPaths`` — they land wherever the
    session happened to write them — so this discovers them by convention
    instead of assuming a fixed name.
    """
    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and child.name not in _KNOWN_SUBDIRS and list(child.glob("*.vhdr"))
    }


def latest_live_run(phase2_live_dir: Path) -> Optional[Path]:
    """The most recent ``phase2_live/<run>/`` directory (timestamp-sorted), or ``None``."""
    if not phase2_live_dir.is_dir():
        return None
    runs = sorted(d for d in phase2_live_dir.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def _manifest_raw_dirs(root: Path) -> dict[str, Path]:
    """Raw dirs *referenced* (not physically nested) by a debug-profile manifest.

    A seeded ``debug_snapshots/<name>/`` profile doesn't copy its recordings
    in — it references them via ``manifest.yaml``'s ``raw_data_dir`` (the FL
    localizer) and, optionally, ``task_data_dir`` (a second, held-out-task
    recording — see ``frontend.debug.profiles``). Both are keyed by their own
    directory basename, matching :func:`find_raw_dirs`' physical-scan
    convention (e.g. ``raw_data_dir=.../functinal_localizer`` -> key
    ``"functinal_localizer"``), so a manifest-based root and a physically-
    nested one resolve the same way downstream. A plain output directory like
    ``data/sub_001`` has no ``manifest.yaml`` and returns ``{}`` here — it's
    expected to have its raw dirs physically nested instead.
    """
    if not (root / "manifest.yaml").is_file():
        return {}
    from frontend.debug.profiles import load_profile

    profile = load_profile(root.name, root=root.parent)
    dirs = {profile.raw_data_dir.name: profile.raw_data_dir}
    if profile.task_data_dir is not None:
        dirs[profile.task_data_dir.name] = profile.task_data_dir
    return dirs


@dataclass
class AnalysisContext:
    """Everything a notebook needs from one analysis root.

    ``artifact.models`` are the FL-trained decoders. Build a fresh inference
    engine per model set with :meth:`engine`.
    """

    repo_root: Path
    paths: Any                        # backend.core.session_paths.SessionPaths
    settings: Any                     # SettingsManager
    epoch_tmin: float                 # from backend.core.preprocessing_constants
    epoch_tmax: float
    event_mapping: dict[str, int]
    name_by_code: dict[int, str]
    artifact: Any                     # DecoderPipelineArtifact
    preproc: Any                      # OnlinePreprocessor
    raw_dirs: dict[str, Path]         # e.g. {"functinal_localizer": ..., "task": ...}
    live_run_dir: Optional[Path]      # latest (or chosen) phase2_live/<run>/
    _tp_by_task: dict[str, float] = field(default_factory=dict)
    _default_tp: Optional[float] = None

    def engine(self, models: Optional[dict[str, Any]] = None):
        """A LiveInferenceEngine over ``models`` (default: the shipped models)."""
        from backend.online_phase.live_inference import LiveInferenceEngine

        if models is None:
            return LiveInferenceEngine(self.artifact.models, self.artifact.metadata)
        return LiveInferenceEngine(models)

    def task_tp(self, task: str) -> Optional[float]:
        """The decoding timepoint for ``task`` (per-task, falling back to shared)."""
        return self._tp_by_task.get(task, self._default_tp)


def save_run_summary(ctx, dc, epoched, t_grid, preds, eval_results, *, source, max_seconds):
    """Persist per-decoder×group trajectories + headline tables to a joblib file.

    FL runs save to ``live_summary.joblib`` (what compare_profiles reads); held-out
    task runs save to ``held_out_<source>_summary.joblib`` so they don't clobber it.
    Returns the output path.
    """
    import joblib
    import numpy as np

    def _diag_at_tp(task):
        tp = ctx.task_tp(task)
        ti = int(np.argmin(np.abs(t_grid - tp))) if tp is not None else None
        return {m: (float(epoched[task][m][:, ti].mean())
                    if (ti is not None and epoched[task][m].shape[0]) else None)
                for m in dc.display_markers}

    pos_groups = {t: [g for g in dc.display_markers if dc.is_target(t, g)] for t in preds}
    summary = {
        "root": str(ctx.paths.root), "source": source, "max_seconds": max_seconds,
        "marker_groups": dc.marker_groups, "markers": list(dc.display_markers),
        "tasks": list(preds),
        "task_pos_marker": {t: (pos_groups[t][0] if pos_groups[t] else None) for t in preds},
        "task_pos_markers": pos_groups,
        "tp_by_task": {t: ctx.task_tp(t) for t in preds}, "t_grid": t_grid,
        "pt_mean": {t: {m: (epoched[t][m].mean(0) if epoched[t][m].shape[0] else None)
                        for m in dc.display_markers} for t in preds},
        "pt_sem": {t: {m: (epoched[t][m].std(0) / np.sqrt(epoched[t][m].shape[0])
                           if epoched[t][m].shape[0] else None)
                       for m in dc.display_markers} for t in preds},
        "diag_at_tp": {t: _diag_at_tp(t) for t in preds},
        "n_epochs": {m: int(epoched[next(iter(preds))][m].shape[0]) for m in dc.display_markers},
        "auc_times": eval_results["times"],
        "diagonal_auc": {t: td["diagonal_auc"] for t, td in eval_results["tasks"].items()},
        "peak_auc": {t: float(td["peak_auc"]) for t, td in eval_results["tasks"].items()},
        "average_peak_auc": float(eval_results["average_peak_auc"]),
        "suggested_timepoint": float(eval_results["suggested_timepoint"]),
    }
    fname = "live_summary.joblib" if source == "fl" else f"held_out_{source}_summary.joblib"
    out_path = ctx.paths.root / fname
    joblib.dump(summary, out_path)
    return out_path


@dataclass
class OfflineContext:
    """Settings + paths only -- no trained decoder artifact required.

    Unlike :class:`AnalysisContext`, this never touches
    ``models/decoder_pipeline.joblib`` or builds an ``OnlinePreprocessor`` --
    for analyses (pure offline CV, permutation tests, spatial patterns) that
    only need the config and saved epochs and refit everything from scratch,
    so they shouldn't require a subject to have already been through Phase 1
    training just to be analyzed.
    """

    repo_root: Path
    paths: Any                        # backend.core.session_paths.SessionPaths
    settings: Any                     # SettingsManager

    def task_tp(self, task: str) -> Optional[float]:
        """Always ``None`` -- there is no frozen decoder here, so no 'trained
        timepoint' to mark (only the CV-established peak makes sense)."""
        return None


def load_offline_context(root: str | Path) -> OfflineContext:
    """Resolve settings + paths only, skipping the trained-decoder artifact.

    Still compatible with anything that only needs ``ctx.paths``,
    ``ctx.settings``, and ``ctx.task_tp`` (e.g. :func:`plots.cv_auc`).
    """
    repo_root = bootstrap()

    from backend.core.session_paths import SessionPaths
    from backend.core.settings_manager import SettingsManager

    root = Path(root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    paths = SessionPaths(root)
    settings = SettingsManager(paths.experiment_config_path)

    return OfflineContext(repo_root=repo_root, paths=paths, settings=settings)


def load_context(root: str | Path, *, live_run: str | Path | None = None) -> AnalysisContext:
    """Resolve an analysis root (a ``SessionPaths``-shaped output dir) into an :class:`AnalysisContext`."""
    repo_root = bootstrap()

    from backend.core import preprocessing_constants as pc
    from backend.core.session_paths import SessionPaths
    from backend.core.settings_manager import SettingsManager
    from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
    from backend.online_phase.online_preprocessor import OnlinePreprocessor

    root = Path(root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    paths = SessionPaths(root)

    settings = SettingsManager(paths.experiment_config_path)
    # The preprocessing recipe (incl. the epoch window) is hardcoded in
    # ``backend.core.preprocessing_constants`` — read it straight from there.
    event_mapping = settings.get_event_mapping()

    artifact = load_decoder_pipeline_artifact(paths.decoder_pipeline_path)
    preproc = OnlinePreprocessor(artifact.online_state)

    if live_run is not None:
        live_run_dir = Path(live_run)
        if not live_run_dir.is_absolute():
            live_run_dir = paths.phase2_live_dir / live_run_dir
    else:
        live_run_dir = latest_live_run(paths.phase2_live_dir)

    # Physically-nested raw dirs win; a manifest-referenced one only fills gaps
    # (e.g. a debug snapshot's task_data_dir, never copied into the snapshot).
    raw_dirs = _manifest_raw_dirs(root) | find_raw_dirs(root)

    return AnalysisContext(
        repo_root=repo_root,
        paths=paths,
        settings=settings,
        epoch_tmin=pc.EPOCH_TMIN,
        epoch_tmax=pc.EPOCH_TMAX,
        event_mapping=event_mapping,
        name_by_code={v: k for k, v in event_mapping.items()},
        artifact=artifact,
        preproc=preproc,
        raw_dirs=raw_dirs,
        live_run_dir=live_run_dir,
        _tp_by_task=artifact.metadata.get("decoding_timepoints") or {},
        _default_tp=artifact.metadata.get("decoding_timepoint"),
    )
