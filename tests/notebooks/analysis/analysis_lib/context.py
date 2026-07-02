"""Profile loading + the analysis context shared by every analysis notebook.

Collapses the per-notebook boilerplate (repo-root walk, ``sys.path`` setup,
profile → settings/artifact/preproc/engine) into :func:`load_context`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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


@dataclass
class AnalysisContext:
    """Everything a notebook needs from one seeded profile.

    ``artifact.models`` are the *shipped* decoders (Mode A). Build a fresh
    inference engine per model set with :meth:`engine`.
    """

    repo_root: Path
    profile: Any                      # frontend.debug.profiles.DebugProfile
    settings: Any                     # SettingsManager
    epoch_tmin: float                 # from backend.core.preprocessing_constants
    epoch_tmax: float
    event_mapping: dict[str, int]
    name_by_code: dict[int, str]
    artifact: Any                     # DecoderPipelineArtifact
    preproc: Any                      # OnlinePreprocessor
    recording_dir: Path
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
        "profile": ctx.profile.name, "source": source, "max_seconds": max_seconds,
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
    out_path = ctx.profile.root_dir / fname
    joblib.dump(summary, out_path)
    return out_path


def load_context(profile_name: str, root: Optional[Path] = None) -> AnalysisContext:
    """Resolve a debug profile into a fully-built :class:`AnalysisContext`."""
    repo_root = bootstrap()

    from backend.core import preprocessing_constants as pc
    from backend.core.settings_manager import SettingsManager
    from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact
    from backend.online_phase.online_preprocessor import OnlinePreprocessor
    from frontend.debug.profiles import load_profile

    root = root or (repo_root / "debug_snapshots")
    profile = load_profile(profile_name, root=root)

    settings = SettingsManager(profile.config_path)
    # The preprocessing recipe (incl. the epoch window) is hardcoded in
    # ``backend.core.preprocessing_constants`` — read it straight from there.
    event_mapping = settings.get_event_mapping()

    artifact = load_decoder_pipeline_artifact(profile.pipeline_path)
    preproc = OnlinePreprocessor(artifact.online_state)

    recording_dir = profile.raw_data_dir
    if not recording_dir.is_absolute():
        recording_dir = (repo_root / recording_dir).resolve()

    return AnalysisContext(
        repo_root=repo_root,
        profile=profile,
        settings=settings,
        epoch_tmin=pc.EPOCH_TMIN,
        epoch_tmax=pc.EPOCH_TMAX,
        event_mapping=event_mapping,
        name_by_code={v: k for k, v in event_mapping.items()},
        artifact=artifact,
        preproc=preproc,
        recording_dir=recording_dir,
        _tp_by_task=artifact.metadata.get("decoding_timepoints") or {},
        _default_tp=artifact.metadata.get("decoding_timepoint"),
    )
