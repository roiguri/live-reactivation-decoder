"""Capture and restore ``OfflineOrchestrator`` state for the debug screen.

Dev-only. Production ``frontend.main`` does **not** import this module.

A snapshot is a small joblib pickle holding a subset of the
orchestrator's stateful attributes (``_data_dir``, optionally ``_raw``,
``_epochs``, ``_eval_results``, ``online_state``) — enough to let the
debug screen drop the operator straight into a downstream view as if
the upstream pipeline had just finished. The bulky ``_preprocessor``
instance is intentionally **not** snapshotted: it pickles awkwardly
across MNE versions and the screens we care about only need
``_epochs`` and the result dicts.

The seeder script (``scripts/demo_seed_debug_snapshots.py``) runs the
real pipeline once and writes snapshots; the debug screen reads them.
Snapshots live under a git-ignored ``debug_snapshots/`` directory so
they never end up in source control.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:  # pragma: no cover — type-check only
    from backend.offline_phase.orchestrator import OfflineOrchestrator


# Orchestrator attributes we round-trip. Order matters only for the
# phase inference below; restore order is irrelevant.
_ATTRS: tuple[str, ...] = (
    "_data_dir",
    "_raw",
    "_epochs",
    "_eval_results",
    "online_state",
)


def save_snapshot(
    orchestrator: "OfflineOrchestrator",
    path: Path | str,
    *,
    include_raw: bool = False,
) -> Path:
    """Dump the orchestrator's pickleable state to ``path``.

    Args:
        orchestrator: An ``OfflineOrchestrator`` with state populated up
            to some phase boundary.
        path: Output file path. Parent directories are created.
        include_raw: If ``False`` (default), the (potentially multi-GB)
            ``_raw`` is omitted. Downstream debug screens don't need
            it; the seeder skips it by default.

    Returns:
        The resolved output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    for name in _ATTRS:
        if name == "_raw" and not include_raw:
            continue
        if not hasattr(orchestrator, name):
            continue
        value = getattr(orchestrator, name)
        # Omit unpopulated state so the inferred phase reflects reality
        # (an empty ``online_state = {}`` shouldn't read as "train_done").
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        payload[name] = value

    payload["_phase"] = _infer_phase(payload)
    joblib.dump(payload, out)
    return out


def load_snapshot(
    orchestrator: "OfflineOrchestrator",
    path: Path | str,
) -> dict[str, Any]:
    """Restore previously-saved state onto ``orchestrator`` in-place.

    Returns the snapshot dict so the caller can pass the eval/train
    result fields directly to the corresponding view's ``_on_*_done``
    slot (the slot does UI-side state mutation that the load must
    replay).
    """
    payload: dict[str, Any] = joblib.load(Path(path))
    for name in _ATTRS:
        if name in payload:
            setattr(orchestrator, name, payload[name])
    return payload


def _infer_phase(payload: dict[str, Any]) -> str:
    """Coarse tag describing how far along the pipeline this snapshot is."""
    if payload.get("online_state"):
        return "train_done"
    if "_eval_results" in payload:
        return "eval_done"
    if "_epochs" in payload:
        return "preproc_done"
    if "_raw" in payload:
        return "load_done"
    return "unknown"
