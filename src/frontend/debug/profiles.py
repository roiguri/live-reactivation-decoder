"""Debug *profiles* — named, self-contained scenarios under ``debug_snapshots/``.

Dev-only. Production ``frontend.main`` does **not** import this module.

A **profile** is a directory ``debug_snapshots/<name>/`` bundling everything
needed to reproduce and run one debug scenario:

* ``manifest.yaml`` — the source of truth: ``name``, ``config`` (a path
  relative to the profile dir — the copied-in config), and ``raw_data_dir``
  (a path *only*, recording which raw recording the snapshots were built
  from, for re-seeding and for replay via ``scripts/replay_vhdr_to_lsl.py``).
* ``experiment_config.yaml`` — the config, copied in so the profile is
  self-contained.
* ``preproc_done.joblib`` / ``eval_done.joblib`` / ``train_done.joblib`` —
  the pipeline-boundary snapshots the debug screens restore.
* ``models/decoder_pipeline.joblib`` — the Phase 2 artifact.
* ``epochs/`` — saved epochs from the run.

The snapshot filenames, the pipeline path, and ``epochs/`` are **conventions**
resolved here, not manifest fields. Profiles are *discovered* by listing
subdirectories that contain a ``manifest.yaml`` — there is no central
registry to keep in sync.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

# ── directory contract (conventions, not manifest fields) ────────────────────

DEFAULT_ROOT = Path("debug_snapshots")
MANIFEST_NAME = "manifest.yaml"
CONFIG_NAME = "experiment_config.yaml"
PIPELINE_RELPATH = Path("models") / "decoder_pipeline.joblib"
SNAPSHOT_FILENAMES: dict[str, str] = {
    "preproc": "preproc_done.joblib",
    "eval": "eval_done.joblib",
    "train": "train_done.joblib",
}


@dataclass(frozen=True)
class DebugProfile:
    """A resolved debug scenario with absolute paths.

    ``config_path`` and ``raw_data_dir`` may be overridden (for one-off
    diagnostics) relative to what the manifest records; the snapshot and
    pipeline paths always follow the on-disk conventions under ``root_dir``.
    """

    name: str
    root_dir: Path
    config_path: Path
    raw_data_dir: Path
    pipeline_path: Path
    snapshot_paths: dict[str, Path]


# ── discovery + load ─────────────────────────────────────────────────────────


def list_profiles(root: Path = DEFAULT_ROOT) -> list[str]:
    """Names of every subdirectory of ``root`` that holds a ``manifest.yaml``."""
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir() if d.is_dir() and (d / MANIFEST_NAME).is_file()
    )


def load_profile(name: str, root: Path = DEFAULT_ROOT) -> DebugProfile:
    """Parse ``root/<name>/manifest.yaml`` and resolve it to a ``DebugProfile``.

    Raises ``FileNotFoundError`` if the profile or its manifest is missing,
    and ``ValueError`` if the manifest omits a required field.
    """
    profile_dir = root / name
    manifest_path = profile_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        available = list_profiles(root)
        raise FileNotFoundError(
            f"No debug profile '{name}' at {manifest_path}. "
            f"Available: {available or '(none — run the seeder)'}."
        )

    raw = yaml.safe_load(manifest_path.read_text()) or {}
    config_rel = raw.get("config")
    data_dir = raw.get("raw_data_dir")
    if not config_rel or not data_dir:
        raise ValueError(
            f"{manifest_path} must define both 'config' and 'raw_data_dir'."
        )

    return DebugProfile(
        name=raw.get("name", name),
        root_dir=profile_dir,
        config_path=(profile_dir / config_rel).resolve(),
        raw_data_dir=Path(data_dir),
        pipeline_path=(profile_dir / PIPELINE_RELPATH).resolve(),
        snapshot_paths={
            key: (profile_dir / fn).resolve()
            for key, fn in SNAPSHOT_FILENAMES.items()
        },
    )


def resolve_profile(
    name: str | None = None,
    *,
    root: Path = DEFAULT_ROOT,
    config: Path | None = None,
    data: Path | None = None,
) -> DebugProfile:
    """Select a profile for the debug entry points, applying CLI overrides.

    When ``name`` is ``None``, falls back to a profile literally named
    ``default``; if none exists, the sole profile when there is exactly one;
    otherwise raises ``ValueError`` listing the choices. ``config`` / ``data``
    override the manifest's paths on top of the resolved profile.
    """
    if name is None:
        name = _default_profile_name(root)
    profile = load_profile(name, root)
    if config is not None:
        profile = replace(profile, config_path=Path(config).resolve())
    if data is not None:
        profile = replace(profile, raw_data_dir=Path(data))
    return profile


def _default_profile_name(root: Path) -> str:
    profiles = list_profiles(root)
    if not profiles:
        raise FileNotFoundError(
            f"No debug profiles under {root}/. Run scripts/demo_seed_debug_snapshots.py "
            "to create one."
        )
    if "default" in profiles:
        return "default"
    if len(profiles) == 1:
        return profiles[0]
    raise ValueError(
        f"Multiple debug profiles {profiles}; pass --profile <name> to pick one."
    )


# ── seeder support: create / refresh a profile's manifest ────────────────────


def prepare_profile(
    name: str,
    *,
    root: Path = DEFAULT_ROOT,
    config: Path | None = None,
    data: Path | None = None,
) -> DebugProfile:
    """Create or refresh ``root/<name>/`` and return its ``DebugProfile``.

    Used by the seeder. Two modes:

    * **Bootstrap** (no manifest yet) — ``config`` and ``data`` are required.
      The config is copied into the profile dir as ``experiment_config.yaml``
      and a manifest is written.
    * **Re-seed** (manifest exists) — ``config`` / ``data`` default to the
      recorded values; either may be overridden. A passed ``config`` is
      re-copied into the profile.

    The pipeline/snapshot/epochs paths follow the directory conventions and
    are populated by the seeder run itself, not here.
    """
    profile_dir = root / name
    manifest_path = profile_dir / MANIFEST_NAME
    existing = (
        yaml.safe_load(manifest_path.read_text()) or {}
        if manifest_path.is_file()
        else {}
    )

    config_src = Path(config) if config is not None else None
    if config_src is None and existing.get("config"):
        config_src = profile_dir / existing["config"]
    data_dir = Path(data) if data is not None else None
    if data_dir is None and existing.get("raw_data_dir"):
        data_dir = Path(existing["raw_data_dir"])

    if config_src is None or data_dir is None:
        raise ValueError(
            f"Bootstrapping profile '{name}' requires both --config and --data "
            "(no manifest to read them from)."
        )
    if not config_src.is_file():
        raise FileNotFoundError(f"--config not found: {config_src}")

    profile_dir.mkdir(parents=True, exist_ok=True)
    dest_config = profile_dir / CONFIG_NAME
    # Copy the config in unless it already *is* the in-profile config.
    if config_src.resolve() != dest_config.resolve():
        shutil.copyfile(config_src, dest_config)

    manifest_path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "config": CONFIG_NAME,
                "raw_data_dir": str(data_dir.resolve()),
            },
            sort_keys=False,
        )
    )
    return load_profile(name, root)
