"""Tests for ``frontend.debug.profiles`` — the debug-profile directory contract.

Profiles bind the seeder (writer of ``manifest.yaml`` + config copy) to the
debug entry points (readers). These tests guard discovery, manifest
parse/validation, default-selection, override application, and the
bootstrap/re-seed behaviour of ``prepare_profile``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from frontend.debug import profiles
from frontend.debug.profiles import (
    CONFIG_NAME,
    MANIFEST_NAME,
    DebugProfile,
    list_profiles,
    load_profile,
    prepare_profile,
    resolve_profile,
)


def _write_config(path: Path, text: str = "preprocessing: {}\n") -> Path:
    path.write_text(text)
    return path


def _seed_profile(root: Path, name: str, data_dir: str = "/data/subj") -> Path:
    """Hand-build a minimal profile dir (manifest + copied config)."""
    pdir = root / name
    pdir.mkdir(parents=True)
    _write_config(pdir / CONFIG_NAME)
    (pdir / MANIFEST_NAME).write_text(
        f"name: {name}\nconfig: {CONFIG_NAME}\nraw_data_dir: {data_dir}\n"
    )
    return pdir


# ── discovery ────────────────────────────────────────────────────────────────


def test_list_profiles_finds_only_manifest_dirs(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "alpha")
    _seed_profile(tmp_path, "beta")
    (tmp_path / "not_a_profile").mkdir()  # no manifest → ignored
    (tmp_path / "stray.txt").write_text("x")  # not a dir → ignored
    assert list_profiles(tmp_path) == ["alpha", "beta"]


def test_list_profiles_missing_root_is_empty(tmp_path: Path) -> None:
    assert list_profiles(tmp_path / "nope") == []


# ── load + validation ────────────────────────────────────────────────────────


def test_load_profile_resolves_conventions(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "alpha", data_dir="/data/alpha")
    prof = load_profile("alpha", tmp_path)

    assert isinstance(prof, DebugProfile)
    assert prof.name == "alpha"
    assert prof.config_path == (tmp_path / "alpha" / CONFIG_NAME).resolve()
    assert prof.raw_data_dir == Path("/data/alpha")
    assert prof.pipeline_path == (
        tmp_path / "alpha" / "models" / "decoder_pipeline.joblib"
    ).resolve()
    assert set(prof.snapshot_paths) == {"preproc", "eval", "train"}
    assert prof.snapshot_paths["eval"].name == "eval_done.joblib"


def test_load_profile_task_data_dir_absent_is_none(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "alpha")  # no task_data_dir key
    assert load_profile("alpha", tmp_path).task_data_dir is None


def test_load_profile_task_data_dir_present(tmp_path: Path) -> None:
    pdir = _seed_profile(tmp_path, "alpha")
    manifest = pdir / MANIFEST_NAME
    manifest.write_text(manifest.read_text() + "task_data_dir: /data/alpha/task\n")
    assert load_profile("alpha", tmp_path).task_data_dir == Path("/data/alpha/task")


def test_load_profile_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile("ghost", tmp_path)


def test_load_profile_incomplete_manifest_raises(tmp_path: Path) -> None:
    pdir = tmp_path / "broken"
    pdir.mkdir()
    (pdir / MANIFEST_NAME).write_text("name: broken\n")  # no config / raw_data_dir
    with pytest.raises(ValueError):
        load_profile("broken", tmp_path)


# ── default selection + overrides ────────────────────────────────────────────


def test_resolve_prefers_default_name(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "default")
    _seed_profile(tmp_path, "other")
    assert resolve_profile(None, root=tmp_path).name == "default"


def test_resolve_sole_profile(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "only")
    assert resolve_profile(None, root=tmp_path).name == "only"


def test_resolve_ambiguous_raises(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "a")
    _seed_profile(tmp_path, "b")
    with pytest.raises(ValueError):
        resolve_profile(None, root=tmp_path)


def test_resolve_no_profiles_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_profile(None, root=tmp_path)


def test_resolve_applies_overrides(tmp_path: Path) -> None:
    _seed_profile(tmp_path, "alpha", data_dir="/data/alpha")
    override_cfg = _write_config(tmp_path / "override.yaml")
    prof = resolve_profile(
        "alpha", root=tmp_path, config=override_cfg, data=Path("/elsewhere")
    )
    assert prof.config_path == override_cfg.resolve()
    assert prof.raw_data_dir == Path("/elsewhere")


# ── prepare_profile: bootstrap + re-seed ─────────────────────────────────────


def test_prepare_bootstrap_copies_config_and_writes_manifest(tmp_path: Path) -> None:
    src_cfg = _write_config(tmp_path / "src_config.yaml", "preprocessing: {x: 1}\n")
    data = tmp_path / "rec"
    data.mkdir()

    prof = prepare_profile("new", root=tmp_path, config=src_cfg, data=data)

    copied = tmp_path / "new" / CONFIG_NAME
    assert copied.is_file()
    assert copied.read_text() == "preprocessing: {x: 1}\n"
    assert prof.config_path == copied.resolve()
    assert prof.raw_data_dir == data.resolve()
    assert (tmp_path / "new" / MANIFEST_NAME).is_file()


def test_prepare_bootstrap_without_config_or_data_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prepare_profile("new", root=tmp_path)  # nothing to bootstrap from


def test_prepare_reseed_reuses_manifest_values(tmp_path: Path) -> None:
    src_cfg = _write_config(tmp_path / "src_config.yaml")
    data = tmp_path / "rec"
    data.mkdir()
    prepare_profile("p", root=tmp_path, config=src_cfg, data=data)

    # Re-seed with no overrides: config + data come from the manifest.
    prof = prepare_profile("p", root=tmp_path)
    assert prof.raw_data_dir == data.resolve()
    assert prof.config_path == (tmp_path / "p" / CONFIG_NAME).resolve()


def test_prepare_reseed_data_override(tmp_path: Path) -> None:
    src_cfg = _write_config(tmp_path / "src_config.yaml")
    data = tmp_path / "rec"
    data.mkdir()
    prepare_profile("p", root=tmp_path, config=src_cfg, data=data)

    new_data = tmp_path / "rec2"
    new_data.mkdir()
    prof = prepare_profile("p", root=tmp_path, data=new_data)
    assert prof.raw_data_dir == new_data.resolve()


def test_prepare_without_task_data_leaves_it_unset(tmp_path: Path) -> None:
    src_cfg = _write_config(tmp_path / "src_config.yaml")
    data = tmp_path / "rec"
    data.mkdir()
    prof = prepare_profile("p", root=tmp_path, config=src_cfg, data=data)
    assert prof.task_data_dir is None
    assert "task_data_dir" not in (tmp_path / "p" / MANIFEST_NAME).read_text()


def test_prepare_bootstrap_with_task_data(tmp_path: Path) -> None:
    src_cfg = _write_config(tmp_path / "src_config.yaml")
    data = tmp_path / "rec"
    data.mkdir()
    task_data = tmp_path / "task_rec"
    task_data.mkdir()

    prof = prepare_profile("p", root=tmp_path, config=src_cfg, data=data, task_data=task_data)
    assert prof.task_data_dir == task_data.resolve()

    # Re-seed with no task_data override: keeps the recorded value.
    reseeded = prepare_profile("p", root=tmp_path)
    assert reseeded.task_data_dir == task_data.resolve()
