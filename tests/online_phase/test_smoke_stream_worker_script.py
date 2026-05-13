from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_stream_worker.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_stream_worker", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_process_output_includes_stdout_and_stderr():
    smoke = _load_smoke_module()

    output = smoke._format_process_output("loaded\n", "missing dependency\n")

    assert "stdout:\nloaded" in output
    assert "stderr:\nmissing dependency" in output


def test_parser_supports_preflight_only_flag():
    smoke = _load_smoke_module()

    args = smoke.build_arg_parser().parse_args([
        "--pipeline",
        "decoder_pipeline.joblib",
        "--preflight-only",
    ])

    assert args.preflight_only is True


def test_validate_decoder_pipeline_contract_explains_artifact_mismatch(monkeypatch):
    smoke = _load_smoke_module()

    def _raise_value_error(_path):
        raise ValueError("missing required key: metadata")

    monkeypatch.setattr(smoke, "load_decoder_pipeline_artifact", _raise_value_error)

    with pytest.raises(RuntimeError) as exc_info:
        smoke._validate_decoder_pipeline_contract(Path("decoder_pipeline.joblib"))

    message = str(exc_info.value)
    assert "Phase 2 contract" in message
    assert "'models', 'online_state', and 'metadata'" in message
    assert "flat Phase 1 online_state exports" in message
    assert "missing required key: metadata" in message


def test_validate_replay_dependencies_reports_missing_packages(monkeypatch):
    smoke = _load_smoke_module()

    monkeypatch.setattr(smoke.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError) as exc_info:
        smoke._validate_replay_dependencies()

    message = str(exc_info.value)
    assert "Replay mode requires missing Python package" in message
    assert "pyxdf" in message
    assert "mne_lsl.player" in message


def test_run_preflight_validates_config_pipeline_and_replay_inputs(tmp_path, monkeypatch):
    smoke = _load_smoke_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_info: {name: test}\n")
    xdf_path = tmp_path / "recording.xdf"
    xdf_path.write_bytes(b"xdf")
    calls = []

    monkeypatch.setattr(smoke, "AppSession", lambda path: calls.append(("config", path)))
    monkeypatch.setattr(
        smoke,
        "_validate_decoder_pipeline_contract",
        lambda path: calls.append(("pipeline", path)),
    )
    monkeypatch.setattr(
        smoke,
        "_validate_replay_dependencies",
        lambda: calls.append(("replay_deps", None)),
    )

    smoke._run_preflight(
        SimpleNamespace(
            config=config_path,
            pipeline=Path("decoder_pipeline.joblib"),
            replay_xdf=xdf_path,
        )
    )

    assert calls == [
        ("config", config_path),
        ("pipeline", Path("decoder_pipeline.joblib")),
        ("replay_deps", None),
    ]


def test_run_preflight_rejects_missing_replay_file(tmp_path, monkeypatch):
    smoke = _load_smoke_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_info: {name: test}\n")

    monkeypatch.setattr(smoke, "AppSession", lambda _path: None)
    monkeypatch.setattr(smoke, "_validate_decoder_pipeline_contract", lambda _path: None)

    with pytest.raises(FileNotFoundError, match="Replay XDF file not found"):
        smoke._run_preflight(
            SimpleNamespace(
                config=config_path,
                pipeline=Path("decoder_pipeline.joblib"),
                replay_xdf=tmp_path / "missing.xdf",
            )
        )


def test_ensure_replay_process_running_reports_early_child_failure():
    smoke = _load_smoke_module()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('replay stdout'); "
                "print('replay stderr', file=sys.stderr); "
                "sys.exit(7)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.wait(timeout=3.0)

    with pytest.raises(RuntimeError) as exc_info:
        smoke._ensure_replay_process_running(process)

    message = str(exc_info.value)
    assert "return code 7" in message
    assert "stdout:\nreplay stdout" in message
    assert "stderr:\nreplay stderr" in message


def test_ensure_replay_process_running_accepts_live_child():
    smoke = _load_smoke_module()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        smoke._ensure_replay_process_running(process)
    finally:
        smoke._stop_process(process)
