from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.session import AppSession, LiveStreamSession
from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless smoke test for StreamWorker.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "experiment_config.yaml",
        help="Experiment config YAML used to construct AppSession.",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        required=True,
        help="Path to decoder_pipeline.joblib produced by Phase 1.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="How long to run the live stream session.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("/tmp/smoke_logs"),
        help="Run directory for LiveSessionLogger output "
             "(predictions.csv / markers.csv / manifest.json / predictions.npz).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Raw-rate samples per StreamWorker batch.",
    )
    parser.add_argument(
        "--stream-name",
        default="NeuroneStream",
        help="Target LSL stream name for this smoke run.",
    )
    parser.add_argument(
        "--stream-type",
        default="EEG",
        help="Target LSL stream type for this smoke run.",
    )
    parser.add_argument(
        "--resolve-timeout",
        type=float,
        default=10.0,
        help="How long LSLReceiver.start() should wait for the stream.",
    )
    parser.add_argument(
        "--launch-proxy",
        action="store_true",
        help="Allow LSLReceiver to launch LSLProxy.exe.",
    )
    parser.add_argument(
        "--replay-xdf",
        type=Path,
        default=None,
        help="Optional XDF file to replay as a temporary LSL stream in a child process.",
    )
    parser.add_argument(
        "--replay-startup-wait",
        type=float,
        default=1.5,
        help="How long to wait after starting the replay subprocess before connecting.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config, artifact contract, and replay inputs without starting LSL.",
    )
    return parser


def _validate_replay_dependencies() -> None:
    missing = [
        module_name
        for module_name in ("pyxdf", "mne_lsl.player")
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        raise RuntimeError(
            "Replay mode requires missing Python package(s): "
            + ", ".join(missing)
            + ". Install online_decoder/requirements-dev.txt before replay smoke tests."
        )


def _validate_decoder_pipeline_contract(pipeline_path: Path) -> None:
    try:
        load_decoder_pipeline_artifact(pipeline_path)
    except ValueError as exc:
        raise RuntimeError(
            "Decoder pipeline artifact does not match the Phase 2 contract. "
            "Expected a joblib dictionary with top-level keys 'models', "
            "'online_state', and 'metadata'. Current Phase 2 cannot run from "
            "flat Phase 1 online_state exports until that artifact handoff is "
            f"fixed or converted. Loader error: {exc}"
        ) from exc


def _run_preflight(args: argparse.Namespace) -> None:
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Instantiating AppSession validates the experiment YAML without starting
    # receivers, workers, loggers, or replay processes.
    AppSession(args.config)
    _validate_decoder_pipeline_contract(args.pipeline)

    if args.replay_xdf is not None:
        if not args.replay_xdf.exists():
            raise FileNotFoundError(f"Replay XDF file not found: {args.replay_xdf}")
        _validate_replay_dependencies()


def _spawn_replay_process(xdf_path: Path, stream_name: str) -> subprocess.Popen:
    replay_script = PROJECT_ROOT / "scripts" / "replay_xdf_to_lsl.py"
    return subprocess.Popen(
        [
            sys.executable,
            str(replay_script),
            str(xdf_path),
            "--stream-name",
            stream_name,
            "--repeat",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _format_process_output(stdout: str, stderr: str) -> str:
    sections = []
    if stdout:
        sections.append(f"stdout:\n{stdout.strip()}")
    if stderr:
        sections.append(f"stderr:\n{stderr.strip()}")
    return "\n\n".join(sections) if sections else "No subprocess output captured."


def _ensure_replay_process_running(process: subprocess.Popen) -> None:
    return_code = process.poll()
    if return_code is None:
        return

    stdout, stderr = process.communicate(timeout=1.0)
    details = _format_process_output(stdout or "", stderr or "")
    raise RuntimeError(
        f"Replay subprocess exited before the smoke run could connect "
        f"(return code {return_code}).\n{details}"
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=3.0)


def _configure_receiver_for_smoke(
    live: LiveStreamSession,
    *,
    stream_name: str,
    stream_type: str,
    resolve_timeout_sec: float,
) -> None:
    # Tool-only bridge until Phase 2 runtime LSL settings live in config.
    receiver = live._receiver
    receiver.stream_name = stream_name
    receiver.stream_type = stream_type
    receiver.resolve_timeout_sec = resolve_timeout_sec


def _summarize_csv(run_dir: Path) -> tuple[int, bool, int, list[str]]:
    predictions_path = run_dir / "predictions.csv"
    markers_path = run_dir / "markers.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction log was not created: {predictions_path}")

    with predictions_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return 0, False, 0, []

    header = rows[0]
    data_rows = rows[1:]
    timestamps = np.asarray([float(row[0]) for row in data_rows], dtype=float)
    monotonic = bool(timestamps.size <= 1 or np.all(np.diff(timestamps) >= 0))

    # Markers live in a sidecar (one row per trigger edge, no inline column).
    marker_count = 0
    if markers_path.exists():
        with markers_path.open(newline="") as f:
            marker_count = max(0, len(list(csv.reader(f))) - 1)
    return len(data_rows), monotonic, marker_count, header


def main() -> int:
    args = build_arg_parser().parse_args()
    replay_process: subprocess.Popen | None = None
    live: LiveStreamSession | None = None
    session: AppSession | None = None

    try:
        _run_preflight(args)
        if args.preflight_only:
            print("Preflight OK.")
            return 0

        if args.replay_xdf is not None:
            xdf_path = args.replay_xdf.resolve()
            print(f"Starting replay subprocess from {xdf_path}")
            replay_process = _spawn_replay_process(xdf_path, args.stream_name)
            time.sleep(args.replay_startup_wait)
            _ensure_replay_process_running(replay_process)

        run_dir = args.log.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        session = AppSession(args.config)
        live = session.build_live_stream_session(
            decoder_pipeline_path=args.pipeline,
            log_dir=run_dir,
            batch_size_samples=args.batch_size,
        )
        _configure_receiver_for_smoke(
            live,
            stream_name=args.stream_name,
            stream_type=args.stream_type,
            resolve_timeout_sec=args.resolve_timeout,
        )

        if args.launch_proxy:
            print("Launching LSL proxy...")
            session.start_stream_source()

        print(
            f"Starting live stream session for {args.duration:.1f}s "
            f"(stream={args.stream_name!r}, type={args.stream_type!r})"
        )
        live.start()
        time.sleep(args.duration)
        live.stop()
        live = None

        row_count, monotonic, marker_count, header = _summarize_csv(run_dir)
        print()
        print("Smoke Test Summary")
        print(f"  run_dir: {run_dir}")
        print(f"  row_count: {row_count}")
        print(f"  timestamps_monotonic: {monotonic}")
        print(f"  marker_count: {marker_count}")
        print(f"  header: {header}")

        if row_count == 0:
            print("Result: no prediction rows written.")
            return 2
        if not monotonic:
            print("Result: timestamps are not monotonic.")
            return 3

        print("Result: success.")
        return 0
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    finally:
        if live is not None:
            live.stop()
        if session is not None:
            session.stop_stream_source()
        _stop_process(replay_process)


if __name__ == "__main__":
    raise SystemExit(main())
