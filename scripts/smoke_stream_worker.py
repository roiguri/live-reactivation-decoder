from __future__ import annotations

import argparse
import csv
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
        default=Path("/tmp/smoke.csv"),
        help="CSV output path for PredictionLogger.",
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
    return parser


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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


def _configure_receiver_for_smoke(
    live: LiveStreamSession,
    *,
    stream_name: str,
    stream_type: str,
    resolve_timeout_sec: float,
    launch_proxy: bool,
) -> None:
    # Tool-only bridge until Phase 2 runtime LSL settings live in config.
    receiver = live._receiver
    receiver.stream_name = stream_name
    receiver.stream_type = stream_type
    receiver.resolve_timeout_sec = resolve_timeout_sec
    receiver.launch_proxy = launch_proxy


def _summarize_csv(log_path: Path) -> tuple[int, bool, int, list[str]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Prediction log was not created: {log_path}")

    with log_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return 0, False, 0, []

    header = rows[0]
    data_rows = rows[1:]
    timestamps = np.asarray([float(row[0]) for row in data_rows], dtype=float)
    marker_count = sum(1 for row in data_rows if len(row) > 1 and row[1] != "")
    monotonic = bool(timestamps.size <= 1 or np.all(np.diff(timestamps) >= 0))
    return len(data_rows), monotonic, marker_count, header


def main() -> int:
    args = build_arg_parser().parse_args()
    replay_process: subprocess.Popen | None = None
    live: LiveStreamSession | None = None

    try:
        if args.replay_xdf is not None:
            xdf_path = args.replay_xdf.resolve()
            if not xdf_path.exists():
                raise FileNotFoundError(f"Replay XDF file not found: {xdf_path}")
            print(f"Starting replay subprocess from {xdf_path}")
            replay_process = _spawn_replay_process(xdf_path, args.stream_name)
            time.sleep(args.replay_startup_wait)

        log_path = args.log.resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            log_path.unlink()

        session = AppSession(args.config)
        live = session.build_live_stream_session(
            decoder_pipeline_path=args.pipeline,
            log_path=log_path,
            batch_size_samples=args.batch_size,
        )
        _configure_receiver_for_smoke(
            live,
            stream_name=args.stream_name,
            stream_type=args.stream_type,
            resolve_timeout_sec=args.resolve_timeout,
            launch_proxy=args.launch_proxy,
        )

        print(
            f"Starting live stream session for {args.duration:.1f}s "
            f"(stream={args.stream_name!r}, type={args.stream_type!r})"
        )
        live.start()
        time.sleep(args.duration)
        live.stop()
        live = None

        row_count, monotonic, marker_count, header = _summarize_csv(log_path)
        print()
        print("Smoke Test Summary")
        print(f"  log_path: {log_path}")
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
        _stop_process(replay_process)


if __name__ == "__main__":
    raise SystemExit(main())
