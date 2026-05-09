from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.online_phase.lsl_receiver import LSLReceiver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual smoke test for LSLReceiver.")
    parser.add_argument("--stream-name", default="NeuroneStream", help="Target LSL stream name.")
    parser.add_argument("--stream-type", default="EEG", help="Target LSL stream type.")
    parser.add_argument("--duration", type=float, default=5.0, help="How long to pull data after connection.")
    parser.add_argument(
        "--resolve-timeout",
        type=float,
        default=10.0,
        help="How long LSLReceiver.start() should wait for the stream.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Sleep time between pull_new_data() calls.",
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
        "--verbose",
        action="store_true",
        help="Print per-pull diagnostics during the smoke test.",
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


def main() -> int:
    args = build_arg_parser().parse_args()

    replay_process: subprocess.Popen | None = None
    receiver = LSLReceiver(
        stream_name=args.stream_name,
        stream_type=args.stream_type,
        resolve_timeout_sec=args.resolve_timeout,
        pull_timeout_sec=0.0,
        launch_proxy=args.launch_proxy,
    )

    total_samples = 0
    pull_count = 0
    pulls_with_data = 0
    observed_markers: list[int] = []
    chunk_sizes: list[int] = []
    timestamp_segments: list[np.ndarray] = []
    last_eeg_shape = (0, receiver.eeg_channel_count)

    try:
        if args.replay_xdf is not None:
            xdf_path = args.replay_xdf.resolve()
            if not xdf_path.exists():
                raise FileNotFoundError(f"Replay XDF file not found: {xdf_path}")
            print(f"Starting replay subprocess from {xdf_path}")
            replay_process = _spawn_replay_process(xdf_path, args.stream_name)
            time.sleep(args.replay_startup_wait)

        print(
            f"Connecting to stream name={args.stream_name!r}, "
            f"type={args.stream_type!r}, timeout={args.resolve_timeout:.1f}s"
        )
        try:
            receiver.start()
        except Exception as exc:
            print(f"Connection failed: {exc}")
            return 1

        print("Connected. Pulling data...")
        deadline = time.monotonic() + args.duration

        while time.monotonic() < deadline:
            timestamps, eeg_chunk, markers = receiver.pull_new_data()
            pull_count += 1

            if len(timestamps) > 0:
                pulls_with_data += 1
                total_samples += len(timestamps)
                chunk_sizes.append(len(timestamps))
                timestamp_segments.append(timestamps)
                observed_markers.extend(markers)
                last_eeg_shape = eeg_chunk.shape

                if args.verbose:
                    print(
                        f"pull={pull_count} samples={len(timestamps)} "
                        f"eeg_shape={eeg_chunk.shape} markers={markers}"
                    )
            elif args.verbose:
                print(f"pull={pull_count} samples=0")

            time.sleep(args.poll_interval)

        print()
        print("Smoke Test Summary")
        print(f"  total_pulls: {pull_count}")
        print(f"  pulls_with_data: {pulls_with_data}")
        print(f"  total_samples: {total_samples}")
        print(f"  last_eeg_shape: {last_eeg_shape}")
        print(f"  unique_markers: {sorted(set(observed_markers))}")

        if chunk_sizes:
            histogram = dict(sorted(Counter(chunk_sizes).items()))
            print(f"  chunk_size_histogram: {histogram}")

        if timestamp_segments:
            timestamps = np.concatenate(timestamp_segments)
            if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
                effective_srate = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                print(f"  effective_srate_hz: {effective_srate:.2f}")

        if total_samples == 0:
            print("Result: no samples received.")
            return 2

        print("Result: success.")
        return 0
    finally:
        receiver.stop()
        _stop_process(replay_process)


if __name__ == "__main__":
    raise SystemExit(main())
