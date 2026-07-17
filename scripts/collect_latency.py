"""Collect live-pipeline latency by real-time replay of recordings.

For each recording (any directory containing a ``.vhdr``) belonging to each
subject, this:

  1. Spawns ``replay_vhdr_to_lsl.py <dir> --no-repeat`` as a child process,
     publishing the recording as a real-time NeurOne-like LSL stream that ends
     with the recording.
  2. Builds a headless ``LiveStreamSession`` against that subject's
     ``models/decoder_pipeline.joblib`` (same construction path the app uses).
  3. Subscribes to ``LiveStreamSession.latency_ready`` and appends every emitted
     per-batch payload to ``<out>/<subject>/<recording>_latency.csv``.
  4. Runs until the replay child exits (or ``--max-seconds`` for a smoke run),
     drains briefly, then stops.

No production code is modified: latency is read off the already-emitted
``latency_ready`` signal via a ``DirectConnection`` (the same cross-thread
pattern ``LiveSessionLogger`` uses for predictions), so slots fire from the
worker thread without needing a Qt event loop.

The E2E latency (``sample_to_decision_ms``) is a real-time property, so this
must run on Windows against a real-time replay and each recording takes its own
wall-clock duration. Summarize the CSVs with ``scripts/summarize_latency.py``.

Examples
--------
Smoke (fast, one subject, 20 s cap)::

    python scripts/collect_latency.py --subjects sub_001 --max-seconds 20

Full sweep (all three subjects, whole recordings)::

    python scripts/collect_latency.py
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.session import AppSession, LiveStreamSession  # noqa: E402

DEFAULT_SUBJECTS = ["sub_001", "sub_002", "sub_003"]
STREAM_NAME = "NeuroneStream"
STREAM_TYPE = "EEG"

# Order of the per-batch latency payload written to CSV. Mirrors the dict
# emitted by StreamWorker.latency_ready; a leading wall_time is added here.
LATENCY_FIELDS = [
    "sample_to_decision_ms",
    "total_ms",
    "pull_ms",
    "accumulation_ms",
    "preprocessing_ms",
    "inference_ms",
    "emit_ms",
    "pending_samples",
    "marker_count",
]


class LatencyCsvSink:
    """Append each ``latency_ready`` payload as one CSV row.

    Called from the StreamWorker thread via a DirectConnection, so only this
    thread touches the file after construction — no locking needed.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["wall_time", *LATENCY_FIELDS])
        self.rows = 0

    def write(self, payload: dict) -> None:
        row = [payload.get(field) for field in LATENCY_FIELDS]
        self._writer.writerow([time.time(), *row])
        self.rows += 1

    def close(self) -> None:
        self._file.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root directory holding the per-subject folders (default: ./data).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subject folder names under --data-root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "docs" / "project_docs" / "latency",
        help="Where per-recording latency CSVs are written.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["task"],
        help="Only replay recording folders whose name is in this list "
        "(default: task, the live-deployment recording). Pass a different name "
        "or several to include others (e.g. task functinal_localizer).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Raw-rate samples per StreamWorker batch (default: 40).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Cap each replay at this many seconds (0 = whole recording). "
        "Use a small value for a smoke run.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after spawning the replay before connecting.",
    )
    parser.add_argument(
        "--drain-seconds",
        type=float,
        default=2.0,
        help="Seconds to keep processing after the replay stream ends.",
    )
    parser.add_argument(
        "--resolve-timeout",
        type=float,
        default=15.0,
        help="How long LSLReceiver.start() waits for the replay stream.",
    )
    return parser


def find_subject_recordings(subject_dir: Path, names: list[str]) -> list[Path]:
    """Return the recording directories (one per ``.vhdr``) to replay.

    Only directories whose name is in ``names`` are kept, so a subject's
    live-task folder can be selected without depending on the FL folder's
    inconsistent spelling. Sorted for stable ordering.
    """
    keep = set(names)
    dirs = {
        vhdr.parent
        for vhdr in subject_dir.rglob("*.vhdr")
        if vhdr.parent.name in keep
    }
    return sorted(dirs)


def resolve_config(subject_dir: Path) -> Path:
    """Prefer the subject's own config; fall back to the repo root config."""
    subject_cfg = subject_dir / "experiment_config.yaml"
    if subject_cfg.exists():
        return subject_cfg
    return PROJECT_ROOT / "experiment_config.yaml"


def spawn_replay(recording_dir: Path) -> subprocess.Popen:
    replay_script = PROJECT_ROOT / "scripts" / "replay_vhdr_to_lsl.py"
    return subprocess.Popen(
        [
            sys.executable,
            str(replay_script),
            str(recording_dir),
            "--stream-name",
            STREAM_NAME,
            "--stream-type",
            STREAM_TYPE,
            "--no-repeat",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=3.0)


def configure_receiver(live: LiveStreamSession, *, resolve_timeout: float) -> None:
    """Point the receiver at the replayed stream (matches smoke_stream_worker)."""
    receiver = live._receiver
    receiver.stream_name = STREAM_NAME
    receiver.stream_type = STREAM_TYPE
    receiver.resolve_timeout_sec = resolve_timeout


def collect_one(
    session: AppSession,
    recording_dir: Path,
    pipeline_path: Path,
    out_csv: Path,
    args: argparse.Namespace,
) -> int:
    """Replay one recording and capture its latency CSV. Returns rows written."""
    print(f"\n=== {recording_dir} ===")
    replay = spawn_replay(recording_dir)
    live: LiveStreamSession | None = None
    sink = LatencyCsvSink(out_csv)
    try:
        time.sleep(args.startup_wait)
        if replay.poll() is not None:
            out, err = replay.communicate(timeout=1.0)
            raise RuntimeError(
                f"Replay exited before connect (code {replay.returncode}).\n"
                f"stdout:\n{out}\nstderr:\n{err}"
            )

        live = session.build_live_stream_session(
            decoder_pipeline_path=pipeline_path,
            log_dir=None,
            batch_size_samples=args.batch_size,
        )
        configure_receiver(live, resolve_timeout=args.resolve_timeout)
        live.latency_ready.connect(sink.write, Qt.ConnectionType.DirectConnection)

        live.start()
        started = time.perf_counter()
        capped = args.max_seconds > 0
        # Run until the replay stream ends (child exits) or the cap is hit.
        while replay.poll() is None:
            if capped and (time.perf_counter() - started) >= args.max_seconds:
                break
            time.sleep(0.2)
        time.sleep(args.drain_seconds)  # let buffered samples flush through
        live.stop()
        live = None
    finally:
        if live is not None:
            live.stop()
        stop_process(replay)
        sink.close()

    print(f"  wrote {sink.rows} latency rows -> {out_csv}")
    return sink.rows


def main() -> int:
    args = build_arg_parser().parse_args()
    total_rows = 0
    processed = 0
    for subject in args.subjects:
        subject_dir = args.data_root / subject
        if not subject_dir.is_dir():
            print(f"SKIP {subject}: {subject_dir} not found")
            continue
        pipeline_path = subject_dir / "models" / "decoder_pipeline.joblib"
        if not pipeline_path.exists():
            print(f"SKIP {subject}: no decoder pipeline at {pipeline_path}")
            continue

        recordings = find_subject_recordings(subject_dir, args.recordings)
        if not recordings:
            print(
                f"SKIP {subject}: no {args.recordings} .vhdr recordings under {subject_dir}"
            )
            continue

        session = AppSession(resolve_config(subject_dir))
        for recording_dir in recordings:
            out_csv = args.out_root / subject / f"{recording_dir.name}_latency.csv"
            try:
                total_rows += collect_one(
                    session, recording_dir, pipeline_path, out_csv, args
                )
                processed += 1
            except Exception as exc:  # keep going across recordings
                print(f"  FAILED {recording_dir}: {exc}")

    print(f"\nDone: {processed} recording(s), {total_rows} latency rows total.")
    print(f"Summarize with: python scripts/summarize_latency.py --in-root {args.out_root}")
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
