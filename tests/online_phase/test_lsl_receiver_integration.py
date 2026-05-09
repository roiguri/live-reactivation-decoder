from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from backend.online_phase.lsl_receiver import LSLReceiver


@pytest.mark.skipif(
    os.environ.get("RUN_LSL_INTEGRATION") != "1",
    reason="Set RUN_LSL_INTEGRATION=1 to run the real LSL integration test.",
)
def test_lsl_receiver_can_connect_to_replayed_xdf_stream():
    project_root = Path(__file__).resolve().parents[2]
    replay_script = project_root / "scripts" / "replay_xdf_to_lsl.py"
    recording_path = project_root / "scripts" / "recordings" / "eeg_recording_with_trigger.xdf"
    stream_name = f"NeuroneStream_test_{uuid.uuid4().hex[:8]}"

    replay_process = subprocess.Popen(
        [
            sys.executable,
            str(replay_script),
            str(recording_path),
            "--stream-name",
            stream_name,
            "--repeat",
        ],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    receiver = LSLReceiver(
        stream_name=stream_name,
        resolve_timeout_sec=10.0,
        pull_timeout_sec=0.0,
        launch_proxy=False,
    )

    try:
        time.sleep(1.5)
        receiver.start()

        deadline = time.monotonic() + 5.0
        total_samples = 0
        eeg_width = None
        saw_nonzero_markers = False

        while time.monotonic() < deadline and total_samples == 0:
            timestamps, eeg_chunk, markers = receiver.pull_new_data()
            total_samples += len(timestamps)
            if len(timestamps) > 0:
                eeg_width = eeg_chunk.shape[1]
                saw_nonzero_markers = saw_nonzero_markers or any(marker > 0 for marker in markers)
            else:
                time.sleep(0.05)

        assert total_samples > 0
        assert eeg_width == 64
        assert isinstance(saw_nonzero_markers, bool)
    finally:
        receiver.stop()
        if replay_process.poll() is None:
            replay_process.terminate()
            try:
                replay_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                replay_process.kill()
                replay_process.wait(timeout=3.0)
