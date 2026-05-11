from __future__ import annotations

import argparse
import time
from pathlib import Path

import mne
import numpy as np
import pyxdf
from mne_lsl.player import PlayerLSL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay an XDF EEG recording as a live LSL stream.")
    parser.add_argument("xdf_path", type=Path, help="Path to the XDF file to replay.")
    parser.add_argument(
        "--stream-name",
        default=None,
        help="Override the emitted LSL stream name. Defaults to the original XDF stream name.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Loop the replay forever instead of playing it once.",
    )
    return parser


def load_raw_from_xdf(xdf_path: Path) -> tuple[mne.io.RawArray, str]:
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    if not streams:
        raise RuntimeError(f"No streams found in {xdf_path}.")

    stream = streams[0]
    info = stream["info"]
    time_series = np.asarray(stream["time_series"], dtype=float)
    if time_series.ndim != 2:
        raise RuntimeError(f"Expected 2D time_series in {xdf_path}, got shape {time_series.shape}.")

    data = time_series.T
    n_channels, _ = data.shape
    srate = float(info["nominal_srate"][0])
    source_stream_name = info["name"][0]

    ch_names = [f"EEG {index + 1}" for index in range(n_channels - 1)] + ["STI 014"]
    ch_types = ["eeg"] * (n_channels - 1) + ["stim"]

    mne_info = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types=ch_types)
    raw = mne.io.RawArray(data, mne_info, verbose=False)
    return raw, source_stream_name


def main() -> int:
    args = build_arg_parser().parse_args()
    raw, source_stream_name = load_raw_from_xdf(args.xdf_path)
    stream_name = args.stream_name or source_stream_name

    print(f"Loaded {args.xdf_path}")
    print(f"Replaying as LSL stream: {stream_name}")
    print(f"Channels: {raw.info['nchan']}")
    print(f"Sample rate: {raw.info['sfreq']} Hz")

    player = PlayerLSL(raw, name=stream_name, n_repeat=np.inf if args.repeat else 1)
    player.start()

    print("Stream is live. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping replay...")
    finally:
        player.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
