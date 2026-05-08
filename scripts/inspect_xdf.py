from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyxdf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an XDF recording used for home LSL testing.")
    parser.add_argument("xdf_path", type=Path, help="Path to the XDF file.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    streams, _ = pyxdf.load_xdf(str(args.xdf_path))
    print(f"File: {args.xdf_path}")
    print(f"Streams: {len(streams)}")

    for index, stream in enumerate(streams):
        info = stream["info"]
        time_series = np.asarray(stream["time_series"])
        n_samples = len(time_series)
        n_channels = time_series.shape[1] if time_series.ndim == 2 else 0

        print()
        print(f"Stream {index}")
        print(f"  name: {info['name'][0]}")
        print(f"  type: {info['type'][0]}")
        print(f"  nominal_srate: {info['nominal_srate'][0]}")
        print(f"  declared_channels: {info['channel_count'][0]}")
        print(f"  samples: {n_samples}")
        print(f"  inferred_channels: {n_channels}")

        if n_channels >= 1:
            trigger_channel = time_series[:, -1].astype(np.int64)
            decoded = (trigger_channel >> 8) & 0xFF
            nonzero_triggers = np.unique(decoded[decoded > 0])
            print(f"  nonzero_decoded_triggers: {nonzero_triggers.tolist()}")
            print(f"  nonzero_trigger_samples: {int((decoded > 0).sum())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
