"""Create a short BrainVision test fixture from a real recording.

Crops a window of an existing recording and writes a new .vhdr/.vmrk/.eeg
triplet via pybv. The trigger channel is preserved unchanged so the production
orchestrator load path (and its parallel-port decoder) exercises the same code
on the fixture as on a full recording.

Usage:
    python scripts/create_test_eeg.py \\
        --input  /path/to/source/recording_dir \\
        --output /path/to/data/new_experiment/test_set \\
        --start-s 5400 --duration-s 1080
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import pybv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("create_test_eeg")

DEFAULT_INPUT = Path(
    "/home/roiguri/projects/uni/reactivation-decoder/data/new_experiment/experiment"
)
DEFAULT_OUTPUT = Path(
    "/home/roiguri/projects/uni/reactivation-decoder/data/new_experiment/test_set"
)
DEFAULT_START_S = 5400.0
DEFAULT_DURATION_S = 1080.0


def find_vhdr(input_dir: Path) -> Path:
    vhdrs = list(input_dir.glob("*.vhdr"))
    if not vhdrs:
        raise FileNotFoundError(f"No .vhdr file found in {input_dir}")
    if len(vhdrs) > 1:
        logger.warning("Multiple .vhdr files in %s; using %s", input_dir, vhdrs[0].name)
    return vhdrs[0]


def crop_and_write(
    input_dir: Path, output_dir: Path, start_s: float, duration_s: float
) -> Path:
    vhdr = find_vhdr(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s (lazy)…", vhdr)
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="WARNING")
    sfreq = raw.info["sfreq"]
    total_s = raw.times[-1]
    logger.info(
        "Source: %.0f Hz, %d channels, %.1f min total",
        sfreq, len(raw.ch_names), total_s / 60,
    )

    end_s = start_s + duration_s
    if start_s < 0 or end_s > total_s:
        raise ValueError(
            f"Window [{start_s:.0f}..{end_s:.0f}]s outside source range "
            f"[0..{total_s:.0f}]s; adjust --start-s / --duration-s"
        )

    logger.info(
        "Cropping to window [%.0f..%.0f]s (%.1f min) and preloading…",
        start_s, end_s, duration_s / 60,
    )
    raw.crop(tmin=start_s, tmax=end_s)
    raw.load_data()

    stem = vhdr.stem
    out_vhdr = output_dir / f"{stem}.vhdr"
    logger.info("Writing BrainVision triplet → %s", out_vhdr)
    pybv.write_brainvision(
        data=raw.get_data(),
        sfreq=sfreq,
        ch_names=raw.ch_names,
        fname_base=stem,
        folder_out=str(output_dir),
        overwrite=True,
    )

    out_eeg = output_dir / f"{stem}.eeg"
    size_mb = out_eeg.stat().st_size / (1024 * 1024)
    logger.info("Wrote .eeg (%.1f MB)", size_mb)
    return out_vhdr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-s", type=float, default=DEFAULT_START_S)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    out_vhdr = crop_and_write(args.input, args.output, args.start_s, args.duration_s)
    logger.info("Done. Point the app at %s", out_vhdr.parent)


if __name__ == "__main__":
    main()
