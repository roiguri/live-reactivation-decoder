"""Create a short BrainVision test fixture from a real recording.

Crops a window of an existing recording and writes a new .vhdr/.vmrk/.eeg
triplet via pybv. The trigger channel is preserved unchanged so the production
orchestrator load path (and its parallel-port decoder) exercises the same code
on the fixture as on a full recording.

Usage:
    python scripts/create_test_eeg.py \\
        --input  /path/to/source/recording_dir \\
        --output /path/to/data/new_experiment/test_set \\
        --target 25min_dense \\
        --start-s 5070 --duration-s 1500

Final write location is ``<output>/<target>/`` so multiple fixtures can live
side-by-side under one base. Omit ``--target`` to write directly to ``<output>``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
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


def _find_source_eeg(vhdr: Path) -> Path:
    """Resolve the .eeg file referenced by a .vhdr. Falls back to same-stem.eeg."""
    candidate = vhdr.with_suffix(".eeg")
    if candidate.exists():
        return candidate
    for line in vhdr.read_text().splitlines():
        if line.startswith("DataFile="):
            return vhdr.parent / line.split("=", 1)[1].strip()
    raise FileNotFoundError(f"Could not resolve .eeg companion for {vhdr}")


_UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "µV": 1e-6, "uV": 1e-6, "nV": 1e-9}


def _read_channel_resolutions_volts(vhdr: Path, n_channels: int) -> np.ndarray:
    """Parse per-channel volts-per-binary-unit from the .vhdr's ``Ch<i>=...`` lines.

    Returns an array of shape (n_channels,) such that
    ``data_volts = raw_binary * resolutions[:, None]``.
    """
    import re

    pattern = re.compile(r"^Ch(\d+)=[^,]*,[^,]*,([\d.eE+-]+),([^\s,]+)")
    resolutions = np.zeros(n_channels, dtype=np.float32)
    seen = 0
    for line in vhdr.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # Ch1 → index 0
        if not 0 <= idx < n_channels:
            continue
        scale = float(m.group(2))
        unit = m.group(3).strip()
        if unit not in _UNIT_TO_VOLTS:
            raise ValueError(f"Unknown unit {unit!r} in {vhdr.name} (channel {idx + 1})")
        resolutions[idx] = scale * _UNIT_TO_VOLTS[unit]
        seen += 1
    if seen != n_channels:
        raise ValueError(
            f"Parsed {seen} channel resolutions from {vhdr.name}, expected {n_channels}"
        )
    return resolutions


def crop_and_write(
    input_dir: Path, output_dir: Path, start_s: float, duration_s: float
) -> Path:
    vhdr = find_vhdr(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use MNE only for the lightweight metadata read (sfreq, ch_names) — no data
    # is loaded into memory because preload=False.
    logger.info("Reading %s (metadata only)…", vhdr)
    raw_info = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="WARNING")
    sfreq = raw_info.info["sfreq"]
    ch_names = list(raw_info.ch_names)
    n_channels = len(ch_names)
    total_s = float(raw_info.times[-1])
    del raw_info
    logger.info(
        "Source: %.0f Hz, %d channels, %.1f min total",
        sfreq, n_channels, total_s / 60,
    )

    end_s = start_s + duration_s
    if start_s < 0 or end_s > total_s:
        raise ValueError(
            f"Window [{start_s:.0f}..{end_s:.0f}]s outside source range "
            f"[0..{total_s:.0f}]s; adjust --start-s / --duration-s"
        )

    # Memmap the .eeg as float32 (multiplexed: each sample is one row of
    # n_channels float32 values). Slicing the memmap and transposing into a
    # contiguous block materialises only the window we need — bypasses MNE's
    # float64 upcast and the ~2× peak memory it costs.
    eeg_path = _find_source_eeg(vhdr)
    bytes_per_sample = n_channels * 4  # float32 = 4 bytes
    n_total_samples = eeg_path.stat().st_size // bytes_per_sample
    mmap = np.memmap(
        eeg_path, dtype="<f4", mode="r", shape=(n_total_samples, n_channels)
    )

    start_sample = int(round(start_s * sfreq))
    end_sample = int(round(end_s * sfreq))
    window_samples = end_sample - start_sample
    window_mb = window_samples * n_channels * 4 / (1024 * 1024)
    logger.info(
        "Slicing samples [%d..%d] (%.1f min) directly from .eeg via memmap — "
        "expected window size %.1f MB float32",
        start_sample, end_sample, duration_s / 60, window_mb,
    )

    # Transpose into pybv's expected (n_channels, n_times) layout and force a
    # contiguous copy so pybv can iterate channel-by-channel efficiently.
    window = np.ascontiguousarray(mmap[start_sample:end_sample, :].T)
    del mmap

    # Source .eeg stores raw binary units; per-channel resolution scales to volts
    # (see Ch<i> lines in the .vhdr). pybv expects data already in volts, so
    # apply the scaling in-place before write — no extra allocation.
    resolutions = _read_channel_resolutions_volts(vhdr, n_channels)
    window *= resolutions[:, np.newaxis]

    stem = vhdr.stem
    out_vhdr = output_dir / f"{stem}.vhdr"
    logger.info("Writing BrainVision triplet → %s", out_vhdr)
    pybv.write_brainvision(
        data=window,
        sfreq=sfreq,
        ch_names=ch_names,
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
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Subdirectory name inside --output. Lets you keep multiple fixtures "
             "side-by-side. Final write location is <output>/<target>/.",
    )
    parser.add_argument("--start-s", type=float, default=DEFAULT_START_S)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    output_dir = args.output / args.target if args.target else args.output
    out_vhdr = crop_and_write(args.input, output_dir, args.start_s, args.duration_s)
    logger.info("Done. Point the app at %s", out_vhdr.parent)


if __name__ == "__main__":
    main()
