from __future__ import annotations

import argparse
import time
from pathlib import Path

import mne
import numpy as np
from mne_lsl.player import PlayerLSL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a BrainVision EEG recording (subject folder) as a live LSL stream.",
    )
    parser.add_argument(
        "subject_dir",
        type=Path,
        help="Path to the subject folder containing a .vhdr file.",
    )
    parser.add_argument(
        "--stream-name",
        default="NeuroneStream",
        help="LSL stream name (default: NeuroneStream).",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Loop the replay forever instead of playing it once.",
    )
    return parser


def find_vhdr(subject_dir: Path) -> Path:
    """Find the single .vhdr file inside a subject folder."""
    vhdrs = list(subject_dir.glob("*.vhdr"))
    if not vhdrs:
        raise FileNotFoundError(f"No .vhdr file found in {subject_dir}")
    if len(vhdrs) > 1:
        print(f"WARNING: multiple .vhdr files in {subject_dir}, using {vhdrs[0].name}")
    return vhdrs[0]


def encode_trigger_value(event_code: int) -> int:
    """Encode a plain event code into NeurOne's packed trigger word.

    ``LSLReceiver.decode_trigger_value`` recovers the code via
    ``(raw >> 8) & 0xFF``, so the inverse is ``code << 8``.
    """
    return event_code << 8


def load_raw_with_stim_channel(vhdr_path: Path) -> mne.io.Raw:
    """Load a BrainVision file, synthesize a stim channel from .vmrk
    annotations, and return a Raw with EEG + 1 stim channel.

    BrainVision recordings store triggers in the ``.vmrk`` sidecar, which
    MNE reads as annotations (not a stim channel).  The online pipeline's
    ``LSLReceiver`` expects a trigger channel at position 64, so we build
    one from the annotations and append it.  Trigger values are packed
    into NeurOne format (``code << 8``) so ``decode_trigger_value`` works
    unchanged.  The data is streamed at its native sample rate with no
    resampling.
    """
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Extract events from annotations before any channel manipulation.
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Keep only EEG channels and drop EMG (which is EEG-typed in the .vhdr
    # until the offline preprocessor retypes it).  The live NeurOne stream
    # sends 64 EEG + 1 trigger — no EMG — so we drop it here to match.
    eeg_picks = mne.pick_types(raw.info, eeg=True).tolist()
    keep = [raw.ch_names[i] for i in eeg_picks]
    if "EMG" in keep:
        keep.remove("EMG")
    raw.pick(keep)

    n_eeg = len(raw.ch_names)
    n_samples = raw.n_times

    # Build the stim channel with packed NeurOne trigger values.
    stim_data = np.zeros((1, n_samples), dtype=float)
    for sample, _, event_code in events:
        if 0 <= sample < n_samples:
            stim_data[0, sample] = float(encode_trigger_value(event_code))

    stim_info = mne.create_info(["STI 014"], raw.info["sfreq"], ch_types=["stim"])
    stim_raw = mne.io.RawArray(stim_data, stim_info, verbose=False)
    raw.add_channels([stim_raw], force_update_info=True)

    print(f"Channels: {n_eeg} EEG + 1 stim = {raw.info['nchan']} total")

    n_events = len(events)
    if n_events > 0:
        unique_codes = sorted(set(events[:, 2]))
        print(f"Events: {n_events} total, codes: {unique_codes}")
    else:
        print("WARNING: no events found in annotations")

    return raw


def main() -> int:
    args = build_arg_parser().parse_args()
    subject_dir: Path = args.subject_dir
    if not subject_dir.is_dir():
        print(f"Not a directory: {subject_dir}")
        return 1

    vhdr_path = find_vhdr(subject_dir)
    raw = load_raw_with_stim_channel(vhdr_path)

    print(f"Loaded {vhdr_path}")
    print(f"Replaying as LSL stream: {args.stream_name}")
    print(f"Sample rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} s")

    player = PlayerLSL(
        raw, name=args.stream_name, n_repeat=np.inf if args.repeat else 1,
    )
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
