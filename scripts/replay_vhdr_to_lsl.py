"""Dev helper: replay a BrainVision recording as a NeurOne-like LSL stream.

Publishes a stream that the app's discovery treats exactly like real hardware:
name ``NeuroneStream``, **type ``EEG``**, 65 channels (64 EEG + 1 trigger at
index 64) @ the recording's sample rate. Markers from the ``.vmrk`` sidecar are
encoded into the trigger channel in NeurOne's packed format (``code << 8``), so
the online ``LSLReceiver`` consumes it unchanged.

A raw ``pylsl`` outlet is used rather than ``mne_lsl.PlayerLSL`` because
PlayerLSL derives the LSL ``type`` from MNE's channel kind (lowercase
``"eeg"`` / ``""`` for mixed channels), which the app's ``type == "EEG"``
discovery filter would miss.

Run it in the background, then pick the discovered stream in the live screen:

    python scripts/replay_vhdr_to_lsl.py data/split/functional_localizer

Loops forever by default so the stream stays up for a consumer to attach.
Ctrl-C (or terminating the process) stops it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mne
import numpy as np
import pylsl

DEFAULT_STREAM_NAME = "NeuroneStream"
DEFAULT_STREAM_TYPE = "EEG"
TRIGGER_CHANNEL_INDEX = 64


def find_vhdr(recording_dir: Path) -> Path:
    """Find the single ``.vhdr`` file inside a recording directory."""
    vhdrs = list(recording_dir.glob("*.vhdr"))
    if not vhdrs:
        raise FileNotFoundError(f"No .vhdr file found in {recording_dir}")
    if len(vhdrs) > 1:
        print(f"WARNING: multiple .vhdr files in {recording_dir}, using {vhdrs[0].name}")
    return vhdrs[0]


def encode_trigger_value(event_code: int) -> int:
    """Pack a plain event code into NeurOne's trigger word (inverse of the
    receiver's ``(raw >> 8) & 0xFF``)."""
    return event_code << 8


def build_stream_matrix(vhdr_path: Path) -> tuple[np.ndarray, float]:
    """Return ``(samples, sfreq)`` where samples is ``(n_times, 65)``.

    Columns 0..63 are EEG in **microvolts** (``get_data(units="uV")``,
    matching the NeurOne proxy's wire units so replay reproduces the live
    pipeline); column 64 is the packed trigger channel built from the
    ``.vmrk`` annotations.
    """
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)

    eeg_picks = mne.pick_types(raw.info, eeg=True).tolist()
    keep = [raw.ch_names[i] for i in eeg_picks]
    if "EMG" in keep:
        keep.remove("EMG")
    raw.pick(keep)

    eeg = raw.get_data(units="uV")  # (n_eeg, n_times), µV (mimics proxy)
    n_eeg, n_times = eeg.shape

    trigger = np.zeros((1, n_times), dtype=float)
    for sample, _, event_code in events:
        if 0 <= sample < n_times:
            trigger[0, sample] = float(encode_trigger_value(event_code))

    samples = np.vstack([eeg, trigger]).T.astype(np.float64)  # (n_times, n_eeg+1)
    print(f"Channels: {n_eeg} EEG + 1 trigger = {samples.shape[1]} total")
    if len(events):
        print(f"Events: {len(events)} total, codes: {sorted(set(events[:, 2]))}")
    else:
        print("WARNING: no events found in annotations")
    return samples, float(raw.info["sfreq"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a BrainVision recording directory as a NeurOne-like LSL stream.",
    )
    parser.add_argument(
        "recording_dir",
        type=Path,
        help="Path to the recording folder containing a .vhdr (+ .vmrk) file.",
    )
    parser.add_argument("--stream-name", default=DEFAULT_STREAM_NAME)
    parser.add_argument("--stream-type", default=DEFAULT_STREAM_TYPE)
    parser.add_argument(
        "--chunk-ms",
        type=float,
        default=20.0,
        help="Milliseconds of data per LSL push (default: 20).",
    )
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="Play the recording once instead of looping forever.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Skip this many seconds into the recording before streaming "
        "(and loop back to it), e.g. to jump past a long rest block.",
    )
    parser.add_argument(
        "--start-at-first-event",
        action="store_true",
        help="Start at the first marker/event (and loop back to it), "
        "overriding --start-sec — skips a leading rest block automatically.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.recording_dir.is_dir():
        print(f"Not a directory: {args.recording_dir}")
        return 1

    vhdr_path = find_vhdr(args.recording_dir)
    samples, sfreq = build_stream_matrix(vhdr_path)
    n_times, n_channels = samples.shape

    info = pylsl.StreamInfo(
        name=args.stream_name,
        type=args.stream_type,
        channel_count=n_channels,
        nominal_srate=sfreq,
        channel_format=pylsl.cf_double64,
        source_id=f"replay_{args.recording_dir.name}",
    )
    outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=360)

    chunk = max(1, int(round(sfreq * args.chunk_ms / 1000.0)))
    period = chunk / sfreq

    print(f"Loaded {vhdr_path}")
    print(
        f"Streaming '{args.stream_name}' (type={args.stream_type}), "
        f"{n_channels} ch @ {sfreq:g} Hz, {n_times / sfreq:.1f} s, "
        f"chunk={chunk} samples. Press Ctrl+C to stop."
    )

    if args.start_at_first_event:
        # The trigger channel (col 64) is nonzero exactly at the packed events,
        # so its first nonzero sample is the first marker.
        nonzero = np.flatnonzero(samples[:, TRIGGER_CHANNEL_INDEX])
        start_i = int(nonzero[0]) if nonzero.size else 0
        if start_i:
            print(f"Starting at first event (sample {start_i}, {start_i / sfreq:.1f} s).")
        else:
            print("No events found; starting at 0.")
    else:
        start_i = min(max(0, int(round(args.start_sec * sfreq))), max(0, n_times - 1))
        if start_i:
            print(f"Starting at {args.start_sec:g} s (sample {start_i}).")
    i = start_i
    next_t = time.perf_counter()
    try:
        while True:
            block = samples[i:i + chunk]
            outlet.push_chunk(block.tolist())
            i += block.shape[0]
            if i >= n_times:
                if args.no_repeat:
                    break
                i = start_i
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind (e.g. after a GC pause); resync to avoid drift.
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopping replay...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
