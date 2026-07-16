from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import pylsl
except ImportError:  # pragma: no cover - handled explicitly at runtime
    pylsl = None


@dataclass(frozen=True)
class ChunkRecord:
    """One received LSL chunk plus the local receive time."""

    receive_time: float
    first_timestamp: float | None
    last_timestamp: float | None
    sample_count: int
    channel_count: int


def _require_pylsl():
    if pylsl is None:
        raise RuntimeError(
            "pylsl is required for LSL characterization. Install online_decoder requirements first."
        )
    return pylsl


def _describe_stream(stream) -> str:
    return (
        f"name={stream.name()!r} "
        f"type={stream.type()!r} "
        f"channels={stream.channel_count()} "
        f"srate={stream.nominal_srate():.2f}"
    )


def channel_metadata(stream) -> list[dict[str, str]]:
    """Return per-channel ``{label, unit, type}`` declared in the stream.

    LSL carries this in the ``desc()`` XML (``channels/channel/{label,unit,type}``).
    The NeurOne proxy may or may not populate ``unit`` — an empty value is
    itself the finding we care about when checking the µV/V wire unit. Returns
    an empty list if no channel metadata is advertised.
    """
    channels: list[dict[str, str]] = []
    try:
        node = stream.desc().child("channels").child("channel")
        while not node.empty():
            channels.append(
                {
                    "label": node.child_value("label"),
                    "unit": node.child_value("unit"),
                    "type": node.child_value("type"),
                }
            )
            node = node.next_sibling()
    except Exception:  # pragma: no cover - metadata shapes vary by producer
        return []
    return channels


def list_streams(*, timeout_s: float = 3.0) -> list:
    """Return all currently visible LSL streams."""

    pylsl_module = _require_pylsl()
    return list(pylsl_module.resolve_streams(wait_time=timeout_s))


def resolve_eeg_stream(
    *,
    stream_type: str = "EEG",
    stream_name: str | None = None,
    timeout_s: float = 5.0,
    verbose: bool = False,
):
    """Resolve the first matching EEG stream."""

    pylsl_module = _require_pylsl()
    if verbose:
        print(
            f"Resolving stream with stream_type={stream_type!r}, "
            f"stream_name={stream_name!r}, timeout={timeout_s:.1f}s"
        )

    if stream_type:
        streams = pylsl_module.resolve_byprop("type", stream_type, timeout=timeout_s)
    else:
        streams = pylsl_module.resolve_streams(wait_time=timeout_s)

    if stream_name is not None:
        streams = [stream for stream in streams if stream.name() == stream_name]

    # Home replay tools may emit the correct stream name but leave the LSL type empty.
    if not streams and stream_name is not None:
        fallback_streams = pylsl_module.resolve_streams(wait_time=timeout_s)
        streams = [stream for stream in fallback_streams if stream.name() == stream_name]

    if not streams:
        if verbose:
            visible_streams = list_streams(timeout_s=1.0)
            print(f"Visible streams at failure time: {len(visible_streams)}")
            for stream in visible_streams:
                print(f"  - {_describe_stream(stream)}")
        target = f"type={stream_type!r}"
        if stream_name is not None:
            target += f", name={stream_name!r}"
        raise RuntimeError(f"No LSL stream found for {target} within {timeout_s:.1f}s.")

    if verbose:
        print(f"Resolved stream: {_describe_stream(streams[0])}")

    return streams[0]


def summarize_chunk_records(
    records: Sequence[ChunkRecord],
    *,
    nominal_srate_hz: float | None = None,
) -> dict[str, Any]:
    """Compute batch-sizing metrics from captured chunks."""

    if not records:
        raise ValueError("At least one chunk record is required.")

    chunk_sizes = np.asarray([record.sample_count for record in records], dtype=float)
    receive_times = np.asarray([record.receive_time for record in records], dtype=float)
    inter_arrival_ms = np.diff(receive_times) * 1000.0

    total_samples = int(chunk_sizes.sum())
    first_timestamp = records[0].first_timestamp
    last_timestamp = records[-1].last_timestamp
    lsl_span_s: float | None = None
    effective_srate_hz: float | None = None

    if first_timestamp is not None and last_timestamp is not None and last_timestamp > first_timestamp:
        lsl_span_s = last_timestamp - first_timestamp
        if total_samples > 1:
            effective_srate_hz = (total_samples - 1) / lsl_span_s
    elif len(receive_times) > 1 and receive_times[-1] > receive_times[0]:
        lsl_span_s = receive_times[-1] - receive_times[0]
        effective_srate_hz = total_samples / lsl_span_s

    chunk_histogram = {
        int(size): count
        for size, count in sorted(Counter(int(size) for size in chunk_sizes).items())
    }
    channel_histogram = {
        int(channels): count
        for channels, count in sorted(Counter(record.channel_count for record in records).items())
    }

    summary = {
        "num_chunks": len(records),
        "num_samples": total_samples,
        "nominal_srate_hz": None if nominal_srate_hz is None else float(nominal_srate_hz),
        "effective_srate_hz": effective_srate_hz,
        "lsl_span_s": lsl_span_s,
        "chunk_size_min": int(chunk_sizes.min()),
        "chunk_size_max": int(chunk_sizes.max()),
        "chunk_size_mean": float(chunk_sizes.mean()),
        "chunk_size_median": float(np.median(chunk_sizes)),
        "most_common_chunk_size": max(chunk_histogram, key=chunk_histogram.get),
        "chunk_size_histogram": chunk_histogram,
        "channel_count_histogram": channel_histogram,
        "inter_arrival_ms_mean": None if inter_arrival_ms.size == 0 else float(inter_arrival_ms.mean()),
        "inter_arrival_ms_median": None if inter_arrival_ms.size == 0 else float(np.median(inter_arrival_ms)),
        "inter_arrival_ms_max": None if inter_arrival_ms.size == 0 else float(inter_arrival_ms.max()),
    }

    if nominal_srate_hz is not None and effective_srate_hz is not None:
        summary["effective_srate_error_hz"] = float(effective_srate_hz - nominal_srate_hz)
    else:
        summary["effective_srate_error_hz"] = None

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Render a readable characterization report."""

    lines = [
        f"Stream: {summary.get('stream_name', '<unknown>')} ({summary.get('stream_type', '<unknown>')})",
        f"Chunks: {summary['num_chunks']}",
        f"Samples: {summary['num_samples']}",
        (
            "Chunk size: "
            f"min={summary['chunk_size_min']} "
            f"median={summary['chunk_size_median']:.1f} "
            f"mean={summary['chunk_size_mean']:.2f} "
            f"max={summary['chunk_size_max']} "
            f"mode={summary['most_common_chunk_size']}"
        ),
        f"Chunk histogram: {summary['chunk_size_histogram']}",
        f"Channel histogram: {summary['channel_count_histogram']}",
    ]

    declared_channels = summary.get("declared_channels") or []
    if declared_channels:
        distinct_units = sorted({c.get("unit") or "<none>" for c in declared_channels})
        lines.append(f"Declared channel units: {distinct_units}")
        # Print each channel's label/unit/type for manual inspection of the wire
        # unit (the µV-vs-V question behind LSL_TO_SI_SCALE).
        lines.append("Declared channels (index: label [unit] type):")
        for idx, ch in enumerate(declared_channels):
            lines.append(
                f"  {idx}: {ch.get('label') or '<none>'} "
                f"[{ch.get('unit') or '<none>'}] {ch.get('type') or ''}".rstrip()
            )
    else:
        lines.append("Declared channel units: <stream advertises no channel metadata>")

    if summary["inter_arrival_ms_mean"] is not None:
        lines.append(
            "Inter-arrival ms: "
            f"median={summary['inter_arrival_ms_median']:.2f} "
            f"mean={summary['inter_arrival_ms_mean']:.2f} "
            f"max={summary['inter_arrival_ms_max']:.2f}"
        )

    if summary["effective_srate_hz"] is not None:
        lines.append(
            "Effective sample rate: "
            f"{summary['effective_srate_hz']:.2f} Hz"
            + (
                ""
                if summary["nominal_srate_hz"] is None
                else f" (nominal {summary['nominal_srate_hz']:.2f} Hz)"
            )
        )

    return "\n".join(lines)


def characterize_stream(
    *,
    duration_s: float = 10.0,
    stream_type: str = "EEG",
    stream_name: str | None = None,
    resolve_timeout_s: float = 5.0,
    pull_timeout_s: float = 0.25,
    verbose: bool = False,
    chunk_log_limit: int = 20,
) -> dict[str, Any]:
    """Capture chunk timing statistics from a live LSL stream."""

    pylsl_module = _require_pylsl()
    stream = resolve_eeg_stream(
        stream_type=stream_type,
        stream_name=stream_name,
        timeout_s=resolve_timeout_s,
        verbose=verbose,
    )
    inlet = pylsl_module.StreamInlet(stream, recover=True)
    declared_channels = channel_metadata(stream)

    if verbose:
        print(
            "Connected inlet: "
            f"name={stream.name()!r}, type={stream.type()!r}, "
            f"channels={stream.channel_count()}, srate={stream.nominal_srate():.2f}, "
            f"duration={duration_s:.1f}s, pull_timeout={pull_timeout_s:.3f}s"
        )

    deadline = time.perf_counter() + duration_s
    records: list[ChunkRecord] = []
    suppressed_chunk_logs = False

    while time.perf_counter() < deadline:
        samples, timestamps = inlet.pull_chunk(timeout=pull_timeout_s)
        if not timestamps:
            continue

        chunk = np.asarray(samples)
        if chunk.ndim == 1:
            channel_count = 1
        elif chunk.ndim >= 2:
            channel_count = int(chunk.shape[1])
        else:
            channel_count = int(stream.channel_count())

        records.append(
            ChunkRecord(
                receive_time=time.perf_counter(),
                first_timestamp=float(timestamps[0]),
                last_timestamp=float(timestamps[-1]),
                sample_count=len(timestamps),
                channel_count=channel_count,
            )
        )

        if verbose:
            chunk_index = len(records)
            if chunk_index <= chunk_log_limit:
                print(
                    f"Chunk {chunk_index}: samples={len(timestamps)} "
                    f"channels={channel_count} "
                    f"first_ts={float(timestamps[0]):.6f} "
                    f"last_ts={float(timestamps[-1]):.6f}"
                )
            elif not suppressed_chunk_logs:
                print(
                    f"Chunk log limit reached at {chunk_log_limit}; "
                    "suppressing additional per-chunk lines."
                )
                suppressed_chunk_logs = True

    if not records:
        raise RuntimeError(
            f"Connected to stream {stream.name()!r} but received no samples in {duration_s:.1f}s."
        )

    summary = summarize_chunk_records(records, nominal_srate_hz=stream.nominal_srate())
    summary["stream_name"] = stream.name()
    summary["stream_type"] = stream.type()
    summary["declared_channels"] = declared_channels
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds.")
    parser.add_argument("--stream-type", default="EEG", help="LSL stream type to resolve.")
    parser.add_argument("--stream-name", default=None, help="Optional exact LSL stream name.")
    parser.add_argument(
        "--resolve-timeout",
        type=float,
        default=5.0,
        help="How long to wait for LSL stream resolution.",
    )
    parser.add_argument(
        "--pull-timeout",
        type=float,
        default=0.25,
        help="Timeout per pull_chunk() call.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stream discovery and per-chunk diagnostics.",
    )
    parser.add_argument(
        "--chunk-log-limit",
        type=int,
        default=20,
        help="Maximum number of per-chunk lines to print in verbose mode.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = characterize_stream(
        duration_s=args.duration,
        stream_type=args.stream_type,
        stream_name=args.stream_name,
        resolve_timeout_s=args.resolve_timeout,
        pull_timeout_s=args.pull_timeout,
        verbose=args.verbose,
        chunk_log_limit=args.chunk_log_limit,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
