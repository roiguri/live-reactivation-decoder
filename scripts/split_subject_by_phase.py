"""Split a subject's experiment recording into per-phase BrainVision triplets.

The recording at ``data/subject_<id>/raw_data/EEG/experiment/`` mixes the
functional localizer with the binding/test/partial_retrieval task in a single
file. The two halves use disjoint marker code sets:

    Functional localizer   : S 11/12/13 (colors), S 16/17/18 (scenes),
                             S 31/32 (fixation pair)
    Task (binding+test...) : S 41/43/44, S 51/53/54/55/56/61/62/66/67, S 71

We find the latest FL-code timestamp and the earliest non-FL-code timestamp;
they are separated by a long gap (the subject break between phases). Each phase
is cropped with ``--pad-s`` seconds of headroom on each side and written as its
own BrainVision triplet via :func:`scripts.create_test_eeg.crop_and_write`,
preserving channel resolutions and stimulus markers.

Usage:
    python scripts/split_subject_by_phase.py --subject 101
    python scripts/split_subject_by_phase.py --subject 101 --pad-s 2.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_test_eeg import crop_and_write, find_vhdr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("split_subject_by_phase")

FL_CODES = {11, 12, 13, 16, 17, 18, 31, 32}
TASK_CODES = {41, 43, 44, 51, 53, 54, 55, 56, 61, 62, 66, 67, 71}


def _parse_code(desc: str) -> int | None:
    """Pull the integer N out of a ``"Stimulus/S NN"`` annotation description."""
    if "/" in desc:
        _, tail = desc.split("/", 1)
    else:
        tail = desc
    tail = tail.strip()
    if tail.startswith("S"):
        tail = tail[1:].strip()
    try:
        return int(tail)
    except ValueError:
        return None


def find_phase_windows(
    vhdr: Path, pad_s: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``((fl_start, fl_dur), (task_start, task_dur))`` in seconds.

    Boundaries are derived from the .vmrk annotations: the FL window spans
    [first FL marker, last FL marker]; the task window spans [first non-FL
    marker, last non-FL marker]. Both are extended by ``pad_s`` on each side
    and clamped to the recording range.
    """
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="WARNING")
    total_s = float(raw.times[-1])
    ann = raw.annotations

    fl_onsets: list[float] = []
    task_onsets: list[float] = []
    unknown: set[int] = set()
    for onset, desc in zip(ann.onset, ann.description):
        code = _parse_code(desc)
        if code is None:
            continue
        if code in FL_CODES:
            fl_onsets.append(float(onset))
        elif code in TASK_CODES:
            task_onsets.append(float(onset))
        else:
            unknown.add(code)
    if unknown:
        logger.warning(
            "Ignored marker codes not in FL or task sets: %s",
            sorted(unknown),
        )

    if not fl_onsets:
        raise ValueError(f"No functional-localizer markers found in {vhdr.name}")
    if not task_onsets:
        raise ValueError(f"No task markers found in {vhdr.name}")

    fl_start = max(0.0, min(fl_onsets) - pad_s)
    fl_end = min(total_s, max(fl_onsets) + pad_s)
    task_start = max(0.0, min(task_onsets) - pad_s)
    task_end = min(total_s, max(task_onsets) + pad_s)

    if fl_end > task_start:
        raise ValueError(
            f"FL window [{fl_start:.2f}, {fl_end:.2f}]s overlaps task window "
            f"[{task_start:.2f}, {task_end:.2f}]s — pad is too large or marker "
            "sets are not disjoint for this subject."
        )

    logger.info(
        "FL   window: [%.2f, %.2f]s (%.1f min, %d markers)",
        fl_start, fl_end, (fl_end - fl_start) / 60, len(fl_onsets),
    )
    logger.info(
        "Task window: [%.2f, %.2f]s (%.1f min, %d markers)",
        task_start, task_end, (task_end - task_start) / 60, len(task_onsets),
    )
    logger.info("Inter-phase gap: %.1f s", task_start - fl_end)

    return (fl_start, fl_end - fl_start), (task_start, task_end - task_start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", required=True,
        help="Subject id, e.g. 101 — resolves to data/subject_<id>/...",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data",
        help="Root containing subject_<id>/ directories (default: <repo>/data).",
    )
    parser.add_argument(
        "--pad-s", type=float, default=2.0,
        help="Seconds of headroom on each side of each phase (default: 2.0).",
    )
    args = parser.parse_args()

    subject_dir = args.data_root / f"subject_{args.subject}"
    input_dir = subject_dir / "raw_data" / "EEG" / "experiment"
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    vhdr = find_vhdr(input_dir)
    (fl_start, fl_dur), (task_start, task_dur) = find_phase_windows(vhdr, args.pad_s)

    split_root = subject_dir / "split"
    fl_out = split_root / "functional_localizer"
    task_out = split_root / "task"

    logger.info("Writing functional_localizer triplet → %s", fl_out)
    crop_and_write(input_dir, fl_out, fl_start, fl_dur)
    logger.info("Writing task triplet → %s", task_out)
    crop_and_write(input_dir, task_out, task_start, task_dur)

    logger.info("Done. Split written under %s", split_root)


if __name__ == "__main__":
    main()
