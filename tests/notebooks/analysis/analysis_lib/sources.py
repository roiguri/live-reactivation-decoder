"""``SOURCE`` → labeled epoch samples for the epoched-decoding notebook.

Bridges the two label regimes behind one call so the notebook stays uniform:

- ``"fl"`` — the FL trigger codes *are* the stimulus identity (pulled via
  :func:`analysis_lib.streaming.extract_markers`).
- task sources (``"stage2_learning"`` / ``"stage2_test"`` / ``"stage3"``) — the
  codes are phase events and identity comes from the behavioral metadata (via
  :mod:`analysis_lib.task_labels`), with order-alignment validated by
  ``gap_residuals``.

All return the same ``samples_by_group`` shape (``{display_group: [sample,...]}``).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from analysis_lib import streaming, task_labels

_FMT = "%Y-%m-%d_%H-%M-%S.%f"


def _ts(s: str) -> float:
    return dt.datetime.strptime(s, _FMT).timestamp()


def build_source_samples(
    ctx, raw, sfreq, source, dc, *, n_times, metadata_dir, recording_dir,
    anchor_code=None, label_key="true_label",
):
    """Return ``(samples_by_group, trials, info)`` for ``source``.

    ``trials`` is the per-trial records for task sources (``None`` for FL);
    ``info`` is a one-line human summary (marker counts for FL; the alignment
    residual + a stage-specific note for task sources).
    """
    if source == "fl":
        markers = streaming.extract_markers(raw, ctx.event_mapping, dc.raw_markers, n_times=n_times)
        sbg: dict[str, list[int]] = {}
        for s, c in markers:
            g = dc.code_to_group.get(c)
            if g is not None:
                sbg.setdefault(g, []).append(s)
        counts = {ctx.name_by_code[c]: n for c, n in Counter(c for _, c in markers).items()}
        return sbg, None, f"FL markers: {counts}"

    md = Path(metadata_dir)
    combined = {r["object"]: r
                for r in csv.DictReader(open(next((md / "combined_data").glob("*.csv"))))}
    markers_all = task_labels.parse_vmrk(streaming.find_vhdr(recording_dir).with_suffix(".vmrk"))

    if source == "stage3":
        partial = json.loads(next((md / "partial_retrival").glob("*.json")).read_text())
        true_label_of = {}
        for v in partial.values():
            obj = task_labels.object_of(v)
            true_label_of[obj] = combined[obj][v[obj]["probe"]]
        trials = task_labels.stage3_trials(markers_all, partial,
                                           anchor_code=anchor_code, true_label_of=true_label_of)
        sbg = task_labels.group_samples_by_label(trials, key=label_key)
        anchors = task_labels.samples_of(task_labels.stage3_markers(markers_all), anchor_code)
        ev = [_ts(v["trial_times"]["start_retrival_time"]) for v in partial.values()]
        extra = f"by probe {dict(Counter(t['probe'] for t in trials))}"

    elif source == "stage2_learning":
        ta = json.loads(next((md / "true_answers").glob("*true_answers.json")).read_text())
        trials = task_labels.stage2_learning_trials(markers_all, ta, anchor_code=anchor_code)
        sbg = {**task_labels.group_samples_by_label(trials, key="colour_label"),
               **task_labels.group_samples_by_label(trials, key="scene_label")}
        anchors = task_labels.samples_of(task_labels.stage2_region(markers_all), anchor_code)
        ev = [_ts(ta[k]["trial_times"]["object_appear"]) for k in sorted(ta, key=int)]
        extra = "perception — colour + scene shown together"

    elif source == "stage2_test":
        subj = json.loads(next((md / "subject_answer").glob("*.json")).read_text())
        trials = task_labels.stage2_test_trials(markers_all, subj, combined, anchor_code=anchor_code)
        sbg = {**task_labels.group_samples_by_label(trials, key="colour_label"),
               **task_labels.group_samples_by_label(trials, key="scene_label")}
        anchors = task_labels.samples_of(task_labels.stage2_region(markers_all), anchor_code)
        order = sorted(subj, key=lambda s: int(s.split("_")[1]))
        ev = [_ts(subj[k]["trial_times"]["retrival_question_appear"]) for k in order]
        extra = f"recall — both_correct {sum(t['both_correct'] for t in trials)}/{len(trials)}"

    else:
        raise ValueError(f"unknown source {source!r}")

    res = task_labels.gap_residuals(anchors, ev, sfreq)
    info = f"[{source}] alignment gap residual max={max(res):.3f}s (approx 0 = aligned) | {extra}"
    return sbg, trials, info
