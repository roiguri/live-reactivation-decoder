"""Retrieval-label alignment for the *task* (held-out) recording.

Unlike the functional localizer — where the EEG trigger codes *are* the
stimulus identity (11-18) — the binding-task EEG markers encode trial **phase**
events (BMR parallel-port codes; see
``knowledge_base/02_reference/BMR Data Specification.md``), not which colour or
scene. The identity lives in the behavioral metadata. These helpers parse the
BrainVision ``.vmrk``, split Stage 2 / Stage 3 by their unique anchor codes, and
join each Stage 3 retrieval trial to its colour/scene label so decoder epochs
can be labeled and the FL-trained decoders run as an honest held-out test.

Pure / backend-free: functions take already-loaded markers / dicts so they unit
test without MNE or disk. The notebook owns recording load + streaming.
"""
from __future__ import annotations

import re
from pathlib import Path

# --- BMR trigger codes (data spec, "Trigger Code Reference") ---
SHOW_BINDING = 41     # Stage 2 learning: object (colour+scene) shown  (binding order)
PROBE = 71            # Stage 3: show probe cue           (unique to Stage 3)
SHOW_OBJECT = 51      # shared: show object in retrieval   (Stage 2 test + Stage 3)
START_RETRIEVAL = 53  # shared: start retrieval window     (Stage 2 test + Stage 3)
ANSWER_RETRIEVAL = 54 # shared: answer during retrieval
REMEMBER_Q = 55       # Stage 2 test: "what do you remember?"  (unique to Stage 2 test)

# behavioral keys that sit beside the object key in a trial dict (across files)
_NON_OBJECT_KEYS = ("retrival_success", "retrival_report_color",
                    "retrival_report_scene", "trial_times")


def parse_vmrk_text(text: str) -> list[tuple[int, int]]:
    """Return ``[(code, sample), ...]`` for every ``Stimulus`` marker, in file order."""
    out: list[tuple[int, int]] = []
    for line in text.splitlines():
        m = re.match(r"Mk\d+=Stimulus,S ?(\d+),(\d+),", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def parse_vmrk(path: str | Path) -> list[tuple[int, int]]:
    """``parse_vmrk_text`` over the file at ``path``."""
    return parse_vmrk_text(Path(path).read_text())


def stage3_markers(markers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Markers from the first :data:`PROBE` (71) onward — i.e. all of Stage 3.

    Stage 3 is the only stage that emits code 71, so the first 71 cleanly marks
    where Stage 2 ends and Stage 3 begins (codes 51/53/54 are reused across both).
    """
    idx = next((i for i, (c, _) in enumerate(markers) if c == PROBE), None)
    return markers[idx:] if idx is not None else []


def samples_of(markers: list[tuple[int, int]], code: int) -> list[int]:
    """Sample indices of every marker with trigger ``code``, in order."""
    return [s for c, s in markers if c == code]


def object_of(trial: dict) -> str:
    """The object name key inside a partial_retrival / true_answers trial dict."""
    return next(k for k in trial if k not in _NON_OBJECT_KEYS)


def stage3_trials(
    markers: list[tuple[int, int]],
    partial_retrival: dict[str, dict],
    *,
    anchor_code: int = START_RETRIEVAL,
    true_label_of: dict[str, str] | None = None,
) -> list[dict]:
    """Join Stage 3 anchor markers to ``partial_retrival`` trials, in order.

    Order alignment is reliable because the per-trial marker sequence is fixed
    and the anchor count matches the trial count exactly (validate independently
    with :func:`gap_residuals`). Each record carries the EEG ``sample`` of
    ``anchor_code`` plus the trial's labels/covariates:

    - ``reported_label``: the feature the subject reported (``subject_answer``)
    - ``true_label``: the cued category's ground-truth feature, if ``true_label_of``
      (object -> feature) is supplied; else ``None``
    - ``probe`` (``colors``/``scenes``), ``is_remember``, ``object``, ``trial``

    Raises if the anchor count and trial count disagree (alignment unsafe).
    """
    s3 = stage3_markers(markers)
    anchors = samples_of(s3, anchor_code)
    trials = list(partial_retrival.items())
    if len(anchors) != len(trials):
        raise ValueError(
            f"Stage 3 anchor(code={anchor_code}) count {len(anchors)} "
            f"!= partial_retrival trials {len(trials)} — alignment unsafe"
        )
    out: list[dict] = []
    for sample, (tid, v) in zip(anchors, trials):
        obj = object_of(v)
        d = v[obj]
        out.append({
            "sample": int(sample),
            "trial": tid,
            "object": obj,
            "probe": d["probe"],
            "reported_label": d.get("subject_answer"),
            "true_label": (true_label_of or {}).get(obj),
            "is_remember": d.get("is_remember"),
            "retrival_success": v.get("retrival_success"),
        })
    return out


def stage2_region(markers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Markers *before* the first :data:`PROBE` (71) — the Stage 2 binding block.

    Codes 51/53/54 are reused by Stage 2 test and Stage 3, so the first 71 is the
    clean Stage 2 / Stage 3 boundary. (The functional localizer is a separate
    recording, so on the task file everything before Stage 3 is Stage 2.)
    """
    idx = next((i for i, (c, _) in enumerate(markers) if c == PROBE), len(markers))
    return markers[:idx]


def stage2_learning_trials(
    markers: list[tuple[int, int]],
    true_answers: dict[str, dict],
    *,
    anchor_code: int = SHOW_BINDING,
) -> list[dict]:
    """Join Stage 2 *learning* (encoding) markers to ``true_answers``, in binding order.

    Learning is **perceptual**: the object is shown tinted a colour on a scene
    background, so each trial carries BOTH a ``colour_label`` and a ``scene_label``
    — the closest analogue to the decoders' FL training. Learning markers and
    ``true_answers`` trials are both in binding order (keys ``"1".."45"``).
    Raises if the counts disagree (alignment unsafe).
    """
    anchors = samples_of(stage2_region(markers), anchor_code)
    trials = [true_answers[k] for k in sorted(true_answers, key=int)]
    if len(anchors) != len(trials):
        raise ValueError(
            f"Stage 2 learning anchor(code={anchor_code}) count {len(anchors)} "
            f"!= true_answers trials {len(trials)} — alignment unsafe")
    out: list[dict] = []
    for sample, v in zip(anchors, trials):
        obj = object_of(v)
        d = v[obj]
        out.append({"sample": int(sample), "object": obj,
                    "colour_label": d["colors"], "scene_label": d["scenes"]})
    return out


def stage2_test_trials(
    markers: list[tuple[int, int]],
    subject_answer: dict[str, dict],
    combined_by_object: dict[str, dict],
    *,
    anchor_code: int = REMEMBER_Q,
) -> list[dict]:
    """Join Stage 2 *test* (retrieval) markers to ``subject_answer``, in test order.

    Test is **recall**: the plain object is shown and the subject recalls the
    binding. Labels are the binding's ground truth (``colour_label``/``scene_label``)
    from ``combined_by_object`` (object -> combined.csv row); per-trial correctness
    (``colour_correct``/``scene_correct``/``both_correct``) and ``difficulty`` are
    carried through. Test markers and ``subject_answer`` trials are both in
    test-presentation order (``trial_1..trial_45``). Raises on count mismatch.
    """
    anchors = samples_of(stage2_region(markers), anchor_code)
    trials = [subject_answer[k] for k in sorted(subject_answer, key=lambda s: int(s.split("_")[1]))]
    if len(anchors) != len(trials):
        raise ValueError(
            f"Stage 2 test anchor(code={anchor_code}) count {len(anchors)} "
            f"!= subject_answer trials {len(trials)} — alignment unsafe")
    out: list[dict] = []
    for sample, v in zip(anchors, trials):
        obj = object_of(v)
        r = combined_by_object[obj]
        out.append({"sample": int(sample), "object": obj,
                    "colour_label": r["colors"], "scene_label": r["scenes"],
                    "colour_correct": r["color_correct"] == "TRUE",
                    "scene_correct": r["scene_correct"] == "TRUE",
                    "both_correct": r["both_correct"] == "TRUE",
                    "difficulty": r.get("difficulty")})
    return out


def group_samples_by_label(trials: list[dict], *, key: str = "true_label") -> dict[str, list[int]]:
    """``{label: [sample, ...]}`` from Stage 3 trial records, dropping null labels."""
    out: dict[str, list[int]] = {}
    for t in trials:
        lab = t.get(key)
        if lab is not None:
            out.setdefault(lab, []).append(t["sample"])
    return out


def gap_residuals(
    anchor_samples: list[int], event_times_s: list[float], sfreq: float
) -> list[float]:
    """Per-pair ``|Δgap|`` (seconds) between EEG-marker spacing and metadata spacing.

    Independent check that order-alignment is correct: the time *between*
    consecutive anchor markers (from EEG samples) should match the time between
    the corresponding behavioral timestamps. Near-zero residuals confirm the
    i-th marker is the i-th trial. Lengths must match.
    """
    if len(anchor_samples) != len(event_times_s):
        raise ValueError("anchor / event-time counts differ")
    eeg = [(anchor_samples[i + 1] - anchor_samples[i]) / sfreq
           for i in range(len(anchor_samples) - 1)]
    meta = [event_times_s[i + 1] - event_times_s[i]
            for i in range(len(event_times_s) - 1)]
    return [abs(eeg[i] - meta[i]) for i in range(len(eeg))]
