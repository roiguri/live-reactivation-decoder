"""Encoding/retrieval trial labeling for the real-time task's held-out phases.

**Encoding** markers directly name what was shown (``learning_<category>_NN``)
— same marker-is-identity principle as the FL localizer, just anchored during
the couple-learning block (see :func:`encoding_trials`).

**Retrieval** is harder: a retrieval cue (``retrieval_verb_*``) only names
which verb is being probed, not which image category it was paired with. That
pairing is recovered from the *encoding* markers earlier in the same
recording: ``learning_verb_*`` is immediately followed by
``learning_<category>_NN`` on every encoding repeat of that couple (see
``experiment_config.realtime_animacy.yaml``), so majority-voting each verb's
paired category across its repeats gives a robust ``verb -> category`` map.
Retrieval trials are then labeled with that category as their ground-truth
``true_label``.

A verb's identity is whatever follows ``learning_verb_``/``retrieval_verb_``
in its marker name — an opaque string, not assumed numeric. This lets it work
unchanged whether a config names verbs by bare index (``learning_verb_5``, as
in ``experiment_config.realtime_animacy.yaml``) or spells out the known
category (``learning_verb_animate_1``, as in
``experiment_config.realtime_animacy_verb_labels.yaml``) — same verb, same
identity string, matched consistently between its encoding and retrieval
occurrences either way.

Both let the FL-trained decoders be scored as an honest held-out test —
encoding as a same-modality (perception) sanity check, retrieval as the real
reactivation-from-memory question.

Pure / backend-free: functions take already-parsed ``(t, code, name)`` marker
rows — e.g. straight from a live run's ``markers.csv``
(:func:`analysis_lib.streaming.load_live_run`) — so they unit test without MNE
or disk. ``t`` is whatever time axis the caller used (seconds); only relative
order and marker names matter here.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple, Optional

_IMAGE_RE = re.compile(r"^([a-zA-Z]+)_\d+$")

# Defaults match the marker *names* in experiment_config.realtime_animacy.yaml's
# markers_mapping. They're overridable per-call, and public so a caller can
# validate them against a loaded config's ctx.event_mapping (see
# sources.build_retrieval_epochs) instead of assuming a renamed config still
# matches these string literals. The captured group is a verb's identity —
# whatever follows the prefix, e.g. "5" or "animate_1" — not necessarily numeric.
VERB_RE = re.compile(r"^learning_verb_(.+)$")
RETRIEVAL_VERB_RE = re.compile(r"^retrieval_verb_(.+)$")
RETRIEVAL_END = "retrieval_end"
RECALL_KEY_PRESS = "recall_key_press"


class Marker(NamedTuple):
    t: float
    code: int
    name: str


def category_of(image_name: str) -> Optional[str]:
    """The category prefix of an image name (``"animate_02" -> "animate"``), or ``None``."""
    m = _IMAGE_RE.match(image_name)
    return m.group(1) if m else None


def marker_groups_by_category(names: list[str]) -> dict[str, list[str]]:
    """Pool ``names`` (e.g. ``ctx.event_mapping`` keys) by :func:`category_of`.

    Names that don't match the ``<category>_<NN>`` convention are dropped.
    """
    out: dict[str, list[str]] = {}
    for n in names:
        cat = category_of(n)
        if cat is not None:
            out.setdefault(cat, []).append(n)
    return out


def group_couple_trials(
    markers: list[Marker], *, verb_re: re.Pattern = VERB_RE, image_prefix: str = "learning_"
) -> list[dict]:
    """One record per encoding repeat: pairs a verb cue with the image that follows it.

    Returns ``[{"t", "verb", "image", "category"}, ...]`` in marker order.
    """
    out: list[dict] = []
    for i, m in enumerate(markers):
        vm = verb_re.match(m.name)
        if not vm:
            continue
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        if nxt is None or not nxt.name.startswith(image_prefix):
            continue
        image = nxt.name[len(image_prefix):]
        cat = category_of(image)
        if cat is None:
            continue
        out.append({"t": m.t, "verb": vm.group(1), "image": image, "category": cat})
    return out


def encoding_trials(
    markers: list[Marker], *, verb_re: re.Pattern = VERB_RE, image_prefix: str = "learning_"
) -> list[dict]:
    """One record per encoding image onset — the image marker *is* the stimulus identity.

    Unlike retrieval, no verb-pairing is needed: ``learning_<category>_NN``
    directly names what was shown, same marker-is-identity principle as the
    FL localizer, just anchored during the couple-learning block instead.
    ``learning_verb_N`` cue markers are skipped (they're not image onsets).

    Returns ``[{"t", "image", "true_label"}, ...]`` in marker order.
    """
    out: list[dict] = []
    for m in markers:
        if verb_re.match(m.name) or not m.name.startswith(image_prefix):
            continue
        image = m.name[len(image_prefix):]
        cat = category_of(image)
        if cat is None:
            continue
        out.append({"t": m.t, "image": image, "true_label": cat})
    return out


def verb_categories(couple_trials: list[dict]) -> dict[str, str]:
    """Majority-vote category per verb identity across its encoding repeats.

    Raises if any verb's repeats disagree — the couple is supposed to be fixed
    for the whole session, so a disagreement means something upstream (marker
    parsing, or the recording itself) is misaligned.
    """
    by_verb: dict[str, list[str]] = {}
    for t in couple_trials:
        by_verb.setdefault(t["verb"], []).append(t["category"])
    out: dict[str, str] = {}
    for verb, cats in by_verb.items():
        counts = Counter(cats)
        if len(counts) > 1:
            raise ValueError(f"verb {verb} paired with inconsistent categories: {dict(counts)}")
        out[verb] = cats[0]
    return out


def retrieval_trials(
    markers: list[Marker], verb_category: dict[str, str], *,
    retrieval_re: re.Pattern = RETRIEVAL_VERB_RE,
    end_name: str = RETRIEVAL_END, recall_name: str = RECALL_KEY_PRESS,
) -> list[dict]:
    """One record per retrieval cue.

    Returns ``[{"t", "verb", "true_label", "recalled"}, ...]``: ``t`` is the cue
    onset (start of the recall window), ``true_label`` the category the cued
    verb was encoded with (``None`` if the verb was never seen at encoding),
    and ``recalled`` whether ``recall_name`` fired before the trial's
    ``end_name`` marker.
    """
    out: list[dict] = []
    i, n = 0, len(markers)
    while i < n:
        m = markers[i]
        vm = retrieval_re.match(m.name)
        if not vm:
            i += 1
            continue
        verb = vm.group(1)
        j = i + 1
        recalled = False
        while j < n and markers[j].name != end_name:
            if markers[j].name == recall_name:
                recalled = True
            j += 1
        out.append({
            "t": m.t, "verb": verb,
            "true_label": verb_category.get(verb),
            "recalled": recalled,
        })
        i = j + 1
    return out


def group_samples_by_label(trials: list[dict], *, key: str = "true_label") -> dict[str, list[float]]:
    """``{label: [t, ...]}`` from trial records, dropping null labels."""
    out: dict[str, list[float]] = {}
    for t in trials:
        lab = t.get(key)
        if lab is not None:
            out.setdefault(lab, []).append(t["t"])
    return out
