"""``SOURCE`` -> labeled epochs for the epoched-decoding notebook.

Bridges the three label regimes behind one call so the notebook stays uniform:

- ``"fl"`` — the FL trigger codes *are* the stimulus identity (pulled via
  :func:`analysis_lib.streaming.extract_markers`); paired with a fresh raw-EEG
  replay through the online preprocessor.
- ``"encoding"`` — the real-time task's couple-learning phase. Trial identity
  is directly in the trigger code too (``learning_<category>_NN``), so it's a
  same-modality (perception) held-out sanity check.
- ``"retrieval"`` — the real-time task's retrieval phase. Trial identity isn't
  in the trigger code (``retrieval_verb_N`` only names the cue verb); it's
  recovered from the encoding markers earlier in the same recording via
  :mod:`analysis_lib.task_labels`. This is the real reactivation-from-memory
  question.

All three sources replay raw EEG fresh through ``OnlinePreprocessor`` + a
caller-chosen ``LiveInferenceEngine`` (see ``analysis_lib.context``) — so any
model/hyperparameters/pos-neg classes can be tried for any source, not just
FL. ``"encoding"``/``"retrieval"`` pull their markers directly from the raw
recording's own annotations (like FL does), not from a live run's saved
``predictions.csv`` — that file reflects only whichever specific decoder
produced it, once, in the past.
"""
from __future__ import annotations

from collections import Counter

from analysis_lib import streaming, task_labels


def _check_encoding_markers(event_mapping: dict[str, int]) -> None:
    """Fail loudly if the loaded config doesn't define the markers this source needs."""
    if not any(not task_labels.VERB_RE.match(n) and n.startswith("learning_")
               and task_labels.category_of(n[len("learning_"):]) is not None
               for n in event_mapping):
        raise ValueError(
            "encoding source needs at least one 'learning_<category>_NN' marker "
            f"in the config's markers_mapping; loaded config only has: {sorted(event_mapping)}"
        )


def _check_retrieval_markers(event_mapping: dict[str, int]) -> None:
    """Fail loudly if the loaded config doesn't define the markers this source needs.

    ``task_labels``' verb/end/recall marker names are Python defaults, not
    config content — if a config renames or drops them, matching should raise
    here instead of silently epoching zero trials.
    """
    required = (task_labels.RETRIEVAL_END, task_labels.RECALL_KEY_PRESS)
    missing = [m for m in required if m not in event_mapping]
    if missing:
        raise ValueError(
            f"retrieval source needs {missing} in the config's markers_mapping; "
            f"loaded config only has: {sorted(event_mapping)}"
        )
    for label, pattern in (("learning_verb_*", task_labels.VERB_RE),
                           ("retrieval_verb_*", task_labels.RETRIEVAL_VERB_RE)):
        if not any(pattern.match(n) for n in event_mapping):
            raise ValueError(
                f"retrieval source needs at least one '{label}' marker in the "
                f"config's markers_mapping; loaded config only has: {sorted(event_mapping)}"
            )


def build_fl_samples(ctx, raw, dc, *, n_times=None):
    """Return ``(samples_by_group, info)`` for the functional-localizer trigger stream."""
    markers = streaming.extract_markers(raw, ctx.event_mapping, dc.raw_markers, n_times=n_times)
    sbg: dict[str, list[int]] = {}
    for s, c in markers:
        g = dc.code_to_group.get(c)
        if g is not None:
            sbg.setdefault(g, []).append(s)
    counts = {ctx.name_by_code[c]: n for c, n in Counter(c for _, c in markers).items()}
    return sbg, f"FL markers: {counts}"


def _extract_task_markers(ctx, raw, *, n_times=None) -> list[task_labels.Marker]:
    """Every configured marker name, pulled from ``raw``'s own annotations, sample-indexed.

    Passes every name in ``ctx.event_mapping`` (not a curated subset) — names
    absent from this particular recording (e.g. FL-only image/rest markers on
    a task recording) simply produce no matches; ``task_labels``' pattern
    matching already tolerates unrelated interleaved markers.
    """
    pairs = streaming.extract_markers(raw, ctx.event_mapping, list(ctx.event_mapping), n_times=n_times)
    return [task_labels.Marker(t=s, code=c, name=ctx.name_by_code[c]) for s, c in pairs]


def _epoch_by_group(out_samples, sfreq, fs_out, dc, trials, preds, *, tmin, tmax):
    """Shared plumbing: epoch a freshly-replayed prediction stream around labeled ``trials``.

    ``trials`` are ``{"t": <raw sample index>, "true_label": ...}`` records
    (already resolved by the caller — encoding or retrieval). Returns
    ``(t_grid, epoched)`` where ``epoched`` is ``{task: {group: (n_epochs, n_grid)}}``,
    keyed by ``dc``'s display groups.
    """
    t_grid, epoch_stream = streaming.make_epocher(out_samples, sfreq, fs_out, tmin, tmax)
    epoched = {
        task: {group: epoch_stream(prob, sorted(t["t"] for t in trials if t["true_label"] == group))
               for group in dc.display_markers}
        for task, prob in preds.items()
    }
    return t_grid, epoched


def build_encoding_epochs(ctx, raw, dc, out_samples, sfreq, fs_out, preds, *, tmin, tmax, n_times=None):
    """Return ``(t_grid, epoched, trials, info)`` for the recording's encoding phase.

    ``raw``, ``out_samples``, ``sfreq``, ``fs_out``, ``preds`` are the outputs
    of a fresh replay of ``raw`` through ``OnlinePreprocessor`` + a chosen
    ``LiveInferenceEngine`` (``streaming.load_recording`` ->
    ``streaming.run_online_stream`` -> ``engine.predict``) — exactly what
    ``"fl"`` already computes; this function only labels + epochs it.

    ``epoched`` is ``{task: {category: (n_epochs, n_grid)}}``, keyed by each
    encoding trial's true (shown) category — a same-modality (perception)
    held-out sanity check, since an image really was on screen (unlike
    retrieval). ``dc`` must pool the per-image markers into category-level
    display groups, e.g.
    ``plots.display_config(ctx, marker_groups=task_labels.marker_groups_by_category(ctx.event_mapping))``.
    """
    _check_encoding_markers(ctx.event_mapping)
    markers = _extract_task_markers(ctx, raw, n_times=n_times)
    trials = task_labels.encoding_trials(markers)

    t_grid, epoched = _epoch_by_group(out_samples, sfreq, fs_out, dc, trials, preds, tmin=tmin, tmax=tmax)

    info = (f"[encoding] {len(trials)} image onsets | "
            f"by true category {dict(Counter(t['true_label'] for t in trials))}")
    return t_grid, epoched, trials, info


def build_retrieval_epochs(ctx, raw, dc, out_samples, sfreq, fs_out, preds, *, tmin, tmax, n_times=None):
    """Return ``(t_grid, epoched, trials, info)`` for the recording's retrieval phase.

    Same replay contract as :func:`build_encoding_epochs`. ``epoched`` is
    ``{task: {category: (n_epochs, n_grid)}}``, keyed by each retrieval
    trial's *true* (encoded) category — an honest held-out test of whether
    the decoder reactivates the right category during recall. ``dc`` must
    pool the per-image markers into category-level display groups, e.g.
    ``plots.display_config(ctx, marker_groups=task_labels.marker_groups_by_category(ctx.event_mapping))``.
    """
    _check_retrieval_markers(ctx.event_mapping)
    markers = _extract_task_markers(ctx, raw, n_times=n_times)
    couples = task_labels.group_couple_trials(markers)
    verb_category = task_labels.verb_categories(couples)
    trials = task_labels.retrieval_trials(markers, verb_category)

    t_grid, epoched = _epoch_by_group(out_samples, sfreq, fs_out, dc, trials, preds, tmin=tmin, tmax=tmax)

    n_recalled = sum(t["recalled"] for t in trials)
    info = (f"[retrieval] {len(trials)} cued trials, {n_recalled} with a recall key-press | "
            f"by true category {dict(Counter(t['true_label'] for t in trials))}")
    return t_grid, epoched, trials, info
