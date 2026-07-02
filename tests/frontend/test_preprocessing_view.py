"""Headless tests for PreprocessingView's ICA-title annotation.

``_annotate_ica_titles`` renders ICLabel's category + confidence as each
``plot_components`` axes' xlabel, leaving the subplot title as the bare index
(``ICAxyz``). The critical invariant is that MNE's own title-click handler —
which recovers the component index via ``int(title.split(" ")[0][-3:])``
(mne/viz/topomap.py) — keeps working, so the operator can still toggle
reject/keep.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frontend.views.preprocessing_view import PreprocessingView  # noqa: E402


def _mne_like_fig(n: int):
    """A figure mimicking plot_components: one axis per component, titled
    'ICAxyz', plus a colorbar-like axis whose title is not an ICA name."""
    fig, axes = plt.subplots(1, n + 1)
    for i in range(n):
        axes[i].set_title(f"ICA{i:03d}")
    axes[n].set_title("AU")  # colorbar-style axis — must be left untouched
    return fig


def _mne_recover_index(title: str) -> int:
    """Mirror MNE's onclick_title index parse to prove it still works."""
    return int(title.split(" ")[0][-3:])


def test_label_goes_to_xlabel_index_stays_in_title():
    fig = _mne_like_fig(3)
    labels = [("brain", 0.91), ("eye", 0.99), ("muscle", 0.85)]
    PreprocessingView._annotate_ica_titles(fig, labels)

    # Titles stay the bare index; the classification moves to the xlabel.
    assert [ax.get_title() for ax in fig.axes[:3]] == ["ICA000", "ICA001", "ICA002"]
    assert [ax.get_xlabel() for ax in fig.axes[:3]] == [
        "brain 91%", "eye 99%", "muscle 85%",
    ]
    # Colorbar axis untouched.
    assert fig.axes[3].get_title() == "AU"
    assert fig.axes[3].get_xlabel() == ""
    plt.close(fig)


def test_index_parse_still_works_after_annotation():
    fig = _mne_like_fig(3)
    labels = [("brain", 0.91), ("eye", 0.99), ("muscle", 0.85)]
    PreprocessingView._annotate_ica_titles(fig, labels)

    for expected_idx, ax in enumerate(fig.axes[:3]):
        assert _mne_recover_index(ax.get_title()) == expected_idx
    plt.close(fig)


def test_none_labels_leaves_titles_unchanged():
    fig = _mne_like_fig(2)
    PreprocessingView._annotate_ica_titles(fig, None)
    assert [ax.get_title() for ax in fig.axes[:2]] == ["ICA000", "ICA001"]
    plt.close(fig)


def test_handles_list_of_figs_and_short_label_list():
    fig = _mne_like_fig(3)
    # Only two labels for three components — extra component left as-is, no crash.
    PreprocessingView._annotate_ica_titles([fig], [("brain", 0.9), ("eye", 0.99)])
    assert fig.axes[0].get_xlabel() == "brain 90%"
    assert fig.axes[1].get_xlabel() == "eye 99%"
    # Third component has no label → no xlabel, title still the bare index.
    assert fig.axes[2].get_xlabel() == ""
    assert fig.axes[2].get_title() == "ICA002"
    plt.close(fig)
