"""Unit checks for the pure (backend-free) analysis_lib helpers."""
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_lib import metrics, streaming, task_labels  # noqa: E402


def test_imports_without_backend_on_path():
    from analysis_lib import context  # noqa: F401  (lazy backend imports inside funcs)


def test_make_epocher_grid_and_shape():
    out = np.arange(0, 300)
    t_grid, epoch = streaming.make_epocher(out, sfreq=100.0, fs_out=100.0, tmin=-0.1, tmax=0.5)
    assert len(t_grid) == 61
    rows = epoch(np.linspace(0, 1, 300), [100, 200])
    assert rows.shape == (2, 61)


def test_make_epocher_skips_markers_without_coverage():
    out = np.arange(0, 120)
    _, epoch = streaming.make_epocher(out, sfreq=100.0, fs_out=100.0, tmin=-0.1, tmax=0.5)
    # marker at 1000 has no surrounding samples -> dropped
    rows = epoch(np.linspace(0, 1, 120), [1000])
    assert rows.shape[0] == 0


def test_extract_markers_parses_codes():
    info = mne.create_info(["Cz"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((1, 500)), info, verbose=False)
    raw.set_annotations(mne.Annotations(
        [1.0, 2.0, 3.0], [0, 0, 0], ["Stimulus/S 11", "Stimulus/S 12", "Stimulus/S 11"]))
    markers = streaming.extract_markers(raw, {"red": 11, "green": 12}, ["red"])
    assert [c for _, c in markers] == [11, 11]


def test_winner_confusion_perfect_separation():
    markers, tasks = ["red", "green"], ["red decoder", "green decoder"]
    mot = {"red decoder": "red", "green decoder": "green"}
    t_grid = np.linspace(-0.1, 0.5, 61)
    ep = {
        "red decoder": {"red": np.full((3, 61), 0.9), "green": np.zeros((2, 61))},
        "green decoder": {"red": np.zeros((3, 61)), "green": np.full((2, 61), 0.9)},
    }
    conf = metrics.winner_confusion(ep, markers, tasks, mot, t_grid)
    assert np.array_equal(conf, np.array([[3, 0], [0, 2]]))
    assert metrics.confusion_scores(conf)["accuracy"] == 1.0


def test_winner_confusion_weighted_prob_mode():
    # weighted_prob: winner = argmax of tp-weighted mean P(t). Here the red
    # decoder reads higher on red trials only near its tp, while vote-mode (rank)
    # would agree too — we just check the mode runs and keeps the diagonal.
    markers, tasks = ["red", "green"], ["red decoder", "green decoder"]
    mot = {"red decoder": "red", "green decoder": "green"}
    t_grid = np.linspace(-0.1, 0.5, 61)
    ep = {
        "red decoder": {"red": np.full((3, 61), 0.8), "green": np.full((2, 61), 0.2)},
        "green decoder": {"red": np.full((3, 61), 0.2), "green": np.full((2, 61), 0.8)},
    }
    tps = {"red decoder": 0.2, "green decoder": 0.2}
    conf = metrics.winner_confusion(ep, markers, tasks, mot, t_grid,
                                    mode="weighted_prob", task_tps=tps, sigma=0.05)
    assert np.array_equal(conf, np.array([[3, 0], [0, 2]]))


def test_winner_confusion_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError, match="unknown winner mode"):
        metrics.winner_confusion({}, [], [], {}, np.array([0.0]), mode="bogus")


def test_modality_groups_within_splits_by_modality():
    within = [
        {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow", "rest"]},
        {"name": "green decoder", "pos_labels": ["green"], "neg_labels": ["red", "yellow", "rest"]},
        {"name": "yellow decoder", "pos_labels": ["yellow"], "neg_labels": ["red", "green", "rest"]},
        {"name": "lr decoder", "pos_labels": ["living_room"], "neg_labels": ["bathroom", "kitchen"]},
        {"name": "bath decoder", "pos_labels": ["bathroom"], "neg_labels": ["living_room", "kitchen"]},
        {"name": "kit decoder", "pos_labels": ["kitchen"], "neg_labels": ["living_room", "bathroom"]},
    ]
    groups = metrics.modality_groups(within)
    by_marker = {tuple(gm): gt for gm, gt in groups}
    assert len(groups) == 2
    assert ("red", "green", "yellow") in by_marker
    assert ("living_room", "bathroom", "kitchen") in by_marker


def test_modality_groups_crossmodal_is_single_block():
    # Both decoders' label-set = {red, green, yellow, living_room} -> one block.
    cross = [
        {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow", "living_room"]},
        {"name": "lr decoder", "pos_labels": ["living_room"], "neg_labels": ["red", "green", "yellow"]},
    ]
    assert len(metrics.modality_groups(cross)) == 1


def test_baseline_correct_subtracts_prestim_mean():
    t_grid = np.array([-0.1, 0.0, 0.1, 0.2])     # pre-stim window = t < 0 -> [-0.1]
    ep = {"d": {"m": np.array([[0.3, 0.4, 0.6, 0.8]])}}  # prestim mean = 0.3
    out = metrics.baseline_correct(ep, t_grid)
    np.testing.assert_allclose(out["d"]["m"][0], [0.0, 0.1, 0.3, 0.5])
    # original is untouched (returns a copy)
    np.testing.assert_allclose(ep["d"]["m"][0], [0.3, 0.4, 0.6, 0.8])


def test_category_of():
    assert task_labels.category_of("animate_02") == "animate"
    assert task_labels.category_of("inanimate_11") == "inanimate"
    assert task_labels.category_of("recall_key_press") is None


def _mk(rows):
    return [task_labels.Marker(t, c, n) for t, c, n in rows]


def test_group_couple_trials_pairs_verb_with_following_image():
    markers = _mk([
        (0.0, 205, "learning_verb_5"),
        (2.0, 215, "learning_inanimate_02"),
        (7.0, 83, "encoding_response"),
        (8.0, 201, "learning_verb_1"),
        (11.0, 211, "learning_animate_01"),
        (16.0, 83, "encoding_response"),
    ])
    couples = task_labels.group_couple_trials(markers)
    assert [(c["verb"], c["image"], c["category"]) for c in couples] == [
        ("5", "inanimate_02", "inanimate"),
        ("1", "animate_01", "animate"),
    ]


def test_group_couple_trials_with_animate_inanimate_verb_names():
    """Verb identity is opaque — works the same for a config that spells out
    the category in the verb marker name (experiment_config.realtime_animacy_
    verb_labels.yaml's convention) as for the bare-index convention above."""
    markers = _mk([
        (0.0, 204, "learning_verb_inanimate_1"),
        (2.0, 215, "learning_inanimate_02"),
        (7.0, 83, "encoding_response"),
        (8.0, 201, "learning_verb_animate_1"),
        (11.0, 211, "learning_animate_01"),
        (16.0, 83, "encoding_response"),
    ])
    couples = task_labels.group_couple_trials(markers)
    assert [(c["verb"], c["image"], c["category"]) for c in couples] == [
        ("inanimate_1", "inanimate_02", "inanimate"),
        ("animate_1", "animate_01", "animate"),
    ]


def test_verb_categories_majority_vote_and_conflict():
    couples = [
        {"verb": "1", "category": "animate"},
        {"verb": "1", "category": "animate"},
        {"verb": "2", "category": "inanimate"},
    ]
    assert task_labels.verb_categories(couples) == {"1": "animate", "2": "inanimate"}

    import pytest
    with pytest.raises(ValueError, match="inconsistent categories"):
        task_labels.verb_categories([
            {"verb": "1", "category": "animate"},
            {"verb": "1", "category": "inanimate"},
        ])


def test_encoding_trials_labels_image_onsets_and_skips_verb_cues():
    markers = _mk([
        (0.0, 205, "learning_verb_5"),
        (2.0, 215, "learning_inanimate_02"),
        (7.0, 83, "encoding_response"),
        (8.0, 201, "learning_verb_1"),
        (11.0, 211, "learning_animate_01"),
        (16.0, 83, "encoding_response"),
    ])
    trials = task_labels.encoding_trials(markers)
    assert [(t["t"], t["image"], t["true_label"]) for t in trials] == [
        (2.0, "inanimate_02", "inanimate"),
        (11.0, "animate_01", "animate"),
    ]
    grouped = task_labels.group_samples_by_label(trials)
    assert grouped == {"inanimate": [2.0], "animate": [11.0]}


def test_retrieval_trials_labels_and_recall_flag():
    markers = _mk([
        (0.0, 222, "retrieval_verb_2"),
        (5.0, 9, "recall_key_press"),
        (6.0, 85, "retrieval_end"),
        (6.1, 88, "feature_question"),
        (7.0, 89, "feature_answer"),
        (8.0, 224, "retrieval_verb_4"),
        (13.0, 85, "retrieval_end"),   # no recall_key_press this trial
        (13.1, 88, "feature_question"),
        (14.0, 89, "feature_answer"),
    ])
    trials = task_labels.retrieval_trials(markers, {"2": "animate", "4": "inanimate"})
    assert [(t["verb"], t["true_label"], t["recalled"]) for t in trials] == [
        ("2", "animate", True),
        ("4", "inanimate", False),
    ]
    grouped = task_labels.group_samples_by_label(trials)
    assert grouped == {"animate": [0.0], "inanimate": [8.0]}


def test_retrieval_trials_with_animate_inanimate_verb_names():
    markers = _mk([
        (0.0, 221, "retrieval_verb_animate_1"),
        (5.0, 9, "recall_key_press"),
        (6.0, 85, "retrieval_end"),
        (8.0, 224, "retrieval_verb_inanimate_1"),
        (13.0, 85, "retrieval_end"),   # no recall_key_press this trial
    ])
    trials = task_labels.retrieval_trials(
        markers, {"animate_1": "animate", "inanimate_1": "inanimate"}
    )
    assert [(t["verb"], t["true_label"], t["recalled"]) for t in trials] == [
        ("animate_1", "animate", True),
        ("inanimate_1", "inanimate", False),
    ]


def test_display_config_identity_and_targets():
    from analysis_lib import plots

    class _Settings:
        @staticmethod
        def get_decoder_settings():
            return {"tasks": [
                {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green"]},
                {"name": "green decoder", "pos_labels": ["green"], "neg_labels": ["red"]},
            ]}

    class _Ctx:
        settings = _Settings()
        event_mapping = {"red": 11, "green": 12}

    dc = plots.display_config(_Ctx())
    assert dc.display_markers == ["red", "green"]
    assert dc.code_to_group == {11: "red", 12: "green"}
    assert dc.is_target("red decoder", "red") and not dc.is_target("red decoder", "green")
    assert dc.target_group("green decoder") == "green"


def test_perm_band_shapes():
    markers = ["red", "green"]
    ep = {"red decoder": {"red": np.random.default_rng(0).random((10, 20)),
                          "green": np.random.default_rng(1).random((8, 20))}}
    obs, lo, hi, nmean = metrics.perm_band(ep, "red decoder", "red", markers, n_perm=50)
    assert obs.shape == lo.shape == hi.shape == nmean.shape == (20,)
    assert np.all(lo <= hi)


def test_permutation_auc_pvalue_separable_signal():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(0)
    n = 40
    X = np.concatenate([rng.normal(-2, 0.5, (n, 3)), rng.normal(2, 0.5, (n, 3))])
    y = np.array([0] * n + [1] * n)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    observed, null, p_value = metrics.permutation_auc_pvalue(
        X, y, LogisticRegression(), cv, n_perm=100, rng=np.random.default_rng(1))

    assert null.shape == (100,)
    assert 0.0 <= p_value <= 1.0
    assert observed > 0.9
    assert p_value < 0.05


def test_permutation_auc_pvalue_reports_progress():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(0)
    n = 20
    X = np.concatenate([rng.normal(-2, 0.5, (n, 3)), rng.normal(2, 0.5, (n, 3))])
    y = np.array([0] * n + [1] * n)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

    calls = []
    metrics.permutation_auc_pvalue(
        X, y, LogisticRegression(), cv, n_perm=10, rng=np.random.default_rng(1),
        on_progress=lambda done, total: calls.append((done, total)))

    assert calls == [(i, 10) for i in range(1, 11)]
