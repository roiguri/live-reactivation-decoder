"""Unit checks for the pure (backend-free) analysis_lib helpers."""
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_lib import (  # noqa: E402
    cross_domain,
    metrics,
    online_validation,
    streaming,
    task_labels,
)


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


def test_block_starts_opens_a_block_per_learning_phase():
    """Two learning->retrieval cycles: a block opens at the first learning_verb
    of each learning phase (the first one, and the first after any retrieval)."""
    markers = _mk([
        # block 0 — learning phase (two study cues), then retrieval phase
        (0.0, 201, "learning_verb_1"),
        (2.0, 211, "learning_animate_01"),
        (4.0, 205, "learning_verb_5"),
        (6.0, 215, "learning_inanimate_02"),
        (10.0, 221, "retrieval_verb_1"),
        (16.0, 85, "retrieval_end"),
        (17.0, 225, "retrieval_verb_5"),
        (23.0, 85, "retrieval_end"),
        # block 1 — new learning phase, then retrieval
        (30.0, 201, "learning_verb_1"),
        (32.0, 211, "learning_animate_01"),
        (36.0, 221, "retrieval_verb_1"),
        (42.0, 85, "retrieval_end"),
    ])
    assert task_labels.block_starts(markers) == [0.0, 30.0]


def test_block_starts_empty_when_no_learning_cues():
    markers = _mk([
        (0.0, 221, "retrieval_verb_1"),
        (6.0, 85, "retrieval_end"),
    ])
    assert task_labels.block_starts(markers) == []


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


# ── step 2: online-pipeline validation ──────────────────────────────────────

_SETTINGS = {
    "model": "Logistic", "scale_method": "standard", "params": {},
    "random_state": 0, "cv": {"k": 3},
}


def _separable_timeseries(rng, *, n=30, n_ch=3, n_t=5, sep=3.0):
    """Two-class Gaussian blobs decodable at every timepoint: (2n, n_ch, n_t), y."""
    X = np.concatenate([rng.normal(-sep, 0.5, (n, n_ch, n_t)),
                        rng.normal(sep, 0.5, (n, n_ch, n_t))])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_make_epocher_multichannel_shape_and_interp():
    out = np.arange(0, 300)
    t_grid, epoch = streaming.make_epocher_multichannel(
        out, sfreq=100.0, fs_out=100.0, tmin=-0.1, tmax=0.5)
    assert len(t_grid) == 61
    feats = np.stack([np.linspace(0, 1, 300), np.linspace(1, 2, 300)], axis=1)  # (300, 2)
    arr, kept = epoch(feats, [100, 200])
    assert arr.shape == (2, 2, 61)  # (n_trials, n_ch, n_grid)
    assert kept == [100, 200]
    # t=0 lands exactly on the marker sample -> interpolated value == raw sample
    zero_idx = int(np.argmin(np.abs(t_grid)))
    assert np.isclose(arr[0, 0, zero_idx], feats[100, 0])
    assert np.isclose(arr[0, 1, zero_idx], feats[100, 1])


def test_make_epocher_multichannel_skips_uncovered_marker():
    out = np.arange(0, 120)
    _, epoch = streaming.make_epocher_multichannel(
        out, sfreq=100.0, fs_out=100.0, tmin=-0.1, tmax=0.5)
    feats = np.stack([np.linspace(0, 1, 120)] * 2, axis=1)
    arr, kept = epoch(feats, [1000])  # no surrounding samples -> dropped
    assert arr.shape == (0, 2, 61)
    assert kept == []


def test_pair_by_onset_drops_unmatched():
    pairs = online_validation._pair_by_onset([0.0, 1.0, 2.0], [0.0, 2.0], tol=0.02)
    assert pairs == [(0, 0), (2, 1)]  # the 1.0s offline trial has no online partner


def test_cross_pipeline_tgm_identical_pipelines_match():
    rng = np.random.default_rng(0)
    X, y = _separable_timeseries(rng)
    res = online_validation.cross_pipeline_tgm(X, X, y, _SETTINGS)
    n_t = X.shape[2]
    assert res["tgm_off"].shape == (n_t, n_t)
    # same fits, same test data -> offline and online branches are identical
    assert np.allclose(res["diag_off"], res["diag_on"])
    assert res["diag_off"].max() > 0.9
    assert (res["tgm_on"] >= 0.0).all() and (res["tgm_on"] <= 1.0).all()


def test_cross_pipeline_tgm_degraded_online_loses_auc():
    rng = np.random.default_rng(1)
    X_off, y = _separable_timeseries(rng)
    X_on = X_off + rng.normal(0, 8.0, X_off.shape)  # heavy noise wrecks separability
    res = online_validation.cross_pipeline_tgm(X_off, X_on, y, _SETTINGS)
    assert res["diag_on"].mean() < res["diag_off"].mean()


def test_operating_point_degradation_identical_pipelines_pass():
    rng = np.random.default_rng(2)
    X, y = _separable_timeseries(rng)
    res = online_validation.operating_point_degradation(
        X, X, y, _SETTINGS, timepoint_idx=2, n_boot=200,
        rng=np.random.default_rng(0))
    assert np.isclose(res["auc_off"], res["auc_on"])
    assert abs(res["delta"]) < 1e-9
    assert res["passed"] is True


def test_operating_point_degradation_degraded_online_drops():
    rng = np.random.default_rng(3)
    X_off, y = _separable_timeseries(rng)
    X_on = X_off + rng.normal(0, 8.0, X_off.shape)
    res = online_validation.operating_point_degradation(
        X_off, X_on, y, _SETTINGS, timepoint_idx=2, n_boot=200,
        rng=np.random.default_rng(0))
    assert res["auc_on"] < res["auc_off"]
    assert res["pct_drop"] > 0.0
    assert 0.0 <= res["auc_on"] <= 1.0


# ── step 3: cross-domain generalization ──────────────────────────────────────

def test_binary_test_set_maps_categories_and_drops_irrelevant():
    task_cfg = {
        "name": "animate decoder",
        "pos_labels": ["animate_01", "animate_02"],
        "neg_labels": ["inanimate_01", "rest_fixation"],  # rest -> None, dropped
    }
    X_all = np.arange(4 * 2 * 3).reshape(4, 2, 3)  # (n_trials, n_ch, n_grid)
    cat_labels = ["animate", "inanimate", "scene", "animate"]  # "scene" in neither
    X_test, y_test = cross_domain.binary_test_set(X_all, cat_labels, task_cfg)
    assert X_test.shape == (3, 2, 3)  # the "scene" trial dropped
    assert y_test.tolist() == [1, 0, 1]
    np.testing.assert_array_equal(X_test, X_all[[0, 1, 3]])


def test_binary_test_set_rejects_overlapping_polarity():
    import pytest
    task_cfg = {"name": "bad", "pos_labels": ["animate_01"], "neg_labels": ["animate_02"]}
    with pytest.raises(ValueError, match="both pos and neg"):
        cross_domain.binary_test_set(np.zeros((2, 1, 1)), ["animate", "animate"], task_cfg)


def test_binary_test_set_single_class_raises():
    import pytest
    task_cfg = {"name": "animate decoder", "pos_labels": ["animate_01"],
                "neg_labels": ["inanimate_01"]}
    with pytest.raises(ValueError, match="single class"):
        cross_domain.binary_test_set(np.zeros((2, 1, 1)), ["animate", "animate"], task_cfg)


def test_tgm_auc_matches_roc_auc_score_per_cell_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n, n_fl, n_test = 30, 4, 5
    # integer scores force ties, exercising average-rank handling
    scores = rng.integers(0, 6, size=(n, n_fl, n_test)).astype(float)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    tgm = cross_domain.tgm_auc(scores, y)
    assert tgm.shape == (n_fl, n_test)
    for i in range(n_fl):
        for j in range(n_test):
            assert np.isclose(tgm[i, j], roc_auc_score(y, scores[:, i, j]))


def _scores_with_signal(rng, *, n=40, n_fl=4, n_test=5, fl_row=1, test_col=3, sep=4.0):
    """Noise everywhere except one (fl_row, test_col) cell that separates the classes."""
    scores = rng.normal(0, 1.0, (n, n_fl, n_test))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    scores[y == 1, fl_row, test_col] += sep
    return scores, y, fl_row, test_col


def test_row_permutation_finds_signal_latency_and_is_significant():
    rng = np.random.default_rng(1)
    scores, y, fl_row, test_col = _scores_with_signal(rng)
    res = cross_domain.row_permutation(scores, y, fl_row, n_perm=500,
                                       rng=np.random.default_rng(0))
    assert res["obs_row"].shape == (scores.shape[2],)
    assert res["null_max"].shape == (500,)
    assert res["argmax_test_idx"] == test_col   # latency of the planted effect
    assert res["max_auc"] > 0.9
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] < 0.05


def test_matrix_permutation_locates_cell_both_axes():
    rng = np.random.default_rng(4)
    scores, y, fl_row, test_col = _scores_with_signal(rng, sep=4.0)
    res = cross_domain.matrix_permutation(scores, y, n_perm=500,
                                          rng=np.random.default_rng(0))
    assert res["tgm"].shape == scores.shape[1:]
    assert res["null_max"].shape == (500,)
    assert (res["fl_idx"], res["test_idx"]) == (fl_row, test_col)  # 2-D search finds it
    assert res["max_auc"] > 0.9
    assert res["p_value"] < 0.05
    # whole-map max is >= any single row's max (searches strictly more cells)
    row = cross_domain.row_permutation(scores, y, fl_row, n_perm=1,
                                       rng=np.random.default_rng(0))
    assert res["max_auc"] >= row["max_auc"] - 1e-12


def test_row_permutation_pure_noise_pbounds():
    rng = np.random.default_rng(2)
    n, n_fl, n_test = 40, 3, 4
    scores = rng.normal(0, 1.0, (n, n_fl, n_test))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    res = cross_domain.row_permutation(scores, y, 1, n_perm=300,
                                       rng=np.random.default_rng(0))
    # Efron-Tibshirani floor: never exactly 0
    assert 1 / (300 + 1) <= res["p_value"] <= 1.0


def test_cell_bootstrap_orders_and_brackets_observed():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    scores, y, fl_row, test_col = _scores_with_signal(rng)
    observed = roc_auc_score(y, scores[:, fl_row, test_col])
    lo, hi = cross_domain.cell_bootstrap(scores, y, fl_row, test_col, n_boot=400,
                                         rng=np.random.default_rng(0))
    assert lo <= hi
    assert lo <= observed <= hi


def test_stouffer_combines_and_sharpens():
    z1, p1 = cross_domain.stouffer([0.05, 0.05, 0.05])
    # three concordant results are jointly more significant than any one
    assert p1 < 0.05
    assert z1 > 0.0
    # monotone: smaller inputs -> smaller combined p
    _, p2 = cross_domain.stouffer([0.2, 0.2, 0.2])
    assert p1 < p2 < 1.0


def test_stouffer_rejects_zero_pvalue():
    import pytest
    with pytest.raises(ValueError, match="must be in"):
        cross_domain.stouffer([0.0, 0.1, 0.1])
