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


def test_parse_vmrk_and_stage3_split():
    # Stage 2 reuses 51/53; Stage 3 begins at the first probe (71).
    text = "\n".join([
        "Mk1=Stimulus,S 51,1000,1,0",   # Stage 2 test: show object
        "Mk2=Stimulus,S 53,1100,1,0",   # Stage 2 test: start retrieval
        "Mk3=Stimulus,S 55,1200,1,0",   # Stage 2 only: remember question
        "Mk4=Stimulus,S 71,2000,1,0",   # --- Stage 3 begins ---
        "Mk5=Stimulus,S 53,2200,1,0",   # Stage 3 retrieval start
        "Mk6=Stimulus,S 71,3000,1,0",
        "Mk7=Stimulus,S 53,3200,1,0",
    ])
    markers = task_labels.parse_vmrk_text(text)
    assert markers[0] == (51, 1000)
    s3 = task_labels.stage3_markers(markers)
    assert s3[0] == (71, 2000)                       # split at first probe
    assert task_labels.samples_of(s3, 53) == [2200, 3200]  # only Stage 3 retrievals


def test_stage3_trials_joins_labels_and_rejects_count_mismatch():
    import pytest
    markers = task_labels.parse_vmrk_text("\n".join([
        "Mk1=Stimulus,S 71,2000,1,0",
        "Mk2=Stimulus,S 53,2200,1,0",
        "Mk3=Stimulus,S 71,3000,1,0",
        "Mk4=Stimulus,S 53,3200,1,0",
    ]))
    pr = {
        "trial_1": {"shoe": {"probe": "colors", "is_remember": True, "subject_answer": "red"},
                    "retrival_success": True, "trial_times": {}},
        "trial_2": {"sock": {"probe": "scenes", "is_remember": True, "subject_answer": "kitchen"},
                    "retrival_success": False, "trial_times": {}},
    }
    rows = task_labels.stage3_trials(markers, pr, true_label_of={"shoe": "red", "sock": "living_room"})
    assert [r["sample"] for r in rows] == [2200, 3200]
    assert rows[0]["reported_label"] == "red" and rows[0]["true_label"] == "red"
    assert rows[1]["true_label"] == "living_room"   # ground truth, not the reported "kitchen"
    grouped = task_labels.group_samples_by_label(rows, key="true_label")
    assert grouped == {"red": [2200], "living_room": [3200]}
    # one extra anchor, two trials -> alignment unsafe -> raise
    bad = markers + [(53, 4200)]
    with pytest.raises(ValueError, match="alignment unsafe"):
        task_labels.stage3_trials(bad, pr)


def test_stage2_learning_trials_dual_label():
    # Stage 2 region = markers before the first probe (71). Learning anchored on 41.
    markers = task_labels.parse_vmrk_text("\n".join([
        "Mk1=Stimulus,S 41,1000,1,0",
        "Mk2=Stimulus,S 41,2000,1,0",
        "Mk3=Stimulus,S 71,9000,1,0",   # Stage 3 begins — must be excluded
        "Mk4=Stimulus,S 41,9500,1,0",
    ]))
    ta = {
        "1": {"shoe": {"colors": "red", "scenes": "kitchen"}, "trial_times": {}},
        "2": {"sock": {"colors": "green", "scenes": "bathroom"}, "trial_times": {}},
    }
    rows = task_labels.stage2_learning_trials(markers, ta)
    assert [r["sample"] for r in rows] == [1000, 2000]      # the post-71 41 is excluded
    assert rows[0]["colour_label"] == "red" and rows[0]["scene_label"] == "kitchen"
    # a trial contributes to BOTH a colour and a scene group
    merged = {**task_labels.group_samples_by_label(rows, key="colour_label"),
              **task_labels.group_samples_by_label(rows, key="scene_label")}
    assert merged == {"red": [1000], "green": [2000], "kitchen": [1000], "bathroom": [2000]}


def test_stage2_test_trials_orders_by_test_index_and_carries_correctness():
    markers = task_labels.parse_vmrk_text("\n".join([
        "Mk1=Stimulus,S 55,1000,1,0",
        "Mk2=Stimulus,S 55,2000,1,0",
    ]))
    # subject_answer keyed in test order; object key is first, with report flags beside it
    subj = {
        "trial_1": {"closet": {"colors": "yellow", "scenes": "kitchen"},
                    "retrival_report_color": True, "retrival_report_scene": True, "trial_times": {}},
        "trial_2": {"shoe": {"colors": "red", "scenes": "living_room"},
                    "retrival_success": True, "trial_times": {}},
    }
    combined = {
        "closet": {"colors": "yellow", "scenes": "kitchen",
                   "color_correct": "TRUE", "scene_correct": "TRUE", "both_correct": "TRUE", "difficulty": "2"},
        "shoe": {"colors": "red", "scenes": "living_room",
                 "color_correct": "TRUE", "scene_correct": "FALSE", "both_correct": "FALSE", "difficulty": "3"},
    }
    rows = task_labels.stage2_test_trials(markers, subj, combined)
    assert rows[0]["object"] == "closet" and rows[0]["both_correct"] is True
    assert rows[1]["scene_correct"] is False and rows[1]["colour_label"] == "red"


def test_gap_residuals_zero_when_aligned():
    res = task_labels.gap_residuals([1000, 1500, 2300], [1.0, 1.5, 2.3], sfreq=1000.0)
    assert max(res) < 1e-9


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
