import copy

import pytest
from pydantic import ValidationError

from backend.core.settings_manager import SettingsManager


class TestLoad:
    def test_loads_valid_config(self, sample_config_path):
        sm = SettingsManager(sample_config_path)
        assert sm is not None

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SettingsManager(tmp_path / "nonexistent.yaml")

    def test_raises_on_missing_required_field(self, tmp_config_file):
        # markers_mapping is required — omitting it must fail
        data = {"experiment_info": {"name": "test"}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(data))

    def test_raises_on_wrong_type(self, tmp_config_file):
        data = {
            "experiment_info": {"name": "test"},
            "markers_mapping": {"events": [{"id": "not_an_int", "name": "red"}]},
        }
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(data))

    def test_rejects_extra_top_level_keys(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["unknown_section"] = {"foo": "bar"}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_error_message_contains_filepath(self, tmp_config_file):
        data = {"experiment_info": {"name": "test"}}
        path = tmp_config_file(data)
        with pytest.raises(ValueError, match=str(path)):
            SettingsManager(path)


class TestGetPreprocessingParams:
    def test_returns_dict(self, sample_config_path):
        params = SettingsManager(sample_config_path).get_preprocessing_params()
        assert isinstance(params, dict)

    def test_contains_all_sections(self, sample_config_path):
        params = SettingsManager(sample_config_path).get_preprocessing_params()
        assert {
            "random_state", "resample_filter_stage", "channel_hygiene",
            "highpass", "notch", "ica", "epochs", "lowpass", "final_resample",
        } <= params.keys()

    def test_filter_values(self, sample_config_path):
        params = SettingsManager(sample_config_path).get_preprocessing_params()
        assert params["highpass"]["l_freq"] == 1.0
        assert params["highpass"]["method"] == "iir"
        assert params["lowpass"]["h_freq"] == 40.0
        assert params["notch"]["freq"] == 50.0
        assert params["resample_filter_stage"] == "early"

    def test_epoch_baseline_is_tuple(self, sample_config_path):
        epochs = SettingsManager(sample_config_path).get_preprocessing_params()["epochs"]
        baseline = epochs["baseline"]
        assert isinstance(baseline, (list, tuple))
        assert baseline[0] is None
        assert baseline[1] == 0.0

    def test_defaults_applied_when_section_omitted(self, tmp_config_file, minimal_valid_data):
        params = SettingsManager(tmp_config_file(minimal_valid_data)).get_preprocessing_params()
        assert params["final_resample"]["target_rate"] == 100
        assert params["resample_filter_stage"] == "early"
        assert params["ica"]["n_components"] is None
        assert params["epochs"]["baseline"] is None


class TestGetDecoderSettings:
    def test_returns_dict(self, sample_config_path):
        settings = SettingsManager(sample_config_path).get_decoder_settings()
        assert isinstance(settings, dict)

    def test_tasks_count(self, sample_config_path):
        tasks = SettingsManager(sample_config_path).get_decoder_settings()["tasks"]
        assert len(tasks) == 2

    def test_task_structure(self, sample_config_path):
        task = SettingsManager(sample_config_path).get_decoder_settings()["tasks"][0]
        assert task["name"] == "red decoder"
        assert "red" in task["pos_labels"]
        assert "green" in task["neg_labels"]
        assert "yellow" in task["neg_labels"]

    def test_model_and_params(self, sample_config_path):
        settings = SettingsManager(sample_config_path).get_decoder_settings()
        assert settings["model"] == "LDA"
        assert settings["params"]["solver"] == "lsqr"


class TestGetEventMapping:
    def test_returns_str_keys(self, sample_config_path):
        mapping = SettingsManager(sample_config_path).get_event_mapping()
        assert all(isinstance(k, str) for k in mapping.keys())

    def test_returns_int_values(self, sample_config_path):
        mapping = SettingsManager(sample_config_path).get_event_mapping()
        assert all(isinstance(v, int) for v in mapping.values())

    def test_correct_mappings(self, sample_config_path):
        mapping = SettingsManager(sample_config_path).get_event_mapping()
        assert mapping["red"] == 1
        assert mapping["green"] == 2
        assert mapping["yellow"] == 3

    def test_single_event(self, tmp_config_file, minimal_valid_data):
        # Override to single event (no decoder tasks referencing it)
        minimal_valid_data["markers_mapping"] = {"events": [{"id": 99, "name": "target"}]}
        minimal_valid_data["decoders"] = {"model": "LDA", "tasks": []}
        mapping = SettingsManager(tmp_config_file(minimal_valid_data)).get_event_mapping()
        assert mapping == {"target": 99}


class TestAllowedValues:
    def test_rejects_invalid_highpass_method(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"highpass": {"l_freq": 0.1, "method": "butterworth"}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_invalid_ica_method(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"ica": {"method": "extended_infomax"}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_invalid_resample_filter_stage(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"resample_filter_stage": "middle"}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_invalid_decoder_model(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["model"] = "XGBoost"
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_extra_key_in_highpass(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"highpass": {"l_freq": 0.1, "typo_key": 99}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_removed_reject_criteria_section(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"reject_criteria": {"hard_amplitude": 1e-4}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_unknown_iclabel_drop_label(self, tmp_config_file, minimal_valid_data):
        # ICLabel returns canonical strings with spaces ("muscle artifact",
        # not "muscle"). Short-name typos used to silently match nothing
        # downstream, letting real artifacts through. Validator must catch
        # any string outside ICLabel's known seven categories.
        minimal_valid_data["preprocessing"] = {
            "ica": {"iclabel": {"drop_labels": ["muscle", "eye blink"]}}
        }
        with pytest.raises(ValueError, match="muscle"):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_accepts_canonical_iclabel_labels(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {
            "ica": {"iclabel": {"drop_labels": [
                "muscle artifact", "eye blink", "heart beat",
                "line noise", "channel noise", "other",
            ]}}
        }
        sm = SettingsManager(tmp_config_file(minimal_valid_data))
        assert sm.get_preprocessing_params()["ica"]["iclabel"]["drop_labels"] == [
            "muscle artifact", "eye blink", "heart beat",
            "line noise", "channel noise", "other",
        ]


class TestRangeValidation:
    def test_rejects_non_positive_highpass(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"highpass": {"l_freq": 0.0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_non_positive_lowpass(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"lowpass": {"h_freq": -1.0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_tmin_above_tmax(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"epochs": {"tmin": 0.8, "tmax": -0.2}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_zero_final_resample_rate(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"final_resample": {"target_rate": 0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_negative_final_resample_rate(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"final_resample": {"target_rate": -100}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_cv_k_below_2(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["cv"] = {"k": 1}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_zero_ica_components(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"ica": {"n_components": 0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))


class TestCrossModelValidation:
    def test_rejects_pos_label_not_in_events(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["tasks"] = [
            {"name": "test", "pos_labels": ["blue"], "neg_labels": ["red"]}
        ]
        with pytest.raises(ValueError, match="blue"):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_neg_label_not_in_events(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["tasks"] = [
            {"name": "test", "pos_labels": ["red"], "neg_labels": ["purple"]}
        ]
        with pytest.raises(ValueError, match="purple"):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_error_names_the_offending_task(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["tasks"] = [
            {"name": "my_task", "pos_labels": ["nonexistent"], "neg_labels": ["red"]}
        ]
        with pytest.raises(ValueError, match="my_task"):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_overlapping_pos_neg_labels(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["tasks"] = [
            {"name": "test", "pos_labels": ["red"], "neg_labels": ["red", "green"]}
        ]
        with pytest.raises(ValueError, match="overlap"):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_no_tasks_is_valid(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["tasks"] = []
        sm = SettingsManager(tmp_config_file(minimal_valid_data))
        assert sm.get_decoder_settings()["tasks"] == []
