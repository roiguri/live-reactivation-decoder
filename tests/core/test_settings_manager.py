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
        assert {"random_state", "bandpass", "resample", "ica", "epochs"} <= params.keys()

    def test_bandpass_values(self, sample_config_path):
        bandpass = SettingsManager(sample_config_path).get_preprocessing_params()["bandpass"]
        assert bandpass["l_freq"] == 1.0
        assert bandpass["h_freq"] == 40.0
        assert bandpass["method"] == "iir"
        assert bandpass["notch"] == 50.0

    def test_epoch_baseline_is_tuple(self, sample_config_path):
        epochs = SettingsManager(sample_config_path).get_preprocessing_params()["epochs"]
        baseline = epochs["baseline"]
        assert isinstance(baseline, (list, tuple))
        assert baseline[0] is None
        assert baseline[1] == 0.0

    def test_defaults_applied_when_section_omitted(self, tmp_config_file, minimal_valid_data):
        params = SettingsManager(tmp_config_file(minimal_valid_data)).get_preprocessing_params()
        assert params["resample"]["target_rate"] == 256


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
    def test_rejects_invalid_bandpass_method(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"bandpass": {"l_freq": 1.0, "h_freq": 40.0, "method": "butterworth"}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_invalid_ica_method(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"ica": {"method": "extended_infomax"}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_invalid_decoder_model(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["model"] = "SVM"
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_extra_key_in_bandpass(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"bandpass": {"l_freq": 1.0, "h_freq": 40.0, "typo_key": 99}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))


class TestRangeValidation:
    def test_rejects_l_freq_above_h_freq(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"bandpass": {"l_freq": 40.0, "h_freq": 1.0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_equal_l_h_freq(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"bandpass": {"l_freq": 10.0, "h_freq": 10.0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_tmin_above_tmax(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"epochs": {"tmin": 0.8, "tmax": -0.2}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_zero_reject(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"epochs": {"reject": 0.0}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_negative_reject(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"epochs": {"reject": -1e-4}}
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
