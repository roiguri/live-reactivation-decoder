import copy
import re

import pytest
from pydantic import ValidationError

from backend.core import preprocessing_constants as pc
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
        # re.escape so Windows paths (containing \) don't get treated as
        # regex escape sequences in pytest.raises(match=...).
        with pytest.raises(ValueError, match=re.escape(str(path))):
            SettingsManager(path)


class TestGetRandomState:
    def test_returns_top_level_seed(self, sample_config_path):
        assert SettingsManager(sample_config_path).get_random_state() == 42

    def test_reads_custom_seed(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["random_state"] = 7
        assert SettingsManager(tmp_config_file(minimal_valid_data)).get_random_state() == 7


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


class TestGetSettings:
    """get_settings()['preprocessing'] is the UI's view of the hardcoded recipe.

    The recipe is no longer in the config; get_settings assembles it from
    preprocessing_constants so the frontend keeps reading a complete dict.
    """

    def test_has_all_top_level_sections(self, sample_config_path):
        settings = SettingsManager(sample_config_path).get_settings()
        assert set(settings.keys()) == {"preprocessing", "decoders", "event_mapping"}

    def test_preprocessing_view_has_full_recipe(self, sample_config_path):
        pre = SettingsManager(sample_config_path).get_settings()["preprocessing"]
        for block in (
            "channel_hygiene", "highpass", "notch", "lowpass", "final_resample",
            "epochs", "ica",
        ):
            assert block in pre

    def test_recipe_values_match_constants(self, sample_config_path):
        pre = SettingsManager(sample_config_path).get_settings()["preprocessing"]
        assert pre["channel_hygiene"] == {
            "drop_emg": pc.CHANNEL_DROP_EMG,
            "rename_hegoc_to_heog": pc.CHANNEL_RENAME_HEGOC_TO_HEOG,
            "montage_name": pc.CHANNEL_MONTAGE_NAME,
            "afz_case_fix": pc.CHANNEL_AFZ_CASE_FIX,
        }
        assert pre["highpass"] == {"l_freq": pc.HIGHPASS_L_FREQ, "method": pc.HIGHPASS_METHOD}
        assert pre["notch"] == {"freq": pc.NOTCH_FREQ}
        assert pre["lowpass"] == {"h_freq": pc.LOWPASS_H_FREQ, "method": pc.LOWPASS_METHOD}
        assert pre["final_resample"] == {"target_rate": pc.FINAL_RESAMPLE_RATE}
        assert pre["epochs"] == {
            "tmin": pc.EPOCH_TMIN, "tmax": pc.EPOCH_TMAX, "baseline": pc.EPOCH_BASELINE,
        }
        assert pre["ica"] == {
            "method": pc.ICA_METHOD,
            "extended": pc.ICA_EXTENDED,
            "n_components": pc.ICA_N_COMPONENTS,
            "fit_l_freq": pc.ICA_FIT_L_FREQ,
            "iclabel": {
                "enabled": pc.ICLABEL_ENABLED,
                "drop_labels": list(pc.ICLABEL_DROP_LABELS),
            },
        }


class TestAllowedValues:
    def test_rejects_invalid_decoder_model(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["model"] = "XGBoost"
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))

    def test_rejects_removed_reject_criteria_section(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["preprocessing"] = {"reject_criteria": {"hard_amplitude": 1e-4}}
        with pytest.raises(ValueError):
            SettingsManager(tmp_config_file(minimal_valid_data))


class TestRangeValidation:
    def test_rejects_cv_k_below_2(self, tmp_config_file, minimal_valid_data):
        minimal_valid_data["decoders"]["cv"] = {"k": 1}
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


class TestLogisticPenaltyMigration:
    """sklearn 1.8 deprecated LogisticRegression(penalty=); we use l1_ratio."""

    def test_default_uses_l1_ratio_not_penalty(self):
        from backend.core.config_models import DecoderSettings
        params = DecoderSettings(model="Logistic").params
        assert params["l1_ratio"] == 1      # == old penalty="l1"
        assert params["solver"] == "liblinear"  # required for l1_ratio=1
        assert "penalty" not in params

    def test_penalty_param_is_now_rejected(self):
        from backend.core.config_models import DecoderSettings
        with pytest.raises(ValueError, match="penalty"):
            DecoderSettings(model="Logistic", params={"penalty": "l1"})
