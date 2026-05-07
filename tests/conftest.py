from pathlib import Path

import pytest
import yaml

TESTS_DIR = Path(__file__).parent
SAMPLE_CONFIG_PATH = TESTS_DIR / "data" / "sample_config.yaml"


@pytest.fixture
def sample_config_path() -> Path:
    return SAMPLE_CONFIG_PATH


@pytest.fixture
def tmp_config_file(tmp_path):
    """Returns a helper that writes a dict as YAML to a temp file and returns its path."""
    def _write(data: dict) -> Path:
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(data, f)
        return config_path
    return _write


@pytest.fixture
def minimal_valid_data() -> dict:
    """Minimal config dict that passes all validation rules."""
    return {
        "experiment_info": {"name": "test"},
        "markers_mapping": {
            "events": [
                {"id": 1, "name": "red"},
                {"id": 2, "name": "green"},
            ]
        },
        "decoders": {
            "model": "LDA",
            "tasks": [
                {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green"]},
            ],
        },
    }
