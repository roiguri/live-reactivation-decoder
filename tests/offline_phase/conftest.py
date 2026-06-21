from pathlib import Path

import mne
import numpy as np
import pytest

# 10-20 channel names that have known positions in the standard montage
EEG_CH_NAMES = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz",
                 "P4", "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8", "Oz"]
N_CHANNELS = len(EEG_CH_NAMES)
SFREQ = 1000.0
DURATION_SEC = 10.0


def _make_raw(sfreq: float = SFREQ, duration: float = DURATION_SEC,
              ch_names: list[str] = EEG_CH_NAMES) -> mne.io.RawArray:
    """Helper: creates a synthetic EEG RawArray with a standard 10-20 montage."""
    n_times = int(sfreq * duration)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(ch_names), n_times)) * 10e-6  # ~10 µV noise

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)
    return raw


@pytest.fixture
def synthetic_raw() -> mne.io.RawArray:
    """20-channel EEG RawArray at 1000 Hz, 10 seconds."""
    return _make_raw()


@pytest.fixture
def synthetic_raw_with_events() -> mne.io.RawArray:
    """Synthetic raw with two stimulus annotations injected at known times."""
    raw = _make_raw()
    # Inject annotations: 'red' at 2s and 5s, 'green' at 3s and 7s
    onsets = [2.0, 3.0, 5.0, 7.0]
    durations = [0.0, 0.0, 0.0, 0.0]
    descriptions = ["red", "green", "red", "green"]
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(annotations)
    return raw


@pytest.fixture(autouse=True)
def fast_ica(monkeypatch):
    """Speed/isolation patch for the hardcoded ICA recipe.

    The real recipe (infomax, auto components, ICLabel enabled) is slow and
    triggers real ICLabel inference. Offline tests fit ICA with fastica + 4
    components and ICLabel off, matching the pre-hardcoding fixtures. Tests that
    exercise the ICLabel suggestion re-enable it and mock ``label_components``.
    """
    import backend.offline_phase.preprocessor as preproc
    monkeypatch.setattr(preproc, "ICA_METHOD", "fastica")
    monkeypatch.setattr(preproc, "ICA_N_COMPONENTS", 4)
    monkeypatch.setattr(preproc, "ICLABEL_ENABLED", False)


@pytest.fixture
def make_preprocessor(tmp_path):
    """Factory fixture: returns an OfflinePreprocessor (recipe is hardcoded;
    only the random seed is passed in)."""
    from backend.offline_phase.preprocessor import OfflinePreprocessor

    data_dir = tmp_path / "Sub_001"
    data_dir.mkdir(parents=True)

    return OfflinePreprocessor(data_dir=data_dir, random_state=42)


# ── ModelEvaluator fixtures ───────────────────────────────────────────────────

@pytest.fixture
def synthetic_epochs() -> mne.EpochsArray:
    """3-class EpochsArray: 30 trials × 3 classes, 20 ch, 20 time points at 100 Hz."""
    rng = np.random.default_rng(0)
    n_per_class, n_times, sfreq, tmin = 30, 20, 100.0, -0.1

    data = rng.standard_normal((n_per_class * 3, N_CHANNELS, n_times)) * 10e-6
    info = mne.create_info(ch_names=EEG_CH_NAMES, sfreq=sfreq, ch_types="eeg")

    event_id = {"red": 1, "green": 2, "yellow": 3}
    labels = np.repeat([1, 2, 3], n_per_class)
    events = np.column_stack(
        [np.arange(len(labels)), np.zeros(len(labels), int), labels]
    )
    return mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id,
                           verbose=False)


def _make_evaluator_settings(random_state: int = 42, **overrides) -> dict:
    """Build a DecoderSettings dict with merged classifier defaults via Pydantic."""
    from backend.core.config_models import DecoderSettings
    return {"random_state": random_state, **DecoderSettings(**overrides).model_dump()}


@pytest.fixture
def evaluator_settings() -> dict:
    """Decoder settings for evaluator tests (3-fold CV for speed)."""
    return _make_evaluator_settings(
        model="LDA",
        params={"solver": "lsqr", "shrinkage": "auto"},
        scale_method="standard",
        cv={"k": 3},
        tasks=[
            {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow"]},
            {"name": "yellow decoder", "pos_labels": ["yellow"], "neg_labels": ["green", "red"]},
        ],
    )


@pytest.fixture
def logistic_evaluator_settings() -> dict:
    """Logistic Regression decoder settings (single task, 3-fold, C=1.0 override)."""
    return _make_evaluator_settings(
        model="Logistic",
        params={"C": 1.0},
        scale_method="standard",
        cv={"k": 3},
        tasks=[
            {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow"]},
        ],
    )


@pytest.fixture
def svm_evaluator_settings() -> dict:
    """SVM decoder settings with median scaler (single task, 3-fold)."""
    return _make_evaluator_settings(
        model="SVM",
        params={},
        scale_method="median",
        cv={"k": 3},
        tasks=[
            {"name": "red decoder", "pos_labels": ["red"], "neg_labels": ["green", "yellow"]},
        ],
    )

