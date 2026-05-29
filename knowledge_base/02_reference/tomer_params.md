```python

from pathlib import Path

class Paths:
    DATA_ROOT   = "/Volumes/CENSORLAB$/Tomer Muller/data"
    OUTPUT_ROOT = "/Volumes/CENSORLAB$/Tomer Muller/preprocessed"
    SESSIONS    = ["experiment", "baseline"]


class FilterParams:
    L_FREQ      = 0.1    # high-pass cut-off (Hz)
    H_FREQ      = 40.0   # low-pass cut-off  (Hz)
    NOTCH_FREQ  = 50.0   # power-line noise  (Hz, EU standard)
    RESAMPLE_HZ = 256    # downsample target (Hz)


class ICAParams:
    N_COMPONENTS = 25
    METHOD       = "fastica"
    RANDOM_STATE = 42
    FIT_L_FREQ   = 1.0   # high-pass used only for ICA fitting


class EpochParams:
    TMIN     = -0.2       # epoch start relative to trigger (s)
    TMAX     = 1.0        # epoch end   relative to trigger (s)
    BASELINE = (None, 0)  # baseline correction window


class TriggerCodes:
    BINDING_TRIAL = 1
    TEST_TRIAL    = 2
    PARTIAL_TEST  = 3

    EVENT_ID = {
        "binding_trial": BINDING_TRIAL,
        "test_trial":    TEST_TRIAL,
        "partial_test":  PARTIAL_TEST,
    }


class RejectCriteria:
    HARD_AMPLITUDE  = 150e-6   # hard threshold before AutoReject (V)
    FLAT_THRESHOLD  = 0.5e-6   # channel std below this → flat channel (V)
    NOISY_Z_SCORE   = 3        # channel std z-score above this → noisy channel


class FrequencyBands:
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA  = (13.0, 30.0)
    GAMMA = (30.0, 40.0)

    ALL = {
        "delta": DELTA,
        "theta": THETA,
        "alpha": ALPHA,
        "beta":  BETA,
        "gamma": GAMMA,
    }


class ChannelGroups:
    FRONTAL   = ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"]
    CENTRAL   = ["C3", "C4", "Cz"]
    PARIETAL  = ["P3", "P4", "Pz", "P7", "P8"]
    OCCIPITAL = ["O1", "O2", "Oz"]
    TEMPORAL  = ["T7", "T8", "TP9", "TP10"]

    ALL = {
        "frontal":   FRONTAL,
        "central":   CENTRAL,
        "parietal":  PARIETAL,
        "occipital": OCCIPITAL,
        "temporal":  TEMPORAL,
    }


class FileFormat:
    PREPROCESSED_SUFFIX = "-epo.fif"
    SUBJECT_FOLDER_PREFIX = "subject "


class ParallelPortDict:
    EVENT_DICT = {
        'start_record_baseline': 1,
        'start_functional_localizer': 2,
        'start_binding_learning_block': 3,
        'start_test_phase_block': 4,
        'start_break_game': 5,
        'start_partial_retrieval': 6,
        'show_red': 11,
        'show_green': 12,
        'show_yellow': 13,
        'show_living_room': 16,
        'show_bathroom': 17,
        'show_kitchen': 18,
        'stop_red': 21,
        'stop_green': 22,
        'stop_yellow': 23,
        'stop_living_room': 26,
        'stop_bathroom': 27,
        'stop_kitchen': 28,
        'show_attention_question': 31,
        'answer_attention_question': 32,
        'show_binding_trials': 41,
        'stop_binding_trials': 42,
        'show_difficulty_question': 43,
        'answer_difficulty_question': 44,
        'show_object_in_test_trial': 51,
        'start_retrieval_time': 53,
        'answer_on_retrieval_time': 54,
        'show_retrieval_question': 55,
        'answer_retrieval_question': 56,
        'show_colors_answers': 61,
        'answer_color_question': 62,
        'show_scenes_answers': 66,
        'answer_scene_question': 67,
    }

    PREPRO_ARGS = {
        'resample': 1000,
        'tmin': -0.4,
        'tmax': 5,
        'baseline': None,
        'drop ica': ['eye blink', 'muscle artifact', 'channel noise']
    }
```