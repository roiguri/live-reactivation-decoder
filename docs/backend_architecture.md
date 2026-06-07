# Backend Architecture

Back to [Codebase Structure](README.md) or [Project Index](../START_HERE.md).

---

## Status

This document is the maintained backend contract for `online_decoder`.

It includes both:
- interfaces for **implemented** backend classes
- interfaces for **planned** backend classes that are not yet committed

Code under `online_decoder/src/` is the source of truth for implemented behavior. For classes that are still missing, this document is the interface contract until the code exists.

Last reconciled with code on **2026-05-10**.

**Offline phase implementation status (as of last reconciliation):**
- ✅ `SettingsManager` + Pydantic config models (`src/backend/core/`)
- ✅ `OfflinePreprocessor` (`src/backend/offline_phase/preprocessor.py`) — accepts a pre-loaded `mne.io.Raw` via constructor; **new reference pipeline** (see `docs/Preprocessing_Migration_Plan.md`): four operator-gated steps `run_step1a_filter()` / `set_bad_channels()` / `run_step1b_fit_ica()` / `run_step2_apply_and_save()`. Channel hygiene → HP → notch → (early: LP+resample) → interpolate bads → epoch → average ref → ICA (infomax+extended, HP-only fit copy) → ICLabel suggest → apply → (late: LP+resample) → save. No AutoReject. `export_online_state()` is fully positional (`eeg_chunk_indices`, `bad_indices`; no channel names) and is consumed directly by the migrated `OnlinePreprocessor`.
- ✅ `ModelEvaluator` (`src/backend/offline_phase/evaluator.py`)
- ✅ `ModelTrainer` (`src/backend/offline_phase/trainer.py`)
- ✅ Shared utilities (`src/backend/offline_phase/utils.py`) — `build_classifier`, `get_task_data`
- ✅ `OfflineOrchestrator` (`src/backend/offline_phase/orchestrator.py`) — single frontend entry point for Phase 1; owns state management, bundling, and `decoder_pipeline.joblib` export

# Reactivation Decoder: Application Architecture Plan
## 1. System Overview & Frontend Integration
The application is built on a decoupled architecture. The "Backend" (Python data pipelines, Scikit-Learn, MNE) handles all heavy mathematical lifting, while the "Frontend" (PyQt6) handles user inputs, experiment states, and data visualization.

**How the Frontend connects to the Backend:**

Because EEG processing and live inference are computationally demanding, mixing them directly with the UI thread will cause the app to freeze.
- Phase 1 (Offline) Integration: The UI acts as a State Machine. When the researcher clicks "Start Preprocessing", the UI disables its buttons, shows a loading bar, and calls the backend methods. For tasks requiring user input (like selecting ICA components), the backend halts, returns data to the UI, the UI displays interactive MNE/PyQtGraph plots, and upon user selection, the UI passes the choices back into the backend to resume processing.
- Phase 2 (Online) Integration: The UI uses a Producer-Consumer model via QThread. The UI asks `AppSession` to build a `LiveStreamSession`, connects to `live.prediction_ready`, and then calls `live.start()`. Internally, `StreamWorker` runs the background micro-batch loop, while `LiveStreamSession` owns start/stop cleanup for the receiver, worker, and optional logger.

## 2. Phase 1: Offline Training
**Context:** This phase occurs during the subject's break. Latency is not an issue here. The goal is to clean a large block of recorded .vhdr data, evaluate where the brain signal is strongest, let the user manually reject artifacts, and compile a final set of predictive models.

**Status (2026-05-09):** `SettingsManager`, `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, shared `utils.py`, and `OfflineOrchestrator` are all implemented.

### Data Flow & Communication

1. UI initializes `SettingsManager` and `OfflineOrchestrator`.
2. UI calls `orchestrator.set_file_path(data_dir)` then `orchestrator.load_raw_data()` to load the EEG file from disk.
3. UI calls `orchestrator.run_step1a_filter()` (creates `OfflinePreprocessor` with raw injected). Returns the filtered `Raw`; the UI pops `raw.plot(block=True)` on the main thread for manual bad-channel marking.
4. UI calls `orchestrator.set_bad_channels(raw.info["bads"])`, then `orchestrator.run_step1b_fit_ica()`. Returns `(ica, epochs, suggested)`; the UI pops `ica.plot_sources(epochs, block=True)` (suggestions pre-filled by ICLabel).
5. UI calls `orchestrator.run_step2_apply_and_save(exclude_components)`. Preprocessor applies ICA, finishes, and saves; orchestrator stores `epochs` internally.
6. UI calls `orchestrator.run_evaluation()`. Internally calls `ModelEvaluator`. Returns AUC/TGM arrays for plotting.
7. The researcher picks and confirms each decoder's timepoint in the roster.
8. UI calls `orchestrator.run_training(timepoints)` (per-decoder `{task: seconds}` dict). Internally calls `ModelTrainer`, bundles models with preprocessor's `online_state`, and saves `decoder_pipeline.joblib`. Returns spatial patterns and `mne.Info` for topomap display.

### Component Map
#### **1. Configuration Schema (`config_models.py`)**

* **Role:** The typed contract for `experiment_config.yaml`. Validates preprocessing, decoder, and marker settings before the rest of the backend touches them.

* **Inputs:** Raw YAML content.

* **Outputs:** A validated `ExperimentConfig` object graph.

#### **2. SettingsManager**

* **Role:** The Single Source of Truth. Loads and validates the shared `experiment_config.yaml`.

* **Inputs:** Path to the YAML file.

* **Outputs:** Dictionaries containing preprocessing constraints, marker mappings, and decoder blueprints.

#### **3. OfflinePreprocessor**

* **Role:** The Heavy Cleaner. Channel hygiene, causal IIR filters, ICA on cleaned epochs, and the two manual selections (bad channels, ICA components) gated for MNE's interactive windows. Records the fitted spatial state positionally so Phase 2 can replicate it.

* **Inputs:** Pre-loaded `mne.io.Raw` object (via constructor), preprocessing settings. File I/O is the caller's responsibility (`OfflineOrchestrator._load_eeg_raw()` — read BrainVision with native `.vmrk` markers + keep EEG only; montage/EMG hygiene is the preprocessor's job).

* **Outputs:** Cleaned `mne.Epochs` at the configured `final_resample.target_rate`, and a positional `online_state` (`eeg_chunk_indices`, `bad_indices`, ICA matrices, `pre_whitener` — no channel names).

#### **4. ModelEvaluator**

* **Role:** The Sandbox. Runs Cross-Validation (Sliding Estimator) across all timepoints to see when decoding works best.

* **Inputs:** Cleaned mne.Epochs and decoder settings.

* **Outputs:** Raw NumPy arrays representing AUC scores over time and Temporal Generalization Matrices (TGM) for the UI to plot.

#### **5. ModelTrainer**

* **Role:** The Trainer. Takes the user's chosen timepoint and trains one production-ready classifier per task on 100% of the data. Computes Haufe et al. 2014 activation patterns for GUI topomap verification. Has no knowledge of persistence or Phase 1 artifacts.

* **Inputs:** Chosen timepoint (float), mne.Epochs, decoder settings.

* **Outputs:** Fitted models (dict), spatial patterns (dict of ndarrays), mne.Info. Does **not** write to disk.

#### **6. OfflineOrchestrator**

* **Role:** The Façade. The single backend entry point for the Phase 1 UI. Owns file I/O (raw loading), holds intermediate state (epochs, eval results) between user-triggered steps, calls the individual backend classes in the correct order, and owns the final bundling and `decoder_pipeline.joblib` export.

* **Inputs:** `SettingsManager`, output directory. Caller sets data directory via `set_file_path()`.

* **Outputs:** Per-step return dicts shaped for the UI's specific display needs. Final `online_state` dict available via `get_online_state_for_live_phase()`.

### Components Interface
```python
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# NOTE: src/backend/core/config_models.py is the source of truth. The
# preprocessing schema was migrated to the new reference (see
# docs/Preprocessing_Migration_Plan.md). PreprocessingSettings now holds:
#   resample_filter_stage: Literal["early", "late"] = "early"
#   channel_hygiene: ChannelHygieneSettings   # drop_emg, rename_hegoc_to_heog,
#                                              # montage_name, afz_case_fix
#   highpass: HighpassSettings                # l_freq, method
#   notch:    NotchSettings                   # freq (Optional → null disables)
#   ica:      ICASettings                     # method=infomax, extended=True,
#                                              # n_components: Optional[int]=None,
#                                              # fit_l_freq, iclabel{enabled,drop_labels}
#   epochs:   EpochSettings                   # tmin, tmax,
#                                              # baseline: Optional[(lo,hi)]=None
#   lowpass:  LowpassSettings                 # h_freq, method
#   final_resample: FinalResampleSettings     # target_rate (default 100)
# BandpassSettings / ResampleSettings / RejectCriteriaSettings were removed.


class ICASettings(BaseModel):
    """ICA fitting configuration used during Phase 1 preprocessing."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["infomax", "picard", "fastica"] = "infomax"
    extended: bool = True
    n_components: Optional[int] = Field(default=None, ge=1)  # None → MNE decides
    fit_l_freq: float = Field(default=1.0, gt=0)  # HP-only ICA fit copy


class EpochSettings(BaseModel):
    """
    Epoch extraction parameters for Phase 1.
    Contract: tmin must remain below tmax.
    """

    model_config = ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 1.0
    baseline: tuple[Optional[float], Optional[float]] = (None, 0.0)


class RejectCriteriaSettings(BaseModel):
    """
    Amplitude and channel-quality thresholds for Phase 1 epoch and channel rejection.
    """

    model_config = ConfigDict(extra="forbid")

    hard_amplitude: float = Field(default=150e-6, gt=0)  # epoch amplitude pre-filter before AutoReject (V)
    flat_threshold: float = Field(default=0.5e-6, gt=0)  # channel std below this → flat channel (V)
    noisy_z_score: float = Field(default=3.0, gt=0)      # channel std z-score above this → noisy channel


class PreprocessingSettings(BaseModel):
    """
    Top-level preprocessing block from the experiment config.
    """

    model_config = ConfigDict(extra="forbid")

    bandpass: BandpassSettings = Field(default_factory=BandpassSettings)
    resample: ResampleSettings = Field(default_factory=ResampleSettings)
    reject_criteria: RejectCriteriaSettings = Field(default_factory=RejectCriteriaSettings)
    ica: ICASettings = Field(default_factory=ICASettings)
    epochs: EpochSettings = Field(default_factory=EpochSettings)


class CVSettings(BaseModel):
    """
    Cross-validation settings for offline decoder evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=5, ge=2)


class DecoderTask(BaseModel):
    """
    One decoding task definition with positive and negative label groups.
    Contract: pos_labels and neg_labels must not overlap.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    pos_labels: list[str]
    neg_labels: list[str]


_VALID_PARAMS_BY_MODEL: dict[str, set[str]] = {
    "LDA":      {"solver", "shrinkage", "n_components", "priors"},
    "Logistic": {"C", "l1_ratio", "solver", "class_weight", "max_iter"},
    "SVM":      {"C", "kernel", "gamma", "class_weight", "max_iter"},
}

_CLASSIFIER_DEFAULTS: dict[str, dict] = {
    "LDA":      {},
    "Logistic": {"solver": "liblinear", "class_weight": "balanced",
                 "C": 1000, "l1_ratio": 1, "max_iter": 1000},
    "SVM":      {"kernel": "linear", "class_weight": "balanced", "C": 1.0, "max_iter": 1000},
}


class DecoderSettings(BaseModel):
    """
    Top-level decoder block, including model family, params, scaler, CV, and tasks.
    Validator merges _CLASSIFIER_DEFAULTS into params so callers receive fully-populated params.
    Validator also rejects param keys that are invalid for the chosen model.
    """

    model_config = ConfigDict(extra="forbid")

    model:        Literal["LDA", "Logistic", "SVM"] = "LDA"
    params:       dict[str, Any] = Field(default_factory=dict)
    scale_method: Literal["standard", "median"] | None = "standard"
    cv:           CVSettings = Field(default_factory=CVSettings)
    tasks:        list[DecoderTask] = Field(default_factory=list)


class EventEntry(BaseModel):
    """
    Single trigger/event entry from the markers mapping.
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str


class MarkersMapping(BaseModel):
    """
    Marker mapping block from the experiment config.
    """

    model_config = ConfigDict(extra="forbid")

    events: list[EventEntry]


class ExperimentInfo(BaseModel):
    """
    Metadata about the experiment.
    """

    model_config = ConfigDict(extra="forbid")

    name: str


class ExperimentConfig(BaseModel):
    """
    Fully validated root configuration object loaded from experiment_config.yaml.
    Contract: every label referenced by decoder tasks must exist in markers_mapping.events.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_info: ExperimentInfo
    random_state: int = 42
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    decoders: DecoderSettings = Field(default_factory=DecoderSettings)
    markers_mapping: MarkersMapping


class SettingsManager:
    """
    Loads the shared YAML config, validates it against ExperimentConfig,
    and exposes read-only plain-dict views to the rest of the backend.
    """

    def __init__(self, config_filepath: str | Path) -> None:
        """
        Reads the YAML file from disk and validates it into `_config: ExperimentConfig`.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML structure or values fail validation.
        """
        self.config_filepath = Path(config_filepath)
        self._config: ExperimentConfig

    def get_preprocessing_params(self) -> dict[str, Any]:
        """
        Returns the 'preprocessing' block as a plain dict, with top-level random_state injected.
        """
        pass

    def get_decoder_settings(self) -> dict[str, Any]:
        """
        Returns the 'decoders' block as a plain dict, with top-level random_state injected.
        """
        pass

    def get_event_mapping(self) -> dict[str, int]:
        """
        Returns a flat dictionary mapping event names to integer trigger IDs
        (e.g., {'red': 1, 'green': 2, 'yellow': 3}).
        """
        pass

class OfflinePreprocessor:
    """
    Executes the offline cleaning pipeline for a single subject recording.
    Designed for two-step execution to allow manual ICA artifact rejection between steps.
    Caller is responsible for loading raw data and passing it via the constructor.

    **Status:** ✅ Implemented in `src/backend/offline_phase/preprocessor.py`
    """

    def __init__(
        self,
        data_dir: Path,
        preprocessing_settings: dict[str, Any],
        raw: Optional[mne.io.Raw] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.subject_id = self.data_dir.name
        self.settings = preprocessing_settings
        self.raw: Optional[mne.io.Raw] = raw
        self.epochs: Optional[mne.Epochs] = None
        self.ica: Optional[mne.preprocessing.ICA] = None

    def run_step1_prepare_ica(self) -> list[int]:
        """
        First half of the pipeline: filter, resample, bad-channel detection,
        average reference, ICA fit + auto-detection of EOG/ECG components.

        Raises:
            RuntimeError: if raw has not been set before calling.

        Returns:
            Suggested EOG/ECG component indices for user review.
        """
        pass

    def run_step2_finish_pipeline(
        self,
        exclude_components: list[int],
        event_mapping: dict[str, int],
        output_dir: Path,
    ) -> None:
        """
        Second half of the pipeline: apply ICA, epoch, AutoReject, save .fif.

        Args:
            exclude_components: Final ICA component indices to remove.
            event_mapping: {event_name: trigger_id} — MNE convention.
            output_dir: Directory to write the .fif epochs file.
        """
        pass

    def export_online_state(self) -> dict[str, Any]:
        """
        Extracts the exact spatial transformations (interpolated channels, ICA weights,
        average reference projection) so they can be injected into the Live Inference Engine.
        """
        pass

    def _filter(self) -> None:
        """
        Applies a band-pass (IIR) and notch filter to the continuous data
        using parameters defined in self.settings['bandpass'].
        """
        pass

    def _resample(self) -> None:
        """
        Downsamples the continuous data to the target rate defined in
        self.settings['resample'] if the current rate is higher.
        """
        pass

    def _detect_bad_channels(self) -> None:
        """
        Identifies flat and noisy channels using standard deviation and z-score
        thresholds, flags them as bad, and interpolates them.
        """
        pass

    def _reference(self) -> None:
        """
        Re-references the continuous EEG data to the average of all channels.
        """
        pass

    def _run_ica(self) -> None:
        """
        Fits ICA on a temporarily filtered copy of the data (e.g., 1Hz highpass).
        Automatically detects and excludes EOG/ECG components, then applies
        the cleaned unmixing matrix to the continuous data.
        """
        pass

    def _epoch(self, event_mapping: Dict[int, str]) -> mne.Epochs:
        """
        Extracts events from ``raw.annotations`` (BrainVision-style
        ``Stimulus/S<code>`` descriptions) loaded natively from the ``.vmrk``
        by ``mne.io.read_raw_brainvision``.

        Inverts the shared ID→name mapping into the form required by MNE if needed.
        Slices the data around the triggers defined in the shared settings,
        applying tmin, tmax, baseline, and hard amplitude rejection.
        """
        pass

    def _autoreject(self) -> None:
        """
        Runs the AutoReject algorithm to repair or drop remaining bad epochs
        based on the random state defined in the settings.
        """
        pass

    def _save(self, output_dir: Path) -> None:
        """
        Saves the fully processed mne.Epochs to a .fif file in the specified output directory.
        """
        pass

class ModelEvaluator:
    """
    Evaluates offline decoder performance using Temporal Generalization CV.
    Runs one GeneralizingEstimator pass per task to produce both the TGM and
    the diagonal AUC curve, then surfaces a suggested inference timepoint.

    Single entry point: run_evaluation().

    **Status:** ✅ Implemented in `src/backend/offline_phase/evaluator.py`
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: Dict[str, Any]):
        """
        Args:
            epochs: Cleaned mne.Epochs object.
            decoder_settings: Dict from SettingsManager.get_decoder_settings().
                              Required keys: 'model', 'params', 'scale_method',
                              'cv' ({'k': int}), 'random_state' (injected from
                              top-level ExperimentConfig.random_state by SettingsManager), 'tasks'.
        """
        self.epochs = epochs
        self.settings = decoder_settings
        self.times: np.ndarray = epochs.times

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation for all decoder tasks defined in settings.

        Returns:
            {
                "times": np.ndarray,
                "suggested_timepoint": float,
                "average_peak_auc": float,
                "tasks": {
                    "<task_name>": {
                        "diagonal_auc": np.ndarray,   # shape (n_times,)
                        "tgm_matrix":   np.ndarray,   # shape (n_times, n_times)
                        "peak_auc":     float,
                        "chance_level": float,        # always 0.5 for binary AUC
                    },
                    ...
                },
            }

        Raises:
            ValueError: If settings contain no tasks, or if any task's labels
                        are missing from the epochs or resolve to a single class.
        """
        pass

class ModelTrainer:
    """
    Trains the final decoders at the user-selected timepoint and calculates
    biological spatial patterns for verification.

    Persistence and bundling with Phase 1 artifacts is the responsibility of
    OfflinePhaseOrchestrator, not this class.

    **Status:** ✅ Implemented in `src/backend/offline_phase/trainer.py`

    Single entry point: run_training().
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: Dict[str, Any]):
        self.epochs = epochs
        self.settings = decoder_settings
        self.times: np.ndarray = epochs.times

    def run_training(self, timepoints: float | dict[str, float]) -> Dict[str, Any]:
        """
        Train one classifier per task, each at its own timepoint.

        Args:
            timepoints: A per-task ``{task_name: seconds}`` dict (each decoder
                its own timepoint), or a single float applied to every task.

        Returns:
            {
                "models":           {task_name: fitted_sklearn_pipeline},
                "spatial_patterns": {task_name: np.ndarray},  # (n_channels,) each
                "mne_info":         mne.Info,
            }

        Raises:
            ValueError: If settings contain no tasks, or if any task's labels
                        are missing from the epochs or resolve to a single class.
        """
        pass

    def _extract_features(self, task_cfg: Dict[str, Any], timepoint: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return X_t (n_trials, n_channels) and binary y at the given timepoint."""
        pass

    def _train_classifier(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Build and fit a classifier on 100% of the data via build_classifier()."""
        pass

    def _calculate_spatial_patterns(self, X: np.ndarray, model: Any) -> np.ndarray:
        """
        Haufe et al. 2014: A = Cov(X) @ w / Var(X @ w).
        Weights are transformed to original feature space when a scaler is present.
        """
        pass


class OfflineOrchestrator:
    """
    Façade over the Phase 1 backend classes. Single entry point for the Phase 1 UI.
    Owns file I/O, holds intermediate state between user-triggered steps, and
    bundles the final decoder_pipeline.joblib export.

    **Status:** ✅ Implemented in `src/backend/offline_phase/orchestrator.py`

    Typical call sequence:
        set_file_path(data_dir)
        load_raw_data()
        ica, suggested = run_step1_prepare_ica()
        # user reviews ICA components, selects excluded_components
        stats = run_step2_finish_pipeline(excluded_components)
        eval_results = run_evaluation()
        # operator picks + confirms each decoder's timepoint in the UI
        result = run_training(timepoints)   # {task_name: seconds}
        online_state = get_online_state_for_live_phase()
    """

    def __init__(self, settings_manager: SettingsManager, output_dir: Path) -> None:
        pass

    def set_file_path(self, data_dir: str | Path) -> None:
        """Store the data directory containing the subject's .vhdr file."""
        pass

    def load_raw_data(self) -> None:
        """
        Load the raw EEG file from disk. Stimulus markers are read natively
        from the ``.vmrk`` by ``mne.io.read_raw_brainvision``; non-EEG
        channels (EOG/ECG/stim/misc) are dropped before returning.

        Raises:
            ValueError: if set_file_path() has not been called.
            FileNotFoundError: if no .vhdr file exists in the data directory.
        """
        pass
```

#### Consideration: BrainVision header filename mismatch (data-side)

Some BrainVision recordings in `data/new_experiment/` were observed with a
`.vhdr` whose `DataFile=` / `MarkerFile=` (and the `.vmrk`'s `DataFile=`)
reference a stem that no longer matches the on-disk filenames — for example
`Bindingdecoding102.vhdr` internally points at `subject102.eeg` /
`subject102.vmrk`, which do not exist (the real companions are
`Bindingdecoding102.*`). `mne.io.read_raw_brainvision` follows those internal
pointers and raises `FileNotFoundError`.

**This is a data defect, not a pipeline bug.** The pipeline intentionally does
not paper over it — `_load_eeg_raw` calls `mne.io.read_raw_brainvision`
directly and surfaces the error.

**Mitigation:** development and testing use the `test_set/` recordings, whose
headers are consistent with their filenames; running the offline pipeline on
those works as-is. Production recordings should be delivered with matching
filenames/headers, or fixed up at the data source before being loaded (rename
the triplet so the on-disk stem matches `DataFile=` / `MarkerFile=`, or edit
those two lines in the `.vhdr` and the `DataFile=` line in the `.vmrk`).

If this defect reappears at scale, revisit hardening `_load_eeg_raw` to
tolerate it (e.g. load through a temp directory with corrected header copies
and an `.eeg` symlink).

```python

    def run_step1_prepare_ica(self) -> tuple[mne.preprocessing.ICA, list[int]]:
        """
        Run signal preprocessing and fit ICA.

        Returns:
            (ica_obj, suggested_components)

        Raises:
            RuntimeError: if load_raw_data() has not been called.
        """
        pass

    def run_step2_finish_pipeline(self, excluded_components: list[int]) -> dict[str, Any]:
        """
        Apply ICA, epoch, and run AutoReject.

        Returns:
            {"n_epochs": int}

        Raises:
            RuntimeError: if run_step1_prepare_ica() has not been called.
        """
        pass

    def run_evaluation(self) -> dict[str, Any]:
        """
        Run temporal generalization CV to surface the best decoding timepoint.

        Returns:
            Full evaluator result dict (times, AUC curves, TGMs, suggested_timepoint).

        Raises:
            RuntimeError: if run_step2_finish_pipeline() has not been called.
        """
        pass

    def run_training(self, timepoints: dict[str, float]) -> dict[str, Any]:
        """
        Train each decoder at its own timepoint, bundle online state, save joblib.

        Returns:
            {"model_filepath": Path, "spatial_patterns": dict, "mne_info": mne.Info}

        Raises:
            RuntimeError: if run_evaluation() has not been called.
        """
        pass

    def get_online_state_for_live_phase(self) -> dict[str, Any]:
        """
        Return the bundled online state from RAM (bypasses disk I/O).

        Raises:
            RuntimeError: if run_training() has not been called.
        """
        pass
```

## 3. Phase 2: Online Live Inference
This section defines the **active** Phase 2 backend contract.

Older full-window / `RingBuffer` descriptions are obsolete and are kept only in historical design material. The active design is **stateful micro-batch processing**.

**Status (2026-05-11):**
- `LSLReceiver` is implemented in code
- `DecoderPipelineArtifact` loader is implemented in code
- `OnlinePreprocessor` is implemented in code (`src/backend/online_phase/online_preprocessor.py`)
- `LiveInferenceEngine` is implemented in code
- `StreamWorker` and `LiveSessionLogger` are implemented in code
- Session composition is implemented as `AppSession.build_live_stream_session(...) -> LiveStreamSession`; do not introduce `OnlinePhase` or `session.online`

### **Data Flow (Active Micro-Batch Design)**
Startup/composition code loads the Phase 1 artifact once before the run:
`load_decoder_pipeline_artifact()` returns unwrapped `models`, `online_state`,
and `metadata`. `OnlinePreprocessor` receives only `online_state`;
`LiveInferenceEngine` receives only `models` and model-facing `metadata`.

The required on-disk `decoder_pipeline.joblib` contract is:
`{"models": {...}, "online_state": {...}, "metadata": {...}}`. If Phase 1
exports a flat online-state joblib instead, Phase 2 startup fails before any
LSL connection is attempted. Use
`python online_decoder/scripts/smoke_stream_worker.py --preflight-only --pipeline <path>`
to validate this handoff before replay or lab runs.

#### Phase 1 Artifact Handoff

Phase 2 treats the saved decoder artifact as a boundary between three
responsibilities:

- `models`: fitted decoder models for `LiveInferenceEngine`
- `online_state`: preprocessing state for `OnlinePreprocessor`
- `metadata`: model-facing runtime metadata such as `feature_width` and
  `decoding_timepoints` (per-decoder `{task_name: seconds}`)

`OfflineOrchestrator.run_training()` constructs the artifact via
`DecoderPipelineArtifactSpec` (`backend/core/artifact_models.py`), which
validates shape + cross-field consistency (`metadata.feature_width` matches
`len(online_state["eeg_chunk_indices"])`) at training-end. `_save_to_disk`
calls `spec.model_dump()` so the on-disk envelope matches what
`load_decoder_pipeline_artifact()` validates against. The in-memory handoff
accessor is `OfflineOrchestrator.get_live_artifact_spec()` (returns the same
validated spec).

UI-facing artifacts that the live runtime does not need (`spatial_patterns`,
`mne_info`) live on `orchestrator._ui_state` and are not persisted in
`decoder_pipeline.joblib`.

1. `StreamWorker` asks `LSLReceiver` for all newly available data.
2. If data exists, `StreamWorker` appends it to an internal batch accumulator.
3. When about `40 ms` of samples are available, `StreamWorker` hands one batch to `OnlinePreprocessor.process_batch()`.
4. `OnlinePreprocessor` applies, in order:
   - Positional EEG channel hygiene (`eeg_chunk_indices` slice — drops EMG and any other offline-dropped channels from the raw 64-EEG array).
   - Causal high-pass + notch filter (IIR, persistent `zi`).
   - Then, branching on `settings["preprocessing"]["resample_filter_stage"]`:
     - **"early"**: 40 Hz LP → decimate 1000→100 Hz → bad-channel interp → average reference → ICA.
     - **"late"**: bad-channel interp → average reference → ICA → 40 Hz LP → decimate 1000→100 Hz.
   - The variant flag is configured per training run and stored implicitly via the ICA matrices in `online_state` (matrices fit at the rate the offline pipeline ran at).
5. `LiveInferenceEngine.predict()` scores all decimated outputs from that batch.
6. `StreamWorker` emits probabilities, aligned timestamps, and markers through `prediction_ready`; `LiveSessionLogger` and the UI are consumers of that signal.

### **The Components**

#### **1. LSLReceiver (The Listener)**

**Role:** A pure *consumer* — resolves an LSL stream and provides a clean interface for pulling EEG data and markers. Automatically decodes trigger channel markers and separates them from EEG data. It does **not** manage any subprocess; making the stream appear on the network is the job of a `StreamSource` (see below).

**Key Features:**
- Discovers available LSL streams on the network (`discover_streams`)
- Connects to specific stream by name and type
- Pulls all available data chunks since last call
- Extracts and decodes trigger codes from channel 65
- Implements stateful edge-only marker detection (no duplicate triggers)
- Validates stream properties before connection
- Gracefully handles malformed data chunks
- Comprehensive logging for diagnostics

**Inputs:**
- Configuration parameters (stream name, stream type, timeouts)
- Start/stop commands

**Outputs:**
- `timestamps`: 1D array of LSL arrival times (n_samples,)
- `eeg_chunk`: 2D array of EEG-only data (n_samples, 64)
- `markers`: List of trigger codes extracted from channel 65

**Assumptions:**
- Hardware stream has 65 channels (64 EEG + 1 trigger at index 64)
- Sample rate is 1000 Hz
- Trigger channel uses NeurOne packed format (PsychoPy code in bits 8-15)

**Status:** ✅ Implemented in `src/backend/online_phase/lsl_receiver.py`

#### **1b. StreamSource (The Publisher)**

**Role:** Anything that publishes an LSL stream onto the network, so the `LSLReceiver` can consume it. A `Protocol` with `start()` / `stop()` / `is_running`.

- `LslProxySource` — wraps the `LSLProxy.exe` subprocess that bridges NeurOne to LSL (Windows-only). `start()` is idempotent so discovery and the subsequent run share one proxy without churning the amplifier connection.
- `ReplaySource` (Phase 2) — a sibling that publishes a recorded `.vhdr`/`.vmrk` directory as a live stream.

**Ownership:** `AppSession` owns the active source's lifetime (`start_stream_source` / `stop_stream_source`); it is started during `discover_streams()` and reused by the next run. The per-run `LiveStreamSession` only consumes the stream and never touches the source.

**Status:** ✅ Implemented in `src/backend/online_phase/stream_source.py` (`LslProxySource`); `ReplaySource` pending Phase 2.

#### **2. OnlinePreprocessor (The Cleaner)**

* **Role:** Applies positional EEG channel hygiene, causal filters, and Phase 1 spatial transforms to streaming micro-batches, producing decimated features at the configured target rate (100 Hz).
* **Inputs:** `eeg_batch` `(n_samples, raw_n_channels)` straight from `LSLReceiver` (post-trigger-split, pre-hygiene), aligned timestamps, `preprocessing_settings` (HP/notch/LP/final_resample/`resample_filter_stage`), and `online_state` from Phase 1 (`eeg_chunk_indices`, `bad_indices`, ICA matrices, interp weights, pre_whitener).
* **Outputs:** `features` `(n_out, n_eeg_post_hygiene)` plus aligned output timestamps at `target_sfreq = 100 Hz`. `n_eeg_post_hygiene = len(eeg_chunk_indices)`.
* **Pipeline:** see Data Flow above — variant-flagged (`resample_filter_stage: "early" | "late"`).
* **Status:** ✅ Implemented in `src/backend/online_phase/online_preprocessor.py`.

#### **3. LiveInferenceEngine (The Brain)**

* **Role:** Holds the trained models and generates real-time probabilities for all outputs produced by a batch.
* **Inputs:** Unwrapped decoder models, model-facing metadata, and `clean_features_250hz` from `OnlinePreprocessor`.
* **Outputs:** per-task positive-class probability arrays aligned to the batch outputs.
* **Status:** Implemented.

#### **3a. DecoderPipelineArtifact Loader (Startup Boundary)**

* **Role:** Loads the saved Phase 1 artifact envelope and returns its parts without constructing runtime components.
* **Inputs:** Path to `decoder_pipeline.joblib`.
* **Outputs:** `models`, opaque `online_state`, and `metadata`.
* **Status:** Implemented.

#### **4. StreamWorker (The Conductor)**

* **Role:** The background `QThread` that owns the batch accumulator and runs the micro-batch loop using injected dependencies.
* **Inputs:** injected `LSLReceiver`, `OnlinePreprocessor`, and `LiveInferenceEngine`.
* **Outputs:** Qt signals carrying probabilities, timestamps, markers, optional latency diagnostics, and unrecoverable runtime errors.
* **Status:** Implemented.
* **Does not own:** artifact loading, logger files, receiver start/stop, or frontend lifecycle. Those belong to `AppSession`/`LiveStreamSession`.

#### **5. LiveSessionLogger (Run Sink)**

* **Role:** Optional consumer of `prediction_ready` that persists one decoding run. A plain (non-Qt) callable connected via a direct connection; its only live job is to append to two line-buffered CSVs (the crash-safe source of truth) and own the run manifest. It also keeps the raw batch arrays in memory to emit a numpy bundle at `close()`.
* **Inputs:** run directory, task names, `{code: name}` event map, manifest metadata, and `prediction_ready` payloads.
* **Outputs (one run directory):** `predictions.csv` (`lsl_timestamp, t_sec, <per-task probs>`), `markers.csv` (`lsl_timestamp, t_sec, code, name` — every trigger edge, no grid-snapping), `manifest.json` (schema version, wall-clock + `lsl_t0`, counts, metadata), and `predictions.npz` (full-precision arrays + embedded manifest, written at close).
* **Recovery:** `export_session_npz(run_dir)` rebuilds the `.npz` from the CSVs for sessions that crashed before `close()`.
* **Status:** Implemented in `src/backend/online_phase/session_logger.py`.

#### **6. LiveStreamSession (Online Lifecycle Wrapper)**

* **Role:** Represents one composed live decoding run. It exposes `prediction_ready`, `error_occurred`, optional `latency_ready` diagnostics, `start()`, and `stop()` so the frontend does not manage backend internals.
* **Inputs:** constructed receiver, worker, and optional logger.
* **Lifecycle:** `start()` calls `receiver.start()` then `worker.start()`. `stop()` calls `worker.stop()`, `worker.wait()`, `logger.close()` if present, then `receiver.stop()`. Both methods are idempotent.
* **Status:** Implemented in `src/backend/session.py`.

#### **7. AppSession Phase 2 Factory**

* **Role:** The app-level composition boundary. The frontend imports only `AppSession`.
* **API:** `AppSession.build_live_stream_session(decoder_pipeline_path, log_dir=None, batch_size_samples=40) -> LiveStreamSession`. Log paths are resolved by `AppSession.resolve_phase2_log_dir(decoder_pipeline_path)` → `<artifact_root>/phase2_live/<timestamp>/`.
* **Responsibilities:** load the Phase 1 artifact, construct `LSLReceiver`, `OnlinePreprocessor`, `LiveInferenceEngine`, `StreamWorker`, and optional `LiveSessionLogger`, connect logger if needed, and return the stopped `LiveStreamSession`.

### Backend to Frontend Contract: Live Decoder Output

**Entry point:** `AppSession.build_live_stream_session(decoder_pipeline_path, log_path=None, batch_size_samples=40) -> LiveStreamSession`

**Lifecycle:**
1. Frontend calls `build_live_stream_session(...)` to get a `LiveStreamSession`.
2. Frontend connects UI-side slots to `live.prediction_ready`, `live.error_occurred`, and optionally `live.latency_ready`.
3. Frontend calls `live.start()`.
4. On shutdown, frontend calls `live.stop()`.

**Signals:**
- `StreamWorker.prediction_ready = pyqtSignal(dict, np.ndarray, list)`, exposed as `live.prediction_ready`.
- `StreamWorker.error_occurred = pyqtSignal(str)`, exposed as `live.error_occurred`.
- `StreamWorker.latency_ready = pyqtSignal(dict)`, exposed as `live.latency_ready`.

If `live.error_occurred` fires, the worker loop has exited but external resources are still owned by `LiveStreamSession`; caller code should still call `live.stop()` to close the logger and stop the receiver.

**Payload:**
- `predictions: dict[str, np.ndarray]` — task name to positive-class probability array, shape `(n_rows,)`.
- `timestamps: np.ndarray` — LSL clock seconds, shape `(n_rows,)`, aligned to prediction rows.
- `markers: list[tuple[float, int]]` — `(timestamp, trigger_code)` marker events.
- `error_occurred` payload: concise string identifying the failing runtime stage (receiver pull, batch accumulation, preprocessing, or inference) and exception type.
- `latency_ready` payload: diagnostic dictionary with millisecond timing keys `pull_ms`, `accumulation_ms`, `preprocessing_ms`, `inference_ms`, `emit_ms`, `total_ms`, plus `input_samples`, `emitted_rows`, `marker_count`, and `pending_samples`.

`latency_ready` emits once per processed micro-batch. At the default 40-sample
batch size on a 1000 Hz stream, this is about 25 Hz. UI consumers should
throttle or aggregate it, for example by showing a rolling mean/p95 latency and
pending backlog once per second.

**Frontend rule:** use only `live.prediction_ready`, `live.error_occurred`, `live.latency_ready`, `live.start()`, and `live.stop()` during normal operation. Do not reach into the underlying worker or private live-session members.

### Components Interface

> ⚠️ **Outdated below (pre-StreamSource extraction).** The proxy lifecycle
> (`default_proxy_path`, `_start_proxy_process`, the `proxy_path` / `launch_proxy`
> constructor args, and proxy teardown in `stop()`) has moved out of
> `LSLReceiver` into `LslProxySource` in `src/backend/online_phase/stream_source.py`.
> `LSLReceiver` is now a pure consumer and `AppSession` owns the source. Treat the
> code as the source of truth for these signatures.

#### LSLReceiver Helper Functions

```python
def default_proxy_path() -> Path:
    """
    Returns the default path to LSLProxy.exe relative to the package.
    Path: <package_root>/tools/lslproxy/LSLProxy.exe
    """

def decode_trigger_value(raw_value: float | int) -> int:
    """
    Decodes the PsychoPy trigger code from NeurOne's packed trigger word.
    Uses bit-shift operation: (int(raw_value) >> 8) & 0xFF

    Args:
        raw_value: Raw trigger channel sample value from LSL

    Returns:
        int: Decoded trigger code (0-255)
    """

def extract_markers_from_trigger_channel(
    raw_trigger_values: np.ndarray | list[float] | list[int],
    *,
    previous_trigger_code: int = 0,
) -> tuple[list[int], int]:
    """
    Extracts marker events from trigger channel using edge-only detection.

    Only emits non-zero trigger codes when they CHANGE from the previous code.
    This prevents duplicate markers while a trigger is held high.

    Args:
        raw_trigger_values: Array of raw trigger channel samples
        previous_trigger_code: Last trigger code from previous call (for state continuity)

    Returns:
        Tuple of:
        - markers: List of detected trigger codes (non-zero edges only)
        - last_code: The final trigger code (for next call's state)
    """

def split_eeg_and_markers(
    samples: np.ndarray | list[list[float]],
    *,
    eeg_channel_count: int = 64,
    trigger_channel_index: int = 64,
    previous_trigger_code: int = 0,
) -> tuple[np.ndarray, list[int], int]:
    """
    Removes trigger channel from raw LSL samples and decodes markers.

    Steps:
    1. Validates raw chunk shape (must be 2D with trigger channel present)
    2. Extracts trigger channel at specified index
    3. Removes trigger channel from data
    4. Verifies remaining EEG channels match expected count
    5. Decodes markers from trigger channel using edge detection

    Args:
        samples: Raw LSL chunk with shape (n_samples, n_channels)
        eeg_channel_count: Expected number of EEG channels after removal (default: 64)
        trigger_channel_index: Index of trigger channel in raw data (default: 64)
        previous_trigger_code: Last trigger code for state continuity

    Returns:
        Tuple of:
        - eeg_chunk: 2D array (n_samples, 64) with trigger channel removed
        - markers: List of detected trigger codes
        - last_code: Final trigger code for next call

    Raises:
        ValueError: If chunk shape is invalid or channel count is wrong
    """
```

#### LSLReceiver Class

```python
class LSLReceiver:
    """
    Manages the NeurOne LSL Proxy lifecycle and ingests the high-speed data stream.
    Separates continuous EEG data from the auxiliary marker channel.
    """

    def __init__(
        self,
        proxy_path: str | Path | None = None,
        stream_name: Optional[str] = None,
        *,
        stream_type: str = "EEG",
        eeg_channel_count: int = 64,
        trigger_channel_index: int = 64,
        resolve_timeout_sec: float = 5.0,
        pull_timeout_sec: float = 0.0,
        launch_proxy: bool = True,
    ) -> None:
        """
        Initializes the LSL receiver with configuration parameters.

        Args:
            proxy_path: Path to LSLProxy.exe. If None, uses default_proxy_path().
            stream_name: Name of LSL stream to connect to (default: "NeuroneStream").
                        Can be changed later with set_stream().
            stream_type: LSL stream type to filter by (default: "EEG").
            eeg_channel_count: Expected number of EEG channels after removing trigger (default: 64).
            trigger_channel_index: Index of trigger channel in raw stream (default: 64).
            resolve_timeout_sec: Max seconds to wait for stream resolution (default: 5.0).
            pull_timeout_sec: Timeout for pylsl pull_chunk calls (default: 0.0 = non-blocking).
            launch_proxy: Whether to spawn LSLProxy.exe subprocess (default: True).
                         Set False when proxy is already running externally.
        """

    def discover_streams(self, timeout_sec: float = 3.0) -> list[str]:
        """
        Discovers available LSL streams on the network.

        Optionally launches proxy if launch_proxy=True and proxy not running.
        Queries network for LSL streams and filters by stream_type if configured.

        Args:
            timeout_sec: How long to wait for streams to appear (default: 3.0)

        Returns:
            Sorted list of stream names matching the configured stream type
        """

    def set_stream(self, stream_name: str) -> None:
        """
        Sets the target stream name for connection.

        Args:
            stream_name: Name of LSL stream to connect to (e.g., "NeuroneStream")
        """

    def start(self) -> None:
        """
        Starts the LSL connection sequence.

        Steps:
        1. Launches LSLProxy.exe subprocess if launch_proxy=True
        2. Repeatedly attempts to resolve configured stream (with timeout)
        3. Validates stream properties (sample rate, channel count)
        4. Opens pylsl.StreamInlet
        5. Resets internal trigger state

        Raises:
            RuntimeError: If stream not found within resolve_timeout_sec or pylsl not installed
            ValueError: If stream has wrong sample rate or channel count
            FileNotFoundError: If proxy_path doesn't exist
        """

    def pull_new_data(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Pulls all available data from the LSL inlet since the last call.

        This method drains ALL currently available chunks from the inlet buffer,
        aggregates them, removes the trigger channel, and decodes markers.

        Maintains stateful trigger edge detection across calls - the same trigger
        held high across multiple calls will only be emitted once.

        If a malformed chunk is encountered (wrong shape/channel count), it is
        logged and skipped, allowing data reception to continue.

        Returns:
            Tuple containing:
            - timestamps: 1D array of LSL timestamps, shape (n_samples,)
            - eeg_chunk: 2D array of EEG data, shape (n_samples, 64)
            - markers: List of integer trigger codes detected in this call

            If no data available, returns empty arrays and empty list.

        Raises:
            RuntimeError: If called before start()
        """

    def stop(self) -> None:
        """
        Stops the LSL receiver and cleans up resources.

        Steps:
        1. Closes the pylsl StreamInlet if open
        2. Terminates LSLProxy.exe subprocess if spawned
        3. Waits up to 2 seconds for graceful termination
        4. Kills process if termination times out

        Safe to call multiple times or if never started.
        """

    # --- Private Methods ---

    def _require_pylsl(self):
        """
        Validates that pylsl is available at runtime.

        Raises:
            RuntimeError: If pylsl is not installed
        """

    def _start_proxy_process(self) -> None:
        """
        Spawns LSLProxy.exe as a background subprocess.

        Checks if proxy is already running before spawning.
        Validates proxy path exists and platform compatibility.
        Uses CREATE_NO_WINDOW flag on Windows to suppress console.
        Detects immediate proxy failures and reports diagnostics.

        Raises:
            FileNotFoundError: If proxy executable not found
            RuntimeError: If trying to run .exe on non-Windows platform or if proxy fails to start
        """

    def _resolve_stream(self, timeout_sec: float):
        """
        Attempts to resolve the configured LSL stream.

        Uses type-based resolution first (fast, specific).
        Falls back to name-only resolution if type-based fails.

        Args:
            timeout_sec: Maximum wait time for resolution

        Returns:
            pylsl.StreamInfo or None if not found
        """
```

**Exception Behavior:**

Uses standard Python exceptions:
- `RuntimeError`: Stream not found, proxy failures, operational errors
- `ValueError`: Stream validation failures (wrong sample rate, channel count)
- `FileNotFoundError`: Proxy executable not found

**Usage Examples:**

```python
# Basic Usage
receiver = LSLReceiver(stream_name="NeuroneStream", launch_proxy=False)
receiver.start()

while experiment_running:
    timestamps, eeg_chunk, markers = receiver.pull_new_data()
    if eeg_chunk.shape[0] > 0:
        # Process data
        pass

receiver.stop()

# Stream Discovery
receiver = LSLReceiver(launch_proxy=True)
available_streams = receiver.discover_streams(timeout_sec=5.0)
print(f"Found streams: {available_streams}")
receiver.set_stream(available_streams[0])
receiver.start()

# Integration with StreamWorker
# In StreamWorker.run() loop:
timestamps, eeg_chunk, markers = self.receiver.pull_new_data()
if eeg_chunk.shape[0] > 0:
    # Append to internal batch accumulator
    self._append_to_batch(timestamps, eeg_chunk)
    ready_batch = self._pop_ready_batch()
    if ready_batch is not None:
        batch_timestamps, batch = ready_batch
        # Process with OnlinePreprocessor.process_batch()
```

```python
class OnlinePreprocessor:
    """
    Stateful causal EEG preprocessor for the online phase.

    Consumes the positional online_state exported by Phase 1's
    OfflinePreprocessor and applies the same spatial transforms to
    streaming micro-batches. Pipeline order is variant-flagged by
    settings["preprocessing"]["resample_filter_stage"]:

      "early":  filter → lowpass → decimate → interp → avg_ref → ica
      "late":   filter → interp → avg_ref → ica → lowpass → decimate

    Status: ✅ Implemented in src/backend/online_phase/online_preprocessor.py
    """

    def __init__(
        self,
        preprocessing_settings: Dict[str, Any],
        online_state: Dict[str, Any],
        input_sfreq: float = 1000.0,
    ) -> None:
        """
        Args:
            preprocessing_settings: Settings from YAML, unwrapped from PreprocessingSettings.
                Required keys: highpass.l_freq, highpass.method, notch.freq
                (may be None), lowpass.h_freq, lowpass.method,
                final_resample.target_rate, resample_filter_stage.
            online_state: Phase 1 positional state dict. Required keys:
                eeg_chunk_indices, bad_indices, interp_weights, ica_unmixing,
                ica_mixing, ica_pca_components, ica_pca_mean, ica_exclude,
                pre_whitener. No channel names cross the LSL boundary.
            input_sfreq: LSL stream sample rate (default: 1000.0 Hz).

        Raises:
            ValueError: If input_sfreq % target_rate != 0, if
                eeg_chunk_indices is malformed, or if ICA matrix
                dimensions are inconsistent with len(eeg_chunk_indices).
        """

    @property
    def n_channels(self) -> int: ...   # = len(online_state["eeg_chunk_indices"])

    @property
    def input_sfreq(self) -> float: ...

    @property
    def target_sfreq(self) -> float: ...

    def process_batch(
        self,
        eeg_batch: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the full online preprocessing pipeline to one micro-batch.

        Args:
            eeg_batch: (n_samples, raw_n_channels) straight from
                LSLReceiver — post-trigger-split, pre-hygiene. The
                eeg_chunk_indices slice picks the post-hygiene EEG
                positions inside _apply_filter.
            timestamps: (n_samples,) LSL timestamps.

        Returns:
            Tuple of (features, output_timestamps) at target_sfreq.
            features shape: (n_out, n_eeg_post_hygiene), where
            n_eeg_post_hygiene == len(eeg_chunk_indices). n_out = 0
            for very small batches.

        Raises:
            ValueError: If eeg_batch shape or timestamps length is wrong.
            IndexError: If raw_n_channels <= max(eeg_chunk_indices).
        """

    def reset_state(self) -> None:
        """Reset all causal filter state (call before each new recording run)."""


class DecoderPipelineArtifact:
    """
    Unwrapped Phase 1 decoder pipeline artifact.
    """

    models: Dict[str, Any]
    online_state: Any
    metadata: Dict[str, Any]


def load_decoder_pipeline_artifact(path: str | Path) -> DecoderPipelineArtifact:
    """
    Loads the saved Phase 1 artifact envelope.

    Validates only top-level artifact concerns:
    - artifact file exists
    - joblib payload is a dictionary
    - required keys exist: models, online_state, metadata
    - models is a non-empty dictionary
    - metadata is a dictionary

    Does not validate model prediction behavior or online_state internals.
    """
    pass


class LiveInferenceEngine:
    """
    Holds unwrapped trained Scikit-Learn decoders and predicts positive-class
    probabilities for all decimated outputs produced by a batch.
    """

    def __init__(
        self,
        models: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            models: Dict mapping decoder task names to fitted sklearn-compatible models.
            metadata: Model-facing metadata such as feature_width. Phase 1
                      should train each one-vs-other decoder with 0 = other
                      and 1 = target; positive_class is only needed as an
                      override if that convention changes.
        """
        pass

    def predict(self, clean_features_250hz: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Scores all outputs produced by a batch.

        Args:
            clean_features_250hz: 2D feature array from OnlinePreprocessor.

        Returns:
            Dict mapping task names to selected positive-class probability arrays
            aligned to the rows in clean_features_250hz.
        """
        pass


# Startup composition:
artifact = load_decoder_pipeline_artifact("decoder_pipeline.joblib")
preprocessor = OnlinePreprocessor(
    preprocessing_settings=preprocessing_settings,
    online_state=artifact.online_state,
)
engine = LiveInferenceEngine(
    models=artifact.models,
    metadata=artifact.metadata,
)

from PyQt6.QtCore import QThread, pyqtSignal

class StreamWorker(QThread):
    """
    The background thread that runs the Phase 2 micro-batch loop.

    Dependencies are injected and kept as references for use inside run().
    The worker does not construct them and does not own their lifecycle.
    """

    # Emits: (probabilities_dict, output_timestamps, list_of_markers_found)
    prediction_ready = pyqtSignal(dict, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    latency_ready = pyqtSignal(dict)

    def __init__(
        self,
        receiver: LSLReceiver,
        preprocessor: OnlinePreprocessor,
        inference_engine: LiveInferenceEngine,
        *,
        batch_size_samples: int = 40,
        poll_interval_sec: float = 0.01,
    ):
        """
        Injects the core Phase 2 components and configures micro-batch behavior.
        """
        super().__init__()
        self.receiver = receiver
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self.batch_size_samples = batch_size_samples
        self.poll_interval_sec = poll_interval_sec
        self._is_running = False

    def run(self) -> None:
        """
        The main execution loop.

        Loop logic:
        1. pull_new_data() from receiver
        2. append new samples to the internal batch accumulator
        3. when enough samples are available, pop one ready batch
        4. process_batch() via OnlinePreprocessor
        5. predict() via LiveInferenceEngine
        6. emit probabilities, aligned timestamps, and markers
        """
        pass

    def stop(self) -> None:
        """
        Signals the loop in run() to break. The caller owns worker.wait()
        and receiver/logger cleanup.
        """
        pass

    def _append_to_batch(
        self,
        timestamps: np.ndarray,
        eeg_chunk: np.ndarray,
    ) -> None:
        """
        Appends newly received samples to the internal batch accumulator.
        """
        pass

    def _pop_ready_batch(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Returns the next ready batch when enough samples have accumulated,
        otherwise returns None.
        """
        pass


class LiveStreamSession:
    """
    Lifecycle wrapper for one composed live decoding run.
    """

    @property
    def prediction_ready(self):
        """
        Forward StreamWorker.prediction_ready without exposing worker internals.
        """
        pass

    @property
    def error_occurred(self):
        """
        Forward StreamWorker.error_occurred without exposing worker internals.
        """
        pass

    @property
    def latency_ready(self):
        """
        Forward StreamWorker.latency_ready without exposing worker internals.
        """
        pass

    def start(self) -> None:
        """
        Start receiver, then start worker. Idempotent.
        """
        pass

    def stop(self) -> None:
        """
        Stop worker, wait for it to join, close optional logger,
        then stop receiver. Idempotent.
        """
        pass


class AppSession:
    def build_live_stream_session(
        self,
        decoder_pipeline_path: str | Path,
        log_path: str | Path | None = None,
        batch_size_samples: int = 40,
    ) -> LiveStreamSession:
        """
        Compose the online runtime and return it stopped.
        The frontend connects to live.prediction_ready and live.error_occurred,
        then calls live.start().
        """
        pass
```

## 4. Current Repo Surface And Planned Backend Structure
```plaintext
online_decoder/
│
├── experiment_config.yaml         # Current shared experiment configuration
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Dev/test dependencies
├── pytest.ini                     # Pytest configuration
├── docs/
│   ├── README.md                  # Current implementation-doc index
│   ├── backend_architecture.md    # Maintained backend contract
│   ├── Phase2_Implementation_Plan.md
│   └── backend_plan.md            # Legacy doc, no longer maintained; content moved here
│
├── tools/
│   └── lslproxy/
│       └── LSLProxy.exe           # Fixed path for the LSLReceiver to launch the proxy
│
├── src/
│   ├── frontend/                  # Planned UI layer - not committed
│   └── backend/                   # Data Layer (Engine)
│       ├── core/
│       │   ├── config_models.py    # Pydantic config schema - Implemented
│       │   └── settings_manager.py # SettingsManager - Implemented
│       │
│       ├── offline_phase/         # Phase 1 Classes
│       │   ├── utils.py           # build_classifier, get_task_data - Implemented
│       │   ├── preprocessor.py    # OfflinePreprocessor - Implemented
│       │   ├── evaluator.py       # ModelEvaluator - Implemented
│       │   ├── trainer.py         # ModelTrainer - Implemented
│       │   └── orchestrator.py    # OfflineOrchestrator - Implemented
│       │
│       └── online_phase/          # Phase 2 Classes
│           ├── lsl_receiver.py        # LSLReceiver - Implemented
│           ├── online_preprocessor.py # OnlinePreprocessor - Implemented
│           ├── live_inference.py      # LiveInferenceEngine - Implemented
│           ├── artifact_loader.py     # DecoderPipelineArtifact loader - Implemented
│           ├── session_logger.py      # LiveSessionLogger + export_session_npz - Implemented
│           ├── stream_worker.py       # StreamWorker (QThread) - Implemented
│           └── __init__.py            # Public API exports
│
├── scripts/                       # Current support scripts
│   ├── characterize_lsl.py        # LSL stream characterization
│   ├── replay_xdf_to_lsl.py       # Replay recorded XDF into LSL
│   ├── smoke_test_lsl_receiver.py # Manual Phase 2 smoke test
│   ├── smoke_stream_worker.py     # Headless StreamWorker + logger smoke test
│   ├── benchmark_preprocessor.py  # OnlinePreprocessor latency benchmark
│   └── inspect_xdf.py             # XDF inspection helper
│
└── tests/
    ├── core/
    │   └── test_settings_manager.py
    ├── notebooks/
    │   ├── validate_preprocessor.ipynb
    │   ├── validate_evaluator.ipynb
    │   ├── validate_trainer.ipynb
    │   └── validate_online_preprocessor.ipynb
    ├── offline_phase/
    │   ├── conftest.py
    │   ├── test_preprocessor.py
    │   ├── test_evaluator.py
    │   ├── test_trainer.py
    │   └── test_orchestrator.py
    └── online_phase/
        ├── test_lsl_receiver.py
        ├── test_lsl_receiver_integration.py
        └── test_online_preprocessor.py
```

---

## 5. Frontend Architecture (Phase 1 UI)

`src/frontend/` is planned but not committed. The active design is described in [Phase1_UI_Plan.md](Phase1_UI_Plan.md).

**Summary:**

- **Entry point:** `src/frontend/main.py` — launches two `QFileDialog` calls (config YAML + output dir), constructs `OfflineOrchestrator`, passes it to `Phase1Screen`.
- **Layout:** `Phase1Screen` splits into a left `QStackedWidget` (workspace) and a right 280 px journey panel. All 4 nodes share one orchestrator instance.
- **Threading:** Every blocking backend call runs in a `BaseWorker(QObject)` moved onto a `QThread`. Workers emit `result_ready(object)` and `error_occurred(str)`. The UI re-enables controls and advances the journey panel on `result_ready`.
- **Charts:** All three chart types (AUC curves, TGM heatmaps, spatial topomaps) use **matplotlib** via `FigureCanvasQTAgg`. MNE's `plot_topomap` renders natively inside the embedded canvas.
- **Journey panel animation:** On node completion, `QPropertyAnimation` fills the trail segment between nodes over 500 ms (`InOutCubic` easing).

**Backend calls the UI makes (in order):**

| Step | Call |
|---|---|
| Startup | `OfflineOrchestrator(settings_manager, output_dir)` |
| Node 1 | `set_file_path(data_dir)` → `load_raw_data()` |
| Node 2 step 1 | `run_step1_prepare_ica()` → returns `(ica_obj, suggested_components)` |
| Node 2 step 2 | `run_step2_finish_pipeline(excluded_components)` → returns `{"n_epochs": int}` |
| Node 3 | `run_evaluation()` → returns `{times, suggested_timepoint, tasks}` |
| Node 4 | `run_training(timepoints)` (per-decoder `{task: seconds}` dict) → returns `{"model_filepath", "spatial_patterns", "mne_info"}` |

---

## What This Document Doesn't Cover

This is the authoritative architecture reference for the backend class surface in `online_decoder`. It does not cover:

- **Frontend/UI implementation details**: See [Phase1_UI_Plan.md](Phase1_UI_Plan.md) for the full Phase 1 UI plan. See [../../knowledge_base/01_timeline/03_online_stage_design/Reactivation Decoder PRD.md](../../knowledge_base/01_timeline/03_online_stage_design/Reactivation%20Decoder%20PRD.md) for the operator workflow and UI intent
- **Development workflow and testing**: See [../../knowledge_base/03_codebase/online_architecture.md](../../knowledge_base/03_codebase/online_architecture.md) for pytest usage, dependency management, and running tests
- **Current experiment-specific values**: See [../experiment_config.yaml](../experiment_config.yaml) for the concrete YAML used in this repository
- **Experiment design and requirements**: See [../../knowledge_base/01_timeline/03_online_stage_design/Reactivation Decoder PRD.md](../../knowledge_base/01_timeline/03_online_stage_design/Reactivation%20Decoder%20PRD.md) for product requirements
- **Hardware integration details**: See [../../knowledge_base/01_timeline/03_online_stage_design/Lab Equipment & LSL.md](../../knowledge_base/01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md) for NeurOne hardware and LSL streaming specifics
- **Real-time ICA approaches**: See [../../knowledge_base/02_reference/ICA_real_time.md](../../knowledge_base/02_reference/ICA_real_time.md) for static vs dynamic ICA implementation options
