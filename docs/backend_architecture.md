# Backend Architecture

Back to [Codebase Structure](README.md) or [Project Index](../START_HERE.md).

---

## Status

This document is the maintained backend contract for `online_decoder`.

It includes both:
- interfaces for **implemented** backend classes
- interfaces for **planned** backend classes that are not yet committed

Code under `online_decoder/src/` is the source of truth for implemented behavior. For classes that are still missing, this document is the interface contract until the code exists.

Last reconciled with code on **2026-05-09**.

**Offline phase implementation status (as of last reconciliation):**
- ✅ `SettingsManager` + Pydantic config models (`src/backend/core/`)
- ✅ `OfflinePreprocessor` (`src/backend/offline_phase/preprocessor.py`)
- ✅ `ModelEvaluator` (`src/backend/offline_phase/evaluator.py`)
- ✅ `ModelTrainer` (`src/backend/offline_phase/trainer.py`)
- ✅ Shared utilities (`src/backend/offline_phase/utils.py`) — `build_classifier`, `get_task_data`
- 🔲 `OfflinePhaseOrchestrator` — planned; will own state management, bundling, and persistence

# Reactivation Decoder: Application Architecture Plan
## 1. System Overview & Frontend Integration
The application is built on a decoupled architecture. The "Backend" (Python data pipelines, Scikit-Learn, MNE) handles all heavy mathematical lifting, while the "Frontend" (PyQt6) handles user inputs, experiment states, and data visualization.

**How the Frontend connects to the Backend:**

Because EEG processing and live inference are computationally demanding, mixing them directly with the UI thread will cause the app to freeze.
- Phase 1 (Offline) Integration: The UI acts as a State Machine. When the researcher clicks "Start Preprocessing", the UI disables its buttons, shows a loading bar, and calls the backend methods. For tasks requiring user input (like selecting ICA components), the backend halts, returns data to the UI, the UI displays interactive MNE/PyQtGraph plots, and upon user selection, the UI passes the choices back into the backend to resume processing.
- Phase 2 (Online) Integration: The UI uses a Producer-Consumer model via QThread. The UI launches the backend StreamWorker in a separate background thread. This worker accumulates small EEG micro-batches, preprocesses them with persistent state, predicts on the decimated outputs, and communicates with the UI strictly via `pyqtSignal`, keeping the frontend responsive while the backend processes the 1000 Hz stream.

## 2. Phase 1: Offline Training
**Context:** This phase occurs during the subject's break. Latency is not an issue here. The goal is to clean a large block of recorded .vhdr data, evaluate where the brain signal is strongest, let the user manually reject artifacts, and compile a final set of predictive models.

**Status (2026-05-09):** `SettingsManager`, `OfflinePreprocessor`, `ModelEvaluator`, `ModelTrainer`, and shared `utils.py` are implemented. `OfflinePhaseOrchestrator` is planned but not committed — it will be the single frontend-facing entry point for Phase 1.

### Data Flow & Communication

1. UI initializes `SettingsManager` and passes its outputs to `OfflinePhaseOrchestrator`.
2. UI calls `orchestrator.run_step1_prepare_ica()`. Internally calls `OfflinePreprocessor`. Returns suggested bad components.
3. UI presents these components. The researcher selects which to drop.
4. UI calls `orchestrator.run_step2_finish_pipeline(exclude_components)`. Preprocessor finishes; orchestrator stores `epochs` and `online_state` internally.
5. UI calls `orchestrator.run_evaluation()`. Internally calls `ModelEvaluator`. Returns AUC/TGM arrays for plotting.
6. The researcher clicks a specific timepoint on the graph.
7. UI calls `orchestrator.run_training(timepoint, output_dir)`. Internally calls `ModelTrainer`, then bundles the returned models with the stored `online_state` and saves `decoder_pipeline.joblib`. Returns spatial patterns and `mne.Info` for topomap display.

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

* **Role:** The Heavy Cleaner. Loads continuous data, applies zero-phase filters, calculates ICA, and epochs the data. Crucially, it records exactly what it did (the online_state) so Phase 2 can replicate the spatial transforms.

* **Inputs:** .vhdr file path, preprocessing settings.

* **Outputs:** Cleaned mne.Epochs and online_state (ICA weights, dropped channels).

#### **4. ModelEvaluator**

* **Role:** The Sandbox. Runs Cross-Validation (Sliding Estimator) across all timepoints to see when decoding works best.

* **Inputs:** Cleaned mne.Epochs and decoder settings.

* **Outputs:** Raw NumPy arrays representing AUC scores over time and Temporal Generalization Matrices (TGM) for the UI to plot.

#### **5. ModelTrainer**

* **Role:** The Trainer. Takes the user's chosen timepoint and trains one production-ready classifier per task on 100% of the data. Computes Haufe et al. 2014 activation patterns for GUI topomap verification. Has no knowledge of persistence or Phase 1 artifacts.

* **Inputs:** Chosen timepoint (float), mne.Epochs, decoder settings.

* **Outputs:** Fitted models (dict), spatial patterns (dict of ndarrays), mne.Info. Does **not** write to disk.

#### **6. OfflinePhaseOrchestrator** *(planned)*

* **Role:** The Façade. The single backend entry point for the Phase 1 UI. Holds intermediate state (epochs, online_state, eval results) between user-triggered steps, calls the individual backend classes in the correct order, and owns the final bundling and `decoder_pipeline.joblib` export.

* **Inputs:** data directory, SettingsManager.

* **Outputs:** Per-step return dicts shaped for the UI's specific display needs.

### Components Interface
```python
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BandpassSettings(BaseModel):
    """
    Validated bandpass/notch configuration for preprocessing.
    Contract: l_freq must remain below h_freq.
    """

    model_config = ConfigDict(extra="forbid")

    l_freq: float = Field(default=0.1, gt=0)
    h_freq: float = 40.0
    method: Literal["iir", "fir"] = "iir"
    notch: Optional[float] = 50.0


class ResampleSettings(BaseModel):
    """
    Target sample rate for Phase 1 outputs and Phase 2 model-facing features.
    """

    model_config = ConfigDict(extra="forbid")

    target_rate: int = Field(default=256, ge=1)


class ICASettings(BaseModel):
    """
    ICA fitting configuration used during Phase 1 preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    n_components: int = Field(default=25, ge=1)
    method: Literal["fastica", "infomax", "picard"] = "fastica"
    fit_l_freq: float = Field(default=1.0, gt=0)  # HP freq for the ICA fitting copy


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
    "Logistic": {"C", "penalty", "solver", "class_weight", "max_iter"},
    "SVM":      {"C", "kernel", "gamma", "class_weight", "max_iter"},
}

_CLASSIFIER_DEFAULTS: dict[str, dict] = {
    "LDA":      {},
    "Logistic": {"solver": "liblinear", "class_weight": "balanced",
                 "C": 1000, "penalty": "l1", "max_iter": 1000},
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
    Executes the offline cleaning pipeline for a single subject and session.
    Designed for a two-step execution to allow manual GUI intervention during ICA.

    **Status:** ✅ Implemented in `src/backend/offline_phase/preprocessor.py`
    """

    def __init__(self, subject_dir: Path, session: str, preprocessing_settings: Dict[str, Any]):
        self.subject_dir = Path(subject_dir)
        self.session = session
        self.subject_id = self.subject_dir.name
        self.settings = preprocessing_settings

        self.vhdr: Optional[Path] = self._find_vhdr()
        self.raw: Optional[mne.io.Raw] = None
        self.epochs: Optional[mne.Epochs] = None
        self.ica: Optional[mne.preprocessing.ICA] = None

    def run_step1_prepare_ica(self) -> List[int]:
        """
        Executes the first half of the preprocessing pipeline up to ICA fitting.

        Steps taken:
        1. Load Raw: Finds and loads the .vhdr file, applying a 10-20 montage.
        2. Filter: Applies bandpass (IIR) and notch filters.
        3. Resample: Downsamples the continuous data to the target rate.
        4. Detect Bad Channels: Flags flat/noisy channels via Z-scores and interpolates them.
        5. Reference: Re-references the EEG data to the average of all channels.
        6. Fit ICA: Fits the ICA model on a temporarily 1Hz-highpassed copy of the data.
        7. Auto-detect Artifacts: Uses MNE's find_bads_eog and find_bads_ecg to find noisy components.

        Returns:
            List[int]: The indices of the auto-detected EOG/ECG components (suggestions for the user).
        """
        pass

    def run_step2_finish_pipeline(self,
                                  exclude_components: List[int],
                                  event_mapping: Dict[int, str],
                                  output_dir: Path) -> None:
        """
        Executes the second half of the pipeline using the finalized ICA components.

        Steps taken:
        8. Apply ICA: Removes the specified components from the continuous continuous raw data.
        9. Epoch: Slices the continuous data around triggers (using event_mapping),
           applying tmin, tmax, baseline correction, and hard amplitude rejection.
        10. AutoReject: Runs AutoReject to repair or drop remaining bad epochs.
        11. Save: Exports the finalized mne.Epochs to a .fif file.

        Args:
            exclude_components: The final list of ICA indices to remove (user overrides included).
            event_mapping: Dictionary mapping integer trigger IDs to event names.
            output_dir: Path to save the final .fif file.
        """
        pass

    def export_online_state(self) -> Dict[str, Any]:
        """
        Extracts the exact spatial transformations (interpolated channels, ICA weights,
        average reference projection) so they can be injected into the Live Inference Engine.
        """
        pass

    def _find_vhdr(self) -> Optional[Path]:
        """
        Scans the subject_dir/session directory for a .vhdr file.
        Logs a warning and returns the first match if multiple are found.
        """
        pass

    def _load_raw(self) -> mne.io.Raw:
        """
        Loads the BrainVision file into memory and applies a standard 10-20 montage.
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
        Extracts events from the MNE annotations (BrainVision markers).
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

    def run_training(self, timepoint: float) -> Dict[str, Any]:
        """
        Train one classifier per task at the given timepoint.

        Args:
            timepoint: Time in seconds to extract features from (e.g. 0.350).

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


# ── Planned: OfflinePhaseOrchestrator ────────────────────────────────────────

class OfflinePhaseOrchestrator:
    """
    Façade over the Phase 1 backend classes. The single entry point for the
    Phase 1 UI. Holds intermediate state (epochs, online_state, eval_results)
    between user-triggered steps, and owns the final artifact bundling and export.

    **Status:** 🔲 Planned, not committed.
    """

    def __init__(self, data_dir: Path, settings_manager: "SettingsManager") -> None:
        """
        Args:
            data_dir: Directory containing the subject's raw .vhdr file.
            settings_manager: Validated settings instance.
        """
        pass

    def run_step1_prepare_ica(self) -> Dict[str, Any]:
        """
        Delegates to OfflinePreprocessor.run_step1_prepare_ica().

        Returns:
            {"suggested_components": list[int]}
        """
        pass

    def run_step2_finish_pipeline(self, exclude_components: List[int]) -> Dict[str, Any]:
        """
        Delegates to OfflinePreprocessor.run_step2_finish_pipeline(), then
        stores epochs and online_state internally for subsequent steps.

        Returns:
            {"n_epochs": int, "event_counts": dict[str, int], "channel_names": list[str]}
        """
        pass

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Delegates to ModelEvaluator.run_evaluation() using stored epochs.

        Returns:
            The full evaluator result dict (times, AUC curves, TGMs, suggested_timepoint).
        """
        pass

    def run_training(self, timepoint: float, output_dir: Path) -> Dict[str, Any]:
        """
        Delegates to ModelTrainer.run_training(), then bundles the returned models
        with the stored online_state and saves decoder_pipeline.joblib.

        Args:
            timepoint: Time in seconds selected by the researcher.
            output_dir: Directory to write decoder_pipeline.joblib.

        Returns:
            {
                "model_filepath":   Path,
                "spatial_patterns": {task_name: np.ndarray},
                "mne_info":         mne.Info,
            }
        """
        pass
```

## 3. Phase 2: Online Live Inference
This section defines the **active** Phase 2 backend contract.

Older full-window / `RingBuffer` descriptions are obsolete and are kept only in historical design material. The active design is **stateful micro-batch processing**.

**Status (2026-05-09):**
- `LSLReceiver` is implemented in code
- `DecoderPipelineArtifact` loader is implemented in code
- `OnlinePreprocessor` is planned, not committed
- `LiveInferenceEngine` is implemented in code
- `StreamWorker` is planned, not committed

### **Data Flow (Active Micro-Batch Design)**
Startup/composition code loads the Phase 1 artifact once before the run:
`load_decoder_pipeline_artifact()` returns unwrapped `models`, `online_state`,
and `metadata`. `OnlinePreprocessor` receives only `online_state`;
`LiveInferenceEngine` receives only `models` and model-facing `metadata`.

1. `StreamWorker` asks `LSLReceiver` for all newly available data.
2. If data exists, `StreamWorker` appends it to an internal batch accumulator.
3. When about `40 ms` of samples are available, `StreamWorker` hands one batch to `OnlinePreprocessor.process_batch()`.
4. `OnlinePreprocessor` applies causal bandpass/notch filtering with persistent state, fixed bad-channel handling from Phase 1, average reference, fixed ICA transform from Phase 1, and decimation from `1000 Hz` to `250 Hz`.
5. `LiveInferenceEngine.predict()` scores all decimated outputs from that batch.
6. `StreamWorker` emits or logs the probabilities, aligned timestamps, and any markers.

### **The Components**

#### **1. LSLReceiver (The Listener & Proxy Runner)**

**Role:** Manages the LSL proxy subprocess lifecycle and provides a clean interface for pulling EEG data and markers from the hardware stream. Automatically decodes trigger channel markers and separates them from EEG data.

**Key Features:**
- Manages LSLProxy.exe subprocess lifecycle (spawn, monitor, terminate)
- Discovers available LSL streams on the network
- Connects to specific stream by name and type
- Pulls all available data chunks since last call
- Extracts and decodes trigger codes from channel 65
- Implements stateful edge-only marker detection (no duplicate triggers)
- Validates stream properties before connection
- Gracefully handles malformed data chunks
- Comprehensive logging for diagnostics

**Inputs:**
- Configuration parameters (proxy path, stream name, stream type, timeouts)
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

#### **2. OnlinePreprocessor (The Cleaner)**

* **Role:** Applies Phase 1 spatial transforms and live causal filters to streaming micro-batches, then decimates them to the model rate.
* **Inputs:** `eeg_batch_1000hz`, aligned timestamps, and `online_state` from Phase 1.
* **Outputs:** `clean_features_250hz` plus aligned output timestamps.
* **Status:** Planned, not committed.

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

* **Role:** The background `QThread` that owns the batch accumulator and orchestrates components 1 through 3.
* **Inputs:** injected backend components plus GUI Start/Stop control.
* **Outputs:** Qt signals carrying probabilities, timestamps, and markers.
* **Status:** Planned, not committed.

### Components Interface

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
    Applies stateful causal cleaning to streaming EEG micro-batches.
    Uses persistent filter state and persistent decimation phase across batches.
    """

    def __init__(
        self,
        preprocessing_settings: Dict[str, Any],
        online_state: Dict[str, Any],
        *,
        input_sfreq: float = 1000.0,
        output_sfreq: float = 250.0,
    ):
        """
        Args:
            preprocessing_settings: Settings from YAML (e.g., filter frequencies).
            online_state: Exported Phase 1 state, including the fixed spatial transforms.
            input_sfreq: Expected incoming sample rate (default: 1000 Hz).
            output_sfreq: Expected model sample rate (default: 250 Hz).
        """
        self.settings = preprocessing_settings
        self.online_state = online_state

    def process_batch(
        self,
        eeg_batch_1000hz: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Processes one micro-batch using persistent state.

        Steps:
        1. Apply causal bandpass/notch filtering with persistent state.
        2. Apply fixed bad-channel handling from Phase 1.
        3. Apply average reference.
        4. Apply fixed ICA transform from Phase 1.
        5. Decimate from 1000 Hz to 250 Hz while preserving phase across batches.

        Args:
            eeg_batch_1000hz: 2D array of shape (n_samples, n_channels).
            timestamps: 1D array aligned with the input samples.

        Returns:
            Tuple of:
            - clean_features_250hz: 2D array aligned to the decimated outputs
            - output_timestamps: timestamps aligned to the decimated outputs
        """
        pass

    def reset_state(self) -> None:
        """
        Resets persistent filter and decimation state for a new run, if needed.
        """
        pass


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
            metadata: Model-facing metadata such as feature_width and optional
                      positive_class. Phase 1 should train each one-vs-other
                      decoder with 0 = other and 1 = target.
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
    The background Conductor thread that orchestrates the Phase 2 micro-batch loop.
    """

    # Emits: (probabilities_dict, output_timestamps, list_of_markers_found)
    prediction_ready = pyqtSignal(dict, np.ndarray, list)
    stream_error = pyqtSignal(str)

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
        Signals the loop in run() to break, stops the LSLReceiver securely,
        and tears down the worker thread.
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
│       │   └── orchestrator.py    # OfflinePhaseOrchestrator - Planned
│       │
│       └── online_phase/          # Phase 2 Classes
│           ├── lsl_receiver.py    # LSLReceiver - Implemented
│           ├── preprocessor.py    # OnlinePreprocessor - Planned
│           ├── inference.py       # LiveInferenceEngine - Planned
│           └── stream_worker.py   # StreamWorker (QThread) - Planned
│
├── scripts/                       # Current support scripts
│   ├── characterize_lsl.py        # LSL stream characterization
│   ├── replay_xdf_to_lsl.py       # Replay recorded XDF into LSL
│   ├── smoke_test_lsl_receiver.py # Manual Phase 2 smoke test
│   └── inspect_xdf.py             # XDF inspection helper
│
└── tests/
    ├── core/
    │   └── test_settings_manager.py
    ├── notebooks/
    │   ├── validate_preprocessor.ipynb
    │   ├── validate_evaluator.ipynb
    │   └── validate_trainer.ipynb
    ├── offline_phase/
    │   ├── conftest.py
    │   ├── test_preprocessor.py
    │   ├── test_evaluator.py
    │   └── test_trainer.py
    └── online_phase/
        ├── test_lsl_receiver.py
        └── test_lsl_receiver_integration.py
```

---

## What This Document Doesn't Cover

This is the authoritative architecture reference for the backend class surface in `online_decoder`. It does not cover:

- **Frontend/UI implementation details**: See [../../knowledge_base/01_timeline/03_online_stage_design/Reactivation Decoder PRD.md](../../knowledge_base/01_timeline/03_online_stage_design/Reactivation%20Decoder%20PRD.md) for the operator workflow and UI intent
- **Development workflow and testing**: See [../../knowledge_base/03_codebase/online_architecture.md](../../knowledge_base/03_codebase/online_architecture.md) for pytest usage, dependency management, and running tests
- **Current experiment-specific values**: See [../experiment_config.yaml](../experiment_config.yaml) for the concrete YAML used in this repository
- **Experiment design and requirements**: See [../../knowledge_base/01_timeline/03_online_stage_design/Reactivation Decoder PRD.md](../../knowledge_base/01_timeline/03_online_stage_design/Reactivation%20Decoder%20PRD.md) for product requirements
- **Hardware integration details**: See [../../knowledge_base/01_timeline/03_online_stage_design/Lab Equipment & LSL.md](../../knowledge_base/01_timeline/03_online_stage_design/Lab%20Equipment%20%26%20LSL.md) for NeurOne hardware and LSL streaming specifics
- **Real-time ICA approaches**: See [../../knowledge_base/02_reference/ICA_real_time.md](../../knowledge_base/02_reference/ICA_real_time.md) for static vs dynamic ICA implementation options
