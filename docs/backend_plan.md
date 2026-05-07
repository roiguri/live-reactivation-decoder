# Reactivation Decoder: Application Architecture Plan
## 1. System Overview & Frontend Integration
The application is built on a decoupled architecture. The "Backend" (Python data pipelines, Scikit-Learn, MNE) handles all heavy mathematical lifting, while the "Frontend" (PyQt6) handles user inputs, experiment states, and data visualization.

**How the Frontend connects to the Backend:**

Because EEG processing and live inference are computationally demanding, mixing them directly with the UI thread will cause the app to freeze.
- Phase 1 (Offline) Integration: The UI acts as a State Machine. When the researcher clicks "Start Preprocessing", the UI disables its buttons, shows a loading bar, and calls the backend methods. For tasks requiring user input (like selecting ICA components), the backend halts, returns data to the UI, the UI displays interactive MNE/PyQtGraph plots, and upon user selection, the UI passes the choices back into the backend to resume processing.
- Phase 2 (Online) Integration: The UI uses a Producer-Consumer model via QThread. The UI launches the backend StreamWorker in a separate background thread. This thread runs an infinite loop at high speed (~10ms). It communicates with the UI strictly via pyqtSignal. The UI simply "listens" to these signals and updates its PyqtGraph visuals at a smooth 30 FPS, keeping the frontend completely responsive while the backend processes data at 1000Hz.

## 2. Phase 1: Offline Training
**Context:** This phase occurs during the subject's break. Latency is not an issue here. The goal is to clean a large block of recorded .vhdr data, evaluate where the brain signal is strongest, let the user manually reject artifacts, and compile a final set of predictive models.

### Data Flow & Communication

1. UI initializes SettingsManager and passes it to OfflinePreprocessor.
2. UI calls preprocessor.run_step1(). The preprocessor loads data, filters, fits ICA, and returns suggested bad components.
3. UI presents these components. The researcher selects which to drop.
4. UI calls preprocessor.run_step2(user_choices). The preprocessor finishes cleaning and exports the mne.Epochs.
5. UI passes epochs to ModelEvaluator. It runs Cross-Validation and returns arrays. UI plots the AUC graph.
6. The researcher clicks a specific timepoint on the graph.
7. UI passes that timepoint to ModelTrainer, which trains the final models and bundles them with the online_state into a .joblib file.

### Component Map
#### **1. SettingsManager**

* **Role:** The Single Source of Truth. Parses the shared experiment_config.yaml.

* **Inputs:** Path to the YAML file.

* **Outputs:** Dictionaries containing preprocessing constraints, marker mappings, and decoder blueprints.

#### **2. OfflinePreprocessor**

* **Role:** The Heavy Cleaner. Loads continuous data, applies zero-phase filters, calculates ICA, and epochs the data. Crucially, it records exactly what it did (the online_state) so Phase 2 can replicate the spatial transforms.

* **Inputs:** .vhdr file path, preprocessing settings.

* **Outputs:** Cleaned mne.Epochs and online_state (ICA weights, dropped channels).

#### **3. ModelEvaluator**

* **Role:** The Sandbox. Runs Cross-Validation (Sliding Estimator) across all timepoints to see when decoding works best.

* **Inputs:** Cleaned mne.Epochs and decoder settings.

* **Outputs:** Raw NumPy arrays representing AUC scores over time and Temporal Generalization Matrices (TGM) for the UI to plot.

#### **4. ModelTrainer**

* **Role:** The Compiler. Takes the user's final chosen timepoint, trains production-ready One-vs-All models on 100% of the data using balanced weights, and saves the system state.

* **Inputs:** Chosen timepoint (float), mne.Epochs, online_state.

* **Outputs:** decoder_pipeline.joblib (saved to disk) and spatial patterns for the UI to draw verification topomaps.

### Components Interface
```python
class SettingsManager:
    """
    Loads and provides read-only access to the shared experiment configuration (YAML).
    """

    def __init__(self, config_filepath: str | Path):
        """
        Reads the YAML file from disk into the internal `_config` dictionary.
        Raises FileNotFoundError if the file doesn't exist.
        """
        self.config_filepath = Path(config_filepath)
        self._config: Dict[str, Any] = {}

    def get_preprocessing_params(self) -> Dict[str, Any]:
        """
        Returns the 'preprocessing' block containing bandpass, resample, 
        ica, epochs, and autoreject settings.
        """
        pass

    def get_decoder_settings(self) -> Dict[str, Any]:
        """
        Returns the 'decoders' block containing the base model (e.g., LDA),
        its parameters, and the list of specific tasks (e.g., 'red decoder' 
        with pos_labels and neg_labels).
        """
        pass

    def get_event_mapping(self) -> Dict[int, str]:
        """
        Parses 'markers_mapping.events' and returns a flat dictionary mapping 
        integer IDs to event names (e.g., {1: 'red', 2: 'green', 3: 'yellow'}).
        Ignores 'trigger_switch' for Milestone 1.
        """
        pass

class OfflinePreprocessor:
    """
    Executes the offline cleaning pipeline for a single subject and session.
    Designed for a two-step execution to allow manual GUI intervention during ICA.
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
                                  event_mapping: Dict[str, int], 
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
            event_mapping: Dictionary mapping trigger names to integer IDs.
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

    def _epoch(self, event_mapping: Dict[str, int]) -> mne.Epochs:
        """
        Extracts events from the MNE annotations (BrainVision markers).
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
    Evaluates the offline performance of specific decoding tasks using Cross-Validation.
    Acts as a single-entry-point engine that runs all necessary mathematical evaluations 
    and returns a comprehensive dictionary formatted for the GUI dashboard.
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: Dict[str, Any]):
        """
        Initializes the evaluator with cleaned epochs and the blueprint for the decoders.
        
        Args:
            epochs: Cleaned mne.Epochs object.
            decoder_settings: Dictionary containing 'model', 'params', 'tasks', 
                              and 'cv_params' (e.g., {'n_splits': 5}).
        """
        self.epochs = epochs
        self.settings = decoder_settings
        self.times: np.ndarray = epochs.times

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Executes the complete evaluation pipeline for all decoders defined in the settings.
        This includes diagonal cross-validation, temporal generalization matrices, 
        chance level calculations, and peak timepoint suggestions.
        
        Returns:
            Dict[str, Any]: A complete payload for the UI containing all plot data and stats.
            Example structure:
            {
                "times": np.ndarray,             # 1D array (X-axis for plots)
                "suggested_timepoint": float,    # Recommended inference time (e.g., 0.350)
                "average_peak_auc": float,       # The mean AUC across all decoders at peak
                "tasks": {
                    "red decoder": {
                        "diagonal_auc": np.ndarray,  # 1D array for the line chart
                        "tgm_matrix": np.ndarray,    # 2D array for the heatmap
                        "peak_auc": float,           # Max AUC for this specific decoder
                        "chance_level": float        # Calculated baseline / permutation result
                    },
                    "yellow decoder": { ... }
                }
            }
        """
        pass

class ModelTrainer:
    """
    Trains the final decoders at the user-selected timepoint, calculates biological 
    spatial patterns for verification, and exports the complete pipeline.
    """

    def __init__(self, epochs: mne.Epochs, decoder_settings: Dict[str, Any]):
        """
        Args:
            epochs: Fully cleaned mne.Epochs object.
            decoder_settings: The 'decoders' block from the shared settings.
        """
        self.epochs = epochs
        self.settings = decoder_settings

    def run_training(self, 
                     timepoint: float, 
                     online_state: Dict[str, Any], 
                     output_dir: Path) -> Dict[str, Any]:
        """
        The single entry point for the App to train and package the final models.
        
        Args:
            timepoint: The specific millisecond to extract features from (e.g., 0.350).
            online_state: The cleaning rules from Phase 1, passed here ONLY to be 
                          bundled into the final export file.
            output_dir: The directory to save the final pipeline file.
            
        Returns:
            Dict containing verification data for the GUI:
            {
                "model_filepath": Path,          # Where the .pkl/.joblib was saved
                "spatial_patterns": {            # For drawing topomaps in the GUI
                    "red decoder": np.ndarray,   # Shape: (n_channels,)
                    "yellow decoder": np.ndarray
                },
                "mne_info": mne.Info             # Required to plot the topomaps
            }
        """
        pass

    # --- Private Methods (Hidden from the App/UI layer) ---

    def _extract_features(self, task_settings: Dict[str, Any], timepoint: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the relevant features for a specific decoder training task.
        Selects only the trials matching the task's pos_labels and neg_labels, 
        and extracts the 2D numpy array (n_trials, n_channels) at the requested time index.
        """
        pass

    def _train_classifier(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Fits a single Scikit-Learn classifier (e.g., LDA) using the extracted features.
        Explicitly uses class_weight='balanced' to handle unequal event distributions.
        """
        pass

    def _calculate_spatial_patterns(self, X: np.ndarray, model: Any) -> np.ndarray:
        """
        Calculates the true biological activation patterns (A = Cov(X) * W * Cov(S)^-1) 
        from the raw classifier weights so the GUI topomaps accurately reflect brain activity.
        """
        pass

    def _export_pipeline(self, models: Dict[str, Any], timepoint: float, online_state: Dict[str, Any], output_dir: Path) -> Path:
        """
        Bundles the trained models, the timepoint, and the online_state into a 
        single 'decoder_pipeline.joblib' file for the Live Inference Engine.
        """
        pass
```

## 3. Phase 2: Online Live Inference
The system targets a ~10ms loop cycle, processing data decoupled from the UI's rendering framerate to ensure zero dropped samples.

### **Data Flow (Per 10ms Tick)**
1. StreamWorker asks LSLReceiver for new data.  
2. If data exists, it is pushed into RingBuffer.  
3. StreamWorker pulls the full 2-second window from RingBuffer and hands it to OnlinePreprocessor.  
4. OnlinePreprocessor filters, decimates to 250Hz, applies the baseline, multiplies by the ICA matrix, and returns the newest clean features.  
5. StreamWorker passes the clean features to LiveInferenceEngine to get probabilities.  
6. StreamWorker emits the probabilities and any captured markers to the UI.

### **The Components**

#### **1. LSLReceiver (The Listener & Proxy Runner)**

**Role:** Manages the connection to the hardware and the LSL proxy subprocess.

* **Inputs:** Start/Stop commands, path to LSLproxy.exe.  
* **Outputs:** \* timestamps (1D array of LSL arrival times).  
  * eeg\_chunk (2D array: *N new samples* × *64 channels*).  
  * markers (List of trigger IDs extracted from Channel 65).

#### **2. RingBuffer (The Memory)**

* **Role:** High-speed, zero-allocation circular memory holding the rolling temporal window (e.g., the last 2 seconds at 1000Hz).
* **Inputs:** eeg\_chunk (from LSLReceiver).  
* **Outputs:** raw\_window\_1000hz (2D array: *64 channels* × *2000 samples*).

#### **3. OnlinePreprocessor (The Cleaner)**

* **Role:** Applies Phase 1 spatial transforms and live causal filters to the raw window. Downsamples the data to match the decoders.
* **Inputs:** raw\_window\_1000hz (from RingBuffer), online\_state (from Phase 1), and UI commands to calibrate\_baseline.  
* **Outputs:** clean\_features\_250hz (1D array of finalized spatial features representing the newest millisecond).

#### **4. LiveInferenceEngine (The Brain)**

* **Role:** Holds the pre-trained Scikit-Learn models and generates the real-time probabilities.
* **Inputs:** clean\_features\_250hz (from OnlinePreprocessor).  
* **Outputs:** probabilities\_dict (e.g., {"Red": 0.85, "Scene": 0.12}).

#### **5. StreamWorker (The Conductor)**

* **Role:** The isolated background thread (QThread) that orchestrates components 1 through 4 in a continuous while loop.
* **Inputs:** Configuration parameters and GUI Start/Stop events.  
* **Outputs:** Qt Signals emitting (probabilities\_dict, markers) up to the main UI thread safely.

### Components Interface
```python
class LSLReceiver:
    """
    Manages the NeurOne LSL Proxy lifecycle and ingests the high-speed data stream.
    Separates the continuous EEG data from the auxiliary marker channel.
    """

    def __init__(self, proxy_path: str | Path, stream_name: Optional[str] = None):
        """
        Args:
            proxy_path: Fixed path to the LSLproxy.exe in the app directory.
            stream_name: The expected name of the LSL stream to connect to (can be chosen later).
        """
        self.proxy_path = Path(proxy_path)
        self.stream_name = stream_name
        self.proxy_process: Optional[subprocess.Popen] = None
        self.inlet: Optional[StreamInlet] = None
        
    def discover_streams(self, timeout_sec: float = 3.0) -> List[str]:
        """
        Temporarily spins up the proxy (if needed) and scans the local network 
        for available LSL streams using pylsl.resolve_streams().
        
        Returns:
            List[str]: A list of found stream names for the user to choose from.
        """
        pass

    def set_stream(self, stream_name: str) -> None:
        """
        Sets the target stream name chosen by the researcher from the UI.
        """
        pass

    def start(self) -> bool:
        """
        1. Spawns the LSLproxy.exe as a background subprocess (if not already running).
        2. Waits and uses pylsl.resolve_byprop to find self.stream_name.
        3. Opens the StreamInlet.
        
        Returns:
            bool: True if proxy started and stream connected, False if timeout.
        """
        pass

    def pull_new_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Pulls all chunks available in the LSL inlet since the last call.
        Strips Channel 65 (index 64) and parses it for markers.
        
        Returns:
            Tuple containing:
            - timestamps: 1D array of LSL timestamps (n_samples,).
            - eeg_chunk: 2D array of EEG-only data (n_samples, 64).
            - markers: List of integers representing triggers found in this chunk.
        """
        pass

    def stop(self) -> None:
        """
        Closes the pylsl StreamInlet and sends a termination signal (kill) 
        to the proxy subprocess to ensure clean teardown.
        """
        pass


class RingBuffer:
    """
    A high-speed, zero-allocation circular buffer for real-time EEG windows.
    """

    def __init__(self, n_channels: int = 64, buffer_length_sec: float = 2.0, sfreq: float = 1000.0):
        """
        Pre-allocates the memory matrix.
        
        Args:
            n_channels: Number of EEG channels (after stripping markers).
            buffer_length_sec: How many seconds of history to maintain.
            sfreq: The hardware sampling rate (e.g., 1000 Hz).
        """
        self.max_samples = int(buffer_length_sec * sfreq)
        # Pre-allocate shape: (channels, samples) -> e.g., (64, 2000)
        self.buffer = np.zeros((n_channels, self.max_samples))

    def append(self, new_chunk: np.ndarray) -> None:
        """
        Inserts new_chunk at the end of the buffer, shifting old data out.
        If new_chunk has N samples, the oldest N samples are discarded.
        Optimized using memory views or np.roll.
        
        Args:
            new_chunk: 2D array of shape (n_channels, n_new_samples).
        """
        pass

    def get_full_window(self) -> np.ndarray:
        """
        Returns a copy of the entire current buffer for the Preprocessor to filter.
        """
        pass


class OnlinePreprocessor:
    """
    Applies real-time, causal cleaning to the continuous sliding window.
    Executes spatial transforms (ICA, bad channels) learned during Phase 1 
    and decimates the high-speed data to match the offline model's sampling rate.
    """

    def __init__(self, preprocessing_settings: Dict[str, Any], online_state: Dict[str, Any]):
        """
        Args:
            preprocessing_settings: Settings from YAML (e.g., filter frequencies).
            online_state: The exported dict from Phase 1 containing ICA unmixing 
                          matrices and dropped channel indices.
        """
        self.settings = preprocessing_settings
        self.online_state = online_state

    def process_window(self, raw_buffer_1000hz: np.ndarray) -> np.ndarray:
        """
        Cleans the live rolling window (Stateless approach: cleans the whole buffer).
        
        Steps:
        1. Drop bad channels (using indices from self.online_state).
        2. Apply causal IIR bandpass/notch filters to the whole buffer.
        3. Apply Average Reference (if defined).
        4. Decimate the buffer from hardware rate (1000Hz) to model rate (250Hz).
        5. Multiply by the ICA unmixing matrix (from self.online_state).
        
        Args:
            raw_buffer_1000hz: 2D array from the RingBuffer.
            
        Returns:
            np.ndarray: The fully cleaned and downsampled features ready for the model
                        (usually an array representing the newest 250Hz timepoint).
        """
        pass


class LiveInferenceEngine:
    """
    Loads the trained Scikit-Learn decoders and executes sub-millisecond 
    probability predictions on the real-time sliding window.
    """

    def __init__(self, pipeline_filepath: str | Path):
        """
        Args:
            pipeline_filepath: Path to the 'decoder_pipeline.joblib' saved in Phase 1.
        """
        self.pipeline_filepath = Path(pipeline_filepath)
        self.models: Dict[str, Any] = {}
        
    def load_pipeline(self) -> Dict[str, Any]:
        """
        Loads the saved models into memory. 
        
        Returns:
            Dict: The 'online_state' (ICA weights, dropped channels) extracted from 
                  the file, which must be passed to initialize the OnlinePreprocessor.
        """
        pass

    def predict(self, clean_features_250hz: np.ndarray) -> Dict[str, float]:
        """
        Takes the extracted features and runs them through all One-vs-All models.
        
        Args:
            clean_features_250hz: The cleaned feature array from the Preprocessor.
            
        Returns:
            Dict[str, float]: Dictionary mapping task names to their current probability.
                              Example: {"red decoder": 0.85, "yellow decoder": 0.12}
        """
        pass

from PyQt6.QtCore import QThread, pyqtSignal

class StreamWorker(QThread):
    """
    The background Conductor thread that orchestrates the real-time inference loop.
    Runs completely independently of the UI to ensure strict ~10ms timing and zero 
    dropped samples.
    """
    
    # PyQt Signals for safe cross-thread communication to the GUI
    # Emits: (probabilities_dict, list_of_markers_found)
    prediction_ready = pyqtSignal(dict, list)  
    
    # Emits error messages to trigger the Red Alert state in the UI
    stream_error = pyqtSignal(str)             

    def __init__(self, 
                 receiver: LSLReceiver, 
                 ring_buffer: RingBuffer, 
                 preprocessor: OnlinePreprocessor, 
                 inference_engine: LiveInferenceEngine):
        """
        Injects the four core components into the worker.
        """
        super().__init__()
        self.receiver = receiver
        self.ring_buffer = ring_buffer
        self.preprocessor = preprocessor
        self.inference_engine = inference_engine
        self._is_running = False

    def run(self) -> None:
        """
        The main execution loop. Called automatically when thread.start() is invoked 
        by the UI. Runs infinitely until self._is_running is False.
        
        Loop logic (~10ms interval):
        1. pull_new_data() from receiver.
        2. If new data exists, append() to ring_buffer.
        3. get_full_window() from ring_buffer.
        4. process_window() via preprocessor to get clean_features.
        5. predict(clean_features) via inference_engine.
        6. self.prediction_ready.emit(probabilities, markers)
        """
        pass

    def stop(self) -> None:
        """
        Signals the infinite loop in run() to break, stops the LSLReceiver securely, 
        and elegantly tears down the thread.
        """
        pass
```

## 4. Suggested App Structure
```plaintext
ReactivationDecoder_App/
│
├── requirements.txt               # Python dependencies (mne, pylsl, pyqt6, scikit-learn, etc.)
│
├── tools/
│   └── lslproxy/
│       └── LSLProxy.exe           # Fixed path for the LSLReceiver to launch the proxy
│
├── src/
│   ├── main.py                    # Application entry point (Launches the PyQt UI)
│   │
│   ├── frontend/                  # UI Layer (Cockpit) - not described in this document
│   │   ├── app_window.py          # Main PyQt6 window
│   │   ├── widgets/               # Reusable UI components (buttons, status bars)
│   │   └── plots/                 # PyQtGraph classes for AUC, TGM, and Live probabilities
│   │
│   └── backend/                   # Data Layer (Engine)
│       ├── core/
│       │   └── settings_manager.py # Class 1: SettingsManager
│       │
│       ├── offline_phase/         # Phase 1 Classes
│       │   ├── preprocessor.py    # Class 2: OfflinePreprocessor
│       │   ├── evaluator.py       # Class 3: ModelEvaluator
│       │   └── trainer.py         # Class 4: ModelTrainer
│       │
│       └── online_phase/          # Phase 2 Classes
│           ├── lsl_receiver.py    # Class 1: LSLReceiver
│           ├── ring_buffer.py     # Class 2: RingBuffer
│           ├── preprocessor.py    # Class 3: OnlinePreprocessor
│           ├── inference.py       # Class 4: LiveInferenceEngine
│           └── stream_worker.py   # Class 5: StreamWorker (QThread)
│
└── data_vault/                    # Auto-generated save directory for subjects
    ├── Sub_001/
    │   ├── phase1_training/
    │   │   ├── preprocessed/      # subject_001_experiment_epo.fif
    │   │   └── models/
    │   │       └── decoder_pipeline.joblib  # The final compiled model & online_state
    │   │
    │   └── phase2_live/
    │       ├── trigger_logs.csv         # Saved inference probabilities & marker timestamps
    │       └── probabilities (numpy array?)            # Optional backup recording of the stream
    │
    └── Sub_002/ ...
```