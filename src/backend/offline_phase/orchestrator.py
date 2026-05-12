from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import mne

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.evaluator import ModelEvaluator
from backend.offline_phase.preprocessor import OfflinePreprocessor
from backend.offline_phase.trainer import ModelTrainer
from backend.offline_phase.trigger_decoder import decode_parallel_port_channel

logger = logging.getLogger(__name__)


class OfflineOrchestrator:
    """
    Façade over the Phase 1 backend classes. Single entry point for the Phase 1 UI.

    Owns file I/O (raw loading), holds intermediate state between user-triggered
    steps, and bundles the final decoder_pipeline.joblib export.

    Typical call sequence:
        set_file_path(data_dir)
        load_raw_data()                        # IO — user clicks Load
        ica, suggested = run_step1_prepare_ica()  # preprocessing + ICA — user clicks Start
        # UI: user reviews ICA components, selects excluded_components
        stats = run_step2_finish_pipeline(excluded_components)
        eval_results = run_evaluation()
        # UI: user clicks timepoint on TGM plot
        result = run_training(timepoint)
        online_state = get_online_state_for_live_phase()
    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        output_dir: Path,
    ) -> None:
        self._settings = settings_manager
        self._output_dir = Path(output_dir)

        self._data_dir: Optional[Path] = None
        self._raw: Optional[mne.io.Raw] = None
        self._preprocessor: Optional[OfflinePreprocessor] = None
        self._epochs: Optional[mne.Epochs] = None
        self._eval_results: Optional[dict[str, Any]] = None
        self.online_state: dict[str, Any] = {}

    # ── UI interaction ────────────────────────────────────────────────────────

    def set_file_path(self, data_dir: str | Path) -> None:
        """Store the data directory containing the subject's .vhdr file."""
        self._data_dir = Path(data_dir)
        logger.info("Data directory set: %s", self._data_dir)

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def load_raw_data(self) -> None:
        """
        Load the raw EEG file from disk.

        Raises:
            ValueError: if set_file_path() has not been called first.
            FileNotFoundError: if no .vhdr file exists in the data directory.
        """
        if self._data_dir is None:
            raise ValueError("Call set_file_path() before load_raw_data().")

        vhdr = self._find_vhdr()
        self._raw = self._load_eeg_raw(vhdr)
        logger.info("Raw EEG loaded from %s", vhdr)

    def run_step1_prepare_ica(self) -> tuple[mne.preprocessing.ICA, list[int]]:
        """
        Run signal preprocessing (filter, resample, bad channels, reference) and
        fit ICA. Returns the ICA object and auto-suggested artifact components for
        the user to review before committing exclusions.

        Returns:
            (ica_obj, suggested_components) — ica_obj for UI topomap display;
            suggested_components are auto-detected EOG/ECG indices to pre-select.

        Raises:
            RuntimeError: if load_raw_data() has not been called first.
        """
        if self._raw is None:
            raise RuntimeError("Call load_raw_data() before run_step1_prepare_ica().")

        self._preprocessor = OfflinePreprocessor(
            data_dir=self._data_dir,
            preprocessing_settings=self._settings.get_preprocessing_params(),
            raw=self._raw,
        )
        suggested = self._preprocessor.run_step1_prepare_ica()
        logger.info("ICA fitted. %d component(s) auto-suggested.", len(suggested))
        return self._preprocessor.ica, suggested

    def run_step2_finish_pipeline(
        self, excluded_components: list[int]
    ) -> dict[str, Any]:
        """
        Apply ICA, epoch, and run AutoReject using the user-confirmed exclusions.

        Args:
            excluded_components: Final ICA component indices to remove.

        Returns:
            {"n_epochs": int, "event_counts": dict[str, int], "channel_names": list[str]}

        Raises:
            RuntimeError: if run_step1_prepare_ica() has not been called first.
        """
        if self._preprocessor is None or self._preprocessor.ica is None:
            raise RuntimeError(
                "Call run_step1_prepare_ica() before run_step2_finish_pipeline()."
            )

        self._preprocessor.run_step2_finish_pipeline(
            exclude_components=excluded_components,
            event_mapping=self._settings.get_event_mapping(),
            output_dir=self._output_dir / "epochs",
        )
        self._epochs = self._preprocessor.epochs
        logger.info("Preprocessing complete. %d epochs retained.", len(self._epochs))
        # TODO: surface AutoReject drop count + bad-channel count for the UI
        # complete page. `n_dropped` is computed in preprocessor._autoreject
        # (reject_log.bad_epochs.sum()) but currently discarded; store it on the
        # preprocessor (e.g. self._autoreject_dropped: int | None where None =
        # AR was skipped) and forward here as
        # `{"n_epochs": ..., "autoreject_dropped": ..., "bad_channels": [...]}`.
        return {"n_epochs": len(self._epochs)}

    def run_evaluation(self) -> dict[str, Any]:
        """
        Run temporal generalization cross-validation to surface the best decoding timepoint.

        Returns:
            Full evaluator result dict (times, AUC curves, TGMs, suggested_timepoint).

        Raises:
            RuntimeError: if run_step2_finish_pipeline() has not been called first.
        """
        if self._epochs is None:
            raise RuntimeError(
                "Call run_step2_finish_pipeline() before run_evaluation()."
            )

        evaluator = ModelEvaluator(self._epochs, self._settings.get_decoder_settings())
        self._eval_results = evaluator.run_evaluation()
        logger.info(
            "Evaluation complete. Suggested timepoint: %.3fs",
            self._eval_results["suggested_timepoint"],
        )
        return self._eval_results

    def run_training(self, timepoint: float) -> dict[str, Any]:
        """
        Train one classifier per task at the given timepoint, bundle the online state,
        and save decoder_pipeline.joblib to disk.

        Args:
            timepoint: Time in seconds selected by the researcher (e.g. 0.350).

        Returns:
            {
                "model_filepath":   Path,
                "spatial_patterns": {task_name: np.ndarray},
                "mne_info":         mne.Info,
            }

        Raises:
            RuntimeError: if run_evaluation() has not been called first.
        """
        if self._epochs is None:
            raise RuntimeError("Call run_evaluation() before run_training().")

        trainer = ModelTrainer(self._epochs, self._settings.get_decoder_settings())
        training_results = trainer.run_training(timepoint)
        # TODO: change according to information needed in phase2
        self.online_state = {
            **self._preprocessor.export_online_state(),
            "models": training_results["models"],
            "spatial_patterns": training_results["spatial_patterns"],
            "mne_info": training_results["mne_info"],
            "decoding_timepoint": timepoint,
        }

        save_path = self._save_to_disk()
        logger.info("Training complete. Pipeline saved → %s", save_path)
        # TODO: consider removing some of the returned values.
        return {
            "model_filepath": save_path,
            "spatial_patterns": training_results["spatial_patterns"],
            "mne_info": training_results["mne_info"],
        }

    # ── Phase 2 handoff ───────────────────────────────────────────────────────

    def get_online_state_for_live_phase(self) -> dict[str, Any]:
        """
        Return the bundled online state directly from RAM, bypassing disk I/O.

        Raises:
            RuntimeError: if run_training() has not been called first.
        """
        if not self.online_state:
            raise RuntimeError(
                "Cannot transition to Phase 2: run_training() has not completed."
            )
        return self.online_state

    # ── Private: IO ───────────────────────────────────────────────────────────

    def _find_vhdr(self) -> Path:
        """Find the .vhdr file in the data directory."""
        vhdrs = list(self._data_dir.glob("*.vhdr"))
        if not vhdrs:
            raise FileNotFoundError(f"No .vhdr file found in {self._data_dir}")
        if len(vhdrs) > 1:
            logger.warning(
                "Multiple .vhdr files in %s, using %s",
                self._data_dir, vhdrs[0].name,
            )
        return vhdrs[0]

    def _load_eeg_raw(self, vhdr: Path) -> mne.io.Raw:
        """
        Load a BrainVision file and retain only EEG channels with a known
        montage position plus EOG/ECG channels needed for ICA artifact detection.
        """
        # TODO: recordings that exceed available RAM will raise MemoryError here.
        # Consider supporting mne memmap preload (preload="path/to/file.bin") so
        # the OS pages signal data from disk without a contiguous RAM allocation.
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn")

        # Parallel-port triggers are recorded as analog pulses on a dedicated
        # channel (not the .vmrk file). Decode them into Annotations now,
        # before the channel is dropped along with other non-EEG channels.
        annotations = decode_parallel_port_channel(raw)
        # TODO: this overrides any existing anotations on the raw; consider merging with existing ones instead of replacing.
        raw.set_annotations(annotations)

        # TODO: consider this code part - added due to issues with emg channel in test data
        # Keep only EEG channels with a known montage position plus physiological
        # reference channels (EOG/ECG) needed for ICA artifact detection.
        # Everything else (e.g. the EMG trigger channel, stim, misc) is dropped.
        montage_names = {ch.lower() for ch in montage.ch_names}
        eeg_picks = [
            i for i in mne.pick_types(raw.info, eeg=True)
            if raw.ch_names[i].lower() in montage_names
        ]
        eog_picks = mne.pick_types(raw.info, eog=True).tolist()
        ecg_picks = mne.pick_types(raw.info, ecg=True).tolist()
        keep = sorted(set(eeg_picks) | set(eog_picks) | set(ecg_picks))
        dropped = [raw.ch_names[i] for i in range(len(raw.ch_names)) if i not in keep]
        if dropped:
            logger.info("Dropping non-EEG/EOG/ECG channels: %s", dropped)
        raw.pick(keep)
        # End TODO

        return raw

    # ── Private: persistence ──────────────────────────────────────────────────

    def _save_to_disk(self) -> Path:
        # TODO: review safe path
        save_path = self._output_dir / "models" / "decoder_pipeline.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.online_state, save_path)
        return save_path
