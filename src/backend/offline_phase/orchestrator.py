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

    The preprocessing pipeline is split into four operator-gated calls so the
    two manual selections happen on MNE's native interactive windows (which
    must run on the GUI main thread):

        set_file_path(data_dir)
        load_raw_data()                              # IO — user clicks Load
        raw = run_step1a_filter()                    # worker
        # UI: raw.plot(block=True) on the main thread → operator marks bads
        set_bad_channels(raw.info["bads"])           # main thread
        ica, epochs, suggested = run_step1b_fit_ica()  # worker
        # UI: ica.plot_sources(epochs, block=True) → operator toggles excludes
        stats = run_step2_apply_and_save(excluded)   # worker
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

    def run_step1a_filter(self) -> mne.io.Raw:
        """
        Channel hygiene → high-pass → notch → (early variant) low-pass +
        resample. Returns the filtered ``Raw`` for the UI's interactive
        bad-channel window.

        Raises:
            RuntimeError: if load_raw_data() has not been called first.
        """
        if self._raw is None:
            raise RuntimeError("Call load_raw_data() before run_step1a_filter().")

        self._preprocessor = OfflinePreprocessor(
            data_dir=self._data_dir,
            preprocessing_settings=self._settings.get_preprocessing_params(),
            raw=self._raw,
        )
        return self._preprocessor.run_step1a_filter()

    def set_bad_channels(self, bads: list[str]) -> None:
        """Forward the operator's bad-channel selection to the preprocessor.

        Raises:
            RuntimeError: if run_step1a_filter() has not been called first.
        """
        if self._preprocessor is None:
            raise RuntimeError("Call run_step1a_filter() before set_bad_channels().")
        self._preprocessor.set_bad_channels(bads)

    def run_step1b_fit_ica(
        self,
    ) -> tuple[mne.preprocessing.ICA, mne.Epochs, list[int]]:
        """
        Interpolate bads → epoch → average reference → fit ICA → ICLabel.

        Returns:
            (ica, epochs_for_review, suggested_exclude) — feed the first two
            to MNE's interactive component window; ``suggested_exclude``
            pre-populates ``ica.exclude``.

        Raises:
            RuntimeError: if run_step1a_filter() has not been called first.
        """
        if self._preprocessor is None:
            raise RuntimeError("Call run_step1a_filter() before run_step1b_fit_ica().")

        ica, epochs, suggested = self._preprocessor.run_step1b_fit_ica(
            self._settings.get_event_mapping()
        )
        self._epochs = epochs
        logger.info("ICA fitted. %d component(s) suggested.", len(suggested))
        return ica, epochs, suggested

    def run_step2_apply_and_save(
        self, excluded_components: list[int]
    ) -> dict[str, Any]:
        """
        Apply ICA with the user-confirmed exclusions, finish the pipeline, and
        save the cleaned epochs.

        Args:
            excluded_components: Final ICA component indices to remove.

        Returns:
            {"n_epochs": int, "n_excluded": int}

        Raises:
            RuntimeError: if run_step1b_fit_ica() has not been called first.
        """
        if self._preprocessor is None or self._preprocessor.ica is None:
            raise RuntimeError(
                "Call run_step1b_fit_ica() before run_step2_apply_and_save()."
            )

        result = self._preprocessor.run_step2_apply_and_save(
            exclude_components=excluded_components,
            output_dir=self._output_dir / "epochs",
        )
        self._epochs = self._preprocessor.epochs
        logger.info("Preprocessing complete. %d epochs retained.", len(self._epochs))
        return result

    def run_evaluation(self) -> dict[str, Any]:
        """
        Run temporal generalization cross-validation to surface the best decoding timepoint.

        Returns:
            Full evaluator result dict (times, AUC curves, TGMs, suggested_timepoint).

        Raises:
            RuntimeError: if run_step2_apply_and_save() has not been called first.
        """
        if self._epochs is None:
            raise RuntimeError(
                "Call run_step2_apply_and_save() before run_evaluation()."
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
        self.online_state = {
            **self._preprocessor.export_online_state(),
            "models": training_results["models"],
            "spatial_patterns": training_results["spatial_patterns"],
            "mne_info": training_results["mne_info"],
            "decoding_timepoint": timepoint,
        }

        save_path = self._save_to_disk()
        logger.info("Training complete. Pipeline saved → %s", save_path)
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
        Load a BrainVision file, decode the parallel-port trigger channel into
        annotations, and keep only the EEG channels (plus EMG, which channel
        hygiene drops downstream). Montage and EMG handling are the
        preprocessor's responsibility — this method is the pure IO boundary.
        """
        # TODO: recordings that exceed available RAM will raise MemoryError here.
        # Consider supporting mne memmap preload (preload="path/to/file.bin").
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

        # Parallel-port triggers are recorded as analog pulses on a dedicated
        # channel (not the .vmrk file). Decode them into Annotations now,
        # before the channel is dropped along with other non-EEG channels.
        annotations = decode_parallel_port_channel(raw)
        raw.set_annotations(annotations)

        # Keep EEG-typed channels only (EMG is EEG-typed until hygiene retypes
        # it). The decoded trigger channel, EOG/ECG, stim and misc are dropped
        # so the offline channel array is positionally aligned with the
        # 64-channel post-trigger-split LSL EEG array.
        eeg_picks = mne.pick_types(raw.info, eeg=True).tolist()
        keep = [raw.ch_names[i] for i in eeg_picks]
        dropped = [c for c in raw.ch_names if c not in keep]
        if dropped:
            logger.info("Dropping non-EEG channels: %s", dropped)
        raw.pick(keep)
        return raw

    # ── Private: persistence ──────────────────────────────────────────────────

    def _save_to_disk(self) -> Path:
        save_path = self._output_dir / "models" / "decoder_pipeline.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.online_state, save_path)
        return save_path
