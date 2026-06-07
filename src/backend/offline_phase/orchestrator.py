from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import mne

from backend.core.artifact_models import (
    DecoderPipelineArtifactSpec,
    DecoderPipelineMetadata,
)
from backend.core.session_paths import SessionPaths
from backend.core.settings_manager import SettingsManager
from backend.offline_phase.evaluator import ModelEvaluator
from backend.offline_phase.preprocessor import OfflinePreprocessor
from backend.offline_phase.trainer import ModelTrainer

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
        # UI: operator picks + confirms each decoder's timepoint
        result = run_training(timepoints)   # {task_name: seconds}
        artifact = get_live_artifact_spec()
    """

    def __init__(
        self,
        settings_manager: SettingsManager,
        paths: SessionPaths,
    ) -> None:
        self._settings = settings_manager
        self._paths = paths

        self._data_dir: Optional[Path] = None
        self._raw: Optional[mne.io.Raw] = None
        self._preprocessor: Optional[OfflinePreprocessor] = None
        self._epochs: Optional[mne.Epochs] = None
        self._eval_results: Optional[dict[str, Any]] = None
        self._live_artifact_spec: Optional[DecoderPipelineArtifactSpec] = None
        # TODO: debug-only consumer today; revisit once Phase 1 needs persisted UI artifacts.
        self._ui_state: Optional[dict[str, Any]] = None

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

    def get_loaded_data_summary(self) -> dict[str, Any] | None:
        """Return a small summary of the loaded raw recording (or ``None``).

        Used by the UI to surface stage results in the journey panel after
        load completes. Returns ``None`` until ``load_raw_data()`` has run.
        """
        if self._raw is None:
            return None
        info = self._raw.info
        try:
            n_events = len(self._raw.annotations)
        except Exception:
            n_events = 0
        vhdr_name = None
        if self._data_dir is not None:
            vhdrs = list(self._data_dir.glob("*.vhdr"))
            if vhdrs:
                vhdr_name = vhdrs[0].name
        return {
            "file_name": vhdr_name,
            "n_channels": int(info["nchan"]),
            "sfreq": float(info["sfreq"]),
            "duration_s": float(self._raw.n_times / info["sfreq"]),
            "n_events": int(n_events),
        }

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

    def ica_component_labels(self) -> list[tuple[str, float]] | None:
        """Per-component ``(ICLabel category, confidence)``, aligned by ICA
        component index, for the review UI to annotate ``plot_components``
        titles. ``None`` if ICLabel is disabled or ICA has not been fitted.
        Valid after ``run_step1b_fit_ica()``."""
        if self._preprocessor is None:
            return None
        return self._preprocessor.component_labels

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
            output_dir=self._paths.epochs_dir,
        )
        self._epochs = self._preprocessor.epochs
        # Preprocessor already dropped its raw handle; drop ours too so the
        # Raw object's refcount reaches zero and the buffer is freed.
        self._raw = None
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

    def run_training(self, timepoints: dict[str, float]) -> dict[str, Any]:
        """
        Train one classifier per task, each at its operator-chosen timepoint.

        Args:
            timepoints: An explicit per-task ``{task_name: seconds}`` map (from
                the Evaluation UI, each decoder pre-filled with its own
                ``peak_timepoint`` suggestion). Stored verbatim in
                ``metadata.decoding_timepoints``. A task missing from the dict
                raises ``ValueError`` in ``ModelTrainer.run_training``.

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
        if self._eval_results is None:
            raise RuntimeError("Call run_evaluation() before run_training().")

        per_task_timepoints = {k: float(v) for k, v in timepoints.items()}

        trainer = ModelTrainer(self._epochs, self._settings.get_decoder_settings())
        training_results = trainer.run_training(per_task_timepoints)
        preprocessor_state = self._preprocessor.export_online_state()

        self._live_artifact_spec = DecoderPipelineArtifactSpec(
            models=training_results["models"],
            online_state=preprocessor_state,
            metadata=DecoderPipelineMetadata(
                feature_width=len(preprocessor_state["eeg_chunk_indices"]),
                decoding_timepoints=per_task_timepoints,
            ),
        )
        self._ui_state = {
            "spatial_patterns": training_results["spatial_patterns"],
            "mne_info": training_results["mne_info"],
        }

        save_path = self._save_to_disk()
        logger.info("Training complete. Pipeline saved → %s", save_path)
        return {
            "model_filepath": save_path,
            "spatial_patterns": training_results["spatial_patterns"],
            "mne_info": training_results["mne_info"],
        }

    # ── Phase 2 handoff ───────────────────────────────────────────────────────

    def get_live_artifact_spec(self) -> DecoderPipelineArtifactSpec:
        """Return the validated live-phase artifact from RAM. Raises if run_training() hasn't completed."""
        if self._live_artifact_spec is None:
            raise RuntimeError(
                "Cannot transition to Phase 2: run_training() has not completed."
            )
        return self._live_artifact_spec

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
        Load a BrainVision file and keep only the EEG channels (plus EMG, which
        channel hygiene drops downstream). Stimulus markers are read natively
        from the ``.vmrk`` by ``mne.io.read_raw_brainvision`` and left on the
        raw for ``mne.events_from_annotations`` downstream. Montage and EMG
        handling are the preprocessor's responsibility — this method is the
        pure IO boundary.
        """
        # TODO: recordings that exceed available RAM will raise MemoryError here.
        # Consider supporting mne memmap preload (preload="path/to/file.bin").
        # read_raw_brainvision loads the .vmrk Stimulus/Sxx markers as
        # annotations automatically; we deliberately do not overwrite them.
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)

        # Keep EEG-typed channels only (EMG is EEG-typed until hygiene retypes
        # it). EOG/ECG, stim and misc are dropped so the offline channel array
        # is positionally aligned with the 64-channel post-trigger-split LSL
        # EEG array.
        eeg_picks = mne.pick_types(raw.info, eeg=True).tolist()
        keep = [raw.ch_names[i] for i in eeg_picks]
        dropped = [c for c in raw.ch_names if c not in keep]
        if dropped:
            logger.info("Dropping non-EEG channels: %s", dropped)
        raw.pick(keep)
        return raw

    # ── Private: persistence ──────────────────────────────────────────────────

    def _save_to_disk(self) -> Path:
        save_path = self._paths.decoder_pipeline_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._live_artifact_spec.model_dump(), save_path)
        return save_path
