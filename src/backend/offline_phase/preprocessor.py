from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np
from autoreject import AutoReject

logger = logging.getLogger(__name__)


class OfflinePreprocessor:
    """
    Executes the offline cleaning pipeline for a single subject recording.
    Designed for two-step execution to allow manual ICA artifact rejection between steps.

    Step 1 — run_step1_prepare_ica():
        1. Load raw BrainVision file and apply montage.
        2. Band-pass + notch filter.
        3. Resample to target rate.
        4. Detect and interpolate bad channels.
        5. Re-reference to average.
        6. Fit ICA and auto-detect EOG/ECG components.
        Returns suggested component indices for user review.

    Step 2 — run_step2_finish_pipeline():
        7. Apply ICA with user-confirmed component exclusions.
        8. Epoch around stimulus triggers.
        9. AutoReject to repair/drop bad epochs.
        10. Save cleaned epochs to .fif.
    """

    def __init__(
        self,
        data_dir: Path,
        preprocessing_settings: dict[str, Any],
    ) -> None:
        self.data_dir = Path(data_dir)
        self.subject_id = self.data_dir.name
        self.settings = preprocessing_settings

        self.vhdr: Optional[Path] = self._find_vhdr()
        self.raw: Optional[mne.io.Raw] = None
        self.epochs: Optional[mne.Epochs] = None
        self.ica: Optional[mne.preprocessing.ICA] = None
        self._bad_channels: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run_step1_prepare_ica(self) -> list[int]:
        """
        First half of the pipeline:
            1. Load raw BrainVision file and apply montage.
            2. Band-pass + notch filter.
            3. Resample to target rate.
            4. Detect and interpolate bad channels.
            5. Re-reference to average.
            6. Fit ICA and auto-detect EOG/ECG artifact components.

        Returns:
            Suggested EOG/ECG component indices for the user to review.

        Raises:
            FileNotFoundError: if no .vhdr file was found in data_dir.
        """
        if self.vhdr is None:
            raise FileNotFoundError(
                f"No .vhdr file found in {self.data_dir}"
            )

        self.raw = self._load_raw()
        self._filter()
        self._resample()
        self._detect_bad_channels()
        self._reference()
        suggested = self._fit_ica()

        logger.info("ICA fitted. Suggested components: %s", suggested)
        return suggested

    def run_step2_finish_pipeline(
        self,
        exclude_components: list[int],
        event_mapping: dict[str, int],
        output_dir: Path,
    ) -> None:
        """
        Second half of the pipeline:
            7. Apply ICA with user-confirmed component exclusions.
            8. Epoch around stimulus triggers.
            9. AutoReject to repair/drop bad epochs.
            10. Save cleaned epochs to .fif.

        Args:
            exclude_components: Final list of ICA component indices to remove
                                (user-confirmed, may differ from suggestions).
            event_mapping: {event_name: trigger_id} — MNE convention.
            output_dir: Directory to write the .fif epochs file.
        """
        if self.raw is None or self.ica is None:
            raise RuntimeError("run_step1_prepare_ica() must be called first.")

        self.ica.exclude = exclude_components
        self.ica.apply(self.raw, verbose=False)

        self.epochs = self._epoch(event_mapping)
        # TODO: consider handling the case were no epochs are found.
        logger.info("Epochs before AutoReject: %d", len(self.epochs))

        self._autoreject()
        logger.info("Epochs after AutoReject: %d", len(self.epochs))

        self._save(Path(output_dir))

    def export_online_state(self) -> dict[str, Any]:
        """
        Extracts the ICA spatial transforms and channel metadata needed by
        the online preprocessor to replicate offline cleaning on live numpy windows.

        Returns:
            Dict containing ICA matrices, excluded components, bad channel names,
            channel order, and offline sampling rate.

        Raises:
            RuntimeError: if called before run_step2_finish_pipeline().
        """
        if self.ica is None or self.raw is None:
            raise RuntimeError(
                "Both pipeline steps must complete before exporting online state."
            )

        n_comp = self.ica.n_components_
        return {
            "bad_channels": list(self._bad_channels),
            "ica_unmixing": self.ica.unmixing_matrix_.copy(),
            "ica_mixing": self.ica.mixing_matrix_.copy(),
            "ica_pca_components": self.ica.pca_components_[:n_comp].copy(),
            "ica_pca_mean": (
                self.ica.pca_mean_.copy() if self.ica.pca_mean_ is not None else None
            ),
            "ica_exclude": list(self.ica.exclude),
            "ch_names": [self.raw.ch_names[i] for i in mne.pick_types(self.raw.info, eeg=True)],
            "sfreq_offline": float(self.raw.info["sfreq"]),
        }

    # ── Private: Stage 1 ─────────────────────────────────────────────────────

    def _find_vhdr(self) -> Optional[Path]:
        vhdrs = list(self.data_dir.glob("*.vhdr"))
        if not vhdrs:
            logger.warning("No .vhdr found in %s", self.data_dir)
            return None
        if len(vhdrs) > 1:
            logger.warning(
                "Multiple .vhdr files in %s, using %s", self.data_dir, vhdrs[0].name
            )
        return vhdrs[0]

    def _load_raw(self) -> mne.io.Raw:
        # TODO: recordings that exceed available RAM will raise MemoryError here.
        # Consider supporting mne memmap preload (preload="path/to/file.bin") so
        # the OS pages signal data from disk without a contiguous RAM allocation.
        raw = mne.io.read_raw_brainvision(self.vhdr, preload=True, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn")

        # TODO: consider this code part - added due to issues with emg channel in test data
        # Keep only EEG channels with a known montage position plus physiological
        # reference channels (EOG/ECG) needed for ICA artifact detection.
        # Everything else (e.g. EMG recorded as EEG type, stim, misc) is dropped.
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

    def _filter(self) -> None:
        bp = self.settings["bandpass"]
        self.raw.filter(
            l_freq=bp["l_freq"],
            h_freq=bp["h_freq"],
            method=bp["method"],
            verbose=False,
        )
        if bp.get("notch"):
            self.raw.notch_filter(freqs=bp["notch"], verbose=False)

    def _resample(self) -> None:
        target = self.settings["resample"]["target_rate"]
        if self.raw.info["sfreq"] > target:
            self.raw.resample(target, verbose=False)

    def _detect_bad_channels(self) -> None:
        rc = self.settings["reject_criteria"]
        data = self.raw.get_data(picks="eeg")
        stds = data.std(axis=1)

        flat_idx = np.where(stds < rc["flat_threshold"])[0]
        z = (stds - stds.mean()) / stds.std() if stds.std() > 0 else np.zeros_like(stds)
        noisy_idx = np.where(z > rc["noisy_z_score"])[0]

        picks_eeg = mne.pick_types(self.raw.info, eeg=True)
        bads = list({
            self.raw.ch_names[picks_eeg[i]]
            for i in np.concatenate([flat_idx, noisy_idx]).astype(int)
        })

        self.raw.info["bads"] = bads
        self._bad_channels = bads
        if bads:
            logger.info("Bad channels detected: %s", bads)
            self.raw.interpolate_bads(reset_bads=True, verbose=False)

    def _reference(self) -> None:
        self.raw.set_eeg_reference("average", projection=False, verbose=False)

    def _fit_ica(self) -> list[int]:
        ica_s = self.settings["ica"]
        raw_for_ica = self.raw.copy().filter(
            l_freq=ica_s["fit_l_freq"], h_freq=None, verbose=False
        )
        self.ica = mne.preprocessing.ICA(
            n_components=ica_s["n_components"],
            method=ica_s["method"],
            random_state=ica_s["random_state"],
            max_iter="auto",
        )
        self.ica.fit(raw_for_ica, verbose=False)

        ch_types = set(self.raw.get_channel_types())

        eog_idx: list[int] = []
        if "eog" in ch_types:
            eog_idx, _ = self.ica.find_bads_eog(self.raw, verbose=False)

        ecg_idx: list[int] = []
        if "ecg" in ch_types:
            ecg_idx, _ = self.ica.find_bads_ecg(self.raw, verbose=False)

        # TODO: consider richer auto-suggestion when no EOG/ECG channels are present:
        # - mne-icalabel (ICLabel): neural-net classifier that labels components as
        #   brain/eye/muscle/heart/line-noise/channel-noise/other without dedicated
        #   physiological channels. from mne_icalabel import label_components
        # - Synthetic EOG: correlate components against Fp1/Fp2 average as a proxy
        #   for eye movements when no dedicated EOG electrode was recorded.

        suggested = list({int(i) for i in eog_idx + ecg_idx})
        return suggested

    # ── Private: Stage 2 ─────────────────────────────────────────────────────

    def _epoch(self, event_mapping: dict[str, int]) -> mne.Epochs:
        ep = self.settings["epochs"]
        events, found_event_id = mne.events_from_annotations(self.raw, verbose=False)
        valid_event_id = {
            name: code
            for name, code in event_mapping.items()
            if code in found_event_id.values()
        }
        if not valid_event_id:
            logger.warning(
                "None of the expected trigger codes found — falling back to all events"
            )
            valid_event_id = found_event_id

        rc = self.settings["reject_criteria"]
        baseline = tuple(ep["baseline"]) if ep["baseline"] is not None else None
        return mne.Epochs(
            self.raw,
            events,
            event_id=valid_event_id,
            tmin=ep["tmin"],
            tmax=ep["tmax"],
            baseline=baseline,
            reject=dict(eeg=rc["hard_amplitude"]),
            event_repeated="drop", # TODO: consider if this is required.
            preload=True,
            verbose=False,
        )

    def _autoreject(self) -> None:
        # TODO: consider how to handle errors in pipline
        if len(self.epochs) == 0:
            raise RuntimeError(
                "AutoReject received 0 epochs — the event mapping likely does not match "
                "the trigger codes in this recording."
            )

        # AutoReject's cross-validation requires >= 2 epochs per condition.
        # With short data crops this can fail; skip and warn rather than crash.
        # consider a finer fallback — e.g. drop only sparse conditions
        # instead of skipping AutoReject entirely, or make the minimum configurable.
        condition_counts = {
            name: int((self.epochs.events[:, 2] == code).sum())
            for name, code in self.epochs.event_id.items()
        }
        if min(condition_counts.values()) < 2:
            logger.warning(
                "AutoReject skipped — too few epochs per condition for cross-validation: %s. "
                "Epochs saved without AutoReject cleaning.",
                condition_counts,
            )
            return
        # end TODO
        
        ar_s = self.settings["autoreject"]
        ar = AutoReject(random_state=ar_s["random_state"], verbose=False)
        self.epochs, reject_log = ar.fit_transform(self.epochs, return_log=True)
        n_dropped = reject_log.bad_epochs.sum()
        logger.info("AutoReject dropped %d epochs", n_dropped)

    def _save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.subject_id}_epo.fif"
        self.epochs.save(out_path, overwrite=True, verbose=False)
        logger.info("Saved → %s", out_path)
