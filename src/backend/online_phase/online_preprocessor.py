from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)



class OnlinePreprocessor:
    """
    Stateful causal EEG preprocessor for the online phase.

    Replicates the offline pipeline's spatial transforms using matrices
    exported from Phase 1, applied to streaming micro-batches.

    Pipeline order (mirrors offline):
        1. Bandpass + notch filter (causal IIR, persistent zi)
        2. Decimate to target rate (FIR anti-alias + phase tracking)
        3. Interpolate bad channels (fixed weight matrix)
        4. Average reference
        5. Apply ICA (fixed unmixing/mixing matrices)
    """

    def __init__(
        self,
        preprocessing_settings: dict,
        online_state: dict,
        input_sfreq: float = 1000.0,
    ) -> None:
        self._validate_inputs(preprocessing_settings, online_state)

        self._input_sfreq = float(input_sfreq)
        self._target_sfreq = float(preprocessing_settings["resample"]["target_rate"])

        # Spatial transform matrices from Phase 1
        self._ch_names: list[str] = list(online_state["ch_names"])
        self._bad_channels: list[str] = list(online_state["bad_channels"])
        self._interp_weights: Optional[np.ndarray] = online_state["interp_weights"]
        self._ica_unmixing: np.ndarray = np.array(online_state["ica_unmixing"])
        self._ica_mixing: np.ndarray = np.array(online_state["ica_mixing"])
        self._ica_pca_components: np.ndarray = np.array(online_state["ica_pca_components"])
        self._ica_pca_mean: Optional[np.ndarray] = (
            np.array(online_state["ica_pca_mean"])
            if online_state["ica_pca_mean"] is not None
            else None
        )
        self._ica_exclude: list[int] = list(online_state["ica_exclude"])

        # Derive good/bad channel index lists from ch_names
        self._bad_indices: list[int] = [
            self._ch_names.index(ch) for ch in self._bad_channels
        ]
        self._good_indices: list[int] = [
            i for i in range(len(self._ch_names)) if i not in self._bad_indices
        ]

        # Filter coefficients — initialised in later commits
        self._bandpass_sos: Optional[np.ndarray] = None
        self._notch_sos: Optional[np.ndarray] = None
        self._decimate_fir: Optional[np.ndarray] = None

        # Persistent filter state — reset to None each reset_state()
        self._bandpass_zi: Optional[np.ndarray] = None
        self._notch_zi: Optional[np.ndarray] = None
        self._decimate_zi: Optional[np.ndarray] = None
        self._decimate_phase: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        return len(self._ch_names)

    @property
    def target_sfreq(self) -> float:
        return self._target_sfreq

    @property
    def input_sfreq(self) -> float:
        return self._input_sfreq

    def reset_state(self) -> None:
        """Reset all causal filter state to initial values (as if no data has been seen)."""
        self._bandpass_zi = None
        self._notch_zi = None
        self._decimate_zi = None
        self._decimate_phase = 0

    def process_batch(
        self,
        eeg_batch: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the full online preprocessing pipeline to one micro-batch.

        Args:
            eeg_batch: (n_samples, n_channels) at input_sfreq.
            timestamps: (n_samples,) LSL timestamps.

        Returns:
            Tuple of (features, output_timestamps) at target_sfreq.
        """
        raise NotImplementedError

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        preprocessing_settings: dict,
        online_state: dict,
    ) -> None:
        # Cross-validate Phase 1 sample rate against Phase 2 config.
        # A mismatch means filter coefficients would be designed for the wrong rate,
        # producing silently wrong results with no crash.
        target_rate = preprocessing_settings["resample"]["target_rate"]
        sfreq_offline = online_state["sfreq_offline"]
        if abs(float(sfreq_offline) - float(target_rate)) > 1e-6:
            raise ValueError(
                f"online_state['sfreq_offline'] ({sfreq_offline}) does not match "
                f"preprocessing_settings['resample']['target_rate'] ({target_rate}). "
                "Phase 1 and Phase 2 configs are out of sync."
            )

        # Catch channel/ICA dimension mismatch early — otherwise it manifests as a
        # cryptic NumPy broadcast error on the first process_batch() call.
        ch_names = online_state["ch_names"]
        pca_components = online_state["ica_pca_components"]
        if hasattr(pca_components, "shape") and pca_components.shape[1] != len(ch_names):
            raise ValueError(
                f"online_state['ica_pca_components'] has {pca_components.shape[1]} columns "
                f"but online_state['ch_names'] has {len(ch_names)} entries. "
                "ch_names and ICA matrices are inconsistent."
            )
