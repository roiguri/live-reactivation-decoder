from __future__ import annotations

import logging
from math import gcd
from typing import Optional

import mne
import numpy as np
from scipy.signal import firwin, iirnotch, lfilter, sosfilt, sosfilt_zi, tf2sos

logger = logging.getLogger(__name__)



class OnlinePreprocessor:
    """
    Stateful causal EEG preprocessor for the online phase.

    Replicates the offline pipeline's spatial transforms using matrices
    exported from Phase 1, applied to streaming micro-batches.

    Pipeline order (mirrors offline):
        1. High-pass + notch filter (causal IIR, persistent zi)
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
        # Per-channel-type rescaling factor MNE applies inside ICA.fit/apply.
        # Stored as (n_ch, 1); we transpose to (1, n_ch) once for row-major broadcasting.
        self._pre_whitener: np.ndarray = np.array(online_state["pre_whitener"]).reshape(-1, 1).T

        # Derive good/bad channel index lists from ch_names
        self._bad_indices: list[int] = [
            self._ch_names.index(ch) for ch in self._bad_channels
        ]
        self._good_indices: list[int] = [
            i for i in range(len(self._ch_names)) if i not in self._bad_indices
        ]

        # Filter coefficients
        hp = preprocessing_settings["highpass"]
        notch_cfg = preprocessing_settings.get("notch")
        
        iir_params = mne.filter.create_filter(
            data=None,
            sfreq=self._input_sfreq,
            l_freq=hp["l_freq"],
            h_freq=None,
            method=hp.get("method", "iir"),
            verbose=False,
        )
        self._highpass_sos: np.ndarray = iir_params["sos"]
        notch_freq = notch_cfg.get("freq") if notch_cfg is not None else None
        if notch_freq is not None:
            b, a = iirnotch(w0=float(notch_freq), Q=30, fs=self._input_sfreq)
            self._notch_sos: Optional[np.ndarray] = tf2sos(b, a)
        else:
            self._notch_sos = None
        # Decimation: reduce from input_sfreq to target_sfreq
        # Ratio: down_factor / up_factor (e.g. 125/32 for 1000→256 Hz)
        common = gcd(int(self._input_sfreq), int(self._target_sfreq))
        self._up_factor: int = int(self._target_sfreq) // common
        self._down_factor: int = int(self._input_sfreq) // common
        cutoff = 0.9 * self._target_sfreq / 2.0
        n_taps = 10 * self._up_factor + 1
        self._decimate_fir: np.ndarray = firwin(n_taps, cutoff, fs=self._input_sfreq)

        # Persistent filter state — reset to None each reset_state()
        self._highpass_zi: Optional[np.ndarray] = None
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
        self._highpass_zi = None
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
        if eeg_batch.ndim != 2 or eeg_batch.shape[1] != self.n_channels:
            raise ValueError(
                f"eeg_batch must be (n_samples, {self.n_channels}), got {eeg_batch.shape}"
            )
        if timestamps.shape[0] != eeg_batch.shape[0]:
            raise ValueError(
                f"timestamps length {timestamps.shape[0]} != eeg_batch rows {eeg_batch.shape[0]}"
            )
        # Guard must come before _apply_filter — zi warm-start does data[0] and
        # would raise IndexError on an empty array.
        if eeg_batch.shape[0] == 0:
            return np.empty((0, self.n_channels)), np.empty((0,))

        data = eeg_batch.copy().astype(float)
        data = self._apply_filter(data)
        data, out_timestamps = self._decimate(data, timestamps)
        self._apply_bad_channel_interpolation(data)
        self._apply_average_reference(data)
        self._apply_ica(data)
        return data, out_timestamps

    # ── Private: filtering ────────────────────────────────────────────────────

    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply causal high-pass (and optional notch) with persistent zi state.

        Args:
            data: (n_samples, n_channels)

        Returns:
            Filtered array, same shape.
        """
        if self._highpass_zi is None:
            zi_template = sosfilt_zi(self._highpass_sos)  # (n_sections, 2)
            self._highpass_zi = zi_template[:, :, np.newaxis] * data[0]  # (n_sections, 2, n_ch)

        filtered, self._highpass_zi = sosfilt(
            self._highpass_sos, data, axis=0, zi=self._highpass_zi
        )

        if self._notch_sos is not None:
            if self._notch_zi is None:
                zi_template = sosfilt_zi(self._notch_sos)
                self._notch_zi = zi_template[:, :, np.newaxis] * filtered[0]
            filtered, self._notch_zi = sosfilt(
                self._notch_sos, filtered, axis=0, zi=self._notch_zi
            )

        return filtered

    def _decimate(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Anti-alias FIR lowpass + phase-tracked subsampling from input_sfreq to target_sfreq.

        Args:
            data: (n_samples, n_channels)
            timestamps: (n_samples,)

        Returns:
            Tuple of (decimated_data, decimated_timestamps).
        """
        n_in = data.shape[0]
        n_ch = data.shape[1]

        if n_in == 0:
            return np.empty((0, n_ch)), np.empty((0,))

        # Anti-aliasing FIR — zero-init (see online_filtering.md for rationale)
        if self._decimate_zi is None:
            self._decimate_zi = np.zeros((len(self._decimate_fir) - 1, n_ch))
        filtered, self._decimate_zi = lfilter(
            self._decimate_fir, 1.0, data, axis=0, zi=self._decimate_zi
        )

        # Phase-tracked subsampling (see online_filtering.md for algorithm details)
        phase = self._decimate_phase
        n_out = (n_in * self._up_factor + phase) // self._down_factor

        if n_out == 0:
            self._decimate_phase = (phase + n_in * self._up_factor) % self._down_factor
            return np.empty((0, n_ch)), np.empty((0,))

        # k-th output is at input index ceil(((k+1)*down - phase) / up) - 1
        k = np.arange(n_out)
        out_indices = (
            np.ceil(((k + 1) * self._down_factor - phase) / self._up_factor).astype(int) - 1
        )
        out_indices = np.clip(out_indices, 0, n_in - 1)

        self._decimate_phase = (phase + n_in * self._up_factor) % self._down_factor

        return filtered[out_indices], timestamps[out_indices]

    # ── Private: spatial transforms (stateless) ───────────────────────────────

    def _apply_bad_channel_interpolation(self, data: np.ndarray) -> None:
        """Replicate offline interpolate_bads() using precomputed weight matrix. In-place."""
        if not self._bad_indices:
            return
        data[:, self._bad_indices] = data[:, self._good_indices] @ self._interp_weights

    def _apply_average_reference(self, data: np.ndarray) -> None:
        """Subtract mean across all channels. In-place.

        Note: the offline pipeline calls set_eeg_reference("average") after
        interpolate_bads(reset_bads=True), which clears the bad-channel list so the
        average is computed over ALL channels (including interpolated ones). We match
        that behaviour here — using all channels, not just good_indices.
        """
        data -= data.mean(axis=1, keepdims=True)

    def _apply_ica(self, data: np.ndarray) -> None:
        """Apply ICA artifact rejection using frozen Phase 1 matrices. In-place.

        Replicates mne.preprocessing.ICA.apply() via a delta approach:
            divide by pre_whitener → center → project to n_comp PCA subspace →
            ICA unmix → zero excluded → ICA mix → add (cleaned − projected) back →
            re-add mean → multiply by pre_whitener.
        The delta-add (rather than overwrite) preserves PCA residual variance from
        components beyond n_components_.
        """
        data /= self._pre_whitener                                        # (n, n_ch)

        if self._ica_pca_mean is not None:
            data -= self._ica_pca_mean

        projected = data @ self._ica_pca_components.T                    # (n, n_comp)
        sources = projected @ self._ica_unmixing.T                       # (n, n_comp)

        if self._ica_exclude:
            sources[:, self._ica_exclude] = 0.0

        cleaned = sources @ self._ica_mixing.T                           # (n, n_comp)
        data += (cleaned - projected) @ self._ica_pca_components         # delta in PCA space

        if self._ica_pca_mean is not None:
            data += self._ica_pca_mean

        data *= self._pre_whitener

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
