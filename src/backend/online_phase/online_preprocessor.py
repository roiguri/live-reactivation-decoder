from __future__ import annotations

import logging
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

    Pipeline order (mirrors offline). Two variants selected by
    settings["preprocessing"]["resample_filter_stage"]:

      "early" — LP + decimate run before the spatial transforms:
        1. High-pass + notch filter (causal IIR, persistent zi)
        2. Low-pass filter at h_freq Hz (causal IIR, persistent zi)
        3. Decimate to target rate (FIR anti-alias + integer subsample)
        4. Interpolate bad channels (fixed weight matrix)
        5. Average reference
        6. Apply ICA (fixed unmixing/mixing matrices)

      "late" — LP + decimate run after the spatial transforms:
        1. High-pass + notch filter (causal IIR, persistent zi)
        2. Interpolate bad channels (fixed weight matrix)
        3. Average reference
        4. Apply ICA (fixed unmixing/mixing matrices)
        5. Low-pass filter at h_freq Hz (causal IIR, persistent zi)
        6. Decimate to target rate (FIR anti-alias + integer subsample)
    """

    def __init__(
        self,
        preprocessing_settings: dict,
        online_state: dict,
        input_sfreq: float = 1000.0,
        # TODO: LSL streams (NeurOne, XDF replay) deliver EEG in microvolts,
        # but the offline pipeline (MNE) trains models in SI volts. This
        # scale factor converts LSL input to match the offline unit space.
        # Hardcoded for NeurOne; revisit if a non-µV source is added.
        lsl_to_si_scale: float = 1e-6,
    ) -> None:
        self._validate_inputs(preprocessing_settings, online_state)

        self._lsl_to_si_scale = float(lsl_to_si_scale)
        self._input_sfreq = float(input_sfreq)
        self._target_sfreq = float(
            preprocessing_settings["final_resample"]["target_rate"]
        )
        self._resample_filter_stage: str = preprocessing_settings.get(
            "resample_filter_stage", "early"
        )
        if self._resample_filter_stage not in ("early", "late"):
            raise ValueError(
                f"resample_filter_stage must be 'early' or 'late', got "
                f"{self._resample_filter_stage!r}."
            )

        # Spatial transform matrices from Phase 1
        self._eeg_chunk_indices: list[int] = list(online_state["eeg_chunk_indices"])
        self._bad_indices: list[int] = list(online_state["bad_indices"])
        self._interp_weights: Optional[np.ndarray] = online_state["interp_weights"]
        self._ica_unmixing: np.ndarray = np.array(online_state["ica_unmixing"])
        self._ica_mixing: np.ndarray = np.array(online_state["ica_mixing"])
        self._ica_pca_components: np.ndarray = np.array(
            online_state["ica_pca_components"]
        )
        self._ica_pca_mean: Optional[np.ndarray] = (
            np.array(online_state["ica_pca_mean"])
            if online_state["ica_pca_mean"] is not None
            else None
        )
        self._ica_exclude: list[int] = list(online_state["ica_exclude"])
        # Per-channel-type rescaling factor MNE applies inside ICA.fit/apply.
        self._pre_whitener: np.ndarray = (
            np.array(online_state["pre_whitener"]).reshape(-1, 1).T
        )

        # Derive good indices from the post-hygiene channel count and the bad list.
        self._n_eeg: int = len(self._eeg_chunk_indices)
        self._good_indices: list[int] = [
            i for i in range(self._n_eeg) if i not in self._bad_indices
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

        # Low-pass filter (40 Hz default, IIR causal).
        lp = preprocessing_settings["lowpass"]
        lp_params = mne.filter.create_filter(
            data=None,
            sfreq=self._input_sfreq,
            l_freq=None,
            h_freq=lp["h_freq"],
            method=lp.get("method", "iir"),
            verbose=False,
        )
        self._lowpass_sos: np.ndarray = lp_params["sos"]

        # Decimation
        # TODO: currently don't support non-integer decimation ratios
        if int(self._input_sfreq) % int(self._target_sfreq) != 0:
            raise ValueError(
                f"input_sfreq ({self._input_sfreq}) must be an integer multiple "
                f"of target_sfreq ({self._target_sfreq}). Non-integer decimation "
                "ratios are not supported."
            )
        self._decimation: int = int(self._input_sfreq) // int(self._target_sfreq)
        cutoff = 0.9 * self._target_sfreq / 2.0
        n_taps = 10 * self._decimation + 1
        self._decimate_fir: np.ndarray = firwin(n_taps, cutoff, fs=self._input_sfreq)

        # Persistent filter state — reset to None each reset_state()
        self._highpass_zi: Optional[np.ndarray] = None
        self._notch_zi: Optional[np.ndarray] = None
        self._lowpass_zi: Optional[np.ndarray] = None
        self._decimate_zi: Optional[np.ndarray] = None
        self._decimate_phase: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        return self._n_eeg

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
        self._lowpass_zi = None
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
            eeg_batch: (n_samples, raw_n_channels) at input_sfreq. Comes
                directly from the LSL receiver (post-trigger-split, pre-hygiene).
                raw_n_channels is the LSL stream's EEG width (typically 64).
                Width is not explicitly validated here — if it's wrong, the
                eeg_chunk_indices slice below will raise IndexError on the first
                out-of-bounds index.
            timestamps: (n_samples,) LSL timestamps.

        Returns:
            Tuple of (features, output_timestamps) at target_sfreq with
            (n_out_samples, n_channels) where n_channels = len(eeg_chunk_indices).
        """
        if eeg_batch.ndim != 2:
            raise ValueError(
                f"eeg_batch must be 2D (n_samples, n_channels), got shape {eeg_batch.shape}"
            )
        if timestamps.shape[0] != eeg_batch.shape[0]:
            raise ValueError(
                f"timestamps length {timestamps.shape[0]} != eeg_batch rows {eeg_batch.shape[0]}"
            )
        # Guard must come before _apply_filter — zi warm-start does data[0] and
        # would raise IndexError on an empty array.
        if eeg_batch.shape[0] == 0:
            return np.empty((0, self.n_channels)), np.empty((0,))

        # Apply positional EEG hygiene and convert to SI volts.
        data = eeg_batch[:, self._eeg_chunk_indices].astype(float)
        # TODO: review earlier comment and scaling
        if self._lsl_to_si_scale != 1.0:
            data *= self._lsl_to_si_scale
        data = self._apply_filter(data)
        if self._resample_filter_stage == "early":
            # LP + decimate happen before spatial transforms
            data = self._apply_lowpass(data)
            data, out_timestamps = self._decimate(data, timestamps)
            self._apply_bad_channel_interpolation(data)
            self._apply_average_reference(data)
            self._apply_ica(data)
        else:
            # spatial transforms at input_sfreq, then LP + decimate
            self._apply_bad_channel_interpolation(data)
            self._apply_average_reference(data)
            self._apply_ica(data)
            data = self._apply_lowpass(data)
            data, out_timestamps = self._decimate(data, timestamps)
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
            self._highpass_zi = (
                zi_template[:, :, np.newaxis] * data[0]
            )  # (n_sections, 2, n_ch)

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

    def _apply_lowpass(self, data: np.ndarray) -> np.ndarray:
        """Apply causal low-pass with persistent zi state.

        Mirrors _apply_filter's structure but with only the LP stage. Runs
        before _decimate (in either variant) to remove energy above the new
        Nyquist and prevent aliasing.

        Args:
            data: (n_samples, n_channels)

        Returns:
            Filtered array, same shape.
        """
        if self._lowpass_zi is None:
            zi_template = sosfilt_zi(self._lowpass_sos)
            self._lowpass_zi = zi_template[:, :, np.newaxis] * data[0]

        filtered, self._lowpass_zi = sosfilt(
            self._lowpass_sos, data, axis=0, zi=self._lowpass_zi
        )
        return filtered

    def _decimate(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Anti-alias FIR + integer subsampling from input_sfreq to target_sfreq.

        Integer-ratio only (e.g. 1000 -> 100 Hz = decimation factor 10). The
        FIR anti-alias filter carries persistent zi across chunks; the phase
        tracks where the next kept sample lands inside the next chunk.

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

        # Anti-aliasing FIR — zero-init
        if self._decimate_zi is None:
            self._decimate_zi = np.zeros((len(self._decimate_fir) - 1, n_ch))
        filtered, self._decimate_zi = lfilter(
            self._decimate_fir, 1.0, data, axis=0, zi=self._decimate_zi
        )

        phase = self._decimate_phase
        d = self._decimation
        out_indices = np.arange(phase, n_in, d)
        self._decimate_phase = (d - ((n_in - phase) % d)) % d

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
        data /= self._pre_whitener  # (n, n_ch)

        if self._ica_pca_mean is not None:
            data -= self._ica_pca_mean

        projected = data @ self._ica_pca_components.T  # (n, n_comp)
        sources = projected @ self._ica_unmixing.T  # (n, n_comp)

        if self._ica_exclude:
            sources[:, self._ica_exclude] = 0.0

        cleaned = sources @ self._ica_mixing.T  # (n, n_comp)
        data += (cleaned - projected) @ self._ica_pca_components  # delta in PCA space

        if self._ica_pca_mean is not None:
            data += self._ica_pca_mean

        data *= self._pre_whitener

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        preprocessing_settings: dict,
        online_state: dict,
    ) -> None:

        eeg_chunk_indices = list(online_state["eeg_chunk_indices"])
        n_eeg = len(eeg_chunk_indices)

        if any(i < 0 for i in eeg_chunk_indices):
            raise ValueError(
                f"online_state['eeg_chunk_indices'] contains negative values: "
                f"{[i for i in eeg_chunk_indices if i < 0]}."
            )
        if len(set(eeg_chunk_indices)) != n_eeg:
            duplicates = [
                i for i in eeg_chunk_indices if eeg_chunk_indices.count(i) > 1
            ]
            raise ValueError(
                f"online_state['eeg_chunk_indices'] contains duplicates: {sorted(set(duplicates))}."
            )

        bad_indices = list(online_state["bad_indices"])
        if any(i < 0 or i >= n_eeg for i in bad_indices):
            raise ValueError(
                f"online_state['bad_indices'] must be in [0, {n_eeg}); got {bad_indices}."
            )
        if len(set(bad_indices)) != len(bad_indices):
            raise ValueError(
                f"online_state['bad_indices'] contains duplicates: {bad_indices}."
            )

        # Catch channel/ICA dimension mismatch early — otherwise it manifests as a
        # cryptic NumPy broadcast error on the first process_batch() call.
        pca_components = online_state["ica_pca_components"]
        if hasattr(pca_components, "shape") and pca_components.shape[1] != n_eeg:
            raise ValueError(
                f"online_state['ica_pca_components'] has {pca_components.shape[1]} columns "
                f"but eeg_chunk_indices has {n_eeg} entries. "
                "online_state shape is inconsistent."
            )

        n_components = (
            pca_components.shape[0] if hasattr(pca_components, "shape") else 0
        )
        ica_exclude = list(online_state.get("ica_exclude", []))
        if any(i < 0 or i >= n_components for i in ica_exclude):
            raise ValueError(
                f"online_state['ica_exclude'] must be in [0, {n_components}); got {ica_exclude}."
            )

        interp_weights = online_state["interp_weights"]
        if interp_weights is not None and hasattr(interp_weights, "shape"):
            n_good_expected = n_eeg - len(bad_indices)
            if interp_weights.shape != (n_good_expected, len(bad_indices)):
                raise ValueError(
                    f"online_state['interp_weights'] shape {interp_weights.shape} does not "
                    f"match expected ({n_good_expected}, {len(bad_indices)}) for {n_eeg} EEG "
                    f"channels with {len(bad_indices)} bads."
                )
