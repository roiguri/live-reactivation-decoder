from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np
from scipy.signal import firwin, lfilter

from backend.core.preprocessing_constants import LOWPASS_H_FREQ, LOWPASS_METHOD

logger = logging.getLogger(__name__)


class OfflinePreprocessor:
    """
    Executes the offline cleaning pipeline for a single subject recording,
    modelled on ``knowledge_base/02_reference/tomer_preprocessing_new.py`` plus
    the instructor's parameters (see docs/old/preprocessing_migration_plan.md).

    Caller is responsible for loading raw data and passing it via the
    constructor (or assigning ``self.raw`` directly).

    The pipeline is split into four operator-gated steps so the two manual
    selections (bad channels, ICA components) happen on MNE's native
    interactive windows, which must run on the GUI main thread:

    1. ``run_step1a_filter()`` (worker)
       Channel hygiene → high-pass → notch → (if ``resample_filter_stage ==
       "early"``) low-pass + resample on the raw. Returns the ``Raw`` so the
       UI can pop ``raw.plot(block=True)`` for manual bad-channel marking.
    2. ``set_bad_channels(bads)`` (main thread, after the window closes)
       Stores the operator's bad-channel selection.
    3. ``run_step1b_fit_ica(event_mapping)`` (worker)
       Interpolate bads → epoch → average reference → fit ICA (HP-only fit
       copy) → ICLabel pre-suggestion. Returns ``(ica, epochs, suggested)``.
    4. ``run_step2_apply_and_save(exclude, event_mapping, output_dir)`` (worker)
       Apply ICA → (if ``resample_filter_stage == "late"``) low-pass +
       resample on the epochs → save ``{subject}_epo.fif``.
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

        self._bad_channels: list[str] = []
        self._interp_weights: Optional[np.ndarray] = None
        self._suggested_exclude: list[int] = []
        # Per-component (ICLabel category, confidence), aligned by component
        # index. Populated by _iclabel_suggest(); None when ICLabel is
        # disabled. Surfaced to the review UI so the operator sees what
        # ICLabel thought each component was. See component_labels.
        self._component_labels: Optional[list[tuple[str, float]]] = None

        # Positional channel bookkeeping for the online handoff.
        self._original_ch_names: list[str] = []   # pre-hygiene .vhdr order
        self._dropped_channels: list[str] = []    # removed by channel hygiene
        self._post_hygiene_eeg_names: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run_step1a_filter(self) -> mne.io.Raw:
        """
        Channel hygiene → high-pass → notch → (early variant only) low-pass +
        resample on the raw.

        Returns:
            The filtered ``Raw`` for the UI's interactive bad-channel window.

        Raises:
            RuntimeError: if raw data has not been provided.
        """
        if self.raw is None:
            raise RuntimeError(
                "raw must be set before calling run_step1a_filter(). "
                "Pass raw to the constructor or assign self.raw directly."
            )

        self._original_ch_names = list(self.raw.ch_names)
        logger.info("Filtering raw (stage=%s)", self._stage)
        self._channel_hygiene()
        self._highpass()
        self._notch()
        if self._stage == "early":
            self._lowpass(self.raw)
            self.raw = self._resample(self.raw)
        return self.raw

    def set_bad_channels(self, bads: list[str]) -> None:
        """Store the operator's bad-channel selection (read off ``raw.info['bads']``)."""
        self._bad_channels = list(bads)
        logger.info("Operator marked bad channels: %s", self._bad_channels)

    def run_step1b_fit_ica(
        self, event_mapping: dict[str, int]
    ) -> tuple[mne.preprocessing.ICA, mne.Epochs, list[int]]:
        """
        Interpolate bads → epoch → average reference → fit ICA → ICLabel.

        Args:
            event_mapping: {event_name: trigger_id} — MNE convention.

        Returns:
            (ica, epochs_for_review, suggested_exclude) — ``ica`` and
            ``epochs_for_review`` feed MNE's interactive component window;
            ``suggested_exclude`` pre-populates ``ica.exclude``.

        Raises:
            RuntimeError: if run_step1a_filter() has not been called first.
        """
        if self.raw is None or not self._original_ch_names:
            raise RuntimeError("Call run_step1a_filter() before run_step1b_fit_ica().")

        self._interpolate_bads()
        self.epochs = self._epoch(event_mapping)
        logger.info("Epochs created: %d", len(self.epochs))
        self._reference()
        logger.info("Fitting ICA…")
        self._suggested_exclude = self._fit_ica()
        logger.info(
            "ICA fitted (%d components). ICLabel suggested: %s",
            self.ica.n_components_, self._suggested_exclude,
        )
        return self.ica, self.epochs, self._suggested_exclude

    def run_step2_apply_and_save(
        self,
        exclude_components: list[int],
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Apply ICA → (late variant only) low-pass + resample on epochs → save.

        Args:
            exclude_components: Final operator-confirmed ICA component indices.
            output_dir: Directory to write the ``{subject}_epo.fif`` file.

        Returns:
            {"n_epochs": int, "n_excluded": int}

        Raises:
            RuntimeError: if run_step1b_fit_ica() has not been called first.
        """
        if self.ica is None or self.epochs is None:
            raise RuntimeError(
                "Call run_step1b_fit_ica() before run_step2_apply_and_save()."
            )

        self.ica.exclude = list(exclude_components)
        self.ica.apply(self.epochs, verbose=False)

        if self._stage == "late":
            self._lowpass(self.epochs)
            self.epochs = self._resample(self.epochs)

        self._save(Path(output_dir))
        # Raw has done its job (filtered/resampled in place, epoched into
        # self.epochs). Nothing past Step 2 reads it — release the reference
        # so it doesn't masquerade as live state through training/export.
        self.raw = None
        return {"n_epochs": len(self.epochs), "n_excluded": len(self.ica.exclude)}

    def export_online_state(self) -> dict[str, Any]:
        """
        Extract the fitted numerical artifacts needed by the online phase.

        Everything is positional — no channel names cross the LSL boundary.
        Recipe parameters live in ``settings`` and are read there by both
        phases; only fitted *state* is exported here.

        Raises:
            RuntimeError: if called before the pipeline has produced an ICA.
        """
        if self.ica is None:
            raise RuntimeError(
                "Fit ICA before exporting online state."
            )

        n_comp = self.ica.n_components_
        return {
            # Which positions of the pre-hygiene channel array survive hygiene.
            "eeg_chunk_indices": self._compute_eeg_chunk_indices(),
            # Operator-marked bads as positions in the post-hygiene EEG array.
            "bad_indices": self._compute_bad_indices(),
            "interp_weights": self._interp_weights,
            "ica_unmixing": self.ica.unmixing_matrix_.copy(),
            "ica_mixing": self.ica.mixing_matrix_.copy(),
            "ica_pca_components": self.ica.pca_components_[:n_comp].copy(),
            "ica_pca_mean": (
                self.ica.pca_mean_.copy() if self.ica.pca_mean_ is not None else None
            ),
            "ica_exclude": list(self.ica.exclude),
            # Per-channel-type rescaling MNE applies before PCA in ICA.fit/apply.
            # Required for online ICA to match offline numerically.
            "pre_whitener": self.ica.pre_whitener_.copy(),
        }

    # ── Settings helpers ──────────────────────────────────────────────────────

    @property
    def _stage(self) -> str:
        return self.settings.get("resample_filter_stage", "early")

    # ── Private: channel hygiene ──────────────────────────────────────────────

    def _channel_hygiene(self) -> None:
        """EMG drop, HEGOC→HEOG rename, hardware montage with the AFz case fix."""
        ch = self.settings.get("channel_hygiene", {})

        if ch.get("drop_emg", True) and "EMG" in self.raw.ch_names:
            self.raw.set_channel_types({"EMG": "emg"})
            self.raw.drop_channels(["EMG"])
            self._dropped_channels.append("EMG")
            logger.info("Channel hygiene: dropped EMG")

        if ch.get("rename_hegoc_to_heog", True) and "HEGOC" in self.raw.ch_names:
            self.raw.rename_channels({"HEGOC": "HEOG"})
            logger.info("Channel hygiene: renamed HEGOC → HEOG")

        montage_name = ch.get("montage_name", "easycap-M1")
        montage = mne.channels.make_standard_montage(montage_name)
        if ch.get("afz_case_fix", True) and "AFz" in montage.ch_names:
            montage.ch_names[montage.ch_names.index("AFz")] = "Afz"
        self.raw.set_montage(
            montage, match_case=False, on_missing="warn", verbose=False
        )

        self._post_hygiene_eeg_names = [
            self.raw.ch_names[i] for i in mne.pick_types(self.raw.info, eeg=True)
        ]

    # ── Private: filtering / resampling ───────────────────────────────────────

    def _highpass(self) -> None:
        hp = self.settings["highpass"]
        # causal: parity with streaming OnlinePreprocessor (scipy.signal.sosfilt).
        self.raw.filter(
            l_freq=hp["l_freq"], h_freq=None, method=hp.get("method", "iir"),
            phase="forward", verbose=False,
        )
        logger.info("Highpass: l_freq=%s Hz (%s)", hp["l_freq"], hp.get("method", "iir"))

    def _notch(self) -> None:
        freq = self.settings.get("notch", {}).get("freq")
        if freq:
            self.raw.notch_filter(freqs=freq, verbose=False)
            logger.info("Notch: %s Hz", freq)

    def _lowpass(self, inst) -> None:
        # causal: parity with streaming OnlinePreprocessor (scipy.signal.sosfilt).
        inst.filter(
            l_freq=None, h_freq=LOWPASS_H_FREQ, method=LOWPASS_METHOD,
            phase="forward", verbose=False,
        )
        logger.info("Lowpass: h_freq=%s Hz (%s)", LOWPASS_H_FREQ, LOWPASS_METHOD)

    def _resample(self, inst):
        """Causal anti-alias FIR + integer decimation, mirroring online ``_decimate``.

        MNE's built-in ``inst.resample()`` uses a zero-phase polyphase resampler;
        the streaming OnlinePreprocessor cannot. Using the same causal FIR
        recipe here keeps training and inference features aligned.

        Returns the resampled instance (a fresh ``Raw``/``Epochs`` when
        decimation happened, else ``inst`` unchanged). Callers must rebind, e.g.
        ``self.epochs = self._resample(self.epochs)`` — building a new object is
        what keeps the time vector consistent with the decimated data.
        """
        target = float(self.settings["final_resample"]["target_rate"])
        current = float(inst.info["sfreq"])
        if current <= target:
            return inst
        if int(current) % int(target) != 0:
            raise RuntimeError(
                f"causal decimate requires an integer ratio; got "
                f"{current} / {target}"
            )

        decimation = int(current) // int(target)
        logger.info("Resample: %g → %g Hz (decimation %d)", current, target, decimation)
        cutoff_hz = 0.9 * target / 2.0
        n_taps = 10 * decimation + 1
        anti_alias = firwin(n_taps, cutoff_hz, fs=current)

        # Process channel-by-channel for Raw — bulk lfilter() on a multi-hour
        # buffer briefly holds three full-size float64 copies (~5 GB on the FL
        # recording), which OOMs commodity dev boxes. Per-channel keeps the
        # peak at input + output + one channel's filtered buffer.
        data = inst.get_data()
        if data.ndim == 2:
            n_ch, n_samples = data.shape
            n_out = len(range(0, n_samples, decimation))
            decimated = np.empty((n_ch, n_out), dtype=data.dtype)
            for channel_index in range(n_ch):
                filtered_channel = lfilter(anti_alias, 1.0, data[channel_index])
                decimated[channel_index] = filtered_channel[::decimation]
        else:
            filtered = lfilter(anti_alias, 1.0, data, axis=-1)
            decimated = filtered[..., ::decimation]

        new_info = mne.create_info(
            ch_names=inst.ch_names,
            sfreq=target,
            ch_types=inst.get_channel_types(),
            verbose=False,
        )
        new_info["bads"] = list(inst.info["bads"])
        try:
            new_info.set_montage(inst.get_montage(), match_case=False, verbose=False)
        except Exception:
            pass

        if isinstance(inst, mne.io.BaseRaw):
            new_raw = mne.io.RawArray(decimated, new_info, verbose=False)
            new_raw.set_annotations(inst.annotations.copy())
            return new_raw
        if isinstance(inst, mne.BaseEpochs):
            # Return a fresh EpochsArray so its time vector is rebuilt from the
            # decimated data. Copying only _data/info onto `inst` left a stale
            # full-rate `times` (1201 entries on 121-sample data), which
            # corrupted the late-stage pipeline (wrong timepoints, unreadable fif).
            return mne.EpochsArray(
                decimated,
                new_info,
                events=inst.events,
                tmin=inst.tmin,
                event_id=inst.event_id,
                baseline=None,
                verbose=False,
            )
        raise TypeError(
            f"_resample expects Raw or Epochs; got {type(inst).__name__}"
        )

    # ── Private: bad channels ─────────────────────────────────────────────────

    def _interpolate_bads(self) -> None:
        self.raw.info["bads"] = list(self._bad_channels)
        if self._bad_channels:
            logger.info("Interpolating bad channels: %s", self._bad_channels)
            self._interp_weights = self._compute_interp_weights()
            self.raw.interpolate_bads(reset_bads=True, verbose=False)
        else:
            self._interp_weights = None

    def _compute_interp_weights(self) -> Optional[np.ndarray]:
        """Extract spherical-spline interpolation weights for bad channels.

        Returns W of shape (n_good, n_bad) such that, for any data array X
        with shape (n_samples, n_channels):
            X[:, bad_indices] = X[:, good_indices] @ W

        Returns None when no bad channels are present.
        """
        if not self._bad_channels:
            return None

        # exclude=[] so bad channels (just marked in raw.info['bads']) are
        # still picked — we need their geometry to derive interp weights.
        eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=[])
        eeg_ch_names = [self.raw.ch_names[i] for i in eeg_picks]
        n_eeg = len(eeg_ch_names)

        bad_local_indices = [eeg_ch_names.index(ch) for ch in self._bad_channels]
        good_local_indices = [i for i in range(n_eeg) if i not in bad_local_indices]

        # Identity basis: each time point t has channel t = 1, all others = 0.
        # After interpolation, bad_channel[t] tells us the weight of channel t.
        identity_data = np.eye(n_eeg)  # shape (n_eeg, n_eeg) — channels × time

        eeg_info = mne.pick_info(self.raw.info, sel=eeg_picks)
        test_raw = mne.io.RawArray(identity_data, eeg_info, verbose=False)
        test_raw.info["bads"] = list(self._bad_channels)
        test_raw.interpolate_bads(reset_bads=False, verbose=False)

        interp_data = test_raw.get_data()  # (n_eeg, n_eeg)

        # interp_data[bad_k, good_j] = weight of good channel j on bad channel k.
        # Transpose to get W[j, k] = weight of good_j on bad_k → shape (n_good, n_bad).
        weights = interp_data[np.ix_(bad_local_indices, good_local_indices)].T
        return weights

    # ── Private: epoching / reference / ICA ───────────────────────────────────

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

        baseline = ep.get("baseline")
        baseline = tuple(baseline) if baseline is not None else None
        return mne.Epochs(
            self.raw,
            events,
            event_id=valid_event_id,
            tmin=ep["tmin"],
            tmax=ep["tmax"],
            baseline=baseline,
            detrend=0,
            event_repeated="drop",
            preload=True,
            verbose=False,
        )

    def _reference(self) -> None:
        self.epochs.set_eeg_reference("average", projection=False, verbose=False)

    def _fit_ica(self) -> list[int]:
        ica_s = self.settings["ica"]
        # TODO(decision): revisit the ICA fit-copy filter method.
        # We forced method="iir" to avoid MNE's default FIR producing a
        # ~3.3 s kernel on ~1.2 s epochs ("filter_length > signal" warning,
        # subtly distorted fit). IIR is also consistent with the rest of the
        # pipeline (highpass/notch/lowpass all IIR) and with the future
        # causal online side. Trade-off vs FIR:
        #   - IIR Butterworth order 4 zero-phase: works on short epochs,
        #     smoother roll-off, no minimum signal length.
        #   - FIR (default): sharper transition band, needs a longer signal.
        # If we ever move ICA fitting to full-rate raw or longer epochs,
        # FIR may again be preferable. Consider exposing this as
        # settings.preprocessing.ica.fit_method instead of hardcoding.
        fit_epochs = self.epochs.copy().filter(
            l_freq=ica_s["fit_l_freq"], h_freq=None, method="iir", verbose=False
        )

        fit_params = None
        if ica_s["method"] == "infomax":
            fit_params = dict(extended=ica_s.get("extended", True))
        self.ica = mne.preprocessing.ICA(
            n_components=ica_s.get("n_components"),
            method=ica_s["method"],
            fit_params=fit_params,
            random_state=self.settings["random_state"],
            max_iter="auto",
        )
        self.ica.fit(fit_epochs, verbose=False)

        return self._iclabel_suggest(fit_epochs)

    def _iclabel_suggest(self, fit_epochs: mne.Epochs) -> list[int]:
        """Pre-select components whose ICLabel class is in ``ica.iclabel.drop_labels``."""
        ic = self.settings["ica"].get("iclabel", {})
        if not ic.get("enabled", True):
            self._component_labels = None
            return []

        from mne_icalabel import label_components

        drop = set(ic.get("drop_labels", []))
        # TODO(decision): ICLabel was trained on EEG bandpassed [1, 100] Hz
        # and prints a calibration warning here because our pipeline runs at
        # [0.1, 40] Hz (paper-aligned: settings.preprocessing.lowpass.h_freq
        # = 40, final_resample.target_rate = 100). Predictions still come
        # through — confidence near band edges may be lower. Options when
        # we revisit:
        #   (a) accept the warning (current); document it.
        #   (b) pass a separate fit_epochs filtered to [1, min(45, nyquist)]
        #       Hz only into label_components, leaving the actual ICA fit
        #       copy unchanged.
        #   (c) raise our LP / target_rate to widen the band for ICLabel —
        #       but that's a paper-deviation, not just a comfort fix.
        result = label_components(fit_epochs, self.ica, method="iclabel")
        labels = result["labels"]
        proba = result["y_pred_proba"]
        # Keep the full per-component categorisation (category + the model's
        # confidence in that category) so the review UI can append it to each
        # plot_components subplot title — the operator sees "what ICLabel
        # thought this is", not just an implicit greyed-out reject. Aligned by
        # component index. See component_labels and PreprocessingView.
        self._component_labels = [
            (lbl, float(p)) for lbl, p in zip(labels, proba)
        ]
        suggested = [i for i, lbl in enumerate(labels) if lbl in drop]
        return suggested

    @property
    def component_labels(self) -> Optional[list[tuple[str, float]]]:
        """Per-component ``(ICLabel category, confidence)``, aligned by ICA
        component index. ``None`` when ICLabel was disabled. Valid only after
        ``run_step1b_fit_ica()`` (populated during the ICA fit)."""
        return self._component_labels

    def _save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.subject_id}_epo.fif"
        self.epochs.save(out_path, overwrite=True, verbose=False)
        logger.info("Saved → %s", out_path)

    # ── Private: positional handoff ───────────────────────────────────────────

    def _compute_eeg_chunk_indices(self) -> list[int]:
        """Positions of the pre-hygiene channel array that survived hygiene.

        Encodes "drop EMG" (and any other offline channel drops) positionally
        so channel names never have to cross the LSL boundary.
        """
        dropped = set(self._dropped_channels)
        return [
            i for i, name in enumerate(self._original_ch_names)
            if name not in dropped
        ]

    def _compute_bad_indices(self) -> list[int]:
        """Operator-marked bads as positions in the post-hygiene EEG array."""
        names = self._post_hygiene_eeg_names
        return [names.index(ch) for ch in self._bad_channels if ch in names]
