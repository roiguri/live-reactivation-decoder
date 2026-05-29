```python
import mne
import numpy as np
from pathlib import Path
from typing import Optional
from autoreject import AutoReject

from src.analysis.enums.analysis_enums import (Paths, FilterParams, ICAParams, EpochParams, TriggerCodes,
                                               RejectCriteria, FileFormat)


class Preprocessing:
    def __init__(self, subject_dir: Path, session: str):
        """input: subject_dir: path to the subject folder (e.g. .../subject 101)
                  session: session name ('experiment' or 'baseline')
           1. find the .vhdr file in subject_dir/session/
           2. load raw data and store subject_id"""
        self.subject_dir = subject_dir
        self.session     = session
        self.subject_id  = subject_dir.name
        self.vhdr        = self._find_vhdr()
        self.raw: mne.io.Raw
        self.epochs: mne.Epochs

    def run(self):
        """run the full preprocessing pipeline and save the result:
            1. load raw
            2. filter
            3. resample
            4. detect and interpolate bad channels
            5. re-reference to average
            6. ICA artefact removal
            7. epoch around triggers
            8. AutoReject
            9. save epochs to .fif"""
        if self.vhdr is None:
            return

        print(f"\n{'='*60}")
        print(f"  Subject: {self.subject_id}  |  Session: {self.session}")
        print(f"  File:    {self.vhdr.name}")
        print(f"{'='*60}")

        self.raw = self._load_raw()
        print(f"  Loaded: {len(self.raw.ch_names)} channels, {self.raw.info['sfreq']} Hz, {self.raw.times[-1]:.1f} s")

        self._filter()
        self._resample()
        self._detect_bad_channels()
        self._reference()
        self._run_ica()

        self.epochs = self._epoch()
        print(f"  Epochs before AutoReject: {len(self.epochs)}")
        self._autoreject()
        print(f"  Epochs after  AutoReject: {len(self.epochs)}")

        self._save()

    def _find_vhdr(self) -> Optional[Path]:
        """return the .vhdr file inside subject_dir/session/, or None if missing"""
        session_dir = self.subject_dir / self.session
        vhdrs = list(session_dir.glob("*.vhdr"))
        if not vhdrs:
            print(f"  [skip] no .vhdr found in {session_dir}")
            return None
        if len(vhdrs) > 1:
            print(f"  [warn] multiple .vhdr files in {session_dir}, using {vhdrs[0].name}")
        return vhdrs[0]

    def _load_raw(self) -> mne.io.Raw:
        """load BrainVision file and apply standard 10-20 montage"""
        raw = mne.io.read_raw_brainvision(self.vhdr, preload=True, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn")
        return raw

    def _filter(self):
        """band-pass + notch filter"""
        self.raw.filter(l_freq=FilterParams.L_FREQ, h_freq=FilterParams.H_FREQ, method="iir", verbose=False)
        self.raw.notch_filter(freqs=FilterParams.NOTCH_FREQ, verbose=False)

    def _resample(self):
        """downsample to RESAMPLE_HZ if current rate is higher"""
        if FilterParams.RESAMPLE_HZ and self.raw.info["sfreq"] > FilterParams.RESAMPLE_HZ:
            self.raw.resample(FilterParams.RESAMPLE_HZ, verbose=False)

    def _detect_bad_channels(self):
        """flag flat and noisy channels and interpolate them"""
        data = self.raw.get_data(picks="eeg")
        stds = data.std(axis=1)

        flat_idx  = np.where(stds < RejectCriteria.FLAT_THRESHOLD)[0]
        z         = (stds - stds.mean()) / stds.std()
        noisy_idx = np.where(z > RejectCriteria.NOISY_Z_SCORE)[0]

        bads = list(set(
            [self.raw.ch_names[i] for i in flat_idx] +
            [self.raw.ch_names[i] for i in noisy_idx]
        ))
        if bads:
            print(f"  [bad channels] {bads}")
        self.raw.info["bads"] = bads
        self.raw.interpolate_bads(reset_bads=True, verbose=False)

    def _reference(self):
        """re-reference to average"""
        self.raw.set_eeg_reference("average", projection=False, verbose=False)

    def _run_ica(self):
        """run ICA and auto-label ocular & cardiac artefacts"""
        raw_for_ica = self.raw.copy().filter(l_freq=ICAParams.FIT_L_FREQ, h_freq=None, verbose=False)

        ica = mne.preprocessing.ICA(
            n_components=ICAParams.N_COMPONENTS,
            method=ICAParams.METHOD,
            random_state=ICAParams.RANDOM_STATE,
            max_iter="auto",
        )
        ica.fit(raw_for_ica, verbose=False)

        eog_idx, _ = ica.find_bads_eog(self.raw, verbose=False)
        ecg_idx: list[int] = []
        if "ECG" in self.raw.get_channel_types():
            ecg_idx, _ = ica.find_bads_ecg(self.raw, verbose=False)

        ica.exclude = list(set(int(i) for i in eog_idx + ecg_idx))
        if ica.exclude:
            print(f"  [ICA] excluding components: {ica.exclude}")

        ica.apply(self.raw, verbose=False)

    def _epoch(self) -> mne.Epochs:
        """extract epochs around stimulus triggers"""
        events, found_event_id = mne.events_from_annotations(self.raw, verbose=False)

        valid_event_id = {k: v for k, v in TriggerCodes.EVENT_ID.items() if v in found_event_id.values()}
        if not valid_event_id:
            print("  [warn] none of the expected trigger codes found — using all events")
            valid_event_id = found_event_id

        return mne.Epochs(
            self.raw,
            events,
            event_id=valid_event_id,
            tmin=EpochParams.TMIN,
            tmax=EpochParams.TMAX,
            baseline=EpochParams.BASELINE,
            reject=dict(eeg=RejectCriteria.HARD_AMPLITUDE),
            preload=True,
            verbose=False,
        )

    def _autoreject(self):
        """run AutoReject to repair / drop remaining bad epochs"""
        ar = AutoReject(random_state=ICAParams.RANDOM_STATE, verbose=False)
        self.epochs, reject_log = ar.fit_transform(self.epochs, return_log=True)
        n_dropped = reject_log.bad_epochs.sum()
        print(f"  [AutoReject] dropped {n_dropped} epochs")

    def _save(self):
        """save cleaned epochs to .fif file"""
        out_dir = Path(Paths.OUTPUT_ROOT) / self.subject_id / self.session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.subject_id}_{self.session}{FileFormat.PREPROCESSED_SUFFIX}"
        self.epochs.save(out_path, overwrite=True, verbose=False)
        print(f"  Saved → {out_path}")


def run_pipeline():
    """run preprocessing for all subjects and sessions"""
    data_root    = Path(Paths.DATA_ROOT)
    subject_dirs = sorted(data_root.glob(f"{FileFormat.SUBJECT_FOLDER_PREFIX}*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders found in {data_root}")

    for subject_dir in subject_dirs:
        for session in Paths.SESSIONS:
            pre_processing = Preprocessing(subject_dir=subject_dir, session=session)
            pre_processing.run()


if __name__ == "__main__":
    run_pipeline()

```