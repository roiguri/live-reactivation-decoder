import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from autoreject import AutoReject
from src.analysis.enums.analysis_enums import ParallelPortDict
from src.analysis.pre_processing.metadata_enrichment import MetadataEnricher
from src.analysis.pre_processing.ica_handler import ICAHandler


class OrPipeline:
    def __init__(self, path: Path, subject_id: str):
        self.subject_id   = subject_id
        self.path         = path
        self.eeg_path     = self.path / f"subject {self.subject_id}" / "EEG" / "experiment"
        self.results_path = self.path / f"subject {self.subject_id}" / "post process"
        self.fig_path     = self.path / f"subject {self.subject_id}" / "figures"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)

    def run(self, do_ica: bool = True, do_auto_reject: bool = False):
        self._log_path   = self.results_path / f"{self.subject_id}_preprocessing.log"
        file_handler     = logging.FileHandler(self._log_path)
        file_handler.setLevel(logging.INFO)
        self._log_handler = file_handler
        logging.getLogger().addHandler(file_handler)

        raw = self._load_vhdr()
        self._resample_and_filtering(raw)
        self._handle_bad_channels(raw)
        self._log_event_counts(raw)
        self._save_raw(raw)
        epochs = self._make_epochs(raw)
        epochs = MetadataEnricher(self.path, self.subject_id).enrich(epochs, raw)
        del raw
        self._save_epochs(epochs, suffix='initial')
        epochs = self._rereference(epochs)
        if do_auto_reject:
            epochs = self._auto_reject(epochs)
        self._save_epochs(epochs, suffix='after_autoreject')
        if do_ica:
            epochs = ICAHandler(epochs, self.fig_path, self.results_path).run()
        self._final_resample_and_filtering(epochs)
        self._save_epochs(epochs)

        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    def _load_vhdr(self, EOG_ch: bool = False) -> mne.io.BaseRaw:
        """Load BrainVision file, fix channel names/types, and set easycap-M1 montage."""
        vhdrs = list(self.eeg_path.glob("*.vhdr"))
        if not vhdrs:
            raise FileNotFoundError(f"No .vhdr file found in {self.eeg_path}")
        raw = mne.io.read_raw_brainvision(vhdrs[0], verbose=False)
        raw.load_data()
        raw.set_channel_types({"EMG": 'emg'})
        if 'HEGOC' in raw.ch_names:
            raw.rename_channels({'HEGOC': 'HEOG'})
        if EOG_ch:
            raw.set_channel_types({"HEOG": 'eog', "VEOG": 'eog'})
        montage = mne.channels.make_standard_montage('easycap-M1')
        montage.ch_names[montage.ch_names.index("AFz")] = "Afz"
        raw.set_montage(montage, verbose=False)
        return raw

    def _resample_and_filtering(self, raw: mne.io.BaseRaw) -> None:
        self._move_out_emg_electrode(raw)
        self._high_pass_filter(raw)
        self._notch_filter(raw)
        raw.resample(ParallelPortDict.PREPRO_ARGS['resample'])
        logging.info(f"Resampled to {ParallelPortDict.PREPRO_ARGS['resample']} Hz")

    @staticmethod
    def _handle_bad_channels(raw: mne.io.BaseRaw) -> None:
        raw.plot(block=True)
        bad_channels = raw.info['bads']
        raw.interpolate_bads(reset_bads=True)
        logging.info(f"Manual: interpolated bad channels {bad_channels}")
        print(f"Interpolated bad channels: {bad_channels}")

    @staticmethod
    def _log_event_counts(raw: mne.io.BaseRaw) -> None:
        id_to_name  = {v: k for k, v in ParallelPortDict.EVENT_DICT.items()}
        events, _   = mne.events_from_annotations(raw, verbose=False)
        unique, counts = np.unique(events[:, 2], return_counts=True)
        logging.info("--- Trigger counts ---"); print("--- Trigger counts ---")
        for event_id, count in zip(unique, counts):
            msg = f"  [{event_id:>3}] {id_to_name.get(event_id, f'unknown ({event_id})')}: {count}"
            logging.info(msg); print(msg)
        logging.info("----------------------"); print("----------------------")

    @staticmethod
    def _make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
        events_from_annot, _ = mne.events_from_annotations(raw)
        selected_events = np.array([x for x in events_from_annot if x[2] in ParallelPortDict.EVENT_DICT.values()])
        logging.info(f"Found {len(selected_events)} events for epoching (tmin={ParallelPortDict.PREPRO_ARGS['tmin']}, tmax={ParallelPortDict.PREPRO_ARGS['tmax']})")
        metadata, _, _ = mne.epochs.make_metadata(selected_events, event_id=ParallelPortDict.EVENT_DICT,
                                                   tmin=0, tmax=0, sfreq=raw.info['sfreq'])
        return mne.Epochs(raw, events=selected_events, event_id=ParallelPortDict.EVENT_DICT,
                          tmin=ParallelPortDict.PREPRO_ARGS['tmin'], tmax=ParallelPortDict.PREPRO_ARGS['tmax'],
                          baseline=ParallelPortDict.PREPRO_ARGS['baseline'], preload=True,
                          detrend=0, metadata=metadata, on_missing='warn')

    @staticmethod
    def _rereference(epochs: mne.Epochs) -> mne.Epochs:
        logging.info("Applying average reference")
        return mne.set_eeg_reference(epochs)[0]

    def _auto_reject(self, epochs: mne.Epochs) -> mne.Epochs:
        ar = AutoReject(n_jobs=1, random_state=11, n_interpolate=[1, 2, 3, 4])
        ar.fit(epochs[:100])
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        fig = epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        try:
            fig.grab().save(str(self.fig_path / "Autoreject_Bad_Epochs.png"))
        except Exception as e:
            print(f"Skipping grab() save due to: {e}")
        reject_log.plot('horizontal')
        plt.savefig(self.fig_path / "Autoreject_Reject_LOG"); plt.close()
        bad_idx    = np.where(reject_log.bad_epochs)[0].tolist()
        id_to_name = {v: k for k, v in ParallelPortDict.EVENT_DICT.items()}
        bad_events = [id_to_name.get(epochs.events[i, 2], f"unknown({epochs.events[i, 2]})") for i in bad_idx]
        logging.info(f"Autoreject: removed {len(bad_idx)} epochs — indices: {bad_idx}")
        logging.info(f"Autoreject: removed event types: {bad_events}")
        self._plot_autoreject_by_condition(bad_events)
        return epochs_ar

    def _final_resample_and_filtering(self, epochs: mne.Epochs) -> None:
        self._low_pass_filter(epochs)
        self._final_resample(epochs)

    def _snapshot_log(self, saved_path: Path) -> None:
        self._log_handler.flush()
        shutil.copy2(self._log_path, saved_path.with_suffix('.log'))

    def _save_raw(self, raw: mne.io.BaseRaw) -> None:
        raw_fname = f'{self.subject_id}_filtered-raw.fif'
        raw_path  = self.results_path / raw_fname
        raw.save(raw_path, overwrite=True)
        logging.info(f"Raw saved to {raw_path}")
        self._snapshot_log(raw_path)

    def _save_epochs(self, epochs: mne.Epochs, suffix: str = 'prepro') -> None:
        ep_fname  = f'{self.subject_id}_{suffix}-epo.fif'
        ep_path   = self.results_path / ep_fname
        epochs.save(ep_path, overwrite=True)
        logging.info(f"Epochs saved to {ep_path}")
        self._snapshot_log(ep_path)

    def _plot_autoreject_by_condition(self, bad_events: list) -> None:
        unique, counts = np.unique(bad_events, return_counts=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(unique, counts)
        ax.set(xlabel='Event type', ylabel='Rejected epochs', title='Autoreject — rejected epochs per condition')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.fig_path / "Autoreject_by_condition.png"); plt.close()

    def plot_signal_snapshot(self, data, step_name: str) -> None:
        """Save a PSD + time-domain snapshot figure for a given pipeline step."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        data.compute_psd().plot(axes=axes[0], show=False)
        axes[0].set_title(f'PSD — {step_name}')
        if isinstance(data, mne.io.BaseRaw):
            segment = data.get_data(picks='eeg', start=0, stop=int(data.info['sfreq'] * 10))
            times   = np.linspace(0, 10, segment.shape[1])
        else:
            segment = data.get_data(picks='eeg').mean(axis=0)
            times   = data.times
        axes[1].plot(times, segment.T, color='steelblue', alpha=0.3, linewidth=0.5)
        axes[1].set(xlabel='Time (s)', ylabel='Amplitude (V)', title=f'Signal — {step_name}')
        plt.tight_layout()
        plt.savefig(self.fig_path / f"snapshot_{step_name.replace(' ', '_')}.png"); plt.close()

    @staticmethod
    def _high_pass_filter(raw: mne.io.BaseRaw, l_freq: float = 0.1) -> None:
        raw.filter(l_freq=l_freq, h_freq=None)
        logging.info(f"High-pass filter applied: {l_freq} Hz")

    @staticmethod
    def _notch_filter(raw: mne.io.BaseRaw, freqs: float = 50.0) -> None:
        raw.notch_filter(freqs=freqs)
        logging.info(f"Notch filter applied: {freqs} Hz")

    @staticmethod
    def _low_pass_filter(epochs: mne.Epochs, h_freq: float = 100.0) -> None:
        epochs.filter(l_freq=None, h_freq=h_freq)
        logging.info(f"Low-pass filter applied: {h_freq} Hz")

    @staticmethod
    def _final_resample(epochs: mne.Epochs, sfreq: float = 500.0) -> None:
        epochs.resample(sfreq)
        logging.info(f"Final resample to {sfreq} Hz")

    @staticmethod
    def _move_out_emg_electrode(raw: mne.io.BaseRaw) -> None:
        if 'EMG' in raw.ch_names:
            raw.drop_channels(['EMG']); logging.info("Dropped channel: EMG")
        else:
            logging.info("No EMG channel found, skipping")


if __name__ == '__main__':
    data_path = Path("/Users/tomermuller/Desktop/data/")
    pipe = OrPipeline(path=data_path, subject_id="102")
    pipe.run()
