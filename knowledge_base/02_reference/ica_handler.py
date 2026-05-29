import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne_icalabel import label_components
from src.analysis.enums.analysis_enums import ParallelPortDict


class ICAHandler:
    def __init__(self, epochs: mne.Epochs, fig_path: Path, save_path: Path = None):
        self.epochs = epochs
        self.fig_path = fig_path
        self.save_path = save_path

    def run(self, do_auto: bool = False) -> mne.Epochs:
        epochs_for_ica = self.epochs.copy().filter(l_freq=1, h_freq=None)
        ica = self._fit_ica(epochs_for_ica)
        self._save_ica(ica)
        labels, probs  = self._get_labels(ica, epochs_for_ica)
        self._plot_components_labeled(ica, labels, probs, epochs_for_ica)
        if do_auto:
            plt.close()
            self._auto_apply(ica, labels)
        else:
            self._manual_apply(ica, labels, probs, epochs_for_ica)
        return self.epochs

    def _auto_apply(self, ica: mne.preprocessing.ICA, labels: list) -> None:
        exclude_idx = [i for i, lbl in enumerate(labels) if lbl in ParallelPortDict.PREPRO_ARGS['drop ica']]
        ica.apply(self.epochs, exclude=exclude_idx)
        ica.plot_overlay(self.epochs.average(), exclude=exclude_idx)
        plt.savefig(self.fig_path / "ICA_Evoked_Overlay.png"); plt.close()
        logging.info(f"ICA: auto rejected {exclude_idx}")

    def _manual_apply(self, ica: mne.preprocessing.ICA, labels: list,
                      probs: list, epochs_for_ica: mne.Epochs) -> None:
        for i, (lbl, prob) in enumerate(zip(labels, probs)):
            print(f"  IC{i:>2}: {lbl} ({float(np.max(np.atleast_1d(prob))):.0%})")
        ica.exclude = [i for i, lbl in enumerate(labels) if lbl in ParallelPortDict.PREPRO_ARGS['drop ica']]
        plt.show(block=True)
        plt.close()
        ica.plot_sources(epochs_for_ica, block=True)
        exclude_idx = ica.exclude
        ica.apply(self.epochs, exclude=exclude_idx)
        ica.plot_overlay(self.epochs.average(), exclude=exclude_idx)
        plt.savefig(self.fig_path / "ICA_Evoked_Overlay.png"); plt.close()
        logging.info(f"ICA: manually rejected {exclude_idx}")

    @staticmethod
    def _fit_ica(epochs_for_ica: mne.Epochs) -> mne.preprocessing.ICA:
        ica = mne.preprocessing.ICA(n_components=None, method='infomax',
                                     fit_params=dict(extended=True), random_state=100)
        ica.fit(epochs_for_ica)
        return ica

    def _save_ica(self, ica: mne.preprocessing.ICA) -> None:
        if self.save_path is None:
            return
        ica_path = self.save_path / "ica-ica.fif"
        ica.save(ica_path, overwrite=True)
        logging.info(f"ICA saved to {ica_path}")

    @staticmethod
    def _get_labels(ica: mne.preprocessing.ICA,
                    epochs_for_ica: mne.Epochs) -> tuple:
        ic_labels = label_components(epochs_for_ica, ica, method='iclabel')
        labels, probs = ic_labels['labels'], ic_labels['y_pred_proba']
        logging.info(f"ICA: {ica.n_components_} components — top {min(20, ica.n_components_)} labels:")
        for i in range(min(20, ica.n_components_)):
            conf = float(np.max(np.atleast_1d(probs[i])))
            logging.info(f"  IC{i:>2}: {labels[i]} ({conf:.0%})")
        return labels, probs

    def _plot_components_labeled(self, ica, labels, probs, epochs_for_ica: mne.Epochs) -> None:
        n_show = min(20, ica.n_components_)
        n_cols = 5
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()
        components = ica.get_components()
        for idx in range(n_show):
            mne.viz.plot_topomap(components[:, idx], epochs_for_ica.info, axes=axes[idx], show=False)
            lbl   = labels[idx]
            color = 'red' if lbl in ParallelPortDict.PREPRO_ARGS['drop ica'] else 'green'
            conf = float(np.max(np.atleast_1d(probs[idx])))
            axes[idx].set_title(f"IC{idx}: {lbl}\n{conf:.0%}", fontsize=8, color=color)
        for idx in range(n_show, len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle("ICA Components — red=excluded, green=kept", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.fig_path / "ICA_top20_labeled.png", dpi=150)
