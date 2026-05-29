import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from src.analysis.enums.analysis_enums import MetadataConfig, StageEnum


class MetadataEnricher:
    def __init__(self, path: Path, subject_id: str):
        self.path       = path
        self.subject_id = subject_id

    def enrich(self, epochs: mne.Epochs, raw: mne.io.BaseRaw) -> mne.Epochs:
        """Load behavioral CSVs and enrich epochs.metadata with per-trial behavioral data."""
        fl_df, bl_df, tp_df = self._load_behavioral_csvs()
        trial_map = self._assign_trial_labels(epochs)
        epochs    = self._enrich_metadata(epochs, trial_map, fl_df, bl_df, tp_df)
        self._validate(epochs, raw, fl_df, bl_df, tp_df)
        return epochs

    def _load_behavioral_csvs(self) -> tuple:
        """Load and sort the three behavioral DataFrames for this subject."""
        meta_path = self.path / f"subject {self.subject_id}" / "META DATA"
        fl_df     = self._read_first_csv(meta_path / MetadataConfig.FL_CSV_FOLDER)
        fl_df     = fl_df.sort_values(MetadataConfig.FL_SORT_COL).reset_index(drop=True)
        combined  = self._read_first_csv(meta_path / MetadataConfig.COMBINED_CSV_FOLDER)
        bl_df     = combined.sort_values(MetadataConfig.BL_SORT_COL).reset_index(drop=True)
        return fl_df, bl_df, self._sort_test_df(combined)

    @staticmethod
    def _read_first_csv(folder: Path) -> pd.DataFrame:
        """Return the first CSV found in folder alphabetically, raise if none exists."""
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in {folder}")
        return pd.read_csv(csvs[0])

    @staticmethod
    def _sort_test_df(combined_df: pd.DataFrame) -> pd.DataFrame:
        """Sort combined_df by test_trial number for test-phase sequential matching."""
        df       = combined_df.copy()
        df['_k'] = df[MetadataConfig.TEST_TRIAL_COL].str.extract(r'(\d+)').astype(float)
        return df.sort_values('_k').drop(columns='_k').reset_index(drop=True)

    @staticmethod
    def _assign_trial_labels(epochs: mne.Epochs) -> dict:
        """Map each epoch index to (stage, trial_num) based on anchor trigger sequences."""
        event_ids = epochs.events[:, 2]
        trial_map, fl, bl, tp = {}, -1, -1, -1
        for i, eid in enumerate(event_ids):
            fl, bl, tp = MetadataEnricher._update_counters(eid, fl, bl, tp)
            assignment = MetadataEnricher._get_assignment(eid, fl, bl, tp)
            if assignment:
                trial_map[i] = assignment
        return trial_map

    @staticmethod
    def _update_counters(eid: int, fl: int, bl: int, tp: int) -> tuple:
        """Increment the trial counter of whichever stage's anchor trigger was seen."""
        if eid in MetadataConfig.FL_ANCHORS:   fl += 1
        elif eid == MetadataConfig.BL_ANCHOR:  bl += 1
        elif eid == MetadataConfig.TP_ANCHOR:  tp += 1
        return fl, bl, tp

    @staticmethod
    def _get_assignment(eid: int, fl: int, bl: int, tp: int):
        """Return (stage, trial_num) for this trigger, or None if unrecognized."""
        if (eid in MetadataConfig.FL_ANCHORS or eid in MetadataConfig.FL_WITHIN) and fl >= 0:
            return StageEnum.FL, fl
        if (eid == MetadataConfig.BL_ANCHOR or eid in MetadataConfig.BL_WITHIN) and bl >= 0:
            return StageEnum.BL, bl
        if (eid == MetadataConfig.TP_ANCHOR or eid in MetadataConfig.TP_WITHIN) and tp >= 0:
            return StageEnum.TP, tp
        return None

    @staticmethod
    def _enrich_metadata(epochs: mne.Epochs, trial_map: dict,
                         fl_df: pd.DataFrame, bl_df: pd.DataFrame, tp_df: pd.DataFrame) -> mne.Epochs:
        """Add stage, trial_num, and prefixed behavioral columns to epochs.metadata."""
        meta = epochs.metadata.copy()
        stage_arr, trial_arr = MetadataEnricher._build_label_arrays(len(meta), trial_map)
        meta[MetadataConfig.STAGE_COL]     = stage_arr
        meta[MetadataConfig.TRIAL_NUM_COL] = trial_arr
        meta = MetadataEnricher._fill_stage_cols(meta, stage_arr, trial_arr, StageEnum.FL, fl_df, MetadataConfig.FL_COLS, MetadataConfig.FL_PREFIX)
        meta = MetadataEnricher._fill_stage_cols(meta, stage_arr, trial_arr, StageEnum.BL, bl_df, MetadataConfig.BL_COLS, MetadataConfig.BL_PREFIX)
        meta = MetadataEnricher._fill_stage_cols(meta, stage_arr, trial_arr, StageEnum.TP, tp_df, MetadataConfig.TP_COLS, MetadataConfig.TP_PREFIX)
        epochs.metadata = meta
        return epochs

    @staticmethod
    def _build_label_arrays(n: int, trial_map: dict) -> tuple:
        """Build stage and trial_num numpy arrays from trial_map."""
        stage_arr = np.full(n, None, dtype=object)
        trial_arr = np.full(n, -1, dtype=np.int32)
        for i, (stage, t) in trial_map.items():
            stage_arr[i] = stage
            trial_arr[i] = t
        return stage_arr, trial_arr

    @staticmethod
    def _fill_stage_cols(meta: pd.DataFrame, stage_arr: np.ndarray, trial_arr: np.ndarray,
                         stage: str, df: pd.DataFrame, cols: list, prefix: str) -> pd.DataFrame:
        """Copy prefixed CSV columns into meta for all epochs belonging to the given stage."""
        mask  = stage_arr == stage
        idxs  = trial_arr[mask]
        valid = (idxs >= 0) & (idxs < len(df))
        pos   = np.where(mask)[0][valid]
        rows  = idxs[valid]
        for col in cols:
            series = pd.Series([None] * len(meta), dtype=object)
            series.iloc[pos] = df[col].iloc[rows].values
            meta[f'{prefix}{col}'] = series
        return meta

    def _validate(self, epochs: mne.Epochs, raw: mne.io.BaseRaw,
                  fl_df: pd.DataFrame, bl_df: pd.DataFrame, tp_df: pd.DataFrame) -> None:
        """Run four independent checks to confirm EEG–CSV matching; log and print results."""
        logging.info("--- Metadata validation ---"); print("--- Metadata validation ---")
        self._check_counts(epochs.events[:, 2], fl_df, bl_df, tp_df)
        self._check_feature_codes(epochs)
        self._check_timestamps(epochs, raw)
        self._check_question_order(epochs)
        logging.info("--- End metadata validation ---"); print("--- End metadata validation ---")

    @staticmethod
    def _check_counts(eids: np.ndarray, fl_df: pd.DataFrame,
                      bl_df: pd.DataFrame, tp_df: pd.DataFrame) -> None:
        """Check that anchor trigger counts match CSV row counts for each stage."""
        checks = [
            (int(np.isin(eids, list(MetadataConfig.FL_ANCHORS)).sum()), len(fl_df), "Functional Localizer"),
            (int((eids == MetadataConfig.BL_ANCHOR).sum()),             len(bl_df), "Binding Learning    "),
            (int((eids == MetadataConfig.TP_ANCHOR).sum()),             len(tp_df), "Test Phase          "),
        ]
        for n_eeg, n_csv, label in checks:
            ok  = n_eeg == n_csv
            msg = f"  [{'OK' if ok else 'WARN'}] Count {label}: EEG={n_eeg}, CSV={n_csv}"
            (logging.info if ok else logging.warning)(msg); print(msg)

    @staticmethod
    def _check_feature_codes(epochs: mne.Epochs) -> None:
        """Check each FL show-trigger code matches the fl_feature value in its metadata row."""
        meta       = epochs.metadata
        eids       = epochs.events[:, 2]
        fl_mask    = np.isin(eids, list(MetadataConfig.FL_ANCHORS))
        mismatches = sum(
            MetadataConfig.TRIGGER_TO_FEATURE[eid] != row.get(MetadataConfig.FL_FEATURE_COL)
            for eid, (_, row) in zip(eids[fl_mask], meta[fl_mask].iterrows())
        )
        ok  = mismatches == 0
        msg = f"  [{'OK' if ok else 'WARN'}] Feature-code check: {fl_mask.sum()} epochs, {mismatches} mismatch(es)"
        (logging.info if ok else logging.warning)(msg); print(msg)

    @staticmethod
    def _check_timestamps(epochs: mne.Epochs, raw: mne.io.BaseRaw) -> None:
        """Check EEG trigger times are within 1 s of CSV feature_appear (skip if no meas_date)."""
        meas_date = raw.info.get('meas_date')
        if meas_date is None:
            msg = "  [SKIP] Timestamp check: no meas_date"; logging.info(msg); print(msg); return
        if meas_date.tzinfo is None:
            meas_date = meas_date.replace(tzinfo=datetime.timezone.utc)
        deltas = MetadataEnricher._compute_fl_time_deltas(epochs, raw.info['sfreq'], meas_date)
        max_d  = max(deltas) if deltas else 0.0
        ok     = max_d < MetadataConfig.TIMESTAMP_MAX_S
        msg    = f"  [{'OK' if ok else 'WARN'}] Timestamp check: max offset {max_d:.3f}s over {len(deltas)} epochs"
        (logging.info if ok else logging.warning)(msg); print(msg)

    @staticmethod
    def _compute_fl_time_deltas(epochs: mne.Epochs, sfreq: float,
                                 meas_date: datetime.datetime) -> list:
        """Return absolute time deltas (s) between EEG sample and CSV feature_appear per FL epoch."""
        meta    = epochs.metadata
        eids    = epochs.events[:, 2]
        fl_mask = np.isin(eids, list(MetadataConfig.FL_ANCHORS))
        deltas  = []
        for event, (_, row) in zip(epochs.events[fl_mask], meta[fl_mask].iterrows()):
            ts = row.get(MetadataConfig.FL_FEATURE_APPEAR_COL)
            if not ts:
                continue
            try:
                csv_dt = datetime.datetime.strptime(str(ts), MetadataConfig.TIMESTAMP_FMT).replace(tzinfo=datetime.timezone.utc)
                eeg_dt = meas_date + datetime.timedelta(seconds=float(event[0]) / sfreq)
                deltas.append(abs((eeg_dt - csv_dt).total_seconds()))
            except ValueError:
                continue
        return deltas

    @staticmethod
    def _check_question_order(epochs: mne.Epochs) -> None:
        """Check that the first of triggers 61/66 per test trial matches tp_first_question."""
        meta    = epochs.metadata
        eids    = epochs.events[:, 2]
        tp_mask = np.isin(eids, [MetadataConfig.TP_ANCHOR, MetadataConfig.SHOW_COLORS, MetadataConfig.SHOW_SCENES])
        first_q = meta[eids == MetadataConfig.TP_ANCHOR][MetadataConfig.TP_FIRST_Q_COL].reset_index(drop=True)
        n_mis   = MetadataEnricher._count_order_mismatches(eids[tp_mask], epochs.events[tp_mask, 0], first_q)
        ok      = n_mis == 0
        msg     = f"  [{'OK' if ok else 'WARN'}] Question-order check: {n_mis} mismatch(es)"
        (logging.info if ok else logging.warning)(msg); print(msg)

    @staticmethod
    def _count_order_mismatches(tp_eids: np.ndarray, tp_times: np.ndarray,
                                 tp_first_q: pd.Series) -> int:
        """Count test trials where EEG trigger order disagrees with tp_first_question."""
        tp_eids    = np.append(tp_eids, MetadataConfig.TP_ANCHOR)
        mismatches = trial_idx = 0
        col_t = scene_t = None
        for j, eid in enumerate(tp_eids):
            if eid == MetadataConfig.TP_ANCHOR:
                if trial_idx < len(tp_first_q) and MetadataEnricher._is_order_mismatch(col_t, scene_t, tp_first_q.iloc[trial_idx]):
                    mismatches += 1
                trial_idx += 1; col_t = scene_t = None
            elif eid == MetadataConfig.SHOW_COLORS: col_t = tp_times[j]
            elif eid == MetadataConfig.SHOW_SCENES: scene_t = tp_times[j]
        return mismatches

    @staticmethod
    def _is_order_mismatch(col_t, scene_t, csv_first) -> bool:
        """Return True if EEG question order disagrees with the CSV first_question value."""
        if col_t is None or scene_t is None or not pd.notna(csv_first):
            return False
        eeg_first = MetadataConfig.COLORS_STR if col_t < scene_t else MetadataConfig.SCENES_STR
        return eeg_first != csv_first
