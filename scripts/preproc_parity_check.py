"""Compare offline-trained preprocessing against online inference preprocessing.

Approach B of the online-prediction investigation. Loads the FL VHDR file,
runs it through both pipelines, epochs both outputs at the same stimulus
markers, and reports per-trial divergence metrics plus a stage-bisection
sweep that isolates which preprocessing stage (filter, ICA, bad-channel
interpolation, resample_filter_stage) drives any divergence.

See ``docs/`` (debug plan) for the full investigation context.

This script is *read-only* with respect to ``src/backend/`` — it patches
attributes on freshly-constructed ``OnlinePreprocessor`` instances during the
stage-bisection sweep but never edits production code.

The module is structured so the helpers can be imported and reused from a
notebook ([tests/notebooks/validate_preproc_parity.ipynb]) for visual
diagnosis. The CLI entry point produces a machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import correlate

try:
    # Auto-picks the notebook-style bar in Jupyter, falls back to a terminal
    # bar elsewhere. Keep the import optional so the helper module still
    # loads in slim test envs.
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised only without tqdm
    def _tqdm(iterable, *_args, **_kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.preprocessor import OfflinePreprocessor
from backend.online_phase.artifact_loader import (
    DecoderPipelineArtifact,
    load_decoder_pipeline_artifact,
)
from backend.online_phase.live_inference import LiveInferenceEngine
from backend.online_phase.online_preprocessor import OnlinePreprocessor

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("preproc_parity")


# Names that channel-hygiene rewrites in the offline path so the offline
# epochs' channel names differ from the raw .vhdr names. Used by the
# channel-alignment helper to translate before comparison.
HYGIENE_RENAMES = {"HEGOC": "HEOG"}


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TrialMetrics:
    max_abs_diff: float
    mean_channel_corr: float
    aligned_corr: float
    best_lag_ms: float


@dataclass
class BranchSummary:
    label: str
    n_trials: int
    median_max_abs_diff: float
    median_mean_channel_corr: float
    median_aligned_corr: float
    median_best_lag_ms: float
    trials: list[TrialMetrics] = field(default_factory=list)


@dataclass
class ChannelAlignmentRow:
    position: int
    offline_name: str | None
    derived_name: str | None
    match: bool


@dataclass
class ChannelAlignment:
    all_match: bool
    n_positions: int
    n_match: int
    rows: list[ChannelAlignmentRow] = field(default_factory=list)


@dataclass
class ProbabilityTable:
    """Median P(positive) per decoder × marker for a single branch.

    ``cells[task_name][marker_name]`` is the median P(positive) over trials
    whose marker matched ``marker_name``. NaN if no trials matched.
    """

    branch_label: str
    task_names: list[str]
    marker_names: list[str]
    cells: dict[str, dict[str, float]]


@dataclass
class ParityReport:
    offline_branch: dict  # epoch shape, sfreq, n_trials
    channel_alignment: ChannelAlignment | None
    baseline: BranchSummary
    stage_bisection: list[BranchSummary]
    probability_tables: list[ProbabilityTable] = field(default_factory=list)
    alt_stage: dict | None = None
    flags: list[str] = field(default_factory=list)


# ── Pipeline helpers ─────────────────────────────────────────────────────────


def _ensure_iclabel_disabled(settings: dict) -> dict:
    """Disable ICLabel because ``mne_icalabel`` is not a runtime dependency."""
    safe = copy.deepcopy(settings)
    ica = safe.setdefault("ica", {})
    iclabel = ica.setdefault("iclabel", {})
    iclabel["enabled"] = False
    return safe


def _load_raw_cropped(vhdr_path: Path, max_seconds: float | None) -> mne.io.BaseRaw:
    """Load a VHDR with the smallest memory footprint achievable.

    When ``max_seconds`` is set, read without ``preload`` first so the full
    multi-hour buffer never lives in RAM. Then crop the on-disk view and
    only then materialise the cropped slice. This avoids the 1.6 GB peak
    of ``read_raw_brainvision(preload=True)`` on a 3267 s recording.
    """
    if max_seconds is None:
        return mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose="ERROR")
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose="ERROR")
    raw.crop(tmin=0.0, tmax=float(max_seconds))
    raw.load_data(verbose="ERROR")
    return raw


def _override_stage(settings: dict, stage: str) -> dict:
    """Return a deep-copy of ``settings`` with ``resample_filter_stage`` flipped."""
    if stage not in {"early", "late"}:
        raise ValueError(f"resample_filter_stage must be 'early' or 'late', got {stage!r}")
    out = copy.deepcopy(settings)
    out["resample_filter_stage"] = stage
    return out


def _bad_channel_names_from_artifact(
    raw: mne.io.BaseRaw,
    artifact: DecoderPipelineArtifact,
) -> list[str]:
    """Map artifact ``bad_indices`` → channel names against ``raw.ch_names``.

    ``bad_indices`` are positions in the post-hygiene EEG array;
    ``eeg_chunk_indices`` are positions in the pre-hygiene full-channel array
    that survive hygiene. Composition gives the original channel name.
    """
    state = artifact.online_state
    eeg_chunk_indices = list(state["eeg_chunk_indices"])
    bad_indices = list(state["bad_indices"])
    pre_hygiene_names = list(raw.ch_names)
    names: list[str] = []
    for index in bad_indices:
        original_position = eeg_chunk_indices[int(index)]
        names.append(pre_hygiene_names[int(original_position)])
    return names


def derive_online_channel_names(
    raw: mne.io.BaseRaw,
    artifact: DecoderPipelineArtifact,
    renames: dict[str, str] | None = None,
) -> list[str]:
    """Return the channel-name list the online preprocessor implicitly operates on.

    Indexes ``raw.ch_names`` (pre-hygiene order, as the LSL stream would
    arrive) with ``online_state["eeg_chunk_indices"]``, then applies the
    offline hygiene renames so the result is comparable to
    ``offline_epochs.ch_names`` (post-hygiene order).
    """
    if renames is None:
        renames = HYGIENE_RENAMES
    eeg_chunk_indices = list(artifact.online_state["eeg_chunk_indices"])
    pre_hygiene_names = list(raw.ch_names)
    online_names = [pre_hygiene_names[int(index)] for index in eeg_chunk_indices]
    return [renames.get(name, name) for name in online_names]


def check_channel_alignment(
    offline_ch_names: list[str],
    derived_online_names: list[str],
) -> ChannelAlignment:
    """Position-by-position diff of two channel-name lists.

    Returns a structured report. ``all_match`` is True iff both lists are the
    same length and every position holds the same name. Use this before
    trusting any per-channel metric — channel-name permutation silently
    produces apples-to-oranges comparisons.
    """
    n_positions = max(len(offline_ch_names), len(derived_online_names))
    rows: list[ChannelAlignmentRow] = []
    n_match = 0
    for position in range(n_positions):
        offline_name = offline_ch_names[position] if position < len(offline_ch_names) else None
        derived_name = (
            derived_online_names[position] if position < len(derived_online_names) else None
        )
        match = offline_name is not None and offline_name == derived_name
        if match:
            n_match += 1
        rows.append(
            ChannelAlignmentRow(
                position=position,
                offline_name=offline_name,
                derived_name=derived_name,
                match=match,
            )
        )
    all_match = n_match == n_positions and len(offline_ch_names) == len(derived_online_names)
    return ChannelAlignment(
        all_match=all_match,
        n_positions=n_positions,
        n_match=n_match,
        rows=rows,
    )


def run_offline_branch(
    raw: mne.io.BaseRaw,
    settings: SettingsManager,
    artifact: DecoderPipelineArtifact,
    output_dir: Path,
    preprocessing_settings_override: dict | None = None,
) -> mne.BaseEpochs:
    """Reproduce the training-time offline preprocessor end-to-end.

    ``preprocessing_settings_override`` lets the alt-stage caller flip
    ``resample_filter_stage`` (or any other key) without mutating the
    SettingsManager-owned dict.
    """
    base = (
        preprocessing_settings_override
        if preprocessing_settings_override is not None
        else settings.get_preprocessing_params()
    )
    preprocessor = OfflinePreprocessor(
        data_dir=output_dir,
        preprocessing_settings=_ensure_iclabel_disabled(base),
        raw=raw,
    )
    print("[offline 1/3] filtering (HP + notch + LP/resample if early-stage)...", flush=True)
    preprocessor.run_step1a_filter()
    bads = _bad_channel_names_from_artifact(raw, artifact)
    preprocessor.set_bad_channels(bads)
    print(f"[offline 2/3] fitting ICA ({len(bads)} bad channels seeded from artifact)...", flush=True)
    preprocessor.run_step1b_fit_ica(settings.get_event_mapping())
    print("[offline 3/3] applying ICA + epoching...", flush=True)
    preprocessor.run_step2_apply_and_save(
        exclude_components=list(artifact.online_state["ica_exclude"]),
        output_dir=output_dir / "epochs",
    )
    if preprocessor.epochs is None:
        raise RuntimeError("Offline preprocessor produced no epochs.")
    print(f"[offline done] {len(preprocessor.epochs)} epochs", flush=True)
    return preprocessor.epochs


def run_online_branch(
    raw: mne.io.BaseRaw,
    artifact: DecoderPipelineArtifact,
    preprocessing_settings: dict,
    chunk_samples: int,
    patches: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the raw through the online preprocessor in chunks.

    Returns ``(trace, out_timestamps)`` with shapes ``(n_out_samples,
    n_channels)`` and ``(n_out_samples,)``. ``patches`` lets the caller
    override attributes on the fresh instance for stage bisection.
    """
    online = OnlinePreprocessor(
        preprocessing_settings=preprocessing_settings,
        online_state=artifact.online_state,
        input_sfreq=raw.info["sfreq"],
    )
    if patches:
        for name, value in patches.items():
            setattr(online, name, value)
        online.reset_state()

    data = raw.get_data().T  # (n_samples, n_raw_channels)
    timestamps = np.arange(data.shape[0]) / raw.info["sfreq"]

    out_data_chunks: list[np.ndarray] = []
    out_ts_chunks: list[np.ndarray] = []
    n_chunks = (data.shape[0] + chunk_samples - 1) // chunk_samples
    iterator = _tqdm(
        range(0, data.shape[0], chunk_samples),
        total=n_chunks,
        desc="online streaming",
        leave=False,
        unit="batch",
    )
    for start in iterator:
        chunk = data[start : start + chunk_samples]
        chunk_ts = timestamps[start : start + chunk_samples]
        out_chunk, out_chunk_ts = online.process_batch(chunk, chunk_ts)
        if out_chunk.shape[0] > 0:
            out_data_chunks.append(out_chunk)
            out_ts_chunks.append(out_chunk_ts)

    if not out_data_chunks:
        raise RuntimeError("Online preprocessor produced no output samples.")
    return np.concatenate(out_data_chunks, axis=0), np.concatenate(out_ts_chunks, axis=0)


# ── Epoching + metrics ──────────────────────────────────────────────────────


def epoch_trace_at_event_times(
    trace: np.ndarray,
    out_timestamps: np.ndarray,
    event_times_s: np.ndarray,
    tmin: float,
    tmax: float,
    target_sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Epoch a continuous trace at given event times.

    Returns ``(epochs, kept_event_mask)`` where ``epochs`` has shape
    ``(n_kept_trials, n_channels, n_samples)``.
    """
    n_samples = int(round((tmax - tmin) * target_sfreq)) + 1
    start_offset = int(round(tmin * target_sfreq))
    end_offset_exclusive = start_offset + n_samples
    epochs: list[np.ndarray] = []
    kept_mask = np.zeros(event_times_s.size, dtype=bool)
    for event_index, event_time in enumerate(event_times_s):
        sample_in_trace = int(np.argmin(np.abs(out_timestamps - event_time)))
        window_start = sample_in_trace + start_offset
        window_end = sample_in_trace + end_offset_exclusive
        if window_start < 0 or window_end > trace.shape[0]:
            continue
        epoch = trace[window_start:window_end].T  # → (n_channels, n_samples)
        epochs.append(epoch)
        kept_mask[event_index] = True
    if not epochs:
        return np.empty((0, trace.shape[1], n_samples)), kept_mask
    return np.stack(epochs, axis=0), kept_mask


def trial_metrics(
    offline_epoch: np.ndarray,
    online_epoch: np.ndarray,
    sfreq: float,
    max_lag_ms: float = 50.0,
) -> TrialMetrics:
    """Per-trial parity metrics between offline and online epochs.

    Shapes: ``(n_channels, n_samples)`` each.
    """
    diff = offline_epoch - online_epoch
    max_abs_diff = float(np.max(np.abs(diff)))

    n_channels = offline_epoch.shape[0]
    corrs = np.zeros(n_channels)
    for channel_index in range(n_channels):
        offline_signal = offline_epoch[channel_index]
        online_signal = online_epoch[channel_index]
        if np.std(offline_signal) < 1e-18 or np.std(online_signal) < 1e-18:
            corrs[channel_index] = np.nan
            continue
        corrs[channel_index] = float(np.corrcoef(offline_signal, online_signal)[0, 1])
    mean_channel_corr = float(np.nanmean(corrs))

    max_lag_samples = max(1, int(round(max_lag_ms * 1e-3 * sfreq)))
    aligned_corrs = np.zeros(n_channels)
    best_lags = np.zeros(n_channels)
    for channel_index in range(n_channels):
        offline_signal = offline_epoch[channel_index]
        online_signal = online_epoch[channel_index]
        if np.std(offline_signal) < 1e-18 or np.std(online_signal) < 1e-18:
            aligned_corrs[channel_index] = np.nan
            best_lags[channel_index] = np.nan
            continue
        cross = correlate(
            offline_signal - offline_signal.mean(),
            online_signal - online_signal.mean(),
            mode="full",
        )
        zero_lag_index = offline_signal.size - 1
        window_start = zero_lag_index - max_lag_samples
        window_end_inclusive = zero_lag_index + max_lag_samples
        window = cross[window_start : window_end_inclusive + 1]
        best_in_window = int(np.argmax(window))
        best_lag_samples = best_in_window - max_lag_samples
        rolled_online = np.roll(online_signal, best_lag_samples)
        if best_lag_samples > 0:
            rolled_online[:best_lag_samples] = online_signal[0]
        elif best_lag_samples < 0:
            rolled_online[best_lag_samples:] = online_signal[-1]
        aligned_corrs[channel_index] = float(np.corrcoef(offline_signal, rolled_online)[0, 1])
        best_lags[channel_index] = best_lag_samples * 1000.0 / sfreq

    return TrialMetrics(
        max_abs_diff=max_abs_diff,
        mean_channel_corr=mean_channel_corr,
        aligned_corr=float(np.nanmean(aligned_corrs)),
        best_lag_ms=float(np.nanmean(best_lags)),
    )


def summarise(label: str, trials: list[TrialMetrics]) -> BranchSummary:
    if not trials:
        return BranchSummary(
            label=label, n_trials=0,
            median_max_abs_diff=float("nan"),
            median_mean_channel_corr=float("nan"),
            median_aligned_corr=float("nan"),
            median_best_lag_ms=float("nan"),
            trials=[],
        )
    max_abs = np.array([trial.max_abs_diff for trial in trials])
    mean_corr = np.array([trial.mean_channel_corr for trial in trials])
    aligned = np.array([trial.aligned_corr for trial in trials])
    lag = np.array([trial.best_lag_ms for trial in trials])
    return BranchSummary(
        label=label,
        n_trials=len(trials),
        median_max_abs_diff=float(np.nanmedian(max_abs)),
        median_mean_channel_corr=float(np.nanmedian(mean_corr)),
        median_aligned_corr=float(np.nanmedian(aligned)),
        median_best_lag_ms=float(np.nanmedian(lag)),
        trials=trials,
    )


def compare_branches(
    offline_epochs_array: np.ndarray,
    online_epochs_array: np.ndarray,
    sfreq: float,
    label: str,
) -> BranchSummary:
    """Per-trial metrics across two aligned epoch arrays.

    Trims both arrays to the common trial count *and* the common per-epoch
    sample count. The sample-count trim is load-bearing for
    ``resample_filter_stage == "late"``: MNE's ``Epochs.resample(target)``
    drops one sample from a 1.2 s × 1000 Hz epoch (1201 → 120), while the
    streaming online path produces 121 samples — an off-by-one that we
    silently align here rather than failing the broadcast in
    ``trial_metrics``.
    """
    n_trials = min(offline_epochs_array.shape[0], online_epochs_array.shape[0])
    n_samples = min(offline_epochs_array.shape[2], online_epochs_array.shape[2])
    trials = [
        trial_metrics(
            offline_epochs_array[trial_index, :, :n_samples],
            online_epochs_array[trial_index, :, :n_samples],
            sfreq,
        )
        for trial_index in range(n_trials)
    ]
    return summarise(label, trials)


# ── Probability-level helpers ───────────────────────────────────────────────


def predict_at_timepoint(
    epochs_array: np.ndarray,
    decoding_timepoint: float,
    tmin: float,
    target_sfreq: float,
    engine: LiveInferenceEngine,
) -> dict[str, np.ndarray]:
    """Extract feature vector at the trained timepoint and run inference.

    ``epochs_array`` has shape ``(n_trials, n_channels, n_samples)``.
    Returns ``{task_name: P_positive[n_trials]}``.
    """
    n_samples = epochs_array.shape[2]
    t_idx = int(round((decoding_timepoint - tmin) * target_sfreq))
    if t_idx < 0 or t_idx >= n_samples:
        raise ValueError(
            f"decoding_timepoint {decoding_timepoint:.3f}s with tmin={tmin} maps to "
            f"index {t_idx}, out of range [0, {n_samples})."
        )
    features = epochs_array[:, :, t_idx]  # (n_trials, n_channels)
    return engine.predict(features)


def probability_table_from_predictions(
    branch_label: str,
    predictions: dict[str, np.ndarray],
    event_codes_per_trial: np.ndarray,
    event_code_to_name: dict[int, str],
    marker_names: list[str],
) -> ProbabilityTable:
    """Aggregate per-trial P(positive) into a decoder × marker median table."""
    name_to_code = {name: code for code, name in event_code_to_name.items()}
    task_names = list(predictions.keys())
    cells: dict[str, dict[str, float]] = {}
    for task in task_names:
        cells[task] = {}
        per_trial = predictions[task]
        for marker in marker_names:
            code = name_to_code.get(marker)
            if code is None:
                cells[task][marker] = float("nan")
                continue
            mask = event_codes_per_trial == code
            if not mask.any():
                cells[task][marker] = float("nan")
            else:
                cells[task][marker] = float(np.median(per_trial[mask]))
    return ProbabilityTable(
        branch_label=branch_label,
        task_names=task_names,
        marker_names=list(marker_names),
        cells=cells,
    )


# ── Threshold checks ────────────────────────────────────────────────────────


def check_thresholds(
    baseline: BranchSummary,
    offline_std: float,
    aligned_corr_min: float,
    lag_max_ms: float,
    diff_ratio_max: float,
) -> list[str]:
    flags: list[str] = []
    if baseline.n_trials == 0:
        flags.append("baseline produced 0 trials")
        return flags
    if baseline.median_aligned_corr < aligned_corr_min:
        flags.append(
            f"aligned_corr {baseline.median_aligned_corr:.3f} < {aligned_corr_min:.3f}"
        )
    if abs(baseline.median_best_lag_ms) > lag_max_ms:
        flags.append(
            f"best_lag_ms |{baseline.median_best_lag_ms:.1f}| > {lag_max_ms:.1f}"
        )
    if offline_std > 0 and baseline.median_max_abs_diff / offline_std > diff_ratio_max:
        flags.append(
            f"max_abs_diff / offline_std "
            f"{baseline.median_max_abs_diff / offline_std:.3f} > {diff_ratio_max:.3f}"
        )
    return flags


# ── Orchestration ───────────────────────────────────────────────────────────


def _build_event_code_to_name(settings: SettingsManager) -> dict[int, str]:
    return {code: name for name, code in settings.get_event_mapping().items()}


def _marker_names_from_settings(settings: SettingsManager) -> list[str]:
    """All decoders' positive labels, deduplicated, in stable order."""
    decoder_settings = settings.get_decoder_settings()
    seen: dict[str, None] = {}
    for task in decoder_settings["tasks"]:
        for label in task["pos_labels"]:
            seen.setdefault(label, None)
    return list(seen.keys())


def run_single_stage_pass(
    vhdr_path: Path,
    settings: SettingsManager,
    artifact: DecoderPipelineArtifact,
    work_dir: Path,
    chunk_samples: int,
    preprocessing_settings: dict,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Run both branches with one ``preprocessing_settings`` dict.

    ``max_seconds`` crops both raw inputs to the first N seconds. Useful when
    the full recording exhausts memory (the offline pipeline + ICA fit on a
    multi-hour recording can exceed 8 GB).

    Returns a dict with everything downstream comparison + probability code
    needs:

    - ``offline_epochs``         : the ``mne.Epochs``
    - ``offline_array``          : ``(n_trials, n_ch, n_samples)``
    - ``offline_event_times_s``  : ``(n_trials,)``
    - ``offline_event_codes``    : ``(n_trials,)``
    - ``online_trace``           : ``(n_samples, n_ch)``
    - ``online_timestamps``      : ``(n_samples,)``
    - ``baseline_online_array``  : ``(n_trials, n_ch, n_samples)`` aligned to
                                   the offline events (default patches=None)
    - ``preprocessing_settings`` : the dict actually used
    """
    raw_for_offline = _load_raw_cropped(vhdr_path, max_seconds)
    offline_epochs = run_offline_branch(
        raw=raw_for_offline,
        settings=settings,
        artifact=artifact,
        output_dir=work_dir,
        preprocessing_settings_override=preprocessing_settings,
    )
    offline_array = offline_epochs.get_data(picks="eeg")
    offline_event_times_s = offline_epochs.events[:, 0] / offline_epochs.info["sfreq"]
    offline_event_codes = offline_epochs.events[:, 2]

    raw_for_online = _load_raw_cropped(vhdr_path, max_seconds)
    online_trace, online_timestamps = run_online_branch(
        raw=raw_for_online,
        artifact=artifact,
        preprocessing_settings=preprocessing_settings,
        chunk_samples=chunk_samples,
        patches=None,
    )

    target_sfreq = float(preprocessing_settings["final_resample"]["target_rate"])
    tmin = float(preprocessing_settings["epochs"]["tmin"])
    tmax = float(preprocessing_settings["epochs"]["tmax"])
    baseline_online_array, _ = epoch_trace_at_event_times(
        trace=online_trace,
        out_timestamps=online_timestamps,
        event_times_s=offline_event_times_s,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
    )

    return {
        "offline_epochs": offline_epochs,
        "offline_array": offline_array,
        "offline_event_times_s": offline_event_times_s,
        "offline_event_codes": offline_event_codes,
        "online_trace": online_trace,
        "online_timestamps": online_timestamps,
        "baseline_online_array": baseline_online_array,
        "preprocessing_settings": preprocessing_settings,
        "raw_pre_hygiene_ch_names": list(raw_for_offline.ch_names),
    }


def _epoch_online_with_patches(
    vhdr_path: Path,
    artifact: DecoderPipelineArtifact,
    preprocessing_settings: dict,
    chunk_samples: int,
    patches: dict,
    event_times_s: np.ndarray,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Re-stream with attribute patches and epoch at the same event times."""
    raw = _load_raw_cropped(vhdr_path, max_seconds)
    trace, out_timestamps = run_online_branch(
        raw=raw,
        artifact=artifact,
        preprocessing_settings=preprocessing_settings,
        chunk_samples=chunk_samples,
        patches=patches,
    )
    target_sfreq = float(preprocessing_settings["final_resample"]["target_rate"])
    tmin = float(preprocessing_settings["epochs"]["tmin"])
    tmax = float(preprocessing_settings["epochs"]["tmax"])
    epochs, _ = epoch_trace_at_event_times(
        trace=trace,
        out_timestamps=out_timestamps,
        event_times_s=event_times_s,
        tmin=tmin,
        tmax=tmax,
        target_sfreq=target_sfreq,
    )
    return epochs


def run_bisection_sweep(
    pass_outputs: dict[str, Any],
    vhdr_path: Path,
    artifact: DecoderPipelineArtifact,
    chunk_samples: int,
    max_seconds: float | None = None,
) -> list[BranchSummary]:
    """Stage-by-stage bisection, plus the offline-baseline-off sanity check.

    Returns one ``BranchSummary`` per variant.
    """
    preprocessing_settings = pass_outputs["preprocessing_settings"]
    offline_array = pass_outputs["offline_array"]
    offline_event_times_s = pass_outputs["offline_event_times_s"]
    target_sfreq = float(preprocessing_settings["final_resample"]["target_rate"])

    summaries: list[BranchSummary] = []

    # 1. ICA off — clear the exclude set so ICA is effectively identity.
    # 2. bad_interp_off — no bad channels marked for interpolation.
    # 3. notch_off — disable the notch SOS.
    n_ch = int(artifact.online_state["pre_whitener"].shape[0])
    stage_patches: dict[str, dict] = {
        "ica_off": {"_ica_exclude": []},
        "bad_interp_off": {
            "_bad_indices": [],
            "_good_indices": list(range(n_ch)),
        },
        "notch_off": {"_notch_sos": None},
    }
    print(f"[bisection] running {len(stage_patches)} online-side variants + 1 offline-side variant...", flush=True)
    for label, patches in _tqdm(stage_patches.items(), total=len(stage_patches), desc="bisection", leave=False):
        try:
            online_array = _epoch_online_with_patches(
                vhdr_path=vhdr_path,
                artifact=artifact,
                preprocessing_settings=preprocessing_settings,
                chunk_samples=chunk_samples,
                patches=patches,
                event_times_s=offline_event_times_s,
                max_seconds=max_seconds,
            )
            summary = compare_branches(offline_array, online_array, target_sfreq, label=label)
            summaries.append(summary)
        except Exception as exc:  # surface but don't abort the bisection sweep
            logger.warning("Stage '%s' failed: %s", label, exc)
            summaries.append(_empty_summary(f"{label} (failed: {exc})"))

    # 4. offline_baseline_off — strip MNE's baseline param from the offline
    #    epochs and re-compare against the default online output. With the
    #    paper-aligned config (``baseline: null``) this is effectively a
    #    no-op sanity check; it produces metric drift only if a baseline
    #    actually was applied at epoching time.
    try:
        offline_epochs_no_baseline = pass_outputs["offline_epochs"].copy().apply_baseline(None)
        offline_array_no_baseline = offline_epochs_no_baseline.get_data(picks="eeg")
        summary = compare_branches(
            offline_array_no_baseline,
            pass_outputs["baseline_online_array"],
            target_sfreq,
            label="offline_baseline_off",
        )
        summaries.append(summary)
    except Exception as exc:
        logger.warning("Stage 'offline_baseline_off' failed: %s", exc)
        summaries.append(_empty_summary(f"offline_baseline_off (failed: {exc})"))

    return summaries


def _empty_summary(label: str) -> BranchSummary:
    return BranchSummary(
        label=label,
        n_trials=0,
        median_max_abs_diff=float("nan"),
        median_mean_channel_corr=float("nan"),
        median_aligned_corr=float("nan"),
        median_best_lag_ms=float("nan"),
    )


def build_probability_tables(
    pass_outputs: dict[str, Any],
    artifact: DecoderPipelineArtifact,
    settings: SettingsManager,
    extra_online_arrays: dict[str, np.ndarray] | None = None,
) -> list[ProbabilityTable]:
    """Median P(positive) per decoder × marker for offline + every online variant."""
    engine = LiveInferenceEngine(artifact.models, artifact.metadata)
    decoding_timepoint = float(artifact.metadata["decoding_timepoint"])
    preprocessing_settings = pass_outputs["preprocessing_settings"]
    target_sfreq = float(preprocessing_settings["final_resample"]["target_rate"])
    tmin = float(preprocessing_settings["epochs"]["tmin"])
    code_to_name = _build_event_code_to_name(settings)
    marker_names = _marker_names_from_settings(settings)
    event_codes = pass_outputs["offline_event_codes"]

    tables: list[ProbabilityTable] = []

    # Offline branch — the reference. Diagonal should dominate on training data.
    offline_preds = predict_at_timepoint(
        pass_outputs["offline_array"], decoding_timepoint, tmin, target_sfreq, engine
    )
    tables.append(
        probability_table_from_predictions(
            "offline", offline_preds, event_codes, code_to_name, marker_names
        )
    )

    # Online baseline.
    online_preds = predict_at_timepoint(
        pass_outputs["baseline_online_array"],
        decoding_timepoint, tmin, target_sfreq, engine,
    )
    tables.append(
        probability_table_from_predictions(
            "online_baseline", online_preds, event_codes, code_to_name, marker_names
        )
    )

    # Any extra online variants the caller passes in (e.g. bisection arrays).
    for label, online_array in (extra_online_arrays or {}).items():
        if online_array.shape[0] == 0:
            tables.append(
                ProbabilityTable(
                    branch_label=label,
                    task_names=list(offline_preds.keys()),
                    marker_names=list(marker_names),
                    cells={t: {m: float("nan") for m in marker_names} for t in offline_preds},
                )
            )
            continue
        preds = predict_at_timepoint(
            online_array, decoding_timepoint, tmin, target_sfreq, engine
        )
        tables.append(
            probability_table_from_predictions(
                label, preds, event_codes, code_to_name, marker_names
            )
        )

    return tables


def run_parity_check(args: argparse.Namespace) -> ParityReport:
    if not args.vhdr.exists():
        raise FileNotFoundError(f"VHDR not found: {args.vhdr}")
    if not args.pipeline.exists():
        raise FileNotFoundError(f"Pipeline not found: {args.pipeline}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    settings = SettingsManager(args.config)
    artifact = load_decoder_pipeline_artifact(args.pipeline)
    preprocessing_settings = settings.get_preprocessing_params()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Default-stage pass ───────────────────────────────────────────────────
    pass_outputs = run_single_stage_pass(
        vhdr_path=args.vhdr,
        settings=settings,
        artifact=artifact,
        work_dir=args.work_dir,
        chunk_samples=args.chunk_samples,
        preprocessing_settings=preprocessing_settings,
        max_seconds=args.max_seconds,
    )

    offline_array = pass_outputs["offline_array"]
    target_sfreq = float(preprocessing_settings["final_resample"]["target_rate"])
    offline_std = float(np.std(offline_array))

    # ── Channel-name alignment ──────────────────────────────────────────────
    # ch_names are part of the .vhdr header; no need to preload the .eeg payload.
    raw_for_channels = mne.io.read_raw_brainvision(str(args.vhdr), preload=False, verbose="ERROR")
    derived_names = derive_online_channel_names(raw_for_channels, artifact)
    offline_ch_names = pass_outputs["offline_epochs"].ch_names
    alignment = check_channel_alignment(offline_ch_names, derived_names)

    # ── Baseline metric ─────────────────────────────────────────────────────
    baseline = compare_branches(
        offline_array,
        pass_outputs["baseline_online_array"],
        target_sfreq,
        label="baseline",
    )

    # ── Stage bisection (incl. offline_baseline_off) ────────────────────────
    stage_summaries = run_bisection_sweep(
        pass_outputs=pass_outputs,
        vhdr_path=args.vhdr,
        artifact=artifact,
        chunk_samples=args.chunk_samples,
        max_seconds=args.max_seconds,
    )

    # ── Probability-level tables ────────────────────────────────────────────
    probability_tables = build_probability_tables(
        pass_outputs=pass_outputs,
        artifact=artifact,
        settings=settings,
    )

    # ── Alt-stage twin ──────────────────────────────────────────────────────
    alt_stage_report: dict | None = None
    if args.alt_stage:
        current_stage = preprocessing_settings.get("resample_filter_stage", "early")
        flipped = "late" if current_stage == "early" else "early"
        try:
            alt_settings = _override_stage(preprocessing_settings, flipped)
            alt_pass = run_single_stage_pass(
                vhdr_path=args.vhdr,
                settings=settings,
                artifact=artifact,
                work_dir=args.work_dir / f"alt_{flipped}",
                chunk_samples=args.chunk_samples,
                preprocessing_settings=alt_settings,
                max_seconds=args.max_seconds,
            )
            alt_baseline = compare_branches(
                alt_pass["offline_array"],
                alt_pass["baseline_online_array"],
                float(alt_settings["final_resample"]["target_rate"]),
                label=f"baseline_{flipped}",
            )
            alt_tables = build_probability_tables(
                pass_outputs=alt_pass,
                artifact=artifact,
                settings=settings,
            )
            alt_stage_report = {
                "stage": flipped,
                "baseline": asdict(alt_baseline),
                "probability_tables": [asdict(t) for t in alt_tables],
            }
        except Exception as exc:
            logger.warning("Alt-stage twin failed: %s", exc)
            alt_stage_report = {"stage": flipped, "error": str(exc)}

    flags = check_thresholds(
        baseline=baseline,
        offline_std=offline_std,
        aligned_corr_min=args.aligned_corr_min,
        lag_max_ms=args.lag_max_ms,
        diff_ratio_max=args.diff_ratio_max,
    )
    if not alignment.all_match:
        flags.append(
            f"channel_alignment: {alignment.n_match}/{alignment.n_positions} positions match"
        )

    return ParityReport(
        offline_branch={
            "n_trials": int(offline_array.shape[0]),
            "n_channels": int(offline_array.shape[1]),
            "n_samples": int(offline_array.shape[2]),
            "sfreq": target_sfreq,
            "tmin": float(preprocessing_settings["epochs"]["tmin"]),
            "tmax": float(preprocessing_settings["epochs"]["tmax"]),
            "std": offline_std,
        },
        channel_alignment=alignment,
        baseline=baseline,
        stage_bisection=stage_summaries,
        probability_tables=probability_tables,
        alt_stage=alt_stage_report,
        flags=flags,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vhdr", type=Path, required=True, help="Source BrainVision .vhdr (FL recording).")
    parser.add_argument("--pipeline", type=Path, required=True, help="decoder_pipeline.joblib path.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config YAML.")
    parser.add_argument("--chunk-samples", type=int, default=40, help="Online micro-batch size in samples.")
    parser.add_argument(
        "--alt-stage",
        action="store_true",
        help="Also run a twin parity check with resample_filter_stage flipped.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("debug_snapshots/parity_report.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/parity_check_work"),
        help="Scratch directory for the offline preprocessor's saved epochs.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Crop both raw inputs to the first N seconds (memory budget).",
    )
    parser.add_argument("--aligned-corr-min", type=float, default=0.95)
    parser.add_argument("--lag-max-ms", type=float, default=50.0)
    parser.add_argument("--diff-ratio-max", type=float, default=0.25)
    return parser.parse_args(argv)


def write_report(report: ParityReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "offline_branch": report.offline_branch,
        "channel_alignment": asdict(report.channel_alignment) if report.channel_alignment else None,
        "baseline": asdict(report.baseline),
        "stage_bisection": [asdict(branch) for branch in report.stage_bisection],
        "probability_tables": [asdict(table) for table in report.probability_tables],
        "alt_stage": report.alt_stage,
        "flags": report.flags,
    }
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=float)


def _format_branch(branch: BranchSummary) -> str:
    return (
        f"  {branch.label:>22s}: "
        f"n={branch.n_trials:3d}  "
        f"corr={branch.median_mean_channel_corr:+.3f}  "
        f"aligned_corr={branch.median_aligned_corr:+.3f}  "
        f"lag={branch.median_best_lag_ms:+.1f}ms  "
        f"max|Δ|={branch.median_max_abs_diff:.3e}"
    )


def _format_probability_table(table: ProbabilityTable) -> str:
    col_w = max(11, max((len(m) for m in table.marker_names), default=11) + 1)
    header = f"  {table.branch_label:<22s}" + "".join(m.rjust(col_w) for m in table.marker_names)
    rows = [header, "  " + "-" * (22 + col_w * len(table.marker_names))]
    for task in table.task_names:
        row = f"  {task:<22s}"
        for marker in table.marker_names:
            value = table.cells[task].get(marker, float("nan"))
            row += (f"{value:.3f}" if np.isfinite(value) else "n/a").rjust(col_w)
        rows.append(row)
    return "\n".join(rows)


def print_summary(report: ParityReport) -> None:
    print("\n=== Offline branch ===")
    for key, value in report.offline_branch.items():
        print(f"  {key}: {value}")

    print("\n=== Channel-name alignment ===")
    alignment = report.channel_alignment
    if alignment is None:
        print("  (no alignment computed)")
    else:
        verdict = "MATCH" if alignment.all_match else "MISMATCH"
        print(f"  {verdict}: {alignment.n_match}/{alignment.n_positions} positions agree")
        if not alignment.all_match:
            print("  First 8 mismatched positions:")
            shown = 0
            for row in alignment.rows:
                if row.match:
                    continue
                print(
                    f"    pos {row.position:>3d}: offline={row.offline_name!s:<8s} "
                    f"derived={row.derived_name!s}"
                )
                shown += 1
                if shown >= 8:
                    break

    print("\n=== Online baseline vs offline ===")
    print(_format_branch(report.baseline))

    print("\n=== Stage bisection (each row disables one stage) ===")
    for summary in report.stage_bisection:
        print(_format_branch(summary))

    print("\n=== Probability tables (median P(positive) per decoder × marker) ===")
    for table in report.probability_tables:
        print(_format_probability_table(table))
        print()

    if report.alt_stage is not None:
        print(f"\n=== Alt-stage twin (resample_filter_stage = {report.alt_stage.get('stage')}) ===")
        if "error" in report.alt_stage:
            print(f"  failed: {report.alt_stage['error']}")
        else:
            alt_baseline = report.alt_stage["baseline"]
            print(
                f"  baseline_{report.alt_stage['stage']}: "
                f"n={alt_baseline['n_trials']:3d}  "
                f"aligned_corr={alt_baseline['median_aligned_corr']:+.3f}  "
                f"lag={alt_baseline['median_best_lag_ms']:+.1f}ms"
            )

    print("\n=== Flags ===")
    if report.flags:
        for flag in report.flags:
            print(f"  ! {flag}")
    else:
        print("  (none)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_parity_check(args)
    write_report(report, args.out)
    print_summary(report)
    print(f"\nWrote report: {args.out}")
    return 1 if report.flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
