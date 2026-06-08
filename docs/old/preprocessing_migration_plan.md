# Migrate preprocessing to `tomer_preprocessing_new.py` reference

> **Status (2026-05-23).** Both phases **done**.
>
> Offline scope: config schema (`config_models.py` + `experiment_config.yaml`),
> `OfflinePreprocessor` (four-step API + new pipeline order), `OfflineOrchestrator`,
> and the frontend (settings + preprocessing screens, MNE interactive windows,
> `ica_component_card.py` deleted) are migrated, with `mne-icalabel` added and
> `autoreject` removed.
>
> Online scope: `OnlinePreprocessor` consumes the positional online_state
> (`eeg_chunk_indices`, `bad_indices`, ICA matrices, interp weights, pre_whitener
> — no channel names) directly, with filter/lowpass/decimate stages variant-flagged
> by `resample_filter_stage`. The offline↔online numerical-parity tests in
> `tests/online_phase/test_online_preprocessor.py::TestApplyICA` and
> `::TestIntegration` are active. The 10 `test_stream_worker.py` errors remain a
> pre-existing missing-`qtbot` environment issue, unrelated.

## Context

The pipeline in [../src/backend/offline_phase/preprocessor.py](../../src/backend/offline_phase/preprocessor.py) and its causal mirror in [../src/backend/online_phase/online_preprocessor.py](../../src/backend/online_phase/online_preprocessor.py) were modelled on an **outdated** reference. The current reference of record is [../../knowledge_base/02_reference/tomer_preprocessing_new.py](../../knowledge_base/02_reference/tomer_preprocessing_new.py), with its ICA helper [../../knowledge_base/02_reference/ica_handler.py](../../knowledge_base/02_reference/ica_handler.py).

The new reference differs structurally — ICA runs on epochs **after** cleaning rather than on raw, filtering is split (HP early / LP late), there is a second resample at the end on epochs, the montage is hardware-specific, and human-in-the-loop steps (bad channels, ICA review) replace the current automatic detection. We must adopt this with these project-specific constraints (from the instructor's latest direction):

1. **Input data is 1000 Hz** — skip the reference's *early* 5 kHz → 1 kHz resample. New recordings will be at 1 kHz from the next subject onwards.
2. **Target sfreq for training = 100 Hz**, **LPF cutoff = 40 Hz** (per the cited replay paper). Both significantly tighter than the reference (which uses ≥250 Hz / 100 Hz LP).
3. **A/B test resample+LPF placement:** the instructor wants two variants compared on the same data:
   - **Variant "early"** — apply 40 Hz LPF + decimate to 100 Hz on the **raw** data, before ICA. ICA then fits on 100 Hz epochs (much faster).
   - **Variant "late"** — keep raw at 1 kHz, fit ICA on 1 kHz epochs, then apply LPF + decimate at the very end (reference order).
   Controlled by `preprocessing.resample_filter_stage: "early" | "late"`. Both offline and online read this directly from `settings`.
4. **No AutoReject** — drop the AutoReject stage entirely. Also drop the `reject=dict(eeg=hard_amplitude)` pre-filter in `mne.Epochs`.
5. **Manual selection via MNE's native interactive windows** — bad-channel marking via `raw.plot(block=True)` (reads `raw.info['bads']` on close — the same flow the reference's updated `_handle_bad_channels` uses, no Qt dialog), and ICA component review via `ica.plot_components(inst=epochs)` + `ica.plot_sources(epochs, block=True)` with `ica.exclude` pre-populated by ICLabel.
6. **ICA fit parameters per instructor** — `n_components=None` (let MNE/infomax pick — the data rank is `n_electrodes - 1` after average-referencing, no `compute_rank` needed); fit copy is **HP-only** (`l_freq=1.0, h_freq=None`), no LPF on the ICA fit copy.
7. **Baseline correction is configurable.** Paper omits it for decoder-friendliness ("To promote better decoding performance, baseline correction was omitted"). Default to `null` (omitted) but expose the field so A/B testing on our own data is possible.

Other answered decisions:

- **MetadataEnricher → skip.** Project labels come from triggers only via `CLASSIFICATION_SCHEMES`; no FL/BL/TP behavioural CSVs exist.
- **Channel hygiene → adopt all three** (`EMG` drop, `HEGOC→HEOG` rename, `easycap-M1` montage with `AFz→Afz` fix) — same hardware as the reference.
- **ICA review → MNE interactive + ICLabel hints.** ICLabel (via `mne-icalabel`) pre-populates `ica.exclude`, the operator overrides in MNE's interactive window.

The outcome: a preprocessing pipeline whose offline output matches the new reference + instructor's parameters for our 1000 Hz data, with a causal `OnlinePreprocessor` that mirrors it for streaming. Supports both "early" and "late" resample/LPF placement via `experiment_config.yaml`.

**Separation of concerns:** `settings` (loaded from YAML via `SettingsManager`) is the single source of truth for **recipe** — every parameter describing *how* the pipeline behaves (filter cutoffs, sample rate target, ICA params, variant flag, channel hygiene flags). `online_state` (inside `decoder_pipeline.joblib`) holds only **state** — the fitted numerical artifacts of one specific training run (ICA matrices, pre-whitener, interpolation weights, `bad_indices`, `ica_exclude`, `eeg_chunk_indices`). The two never overlap, and everything in `online_state` is positional — no channel names. Both offline and online phases read `settings` from the same in-memory `SettingsManager` owned by `AppSession`, so drift between training and inference is prevented by the workflow itself — settings don't change between phases in a single session.

**All currently trained `decoder_pipeline.joblib` artifacts become invalid** (different sfreq, different LP cutoff, different ICA fit data) and must be regenerated.

---

## Target offline pipeline order

Implemented in [../src/backend/offline_phase/preprocessor.py](../../src/backend/offline_phase/preprocessor.py). Stage placement of LPF + final resample depends on `preprocessing.resample_filter_stage`.

Shared prefix (both variants):

1. **Load raw `.vhdr`** — `mne.io.read_raw_brainvision(..., preload=True)`.
2. **Channel hygiene** —
   - `raw.set_channel_types({"EMG": "emg"})` then `raw.drop_channels(["EMG"])`.
   - If `HEGOC` in `ch_names`: `raw.rename_channels({"HEGOC": "HEOG"})`.
   - Build `easycap-M1` montage, swap `"AFz" → "Afz"` in `montage.ch_names`, then `raw.set_montage(...)`.
3. **High-pass filter** — `raw.filter(l_freq=0.1, h_freq=None, method="iir")`.
4. **Notch filter** — `raw.notch_filter(freqs=50.0)`.
5. *(If `resample_filter_stage == "early"`)* **Low-pass filter on raw** — `raw.filter(l_freq=None, h_freq=40.0, method="iir")` **and** **Resample raw to 100 Hz** — `raw.resample(100)`.
6. **[Interactive — main thread]** `raw.plot(block=True)` — operator marks bads by clicking channel names; values are read from `raw.info['bads']` after the window closes.
7. **Interpolate bads** — `raw.interpolate_bads(reset_bads=True)`. Capture spherical-spline weights via the existing `_compute_interp_weights()` helper (preserved).
8. **Make epochs** — `mne.Epochs(raw, events, event_id=..., tmin, tmax, baseline=baseline_from_config, detrend=0, preload=True, metadata=mne.epochs.make_metadata(...))`. No `reject=`. `baseline_from_config` defaults to `None` (paper-aligned) but is config-driven.
9. **Average reference on epochs** — `mne.set_eeg_reference(epochs, "average")`.
10. **Fit ICA on cleaned epochs** — copy with `epochs.copy().filter(l_freq=1.0, h_freq=None)` (HP-only fit copy). `n_components=None` (let MNE/infomax decide — rank is `n_electrodes - 1` after average reference). `method='infomax'`, `fit_params=dict(extended=True)`, `random_state` from config.
11. **ICLabel auto-suggest** — `mne_icalabel.label_components(fit_epochs, ica, method='iclabel')`. Pre-populate `ica.exclude` with indices whose label is in the configured `drop_labels` set.
12. **[Interactive — main thread]** `ica.plot_components(inst=epochs)` then `ica.plot_sources(epochs, block=True)` — operator toggles. Final `ica.exclude` is whatever the window returns.
13. **Apply ICA** — `ica.apply(epochs)`.
14. *(If `resample_filter_stage == "late"`)* **Low-pass filter on epochs** — `epochs.filter(l_freq=None, h_freq=40.0, method="iir")` **and** **Final resample on epochs** — `epochs.resample(100)`.
15. **Save** — `{subject}_epo.fif`.

`export_online_state()` stays minimal — only the fitted state. ICA matrices come from a fit at either 100 Hz (early variant) or 1 kHz (late variant); the variant itself is *not* stored in `online_state` — the online side reads it from `settings.preprocessing.resample_filter_stage`.

**Expected ICA time budget:** Variant "early" fits ICA on ~10× fewer samples than "late" — wall-clock difference will likely be the deciding factor in production unless the two variants produce visibly different decoder accuracy.

---

## Target online pipeline order

Implemented in [../src/backend/online_phase/online_preprocessor.py](../../src/backend/online_phase/online_preprocessor.py). Stage placement of LPF + decimation mirrors `settings.preprocessing.resample_filter_stage`.

Shared causal stages: HP + notch IIR (persistent `zi`) → bad-channel interpolation → average reference → ICA → optionally followed (or preceded) by LP + decimate.

**Variant "early"** — LPF + decimate happen *before* spatial transforms, matching the offline order:
1. HP + notch IIR (1000 Hz, persistent `zi`).
2. **Low-pass IIR** at 40 Hz (persistent `zi`).
3. **Decimate** 1000 → 100 Hz (FIR anti-alias + phase tracking — reuse `_decimate()`; ratio 10).
4. Bad-channel interpolation (now operating at 100 Hz, on matrices fit from 100 Hz offline data).
5. Average reference.
6. Apply ICA (matrices fit from 100 Hz cleaned epochs).

**Variant "late"** — LPF + decimate happen *after* spatial transforms:
1. HP + notch IIR (1000 Hz, persistent `zi`).
2. Bad-channel interpolation (at 1000 Hz).
3. Average reference (at 1000 Hz).
4. Apply ICA (matrices fit from 1000 Hz cleaned epochs).
5. **Low-pass IIR** at 40 Hz (persistent `zi`).
6. **Decimate** 1000 → 100 Hz (FIR anti-alias + phase tracking; ratio 10).

`process_batch()` returns `(features, timestamps)` at **100 Hz** for both variants. The `_apply_*` stages run in an order chosen at construction time from `settings.preprocessing.resample_filter_stage`.

---

## Config schema changes

[../src/backend/core/config_models.py](../../src/backend/core/config_models.py) and [../experiment_config.yaml](../../experiment_config.yaml):

**Remove:**

- `BandpassSettings.h_freq` (replaced by a separate lowpass section).
- `ResampleSettings.target_rate = 256` (replaced).
- `RejectCriteriaSettings` entirely (`hard_amplitude`, `flat_threshold`, `noisy_z_score` — no auto bad-channel detection, no AutoReject, no hard pre-reject).
- `ICASettings.n_components: int` becomes `Literal["auto"] | int` with default `"auto"`.

**Add:**

```yaml
preprocessing:
  # Where to place LPF + decimation in the pipeline.
  # "early" → on raw before ICA (faster ICA, paper-aligned)
  # "late"  → on epochs after ICA (reference order, fits ICA on full-rate data)
  resample_filter_stage: early   # default; the A/B comparison toggles this

  channel_hygiene:
    drop_emg: true
    rename_hegoc_to_heog: true
    montage_name: easycap-M1
    afz_case_fix: true

  highpass:
    l_freq: 0.1
    method: iir          # keep IIR for offline/online causal parity

  notch:
    freq: 50.0

  ica:
    method: infomax
    extended: true
    n_components: null    # let MNE/infomax decide (rank = n_electrodes - 1 after avg ref)
    fit_l_freq: 1.0
    # No fit_h_freq — ICA fit copy is HP-only per instructor.
    iclabel:
      enabled: true
      drop_labels: ["muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"]

  epochs:
    tmin: -0.2
    tmax: 1.0
    baseline: null        # paper-aligned; set to [null, 0] to re-enable pre-stim mean subtraction

  lowpass:
    h_freq: 40.0          # paper-aligned LP for 100 Hz target sfreq
    method: iir

  final_resample:
    target_rate: 100      # paper-aligned training rate
```

Pydantic models: split `BandpassSettings` → `HighpassSettings` + `NotchSettings` + `LowpassSettings`; add `ChannelHygieneSettings`, `IclabelSettings`, `FinalResampleSettings`; delete `RejectCriteriaSettings`. Update `PreprocessingSettings` container. Add a top-level `resample_filter_stage: Literal["early", "late"]` field with `Field(default="early")`. `EpochSettings.baseline` becomes `Optional[tuple[Optional[float], Optional[float]]]` (allow `None` for the whole baseline). `ICASettings.n_components` becomes `Optional[int]` with default `None`.

**Note on IIR vs FIR divergence from reference:** The reference uses MNE's default FIR filters (zero-phase via `filtfilt`). We deliberately use IIR for both offline HP/notch/LP because the **online** side must be causal — using FIR offline and IIR online produces a phase-response train/test mismatch. This is the same trade-off the current code already makes.

---

## Phase 1 → Phase 2 contract (`online_state` / `decoder_pipeline.joblib`)

`online_state` carries **only fitted state** — the numerical artifacts the offline run produced. Recipe parameters (filter cutoffs, sample-rate target, ICA params, variant flag, channel hygiene) live in `settings` and are read directly from there by both phases. No defensive copies are stored in the joblib; the workflow guarantees that the same in-memory `SettingsManager` (owned by `AppSession`) serves both phases within a session, and a YAML edit between sessions is treated as "different experiment, retrain."

### `online_state` schema

Exported by `OfflinePreprocessor.export_online_state()` and consumed by [../src/backend/online_phase/online_preprocessor.py](../../src/backend/online_phase/online_preprocessor.py). The full joblib envelope is built in [../src/backend/offline_phase/orchestrator.py](../../src/backend/offline_phase/orchestrator.py) and unwrapped by `load_decoder_pipeline_artifact()` in [../src/backend/online_phase/artifact_loader.py](../../src/backend/online_phase/artifact_loader.py).

```python
online_state = {
    # Index permutation: which EEG positions to keep from the 64-channel
    # post-trigger-split LSL EEG array. Encodes "drop EMG" (and any other
    # offline channel drops) as a positional list — names never cross.
    # Example: 64 EEG channels with EMG at position 8 → [0..7, 9..63] (63 ints).
    "eeg_chunk_indices": list[int],

    # Bad-channel info, all positional (indices into the post-hygiene EEG array)
    "bad_indices":       list[int],           # operator-marked, post-hygiene positions
    "interp_weights":    np.ndarray | None,   # (n_good, n_bad) spherical-spline

    # Frozen ICA fit
    "ica_unmixing":      np.ndarray,          # (n_components, n_components)
    "ica_mixing":        np.ndarray,          # (n_components, n_components)
    "ica_pca_components": np.ndarray,         # (n_components, n_channels)
    "ica_pca_mean":      np.ndarray | None,   # (n_channels,) or None
    "ica_exclude":       list[int],           # operator's final selection
    "pre_whitener":      np.ndarray,          # (n_channels, 1)
}
```

Everything is positional. No channel names — they don't survive into the LSL world. Bad channels are stored as `bad_indices` (positions in the post-hygiene array) rather than `bad_channels` (names) — same information, no name-to-index lookup needed online.

Shape is *almost* identical to today's `export_online_state()`. Differences: `bad_channels: list[str]` → `bad_indices: list[int]`; no `ch_names`; new `eeg_chunk_indices`. No schema version field, no recipe fields, no provenance block.

### `OnlinePreprocessor.__init__` signature

Unchanged from today:

```python
OnlinePreprocessor(
    preprocessing_settings: dict,   # recipe
    online_state: dict,              # state
    input_sfreq: float = 1000.0,
)
```

Recipe (filter coefficients, decimation ratio, stage ordering) is designed from `preprocessing_settings`. Spatial transforms (bad-channel interp, ICA matrices) are applied from `online_state`. No source overlap.

### Validation rules in `OnlinePreprocessor._validate_inputs`

The cross-check against `sfreq_offline` is removed (it was comparing settings to settings). Remaining checks are pure positional/shape consistency. Let `n_eeg = len(eeg_chunk_indices)` (the post-hygiene channel count, e.g. 63 after dropping EMG):

1. **Dimension cross-check.** `ica_pca_components.shape[1] == n_eeg`. `ica_unmixing.shape == ica_mixing.shape`. `pre_whitener.shape[0] == n_eeg`. `interp_weights.shape[0] + len(bad_indices) == n_eeg` if `interp_weights` is not None.
2. **ICA exclude in range.** Every index in `ica_exclude` must be in `[0, n_components)`.
3. **`eeg_chunk_indices` bounds.** Every index must be in `[0, 64)` (or whatever the LSL receiver's `eeg_channel_count` is). No duplicates.
4. **`bad_indices` bounds.** Every index must be in `[0, n_eeg)`. No duplicates.

### Online channel-hygiene replay

[../src/backend/online_phase/lsl_receiver.py](../../src/backend/online_phase/lsl_receiver.py) (or a thin shim immediately downstream of it) applies `online_state["eeg_chunk_indices"]` as a column selection to the **post-trigger-split EEG array** (the (n_samples, 64) chunk that comes out of `split_eeg_and_markers`, before reaching `OnlinePreprocessor`):

```python
# After existing trigger split:
eeg_chunk = eeg_chunk[:, online_state["eeg_chunk_indices"]]
# Now (n_samples, n_eeg_post_hygiene), with EMG (and any other offline-dropped channels) removed.
```

This replaces my earlier "drop the channel named EMG" prose, which doesn't fit the LSL contract (the stream is purely positional — no channel labels cross the LSL boundary in our setup). The HEGOC→HEOG rename and the easycap-M1 montage are **offline-only** operations: they touch MNE's `info` metadata and the spherical-spline geometry used by `interpolate_bads`, neither of which has any online runtime analog. Their effects are baked into `interp_weights` and the ICA matrices already.

The trigger split layer is unaffected: it stays in the LSL receiver, parameterized by the amplifier's hardware contract (`DEFAULT_TRIGGER_CHANNEL_INDEX = 64`). Hygiene operates strictly downstream of it, so dropping EEG channels can never shift the trigger position.

---

## Frontend / orchestrator restructure

MNE interactive plots **require the main Qt thread**. Today, [../src/frontend/workers/preprocessing_worker.py](../../src/frontend/workers/preprocessing_worker.py) runs the entire two-step pipeline on a `QThread` worker. We split into compute-on-worker / plot-on-main steps.

[../src/backend/offline_phase/orchestrator.py](../../src/backend/offline_phase/orchestrator.py) — replace the current `run_step1_prepare_ica` / `run_step2_finish_pipeline` API with four granular methods:

| Method | Where | What |
|---|---|---|
| `run_step1a_filter()` | Worker | Load raw → channel hygiene → HP+notch. Returns `mne.io.Raw` to UI. |
| `set_bad_channels(bads: list[str])` | Main thread (called after UI's interactive window closes) | Stores operator's selection. |
| `run_step1b_fit_ica()` | Worker | Interpolate bads → make epochs → average reference → fit ICA → ICLabel labels. Returns `(ica, epochs_for_review, suggested_exclude)`. |
| `run_step2_apply_and_save(exclude_components: list[int])` | Worker | Apply ICA → LP → final resample → save → export online state. |

Frontend changes in [../src/frontend/](../../src/frontend/):

- **New screen / step**: `BadChannelReviewStep`. Worker runs `run_step1a_filter()`. On completion, main thread calls `raw.plot(block=True)` — when the user closes the window, `raw.info["bads"]` is read and passed to `orchestrator.set_bad_channels(...)`. Then trigger Step 1B worker.
- **Replace `ICAComponentCard` grid screen** with: after Step 1B worker completes, main thread sets `ica.exclude = suggested_exclude`, calls `ica.plot_components(inst=epochs)` (optional topomap reference window) and `ica.plot_sources(epochs, block=True)`. Final `ica.exclude` is read on close and passed to `run_step2_apply_and_save(...)`.
- Delete [../src/frontend/widgets/ica_component_card.py](../../src/frontend/widgets/ica_component_card.py) and any preprocessing-view screens that wired it up.

Threading discipline: workers emit signals carrying the raw / ICA objects; main thread invokes MNE plot calls. **No MNE plot call from inside a `QThread.run()`** — that deadlocks Qt's event loop. The pattern is: `worker.finished.emit(raw)` → slot on main thread → `raw.plot(block=True)` → `orchestrator.set_bad_channels(raw.info["bads"])` → kick off next worker.

---

## Dependencies

[../requirements.txt](../../requirements.txt):

- **Add** `mne-icalabel` (for `mne_icalabel.label_components`).
- PyQt6 already pinned (`pyqt6>=6.6`), MNE already pinned — no further GUI deps.

---

## Files to modify

| Path | Change |
|---|---|
| [../src/backend/offline_phase/preprocessor.py](../../src/backend/offline_phase/preprocessor.py) | Full rewrite of pipeline order. Drop `_detect_bad_channels`, `_autoreject`. Add channel hygiene, ICLabel suggestion, **`resample_filter_stage` branching** (LP+resample on raw vs on epochs). Preserve `_compute_interp_weights`. **`export_online_state` rewrites its channel fields as positional:** `bad_channels` (names) → `bad_indices` (ints into post-hygiene array); drop `ch_names`; add `eeg_chunk_indices` computed by capturing the .vhdr's original channel order before EMG drop and recording which positions survived. |
| [../src/backend/offline_phase/orchestrator.py](../../src/backend/offline_phase/orchestrator.py) | Replace two-step API with `run_step1a_filter` / `set_bad_channels` / `run_step1b_fit_ica` / `run_step2_apply_and_save`. State machine update. |
| [../src/backend/online_phase/online_preprocessor.py](../../src/backend/online_phase/online_preprocessor.py) | Reconfigure stage 1 to HP-only. Add `_apply_lowpass` (40 Hz, persistent `zi`). Decimation ratio = 1000/100 = 10. **Pipeline ordering branches on `settings.preprocessing.resample_filter_stage`** ("early" → LP+decimate before spatial transforms; "late" → after ICA). `process_batch` returns at 100 Hz. Constructor signature unchanged (`preprocessing_settings`, `online_state`, `input_sfreq`). Drop the `sfreq_offline` validation check. |
| [../src/backend/online_phase/lsl_receiver.py](../../src/backend/online_phase/lsl_receiver.py) | After existing trigger split, apply `online_state["eeg_chunk_indices"]` as a column selection on the EEG chunk before forwarding to `OnlinePreprocessor`. Trigger split stays unchanged — hygiene operates downstream of it, so trigger position is unaffected. |
| [../src/backend/core/config_models.py](../../src/backend/core/config_models.py) | Split bandpass; add highpass/notch/lowpass/channel_hygiene/iclabel/final_resample; delete reject_criteria; ICA `n_components: Literal["auto"] \| int`. |
| [../experiment_config.yaml](../../experiment_config.yaml) | Rewrite `preprocessing:` block to new schema. |
| [../src/frontend/workers/preprocessing_worker.py](../../src/frontend/workers/preprocessing_worker.py) | Replace two workers with three (Step1A, Step1B, Step2). |
| `../src/frontend/screens/preprocessing_view.py` (and related) | Replace ICA card grid with main-thread `raw.plot` + `ica.plot_sources` calls. Add `BadChannelReviewStep`. |
| [../src/frontend/widgets/ica_component_card.py](../../src/frontend/widgets/ica_component_card.py) | **Delete** (replaced by MNE's native window). |
| [../requirements.txt](../../requirements.txt) | Add `mne-icalabel`. |

---

## Tests

| Path | Action |
|---|---|
| [../tests/offline_phase/test_preprocessor.py](../../tests/offline_phase/test_preprocessor.py) | Drop AutoReject + hard-amplitude + auto-bad-channel tests. Add tests for channel hygiene (EMG drop, HEGOC rename, montage), ICLabel-based pre-fill (mock `label_components`), epoch-side LP + resample, manual bad-channel hook accepts bads list. |
| [../tests/offline_phase/test_orchestrator.py](../../tests/offline_phase/test_orchestrator.py) | Update state machine for four-method API; assert ordering. |
| [../tests/online_phase/test_online_preprocessor.py](../../tests/online_phase/test_online_preprocessor.py) | Bandpass test → HP-only test. New LP+decimate test (40 Hz LP, 1000/100 ratio = 10). Drop the `sfreq_offline` cross-check test. **Add ordering tests for both `resample_filter_stage` variants** — verify stages run in the right order based on the settings flag. Add a shape-validation test: corrupt `ica_pca_components` shape → constructor raises before first `process_batch`. |
| [../tests/online_phase/test_lsl_receiver.py](../../tests/online_phase/test_lsl_receiver.py) | New test for index-based hygiene: feed a synthetic (n_samples, 65) chunk with a known column for EMG, apply trigger split + `eeg_chunk_indices` selection, assert the resulting array drops the right column and preserves the trigger column unchanged. |

---

## Verification

End-to-end, in order:

1. **Unit tests** — `pytest online_decoder/tests/ -v`. All offline + online preprocessor + orchestrator tests pass.
2. **Schema check** — `python -c "from src.backend.core.settings_manager import SettingsManager; SettingsManager.load_from_yaml('experiment_config.yaml')"` succeeds with the rewritten YAML.
3. **Offline replay on a known subject** — run the new offline pipeline end-to-end on one recording: filter → mark bads in the popped MNE window → confirm ICA review window opens with ICLabel suggestions pre-filled → close → confirm `{subject}_epo.fif` saves at 100 Hz with expected n_epochs and n_channels.
4. **Train a model** — run the offline orchestrator's train step on the new epochs and confirm `decoder_pipeline.joblib` is produced with the expected ICA matrix shapes and a non-empty `eeg_chunk_indices`.
5. **Online parity smoke test** — `python online_decoder/scripts/smoke_stream_worker.py --pipeline /path/to/new/decoder_pipeline.joblib --duration 5 --log /tmp/smoke.csv` runs without channel-mismatch errors and produces predictions at 100 Hz.
6. **A/B variant comparison** — run the offline pipeline twice on the same subject (once with `resample_filter_stage: early`, once with `late`), train a decoder on each, and compare CV accuracy + ICA wall-clock time. A small helper script in `scripts/ab_compare_resample_stages.py` automates the loop and prints a side-by-side report. This is the comparison the instructor explicitly requested.
7. **Numerical parity check (best-effort)** — replay the same raw through the new offline pipeline and the new online pipeline (using `scripts/characterize_lsl.py` style replay if available); compare a representative window. Causal IIR vs MNE's defaults will differ in phase but cutoffs and ICA matrices should agree to within filter-warmup tolerance.
8. **GUI smoke** — launch the frontend, walk through the full offline phase, confirm the two MNE interactive windows pop on the main thread (no Qt deadlock), and the rest of the existing screens still work.
