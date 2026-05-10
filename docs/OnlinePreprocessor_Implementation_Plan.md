# OnlinePreprocessor Implementation Plan

## Context

`OnlinePreprocessor` is the stateful EEG preprocessing component that bridges `LSLReceiver`
(raw 1000 Hz EEG, shape `(n_samples, 64)`) and `LiveInferenceEngine` (cleaned, decimated
features at ~256 Hz). It must causally replicate the offline pipeline's spatial transforms
using matrices exported from Phase 1, so that the live data distribution matches the
training distribution exactly.

**Offline pipeline order (must be mirrored online):**
1. Bandpass + notch filter at 1000 Hz (IIR)
2. Resample → target rate (256 Hz)
3. Interpolate bad channels (fixed weights)
4. Average reference
5. Apply ICA (fixed unmixing/mixing matrices)

**Key constraints:**
- All filtering must be causal (`sosfilt` with persistent `zi` state — never `filtfilt`)
- Filter and decimation state carry forward across every `process_batch()` call
- No live detection or adaptation — all transforms are fixed from Phase 1
- `OnlinePreprocessor` is the only component that unpacks `online_state`

**Architecture decisions locked:**
- Bad channel handling: export precomputed spherical-spline interpolation weight matrix
  from Phase 1 (exact match to offline). Requires small change to `export_online_state()`.
- Filter coefficients: obtained via `mne.filter.create_filter()` in `__init__` so we use
  exactly the same filter as the offline pipeline.
- Decimation: design FIR anti-aliasing lowpass via `scipy.signal.firwin`, apply with
  `scipy.signal.lfilter` + persistent `zi`, then subsample using a running phase counter.
- ICA: pure NumPy linear algebra — no MNE required at inference time.
- Dev approach: TDD — write failing tests first, then implement.

**Findings from latest pull (2026-05-09):**
- `live_inference.py` was simplified — positive class is now a single
  `metadata.get("positive_class", 1)` with no per-task fallback. No impact on
  `OnlinePreprocessor`.
- `export_online_state()` does not include `interp_weights` yet. Commit 1 is still needed.
- Sample rate is 256 Hz in `experiment_config.yaml` but Phase 2 docs still mark it `[!]`.
  Our plan reads `target_rate` from settings, so it is flexible regardless.
- `OfflinePhaseOrchestrator` is the next planned offline-phase item per `CLAUDE.md` — it
  will own `decoder_pipeline.joblib` export. Our Commit 1 only adds one key to
  `export_online_state()` (additive), so it is compatible.
- ICA matrix shapes are not verified by existing tests (only `isinstance` is checked).
  Commit 5 must add explicit shape assertions before relying on formula correctness.
- `sfreq_offline` in `online_state` can be used to cross-validate that the Phase 1 locked
  sample rate equals `preprocessing_settings["resample"]["target_rate"]`. Add this check
  to the constructor in Commit 2.

---

## Critical Files

| File | Role |
|---|---|
| `online_decoder/src/backend/online_phase/online_preprocessor.py` | New — main implementation |
| `online_decoder/src/backend/online_phase/__init__.py` | Add `OnlinePreprocessor` export |
| `online_decoder/src/backend/offline_phase/preprocessor.py` | Add `interp_weights` to export |
| `online_decoder/tests/online_phase/test_online_preprocessor.py` | New — all unit + integration tests |
| `online_decoder/tests/offline_phase/test_preprocessor.py` | Add test for `interp_weights` in export |

---

## Constructor Signature (agreed interface)

```python
class OnlinePreprocessor:
    def __init__(
        self,
        preprocessing_settings: dict,   # unwrapped from PreprocessingSettings Pydantic model
        online_state: dict,              # as-is from DecoderPipelineArtifact.online_state
        input_sfreq: float = 1000.0,    # LSL enforces this; parameterised for testing
    ) -> None:
```

`process_batch()` signature:
```python
def process_batch(
    self,
    eeg_batch: np.ndarray,   # (n_samples, n_channels) at input_sfreq
    timestamps: np.ndarray,  # (n_samples,) LSL timestamps
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: features (n_out, n_channels), output_timestamps (n_out,)
```

Validation in `__init__` is minimal — only catches errors that would otherwise be silent:
- Cross-validates `online_state["sfreq_offline"]` against `preprocessing_settings["resample"]["target_rate"]`:
  a mismatch means filter coefficients would be designed for the wrong rate.
- Checks `ica_pca_components.shape[1] == len(ch_names)`: a mismatch would surface as a
  cryptic NumPy broadcast error on the first `process_batch()` call.

All other key/type errors are left to Python's natural `KeyError`/`TypeError` — these dicts
come from internal code (Pydantic-validated settings + `export_online_state()`) so upfront
checking is redundant.

---

## Commit 1 — Phase 1: export interpolation weight matrix

**Scope:** `offline_phase/preprocessor.py` + its test file only.

**What:** After `raw.interpolate_bads()` runs in `_detect_bad_channels()`, compute and store
a weight matrix `W` such that, for any future data array `X` (shape `n_samples × n_channels`):
```
X[:, bad_indices] = X[:, good_indices] @ W   # W shape: (n_good, n_bad)
```
This is derived by running MNE interpolation on an identity-like basis — one column at a
time — to extract the linear weights.

**Export addition** in `export_online_state()`:
```python
"interp_weights": self._interp_weights,  # ndarray (n_good, n_bad) or None if no bad channels
```

### Checklist
- [x] Write failing test: `export_online_state()` includes key `interp_weights`
- [x] Write failing test: when no bad channels, `interp_weights` is `None`
- [x] Write failing test: `interp_weights` shape is `(n_good_channels, n_bad_channels)`
- [x] Write failing test: applying `interp_weights` to ground-truth good-channel data
      reproduces what MNE's `interpolate_bads()` produces for bad channels
      (within 1e-10 tolerance — it must be an exact linear transform)
- [x] Implement `_compute_interp_weights()` private method using identity-basis trick
- [x] Call it at end of `_detect_bad_channels()`, store as `self._interp_weights`
- [x] Add `interp_weights` to `export_online_state()` return dict
- [x] All 4 new tests pass; existing preprocessor tests unchanged

---

## Commit 2 — Skeleton, constructor, validation, reset

**Scope:** New file `online_preprocessor.py` + new test file.

**What:** Class skeleton with constructor validation, property accessors, `reset_state()`,
and a `process_batch()` stub that raises `NotImplementedError`. All persistent state
initialised but no processing logic.

**State initialised in `__init__`:**
- `_bandpass_sos`: SOS filter coefficients from `mne.filter.create_filter()`
- `_notch_sos`: SOS notch coefficients (or `None` if `notch` is None)
- `_decimate_fir`: Anti-aliasing FIR lowpass coefficients from `scipy.signal.firwin`
- `_bandpass_zi`: `None` (reset to proper shape on first batch)
- `_notch_zi`: `None`
- `_decimate_zi`: `None`
- `_decimate_phase`: `0` (running index into the up/down cycle)
- Unpacked spatial transform matrices from `online_state`

### Checklist
- [x] Write failing test: valid construction succeeds
- [x] Write failing test: constructor raises `ValueError` if `sfreq_offline` mismatches `target_rate`
- [x] Write failing test: constructor raises `ValueError` if `ica_pca_components` column count
      does not match `len(ch_names)`
- [x] Write failing test: `process_batch()` raises `NotImplementedError` (stub)
- [x] Write failing test: properties return expected values (`n_channels`, `target_sfreq`, `input_sfreq`)
- [x] Write failing test: `reset_state()` resets `_bandpass_zi`, `_notch_zi`, `_decimate_zi`,
      `_decimate_phase` to initial values
- [x] Implement the skeleton — constructor, properties, `reset_state()`, stub `process_batch()`
- [x] All tests pass

---

## Commit 3 — Causal bandpass + notch filter (step 1 of pipeline)

**Scope:** `online_preprocessor.py` + test file. No changes to other files.

**What:** Private method `_apply_filter(data)` that:
1. Applies bandpass SOS filter with `scipy.signal.sosfilt(sos, data, zi=self._bandpass_zi)`
   and updates `self._bandpass_zi` with the returned `zi_new`.
2. If notch is configured, same for notch SOS + `self._notch_zi`.

Filter coefficient design (in `__init__`):
- `mne.filter.create_filter(..., method='iir')` returns a dict; bandpass SOS is at `result['sos']`.
  Applied single-pass with `sosfilt` (causal). Differs from offline `filtfilt` in phase response
  and effective stopband depth (~−34 dB vs ~−68 dB at 2.5× cutoff). See
  `knowledge_base/02_reference/online_filtering.md` for full group delay analysis and rationale.
- Notch: `scipy.signal.iirnotch(w0=notch_freq, Q=30, fs=input_sfreq)` → `tf2sos`. Q=30 gives
  ~3.3 Hz notch width. MNE's `create_filter` does not expose a bandstop output directly.
- Store both SOS arrays as class attributes.

`zi` initialisation: lazy — on first batch, initialise from
`scipy.signal.sosfilt_zi(sos)[:, :, np.newaxis] * data[0]` (warm-start, shape `(n_sections, 2, n_ch)`).

### Checklist
- [x] Write failing test: processing same data in one chunk vs. ten equal-sized chunks
      produces identical output arrays (within 1e-10 — validates `zi` continuity)
- [x] Write failing test: processing same data in irregular chunk sizes (37, 51, 29, …)
      matches one-chunk output (validates `zi` on irregular batches)
- [x] Write failing test: power above `h_freq` is attenuated (>30 dB) after filtering
      (note: −30 dB threshold, not −40 dB — causal single-pass has half the effective order of filtfilt)
- [x] Write failing test: power below `l_freq` is attenuated (>40 dB) after filtering
- [x] Write failing test: when `notch=None`, notch filter is not applied (no extra attenuation)
- [x] Write failing test: when `notch=50.0`, power around 50 Hz is attenuated
- [x] Write failing test: `_bandpass_zi` is not `None` after first call to `_apply_filter()`
- [x] Write failing test: `reset_state()` zeroes `_bandpass_zi` and `_notch_zi` back to `None`
- [x] Implement `_apply_filter()` with lazy `zi` initialisation and state update
- [ ] Wire `_apply_filter()` into `process_batch()` (done in Commit 6)
- [x] All tests pass

---

## Commit 4 — Stateful decimation (step 2 of pipeline)

**Scope:** `online_preprocessor.py` + test file.

**What:** Private method `_decimate(data, timestamps)` that:
1. Applies a FIR anti-aliasing lowpass filter via `scipy.signal.lfilter(b, 1, data,
   axis=0, zi=self._decimate_zi)` and updates `self._decimate_zi`.
2. Subsamples the filtered output using `self._decimate_phase` to track the correct
   positions: collect sample indices where `(global_sample_idx % step_numer) < step_denom`
   (polyphase selection for non-integer ratios like 1000→256).
3. Selects corresponding timestamps by index.

Anti-aliasing FIR design (in `__init__`):
- Cutoff at `0.9 * target_rate / 2` Hz, length = `10 * up_factor + 1` taps, using
  `scipy.signal.firwin`.
- `up_factor = target_rate / gcd(input_sfreq, target_rate)`, `down_factor = input_sfreq / gcd(...)`.
- `_decimate_zi` initialised lazily on first call (shape: `(n_taps - 1, n_channels)`).

**Output sample count formula:** For a batch of `n_in` samples starting at global phase `p`:
`n_out = floor((n_in + p) * target_rate / input_sfreq) - floor(p * target_rate / input_sfreq)`

### Checklist
- [ ] Write failing test: one large batch and many small batches produce same number of
      output samples total (validates phase continuity)
- [ ] Write failing test: output timestamps are a valid subset of input timestamps
      (each output timestamp corresponds to a real input sample)
- [ ] Write failing test: batch of 40 input samples → expected output count
      `floor(40 * 256/1000)` = 10 samples
- [ ] Write failing test: batch of 37 samples at phase offset 3 → correct output count
- [ ] Write failing test: empty input returns empty output and timestamps with correct shape `(0, n_ch)`
- [ ] Write failing test: output shape is `(n_out, n_channels)` — 2D preserved
- [ ] Write failing test: `reset_state()` resets `_decimate_zi` to `None` and `_decimate_phase` to 0
- [ ] Write failing test: after reset, first batch of same data produces same output
      as very first call (stateless restart)
- [ ] Implement `_decimate()` with lazy `zi` init, lfilter, and phase tracking
- [ ] Wire `_decimate()` into `process_batch()` (after filter, before spatial transforms)
- [ ] All tests pass

---

## Commit 5 — Spatial transforms: bad channel interpolation, average reference, ICA

**Scope:** `online_preprocessor.py` + test file. All three steps are stateless.

**Step A — Bad channel interpolation** (`_apply_bad_channel_interpolation(data)`):
```python
data[:, bad_indices] = data[:, good_indices] @ interp_weights  # W: (n_good, n_bad)
```
`bad_indices` and `good_indices` derived from `online_state["ch_names"]` and
`online_state["bad_channels"]` in `__init__`.

**Step B — Average reference** (`_apply_average_reference(data)`):
```python
data -= data.mean(axis=1, keepdims=True)
```
Applied to all channels after interpolation (bad channels now carry interpolated values).

**Step C — ICA application** (`_apply_ica(data)`):
```python
# Exact MNE ICA math — verify against mne.preprocessing.ICA.apply() in tests
centered = data - pca_mean                                            # (n, n_ch)
whitened = centered @ pca_components.T                               # (n, n_comp)
sources  = whitened @ unmixing.T                                     # (n, n_comp)
sources[:, ica_exclude] = 0
reconstructed = sources @ mixing.T                                   # (n, n_comp)
data_clean = reconstructed @ pca_components + pca_mean              # (n, n_ch)
```
Note: the exact matrix shapes and formula must be verified in tests against MNE output
(see checklist). If the formula is wrong, tests will catch it.

### Checklist
- [ ] Write failing test: `_apply_bad_channel_interpolation()` — interpolated bad channel
      values match MNE's `interpolate_bads()` output on a small synthetic example (tol 1e-10)
- [ ] Write failing test: when `bad_channels` is empty list, data is returned unchanged
- [ ] Write failing test: `_apply_average_reference()` — mean across all channels is 0
      for each sample after application
- [ ] Write failing test: `_apply_average_reference()` is idempotent
      (applying twice gives same result as once)
- [ ] Write failing test: verify ICA matrix shapes — `ica_unmixing` and `ica_mixing` are
      `(n_components, n_components)` and `ica_pca_components` is `(n_components, n_channels)`.
      (Existing tests only check `isinstance`; shapes must be explicitly asserted here
      because the ICA formula depends on them.)
- [ ] Write failing test: `_apply_ica()` output matches `mne.preprocessing.ICA.apply()` on
      a small synthetic example using the same matrices (tol 1e-8)
- [ ] Write failing test: `_apply_ica()` with empty `ica_exclude` leaves data unchanged
      (reconstruction should be identity)
- [ ] Write failing test: `_apply_ica()` with `pca_mean=None` skips mean subtraction without error
- [ ] Implement `_apply_bad_channel_interpolation()`, `_apply_average_reference()`, `_apply_ica()`
- [ ] Verify ICA formula by running test against MNE — adjust formula if test fails
      (expected: center → PCA whiten → ICA unmix → zero excl. → ICA mix → PCA unwhiten → add mean)
- [ ] Wire all three into `process_batch()` (after decimation)
- [ ] All tests pass

---

## Commit 6 — `process_batch()` integration and end-to-end tests

**Scope:** `online_preprocessor.py` + test file.

**What:** Remove the `NotImplementedError` from `process_batch()` and wire the full pipeline.
Add end-to-end integration tests.

**process_batch() implementation:**
```python
def process_batch(self, eeg_batch, timestamps):
    # Input validation
    if eeg_batch.ndim != 2 or eeg_batch.shape[1] != self.n_channels:
        raise ValueError(...)
    if timestamps.shape[0] != eeg_batch.shape[0]:
        raise ValueError(...)
    if eeg_batch.shape[0] == 0:
        return np.empty((0, self.n_channels)), np.empty((0,))

    data = eeg_batch.copy().astype(float)
    data, out_timestamps = self._apply_filter_and_decimate(data, timestamps)
    data = self._apply_bad_channel_interpolation(data)
    data = self._apply_average_reference(data)
    data = self._apply_ica(data)
    return data, out_timestamps
```

### Checklist
- [ ] Write failing test: wrong number of channels raises `ValueError`
- [ ] Write failing test: timestamps length mismatches data rows → `ValueError`
- [ ] Write failing test: empty batch `(0, n_ch)` returns `(empty (0, n_ch), empty (0,))`
      without modifying state
- [ ] Write failing test: output shape is `(n_out, n_channels)` for valid input
- [ ] Write failing test: sequential calls — processing data in 40-sample chunks gives
      same total output samples as processing it all at once in one call
      (state continuity across the full pipeline)
- [ ] Write failing test: `reset_state()` after a series of batches, then reprocessing
      the same data from scratch produces identical output to the first run
- [ ] Write failing test: `process_batch()` does not mutate the input `eeg_batch` array
- [ ] Remove `NotImplementedError`, finalise `process_batch()` implementation
- [ ] All tests pass

---

## Commit 7 — Public API export

**Scope:** `online_phase/__init__.py` only.

### Checklist
- [ ] Write failing test: `from backend.online_phase import OnlinePreprocessor` succeeds
- [ ] Write failing test: `OnlinePreprocessor` appears in `online_phase.__all__`
- [ ] Add `OnlinePreprocessor` to `__init__.py` imports and `__all__`
- [ ] All tests pass

---

## Verification (end-to-end smoke test)

Once all commits are done, run manually (or in a notebook):

```python
from backend.online_phase import OnlinePreprocessor, load_decoder_pipeline_artifact

artifact = load_decoder_pipeline_artifact("decoder_pipeline.joblib")
preprocessor = OnlinePreprocessor(
    preprocessing_settings=settings_manager.preprocessing.model_dump(),
    online_state=artifact.online_state,
)

# Simulate 10 micro-batches of 40 samples
rng = np.random.default_rng(0)
for _ in range(10):
    eeg = rng.standard_normal((40, 64))
    timestamps = np.linspace(0, 0.04, 40)
    features, ts_out = preprocessor.process_batch(eeg, timestamps)
    print(features.shape, ts_out.shape)  # should be ~(10, 64) each time
```

All 7 commits' tests should pass with `pytest online_decoder/tests/`.
