# Phase 2 — Preprocessing Migration Execution Plan

## Context

The overall design lives in [Preprocessing_Migration_Plan.md](Preprocessing_Migration_Plan.md) and covers **both phases** (offline training pipeline + online streaming pipeline). This document is the **execution-only** plan for **Phase 2 (Itai's half)**, broken by commit.

**Division of labour:**
- **Itai** — Phase 2: `OnlinePreprocessor`, `LSLReceiver` hygiene replay, online tests, benchmark script.
- **Roi** — Phase 1: `OfflinePreprocessor`, `OfflineOrchestrator`, frontend (workers, preprocessing view, bad-channel + ICA review screens), `mne-icalabel` dep.

Work happens on **separate feature branches**, merged periodically into `online-decoder` as logical chunks complete.

The **config schema** (`config_models.py` + `experiment_config.yaml`) is owned by Roi (Phase 1). Itai's work starts after Roi's schema commit lands on `online-decoder`. See the "Coordination" section below.

---

## ⚠ Pre-commit policy (MUST follow)

**Always stop and ask for explicit permission before running `git commit`.** Do not commit autonomously, do not amend autonomously, do not push autonomously. Even when a logical commit boundary is clearly reached.

Repeat this rule at the start of every commit step in this plan. The agent (me) will *propose* commit messages and *propose* staged file lists, but Itai runs (or explicitly authorises) the `git commit` invocation.

---

## Commit message style

Match the existing repo convention (from `git log --oneline`):

```
<type>(<optional-scope>): <imperative subject>
```

- `type` ∈ {`feat`, `fix`, `docs`, `tools`, `class`, `refactor`, `test`}
- `scope` is optional; recent commits use `(backend)`, `(scripts)`, `(session)`.
- Subject is lowercase, imperative, no trailing period.
- Body (optional) explains the *why*, not the *what*.

Examples already in history: `feat(session): add live stream session factory`, `tools: add StreamWorker smoke preflight checks`, `docs(backend): reconcile live stream implementation status`.

---

## Branch + merge strategy

- **Working branch:** `phase2-prepro-migration` (suggested) — branched from `online-decoder`.
- **Merge target:** `online-decoder`.
- **Merge cadence:** after each *logical chunk* completes (see commit groups below), not after every single commit. This minimizes broken-state windows on `online-decoder` while keeping Roi unblocked.
- **Suggested merge points:**
  1. After **Commits 1–4** (full `OnlinePreprocessor` + `LSLReceiver` rewrite) — merge as one chunk so the new pipeline lands atomically on `online-decoder`.
  2. After **Commits 5–6** (benchmark script, docs) — merge anytime.

Rebase (not merge-commit) when pulling `online-decoder` updates into the feature branch, to keep history linear.

---

## Coordination with Roi

**The config schema migration is Roi's responsibility, not Itai's.** All of the new Pydantic models (`HighpassSettings`, `LowpassSettings`, `ChannelHygieneSettings`, etc.) and the `experiment_config.yaml` rewrite live in Roi's Phase 1 work. **However, Itai can begin Phase 2 work in parallel** — only the *merge to `online-decoder`* is gated on Roi's schema, not the development itself.

### Why parallel work is safe

Itai's code reads `preprocessing_settings["highpass"]["l_freq"]` from a dict. Where that dict comes from is irrelevant to the code:

- **In tests:** the dict is a literal fixture (e.g. `{"highpass": {"l_freq": 0.1, "method": "iir"}, ...}`). The existing test suite already hardcodes settings this way; Itai just updates the fixture field names from day one.
- **In production runtime:** the dict comes from `SettingsManager.load_from_yaml(...)` via Pydantic. This path doesn't exist until Roi merges the schema — but Itai doesn't need it to run unit tests.

What Itai *can* do without waiting: write all code, run all unit tests, run the benchmark script (it synthesises its own settings dict).

What Itai *cannot* do without Roi's schema: run the GUI app end-to-end, run `smoke_stream_worker.py` against a real joblib.

### Merge-order constraint

The constraint is on **merging to `online-decoder`**, not on development. Once Itai's preprocessor code reads `["highpass"]` from a settings dict that still has `["bandpass"]` (the live YAML), the app crashes. So `online-decoder` must contain *both* the new schema and the new code at all times after the first merge that touches either.

Three acceptable merge orderings:
1. **Roi merges schema first**, Itai pulls/rebases `phase2-prepro-migration`, then Itai merges his branch. Cleanest.
2. **Atomic co-merge** — Itai cherry-picks or pulls Roi's schema commit into his branch, then both bodies of work go in via Itai's merge in one shot.
3. **Roi merges while Itai is mid-development**, Itai rebases to pick the schema up before any further work.

### Before Itai pushes any commit to `online-decoder`

1. Confirm Roi's schema has landed (or pull Roi's branch into `phase2-prepro-migration` for an atomic merge).
2. Rebase `phase2-prepro-migration` on the latest `online-decoder`.
3. Re-run the full test suite locally.
4. Verify the YAML loads: `python -c "from src.backend.core.settings_manager import SettingsManager; s = SettingsManager.load_from_yaml('experiment_config.yaml'); print(s.preprocessing)"`.

### Before Itai *starts*

Agree with Roi on the exact field paths (`highpass.l_freq`, `notch.freq`, `lowpass.h_freq`, `final_resample.target_rate`, `resample_filter_stage`, `channel_hygiene.*`, `ica.iclabel.*`, etc.) by referencing [Preprocessing_Migration_Plan.md § Config schema changes](Preprocessing_Migration_Plan.md). If those paths drift between Itai's test fixtures and Roi's eventual Pydantic models, the merge will fail loudly — but it's avoidable with five minutes of upfront agreement.

Roi's Phase 1 work also produces the new positional `online_state` shape (`bad_indices`, `eeg_chunk_indices`). Itai's tests use synthetic fixtures of that shape until Roi's joblib exports land.

---

## Commit sequence

Each commit lists: **Scope** (files touched), **Changes** (what), **Tests** (added/updated), **Dependencies** (prereqs), **Verification** (how to know it's done).

### Commit 1 — OnlinePreprocessor: split bandpass into HP-only

> **🛑 Ask before committing.**

**Message:** `feat(online): split bandpass filter into HP-only stage`

**Scope:**
- [../src/backend/online_phase/online_preprocessor.py](../src/backend/online_phase/online_preprocessor.py)
- [../tests/online_phase/test_online_preprocessor.py](../tests/online_phase/test_online_preprocessor.py)

**Changes:**
- Update constructor to read from `preprocessing_settings["highpass"]` and `preprocessing_settings["notch"]` instead of `["bandpass"]`.
- In the `mne.filter.create_filter(...)` call, pass `l_freq=hp["l_freq"]` and `h_freq=None` (no upper bound).
- Rename internal attribute `_bandpass_sos` → `_highpass_sos` if you want symmetry; otherwise just change what it's designed for. (Recommend rename for clarity.)
- Notch handling stays as-is; just read from new field path.

**Tests:**
- Existing bandpass tests in `TestApplyFilter` class — update fixtures to new schema, assert HP-only filter response (no upper rolloff).
- Add: `test_apply_filter_passes_high_frequencies` — confirm a 50 Hz tone passes the HP+notch stage (after notch attenuation) and a 0.05 Hz drift is suppressed.

**Dependencies:** None for development. **Merge gate:** Roi's schema must be on `online-decoder` (or atomically co-merged) before this commit lands. Tests run standalone via dict-literal fixtures.

**Verification:**
```bash
pytest online_decoder/tests/online_phase/test_online_preprocessor.py -v -k "filter"
```

---

### Commit 2 — OnlinePreprocessor: add low-pass stage + 100 Hz decimation

> **🛑 Ask before committing.**

**Message:** `feat(prepro-fix): add lowpass stage and decimate to 100 Hz`

**Scope:**
- [../src/backend/online_phase/online_preprocessor.py](../src/backend/online_phase/online_preprocessor.py)
- [../tests/online_phase/test_online_preprocessor.py](../tests/online_phase/test_online_preprocessor.py)

**Changes:**
- **New `_apply_lowpass(data)` method** mirroring `_apply_filter`'s structure: design `_lowpass_sos` from `preprocessing_settings["lowpass"]["h_freq"]` at input rate via `mne.filter.create_filter`, apply causally with persistent `_lowpass_zi`. Reset state added to `reset_state()`.
- Update decimation target: use `preprocessing_settings["final_resample"]["target_rate"]` (now 100) instead of `["resample"]["target_rate"]` (was 256). Ratio becomes 1000/100 = 10.
- **🚩 Rethink the decimation design from scratch.** The current implementation in `__init__` + `_decimate()` was built for the general-purpose non-integer case (1000→256 reduces to 125/32, with `up_factor=32`, `down_factor=125`, phase-tracked subsampling, and an FIR sized `10 * up_factor + 1 = 321` taps). For the new 1000→100 case the ratio is a clean 10:1 (`up_factor=1`, `down_factor=10`), which makes most of that machinery unnecessary. Specifically:
  - The `up_factor / down_factor` polyphase framing collapses — we're doing pure integer decimation by 10, not rational resampling.
  - The phase-tracked output-index formula (`out_indices = ceil(((k+1)*down - phase) / up) - 1`) collapses to "keep every 10th sample, modulo a running offset across chunks".
  - The `n_taps = 10 * up_factor + 1` heuristic produces only 11 taps when `up_factor=1` — too few for a clean anti-alias rolloff, would let energy alias.
  - Possible simpler designs: (a) explicit integer decimator with persistent phase (current code's algorithmic shape but trimmed of polyphase bookkeeping); (b) `scipy.signal.decimate`-style approach using `lfilter` with persistent `zi` then take `[phase::down_factor]`; (c) leave the general code in place but fix the tap-count heuristic (`n_taps = max(10 * up_factor + 1, 10 * down_factor + 1)`). Pick the design that's clearest to read and easiest to test against the new chunk-boundary continuity assertions.
  - Whichever design lands, the new LP-stage frequency-response tests (passband fidelity, stopband attenuation at 80/120/200 Hz, cutoff at 40 Hz) will catch insufficient anti-alias rolloff. Use those as ground truth.
- Add branching in `process_batch` based on `preprocessing_settings["resample_filter_stage"]`:
  - **"early"** order: HP+notch → **lowpass** → **decimate** → interp → avg-ref → ICA
  - **"late"** order: HP+notch → interp → avg-ref → ICA → **lowpass** → **decimate**
- Update `_validate_inputs`: drop the `sfreq_offline` vs `target_rate` cross-check (it was a tautology under the settings-only-recipe model).

**Tests** — the LP stage is the **new** numerical surface and deserves more than a single tone check. Match the rigour of the existing `TestApplyFilter` suite:

- Existing decimation tests — update ratios (10 instead of 1000/256), keep continuity assertions (chunking invariance, phase tracking).
- **Passband fidelity** — feed a low-frequency tone (e.g. 5 Hz) and assert its amplitude is preserved within ~1 dB after the LP stage. Confirms the filter doesn't accidentally attenuate signals well below cutoff.
- **Stopband attenuation** — feed a high-frequency tone (e.g. 80 Hz, 120 Hz, 200 Hz) and assert each is attenuated by progressively more dB. Asserts the filter actually rolls off, not just "attenuates above cutoff".
- **Cutoff sanity** — feed a tone right at `h_freq` (40 Hz) and assert attenuation is in a tight tolerance band around -3 dB (or whatever MNE's IIR design produces — verify the expected value by hand once, then assert against it).
- **Causality** — feed a unit impulse at sample 0, assert all output samples at indices < 0 (none, since causal) and at indices ≥ 0 that the response is non-zero only from sample 0 onward. Mirrors the existing `test_apply_filter_is_causal` style.
- **Chunk-boundary continuity** — apply the LP to a single long signal in one go, then apply it to the same signal split into 3 chunks, assert the concatenated chunked output equals the one-shot output to within `np.allclose` tolerance. This is the persistent-`zi` invariant — same test pattern as the existing decimation continuity tests.
- **`reset_state()` clears `_lowpass_zi`** — apply LP, reset, apply again to a different signal, assert the second output doesn't show carry-over from the first.
- **Frequency response matches MNE's design** — pick a few probe frequencies, compute the offline frequency response of the same IIR via `scipy.signal.sosfreqz`, assert our online output amplitudes at those frequencies fall within tolerance. Catches design-time bugs (e.g. wrong filter order, wrong sfreq parameterization).

Variant ordering (these go in `TestProcessBatch` or a new `TestVariantOrdering` class):

- `test_process_batch_early_variant_ordering` — patch settings to `resample_filter_stage: "early"`, monkey-patch each `_apply_*` to record the order called and the input shape it received. Assert sequence is `[hp_notch, lowpass, decimate, interp, avg_ref, ica]` and that `_apply_ica` sees input at 100 Hz.
- `test_process_batch_late_variant_ordering` — symmetric: sequence `[hp_notch, interp, avg_ref, ica, lowpass, decimate]`, ICA sees 1000 Hz input.

`sfreq_offline` check:
- Update `test_sfreq_validation` — assert it no longer raises on `target_rate` mismatch (since the check is removed).

**Dependencies:** Commit 1 landed.

**Verification:**
```bash
pytest online_decoder/tests/online_phase/test_online_preprocessor.py -v -k "lowpass or decimat or variant"
```

---

### Commit 3 — OnlinePreprocessor: switch to positional `online_state`

> **🛑 Ask before committing.**

**Message:** `refactor(prepro-fix): use positional indices from online_state`

**Scope:**
- [../src/backend/online_phase/online_preprocessor.py](../src/backend/online_phase/online_preprocessor.py)
- [../tests/online_phase/test_online_preprocessor.py](../tests/online_phase/test_online_preprocessor.py)

**Changes:**
- Replace constructor reads of `online_state["ch_names"]` (compute index lists from names) with direct reads of:
  - `online_state["bad_indices"]: list[int]`
  - `online_state["eeg_chunk_indices"]: list[int]` *(stored but not used here — used by LSLReceiver in Commit 4)*
- Derive `n_eeg = len(eeg_chunk_indices)` instead of `len(ch_names)`.
- Drop the `bad_local_indices = [eeg_ch_names.index(ch) for ch in bad_channels]` name-lookup logic in `_apply_bad_channel_interpolation`. Use `self._bad_indices` directly.
- Update `_validate_inputs`:
  - Drop the `ica_pca_components.shape[1] == len(ch_names)` check → use `len(eeg_chunk_indices)`.
  - Add bounds check on `bad_indices` (must be in `[0, n_eeg)`).
  - Add bounds check on `eeg_chunk_indices` (must be in `[0, 64)`, no dups).
  - Add bounds check on `ica_exclude` (must be in `[0, n_components)`).
  - Interp weights shape: `interp_weights.shape[0] + len(bad_indices) == n_eeg`.

**Tests:**
- Update existing `TestValidateInputs` tests — drop ch_names assertions, add positional bounds tests.
- Update integration tests (`TestOfflineStateIntegration`) — synthesize new online_state shape (with `bad_indices`, `eeg_chunk_indices`, no `ch_names`).
- New: `test_validate_rejects_duplicate_chunk_indices`.
- New: `test_validate_rejects_out_of_range_bad_indices`.
- Mock Roi's offline_state output until Roi's Phase 1 work merges.

**Dependencies:** Commit 2 landed. Roi may not yet have shipped the offline exporter for the new `online_state` shape — so this commit uses test fixtures with the new shape. End-to-end check waits for Roi's merge.

**Verification:**
```bash
pytest online_decoder/tests/online_phase/test_online_preprocessor.py -v -k "validate or offline_state"
```

---

### Commit 4 — OnlinePreprocessor: apply positional EEG hygiene at process_batch entry

> **🛑 Ask before committing.**

**Message:** `feat(prepro-fix): apply positional EEG hygiene inside OnlinePreprocessor`

**Scope:**
- [../src/backend/online_phase/online_preprocessor.py](../src/backend/online_phase/online_preprocessor.py)
- [../tests/online_phase/test_online_preprocessor.py](../tests/online_phase/test_online_preprocessor.py)

**Design decision:** the column selection lives in `OnlinePreprocessor`, **not** `LSLReceiver`. Rationale: every offline-derived artifact (`bad_indices`, ICA matrices, interp weights, `eeg_chunk_indices`) belongs together in the one class that consumes `online_state`. `LSLReceiver` stays pure hardware plumbing — talks to LSL, strips triggers, validates count. It has no business knowing about the offline pipeline. Bonus: smoke tests and future debug/visualization consumers of the receiver get raw 64-channel amplifier output, which is what they want.

**Changes:**
- In `process_batch`, as the very first transform (before `_apply_filter`):
  ```python
  data = eeg_batch[:, self._eeg_chunk_indices].astype(float)
  ```
- Loosen the input-shape check to ndim-only (`eeg_batch.ndim == 2`). The raw EEG width is not validated explicitly — if the configured `eeg_chunk_indices` point past the actual batch width, NumPy raises `IndexError` at slice time. The fail-fast value isn't worth the parameter churn: the LSL receiver already validates the stream has 65 channels (64 EEG + trigger) before any data reaches the preprocessor, so the production input width is guaranteed.
- `self.n_channels` property still returns `len(eeg_chunk_indices)` — the output count. `StreamWorker` and other downstream consumers unaffected.
- `_validate_inputs` keeps the local checks on `eeg_chunk_indices` (non-negativity, no duplicates) but adds no upper bound — the upper bound isn't knowable at construction time without coupling to the receiver.
- `LSLReceiver` and `session.py` get **no changes** — the receiver still hands over (n_samples, 64) EEG arrays as it does today.

**Tests:**
- New `TestProcessBatchHygiene::test_process_batch_applies_chunk_indices_drops_emg_column` — synthesise a 22-column batch with distinct per-column values, configure `eeg_chunk_indices` to drop position 8, assert the output width is 21.
- `test_too_narrow_batch_raises_index_error_at_slice` — a batch narrower than `max(eeg_chunk_indices) + 1` raises `IndexError` at the slice (documents the failure mode in lieu of a custom shape check).
- `test_non_2d_batch_raises` — 1D input still raises `ValueError` (the surviving ndim check).
- No `input_n_channels` kwarg threading through fixtures — existing tests use small synthetic data (`N_CHANNELS=20` or `4`) and `eeg_chunk_indices=list(range(N_CHANNELS))`, which slices any matching-width batch correctly.
- All existing `test_lsl_receiver.py` and `test_lsl_receiver_integration.py` tests stay as they were before this commit (no `eeg_chunk_indices` arg).

**Dependencies:** Commit 3 landed.

**Verification:**
```bash
pytest online_decoder/tests/online_phase/test_online_preprocessor.py -v -k "process_batch or validate or chunk_indices"
pytest online_decoder/tests/online_phase/ --ignore=tests/online_phase/test_lsl_receiver_integration.py
```

---

### Commit 5 — Benchmark script: align defaults with new schema

> **🛑 Ask before committing.**

**Message:** `tools(prepro-fix): align benchmark_preprocessor defaults with new schema`

**Scope:**
- [../scripts/benchmark_preprocessor.py](../scripts/benchmark_preprocessor.py)

**Changes:**
- Update default `--target-sfreq` from 256 → 100.
- Update internal synthetic-state builder to produce the new positional online_state shape (`bad_indices`, `eeg_chunk_indices`, no `ch_names`) and new preprocessing_settings (`highpass`/`notch`/`lowpass`/`final_resample`/`resample_filter_stage`).
- Update any hardcoded filter cutoffs (1 Hz HP / 40 Hz LP) to read from the synthetic settings rather than be baked in.

**Tests:** none (script is a manual perf harness).

**Dependencies:** Commits 1–4.

**Verification:**
```bash
python online_decoder/scripts/benchmark_preprocessor.py --target-sfreq 100 --n-channels 63
# Sanity-check the printed latency and that the script doesn't crash.
```

---

### Commit 6 — Documentation (partial, pulled forward before PR)

> Pulled forward before opening the PR for Commits 1–4 so reviewers see accurate Phase 2 docs. A follow-up commit will refresh the large stale `OnlinePreprocessor` code-block class doc inside `backend_architecture.md` (~line 1100) and the `CLAUDE.md` "Current Backend Scope" bullet if needed.



> **🛑 Ask before committing.**

**Message:** `docs(prepro-fix): document new OnlinePreprocessor pipeline order`

**Scope:**
- [backend_architecture.md](backend_architecture.md)
- [../../knowledge_base/02_reference/online_filtering.md](../../knowledge_base/02_reference/online_filtering.md)
- Possibly [../CLAUDE.md](../CLAUDE.md)

**Changes:**
- Add a short section to `backend_architecture.md` describing the new online pipeline order (both variants), the positional online_state, and how `LSLReceiver` applies hygiene.
- Reference [Preprocessing_Migration_Plan.md](Preprocessing_Migration_Plan.md) for the full design rationale.
- Update `CLAUDE.md`'s "Current Backend Scope" bullet if the architecture summary needs adjustment.
- **Update `online_filtering.md` for Commit 2's changes:**
  - Rewrite the "Decimation: 1000 Hz → 256 Hz" section for the new integer 1000 → 100 Hz case. The polyphase machinery (`up_factor`, `down_factor`, phase-counter loop, "first output at sample 3") is gone; the algorithm is now `np.arange(phase, n_in, decimation)` with simpler phase tracking. The attributes table needs to drop `_up_factor`/`_down_factor` and add `_decimation`.
  - Add a section for the new `_apply_lowpass` stage (40 Hz IIR cascaded after HP+notch). Extend the group-delay table with the LP contribution (~10 ms more delay near the 40 Hz cutoff).
  - Add an explicit note that `out_timestamps` returned from `_decimate` is the timestamp of the *input sample at the kept index*, not the effective time of the filtered value. Document the ~50 ms FIR group delay + ~10 ms IIR LP delay that the timestamps don't compensate for. Note this is a constant offset (consistent across predictions) and existed before Commit 2 — Commit 2 just makes it slightly larger by adding the LP stage. List the three mitigations (subtract FIR delay from timestamps, train classifier with shifted timepoints, or use identical causal filters offline) for future reference.

**Dependencies:** Commits 1–5 landed.

---

## End-to-end verification (after Commit 5, before merging back to `online-decoder`)

1. **Unit tests pass:**
   ```bash
   pytest online_decoder/tests/online_phase/ -v
   pytest online_decoder/tests/core/ -v
   ```
2. **Smoke test with a real (or replayed) joblib** *— requires Roi's Phase 1 work to be at least partially merged*:
   ```bash
   python online_decoder/scripts/smoke_stream_worker.py \
       --pipeline /path/to/decoder_pipeline.joblib --duration 5 --log /tmp/smoke.csv
   ```
   Expect predictions emitted at 100 Hz, no channel-mismatch errors, no dimension errors.
3. **Numerical parity (best-effort):** if Roi ships an offline replay tool, feed the same raw .vhdr through both offline and online pipelines, compare a representative window. Causal-IIR vs FIR phase will differ; cutoff magnitudes and ICA-cleaned components should agree to filter-warmup tolerance.

---

## Risks + open items

- **Roi schema timing.** Development can proceed in parallel; only the *merge* to `online-decoder` is gated on Roi's schema. Risk reduces to: if Itai's planned field paths drift from Roi's actual Pydantic models, the merge breaks. Mitigation: confirm exact field paths upfront (see Coordination section).
- **Mocked online_state in Commit 3 tests.** Until Roi's offline exporter produces the real new shape, tests rely on synthetic fixtures. End-to-end joblib testing has to wait for Roi's merge. Plan for at least one integration-test commit after Roi merges (probably belongs in a follow-up, not in this Phase 2 plan).
- **`mne.filter.create_filter` for the LP stage.** Same library function as HP — should "just work" with `l_freq=None, h_freq=40, method='iir'`. Verify the returned `sos` array shape on first run; surprises here mean an extra commit to wrap `iirfilter` directly.

---

## Quick reference: post-commit ask template

When proposing a commit, say something like:

> Ready to commit. Suggested message: `feat(online): split bandpass filter into HP-only stage`. Staged files: `online_decoder/src/backend/online_phase/online_preprocessor.py`, `online_decoder/tests/online_phase/test_online_preprocessor.py`. Confirm to proceed?

Wait for explicit "yes" / "go" / "do it" before running `git commit`. Never auto-commit.
