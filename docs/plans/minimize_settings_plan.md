# Plan: Minimize the experiment config — hardcode the preprocessing recipe (incremental)

**Branch:** `feat/minimize-settings`
**Goal:** Remove configurability from the fixed parts of `experiment_config.yaml`. The
preprocessing recipe is a faithful, paper-aligned reproduction that is never changed
between runs — it should be **hardcoded as named constants**, not magic numbers, and not
configuration. `decoders` and `markers_mapping` stay fully configurable.

**Approach:** Migrate **one preprocessing block at a time**. Each step *creates that block's
constant(s) as it strips them from the settings* — so the constant being added and the dict
read it replaces sit in the same small diff, easy to review. After every step the codebase
compiles, `pytest tests/` is green, and behavior is byte-identical (the constants equal
today's defaults). Each step is one commit/PR.

**Operator visibility:** the verbose, editable-looking Settings card is removed (Step 1).
Once the minimizing work is complete, a separate final step adds a compact read-only
**"Preprocessing stages" overview** shown *before the operator starts preprocessing* — on
the Node 3 `PreprocessingView` "Ready" page (`_build_ready_page()`, preprocessing_view.py:388),
populated directly from the constants module.

## Motivation

1. **Declutter.** The YAML drops from ~86 lines to ~25; the read-only Settings UI loses a
   ~180-line card displaying values no one edits.
2. **Kill a real footgun.** The online phase reads the *live* config for its preprocessing
   recipe (`session.py:213`), not the artifact that trained the model. If Phase 2 loads a
   different config than training used, the online recipe silently diverges. Importing one
   shared constants module from both phases makes divergence structurally impossible.

## Scope decisions (confirmed)

| Field group | Outcome |
|---|---|
| `experiment_info.name` | stays in YAML |
| `random_state` (top-level) | stays in YAML — seeds decoder CV/training **and** the ICA fit |
| `decoders.*` (model, params, scale_method, cv, tasks) | stays fully configurable |
| `markers_mapping.events` | stays in YAML (decoder tasks reference these names) |
| `preprocessing.*` (all of it, incl. `resample_filter_stage`, all ICA variations, `epochs.baseline: null`) | **hardcoded as constants** |
| SettingsView "Preprocessing" card | **removed entirely** |
| Operator visibility into the recipe | new read-only **overview on the Node 3 "Ready" page**, sourced from the constants — added as a separate final step (after minimizing is done) |

## Who reads which block (sizes each step)

| Block | OfflinePreprocessor | OnlinePreprocessor | Notes | Status |
|---|---|---|---|---|
| `lowpass` | yes | yes | h_freq + method | **done (Step 2)** |
| `final_resample.target_rate` | yes (`_resample`) | yes | scalar | **done (Step 3)** |
| `notch.freq` | yes | yes | scalar, `None` disables | **done (Step 4)** |
| `highpass` | yes | yes | l_freq + method | **done (Step 5)** |
| `resample_filter_stage` | yes (`_stage`) | yes | gated two pipeline paths — **late path removed** | **done (Step 5b)** |
| `epochs` | yes | **no** | tmin/tmax/baseline; baked into matrices offline | **done (Step 6)** |
| `channel_hygiene` | yes | **no** | 4 flags | **done (Step 7)** |
| `ica` (+ iclabel) | yes | **no** | most complex; touches `random_state` | |

The online preprocessor reads `lowpass`/`final_resample`/`notch`/`highpass` (now all
constants); `epochs`, `channel_hygiene`, and `ica` touch the offline preprocessor only.

> **`resample_filter_stage` — removed (Step 5b, 2026-06-21, reversing the earlier deferral).**
> Originally kept configurable because it selected two live LP+decimate orderings
> (`early`/`late`). Per new decision the early ordering is the only one used, so the toggle
> **and the entire `late` code path** were deleted from both preprocessors (along with the
> late-variant tests + the "rejects invalid stage" validation). The online preprocessor now
> reads **nothing** from its `preprocessing_settings` arg (it's vestigial — the final cleanup
> step will drop the arg). This unblocks the full removal of the online-side settings coupling.

## The pattern (every block-migration step does exactly this)

1. Add this block's named constant(s) to `src/backend/core/preprocessing_constants.py`
   (Step 2 creates the module on the first block).
2. Replace the dict reads (`self.settings["x"]` / `preprocessing_settings["x"]`) with the
   new constant — in **both** preprocessors if the table above says both read it.
3. Remove that field's sub-model + field from `config_models.py` (`PreprocessingSettings`).
4. Delete the key from the **3 tracked YAMLs** (`experiment_config.yaml`,
   `experiment_config.full.yaml`, `tests/data/sample_config.yaml`) — mandatory, since
   `extra="forbid"` rejects a leftover key. Also strip it from the **5 git-ignored
   `debug_snapshots/*` configs** (not committed, but they must load under the new schema);
   first check each block's value matches the constant before stripping — a divergent value
   would mean the cached `.joblib` artifact was trained with a different recipe.
5. **Re-attach the block to `SettingsManager._hardcoded_recipe()`** from its constant, in the
   historical config-dict shape (see "Frontend settings access" below). This keeps the
   effective recipe surfaced by `get_settings()` complete and shape-stable.
6. Add/extend `tests/core/test_preprocessing_constants.py` to pin the new constant(s), and the
   `TestGetSettings` assertions in `test_settings_manager.py` to cover the re-attached block.
7. Run `pytest tests/`; confirm green.

> Note: two debug snapshots (`colors_window50`, `colors_window50_restclass`) already fail to
> load under the current schema due to experimental **decoder** keys (`feature_window_ms`,
> `rest_class`) that predate this work — unrelated to the preprocessing migration.

### Frontend settings access (decision 2026-06-13)
The frontend reads config only through the `session.settings` dict (it imports no backend
internals). To keep that contract while the recipe moves to constants, `SettingsManager` has
two distinct surfaces:
- **`get_preprocessing_params()`** — the *backend pipeline* input. Only the fields still in
  the config (shrinks as blocks migrate; ultimately just `random_state` + the not-yet-migrated
  blocks). The preprocessors read the hardcoded blocks from `preprocessing_constants` directly,
  not from this dict.
- **`get_settings()`** — the *frontend's effective view*. Takes the shrinking config params
  and merges the hardcoded recipe back in via `_hardcoded_recipe()` (single source:
  `preprocessing_constants`), in the historical shape. **Its output is shape-stable across the
  whole migration** — values move config→constants under the hood, consumers are untouched.

This decapsulates raw config (backend) from the effective view (frontend). It replaced an
earlier idea of an `AppSession.target_sfreq` property / a typed `AppSettings` class — both
rejected in favour of keeping the existing dict presentation.

---

## Incremental steps

### Step 1 — Remove the SettingsView preprocessing card (UI only, independent)
- Delete `_build_preproc_section()` + its call, and the `pre = settings["preprocessing"]`
  branch of `_update_settings_display` (incl. the None-reset block).
- Leaves Setup + Model Evaluation cards untouched. No backend change; decouples the UI from
  every later step so block migrations never touch the frontend.

### Step 2 — Scaffolding + first block: `lowpass` ✅ DONE
- Created `src/backend/core/preprocessing_constants.py` with a module docstring.
- Added `LOWPASS_H_FREQ = 40.0`, `LOWPASS_METHOD = "iir"`.
- Offline `_lowpass` + online `__init__` (filter build + log line) read the constants.
- Removed `LowpassSettings` + the field from `PreprocessingSettings`; stripped `lowpass`
  from the 3 tracked YAMLs + the 5 debug snapshots (all used 40.0/iir — no divergence).
- Created `tests/core/test_preprocessing_constants.py` pinning the values; updated
  `test_settings_manager.py` (dropped the `lowpass` key/value asserts + the obsolete
  `test_rejects_non_positive_lowpass`); removed the now-inert `lowpass` key from the online
  test fixtures. Full suite green (445 passed, 1 skipped).
- (Pilot chosen over `resample_filter_stage` because lowpass is truly inert — every test
  already used 40.0/iir, and it gates no code path.)
- Recipe re-attachment to `get_settings()._hardcoded_recipe()` landed in Step 3 (when the
  frontend-settings-access design was decided); the lowpass block is now part of the
  shape-stable effective view.

### Step 3 — `final_resample.target_rate` ✅ DONE
- Added `FINAL_RESAMPLE_RATE = 100`; offline `_resample` + online `__init__` read it.
- Removed `FinalResampleSettings` + field; stripped `final_resample` from the 3 tracked YAMLs
  + the 5 debug snapshots (all were 100 — no divergence).
- **Frontend access (decided during this step):** `phase2_screen.py` reads
  `settings["preprocessing"]["final_resample"]["target_rate"]` to size the live chart. Rather
  than a new accessor, `SettingsManager.get_settings()` now re-attaches the hardcoded recipe
  via `_hardcoded_recipe()` (see "Frontend settings access" above), so the screen is
  **unchanged**. `get_preprocessing_params()` (backend input) stays minimal.
- Tests: rewrote `TestDecimateFrequencies` to parametrize **`input_sfreq`** ([1000,500,400,200]
  → factors 10/5/4/2) instead of `target_rate`, since the target is now fixed at 100; rewrote
  `test_raises_on_non_integer_decimation_ratio` to use `input_sfreq=1050`; dropped the
  `target_rate` params from the settings factories; removed the two obsolete `final_resample`
  range-rejection tests; added a `TestGetSettings` class asserting the re-attached recipe;
  updated the session fake to source the rate from the constant. Full suite green
  (448 passed, 1 skipped).

### Step 4 — `notch.freq` ✅ DONE
- Added `NOTCH_FREQ: float | None = 50.0`; offline `_notch` + online `__init__` (+ log line) read it.
- Removed `NotchSettings` + field; stripped `notch` from the 3 tracked YAMLs + 5 debug snapshots
  (all `freq: 50.0` — no divergence). Re-attached to `_hardcoded_recipe()`.
- Kept the `None`-disables guard in both preprocessors (notch is hardcoded **on**; the disable
  branch survives as dev-toggleable/defensive code). Rewrote `test_no_notch_leaves_50hz_intact`
  to monkeypatch `NOTCH_FREQ=None` (the disable path is no longer config-reachable);
  simplified `test_notch_attenuates_50hz` to the default; extended pin + `TestGetSettings`
  coverage. Full suite green (449 passed, 1 skipped).

### Step 5 — `highpass` ✅ DONE
- Added `HIGHPASS_L_FREQ = 0.1`, `HIGHPASS_METHOD = "iir"`; offline `_highpass` + online
  `__init__` (+ log line) read them. Removed `HighpassSettings` + field; re-attached to
  `_hardcoded_recipe()`. Stripped `highpass` from the 3 tracked YAMLs + 5 debug snapshots
  (all `0.1/iir` — no divergence).
- **Resolved the fixture wrinkle:** online fixtures + `sample_config.yaml` used `l_freq = 1.0`
  (a test-convenience value, ≠ real 0.1). The `_apply_ica` offline↔online parity test doesn't
  exercise HP, so the mismatch was harmless. Two HP-cutoff tests (`test_lowfreq_attenuated`,
  the drift half of `test_apply_filter_passes_high_frequencies`) were designed for the 1.0 Hz
  cutoff and assumed sub-1Hz heavy attenuation — invalid at 0.1 Hz, and time-domain probing
  below 0.1 Hz needs impractically long signals. Reworked `test_lowfreq_attenuated` to check
  the HP **SOS frequency response** (`sosfreqz`) at 0.01 Hz (a decade below cutoff → <−40 dB),
  mirroring the existing LP response test; trimmed the passing-only assertion in the other.
- Removed three now-obsolete highpass config-validation tests (method/extra-key/non-positive);
  added `TestHighpass` pin + extended `TestGetSettings`. Full suite green (448 passed, 1 skipped).

### Step 5b — Remove `resample_filter_stage` + the `late` code path ✅ DONE
- Reverses the earlier deferral. Deleted the `resample_filter_stage` field from the schema and
  all YAMLs (no constant — there is now a single fixed ordering, not a toggle).
- Offline: removed the `_stage` property; `run_step1a_filter` always does LP+resample on raw,
  `run_step2_apply_and_save` no longer has a late LP+resample branch. (`_resample` stays a
  general Raw/Epochs helper — still unit-tested directly.)
- Online: removed the `_resample_filter_stage` read/validation and the `process_batch`
  early/late branch (kept the early ordering); dropped `stage=` from the ready-log. The
  online preprocessor now reads **nothing** from `preprocessing_settings` (vestigial arg).
- Tests: removed the late-variant ordering test, the both-variants test, the invalid-stage
  tests (online + settings-manager), and the offline late-path tests; renamed the survivors
  (`TestPipelineOrdering`, `test_step1a_resamples_raw_to_target`). Full suite green
  (441 passed, 1 skipped).

### Step 6 — `epochs` (offline only) ✅ DONE
- Added `EPOCH_TMIN = -0.2`, `EPOCH_TMAX = 1.0`, `EPOCH_BASELINE = None` (+ module-level assert
  `EPOCH_TMIN < EPOCH_TMAX`, replacing the old `EpochSettings._tmin_below_tmax` validator).
  Offline `_epoch` reads them; removed `EpochSettings` + field; re-attached to `_hardcoded_recipe()`.
  Stripped `epochs` from the 3 tracked YAMLs + 5 debug snapshots (all `-0.2/1.0/null` — no divergence).
- **Fixture wrinkle (like highpass):** the offline fixtures used `tmin=-0.1, tmax=0.5,
  baseline=[None,0]` (small/fast); the real values are `-0.2/1.0/None`. The 10 s synthetic raw
  (events at 2–7 s) fits the larger window fine, and `baseline=None` actually *removes* the
  baseline-corrected-ICA warnings (18→4 warnings). Reworked `test_baseline_none_supported` →
  `test_baseline_is_none` (no longer mutates settings); removed the obsolete
  `test_epoch_baseline_is_tuple` + `test_rejects_tmin_above_tmax`; added `TestEpochs` pin +
  extended `TestGetSettings`. Full suite green (443 passed, 1 skipped).

### Step 7 — `channel_hygiene` (offline only) ✅ DONE
- Added `CHANNEL_DROP_EMG`, `CHANNEL_RENAME_HEGOC_TO_HEOG`, `CHANNEL_MONTAGE_NAME`,
  `CHANNEL_AFZ_CASE_FIX`; offline `_channel_hygiene` reads them (boolean guards kept, so a dev
  can disable a step by flipping the constant). Removed `ChannelHygieneSettings` + field;
  re-attached to `_hardcoded_recipe()`. Stripped from 3 tracked YAMLs + 5 debug snapshots
  (all identical `true/true/easycap-M1/true` — no divergence).
- Rewrote `test_hygiene_skipped_when_disabled` to monkeypatch `CHANNEL_DROP_EMG=False` (the
  disable path is no longer config-reachable); added `TestChannelHygiene` pin + extended
  `TestGetSettings`. Full suite green (447 passed, 1 skipped). After this, the config's
  `preprocessing:` block holds only `random_state` (model field) + `ica`.

### Step 8 — `ica` (+ iclabel) (offline only; most complex)
- Add `ICA_METHOD`, `ICA_EXTENDED`, `ICA_N_COMPONENTS`, `ICA_FIT_L_FREQ`, `ICLABEL_ENABLED`,
  `ICLABEL_DROP_LABELS`.
- Move `_ICLABEL_VALID_LABELS` from `config_models.py` into the constants module, plus a
  module-level assert that `ICLABEL_DROP_LABELS` ⊆ valid labels (preserves the typo-guard).
- Carry over the two `# TODO(decision)` comments (ICA fit-copy method; ICLabel band mismatch).
- Offline `_fit_ica` / `_iclabel_suggest` read the constants; remove `ICASettings` /
  `IclabelSettings` from `config_models.py`.
- ICA fit keeps reading `random_state` from the settings dict (it stays a top-level knob).

### Step 9 — Cleanup: collapse the preprocessing plumbing
With `resample_filter_stage` now removed (Step 5b), the preprocessing config holds only
`random_state` (+ whatever blocks remain un-migrated at this point — after Steps 6–8, none):
- `PreprocessingSettings` is reduced to `random_state` only (all sub-models gone). The
  `preprocessing:` YAML block disappears (or holds nothing).
- **Online side:** `OnlinePreprocessor` already reads nothing from `preprocessing_settings`
  (Step 5b) — drop the arg from its constructor + `_validate_inputs`, and stop passing it in
  `session.py`. This makes the online phase fully config-independent and kills the
  `session.py:213` footgun (Phase 2 reading live config that may differ from training).
- **Offline side:** `OfflinePreprocessor` still needs `random_state` for ICA — switch its
  constructor to take `random_state: int` directly instead of the settings dict.
- `SettingsManager.get_preprocessing_params()` shrinks to `{random_state}` (or is folded into
  a `get_random_state()`); `get_settings()` keeps assembling the full recipe via
  `_hardcoded_recipe()` for the UI.
- Docs: `CLAUDE.md` Config Schema section + `docs/architecture/backend_architecture.md`
  (+ note in `docs/old/preprocessing_migration_*`).

> **End of the minimizing-settings feature.** The `preprocessing:` config block is gone; the
> recipe is hardcoded constants imported by both phases, and the online phase no longer reads
> live config. Step 10 is a follow-on UI addition.

### Step 10 — Preprocessing-stages overview on the Node 3 "Ready" page (separate, after the above)
- Add a compact read-only **"Preprocessing stages" overview** to
  `PreprocessingView._build_ready_page()` (preprocessing_view.py:388), shown before the
  operator starts preprocessing.
- Render it from `preprocessing_constants.py` (never the config) — an ordered stage list
  reflecting the actual (single, fixed) pipeline:
  channel hygiene → highpass → notch → LP + resample → interpolate/avg-ref → ICA + ICLabel →
  epoching, with the key values per stage.
- Pure UI addition; no backend change. Optionally add a headless render test under
  `tests/frontend/`.

---

## Notes / risks

- **`debug_snapshots/*` configs are git-ignored** (CLAUDE.md) and re-seeded via
  `scripts/demo_seed_debug_snapshots.py`. They are hand-stripped per block (not committed) so
  they keep loading under the shrinking schema; the 3 tracked YAMLs are the committed edits.
  Before stripping a block, verify its value matches the constant (a mismatch means the
  cached artifact used a different recipe).
- **Behavior parity** is the acceptance bar for every step: full `pytest tests/` green, and
  at the end a `demo_seed_debug_snapshots.py` retrain + offline inference check producing
  identical artifacts. The per-step value-pin test catches transcription errors.
- **Ordering rationale:** `lowpass` was the pilot (Step 2) — truly inert, every test already
  used 40.0/iir. Then the remaining online-read scalars (steps 3–5), then the
  `resample_filter_stage` toggle + `late` path removal (Step 5b), then offline-only blocks
  (6–8), then the cleanup (9). Steps 1–9 complete the minimizing feature; the operator-facing
  overview (step 10) is a separate follow-on.

## Resulting `experiment_config.yaml` (target, after step 9)

```yaml
experiment_info:
  name: Reactivation_Study_V1

random_state: 42

decoders:
  model: LDA
  params:
    solver: lsqr
    shrinkage: auto
  scale_method: standard
  cv:
    k: 5
  tasks:
    - name: red decoder
      pos_labels: [red]
      neg_labels: [green, yellow]
    - name: yellow decoder
      pos_labels: [yellow]
      neg_labels: [green, red]

markers_mapping:
  events:
    - id: 11
      name: red
    - id: 12
      name: green
    - id: 13
      name: yellow
```
