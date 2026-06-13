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
| `final_resample.target_rate` | yes (`_resample`) | yes | scalar | |
| `notch.freq` | yes | yes | scalar, `None` disables | |
| `highpass` | yes | yes | l_freq + method | |
| `epochs` | yes | **no** | tmin/tmax/baseline; baked into matrices offline | |
| `channel_hygiene` | yes | **no** | 4 flags | |
| `ica` (+ iclabel) | yes | **no** | most complex; touches `random_state` | |
| `resample_filter_stage` | yes (`_stage`) | yes | scalar, but **gates two pipeline code paths** | **deferred** |

The online preprocessor reads `lowpass`/`final_resample`/`notch`/`highpass`/
`resample_filter_stage`; `epochs`, `channel_hygiene`, and `ica` touch the offline
preprocessor only.

> **Deferred — `resample_filter_stage`.** Unlike the other blocks it selects between two
> live code paths (`early` vs `late` LP+resample ordering) in both preprocessors, with a
> dedicated "late"-variant test suite. Per decision (2026-06-13) the early/late toggle stays
> configurable for now. Consequence: the schema's `PreprocessingSettings` and both
> preprocessors must keep reading the stage flag from settings, so the final cleanup (Step 9)
> can only *shrink* the preprocessing config down to `resample_filter_stage`, not remove it.

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
5. Add/extend `tests/core/test_preprocessing_constants.py` to pin the new constant(s) to
   their previously-shipped value.
6. Run `pytest tests/`; confirm green.

> Note: two debug snapshots (`colors_window50`, `colors_window50_restclass`) already fail to
> load under the current schema due to experimental **decoder** keys (`feature_window_ms`,
> `rest_class`) that predate this work — unrelated to the preprocessing migration.

`get_preprocessing_params()` keeps returning a (shrinking) dict throughout; `random_state`
stays inside it until the final cleanup step.

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

### Step 3 — `final_resample.target_rate`
- Add `FINAL_RESAMPLE_RATE = 100`. Offline `_resample` + online `__init__`. (Both.)
- Note: the online test suite parametrizes `target_rate` (e.g. 256 for the non-integer
  decimation-error test) — those tests must construct the preprocessor with a forced rate
  some other way, or move to asserting the hardcoded ratio.

### Step 4 — `notch.freq`
- Add `NOTCH_FREQ = 50.0` (document `None` disables). Offline `_notch` + online `__init__`. (Both.)

### Step 5 — `highpass`
- Add `HIGHPASS_L_FREQ = 0.1`, `HIGHPASS_METHOD = "iir"`. Offline `_highpass` + online `__init__`. (Both.)
- Note: test fixtures use `l_freq = 1.0` (≠ the real 0.1) — hardcoding shifts their filter;
  check the offline↔online parity tests still hold (they compare the two sides, which both
  move to 0.1 together, so parity is preserved) and update any absolute-value asserts.

### Step 6 — `epochs` (offline only)
- Add `EPOCH_TMIN = -0.2`, `EPOCH_TMAX = 1.0`, `EPOCH_BASELINE = None`
  (+ module-level assert `EPOCH_TMIN < EPOCH_TMAX`). Offline `_epoch`.

### Step 7 — `channel_hygiene` (offline only)
- Add `CHANNEL_DROP_EMG`, `CHANNEL_RENAME_HEGOC_TO_HEOG`, `CHANNEL_MONTAGE_NAME`,
  `CHANNEL_AFZ_CASE_FIX`. Offline `_channel_hygiene`.

### Step 8 — `ica` (+ iclabel) (offline only; most complex)
- Add `ICA_METHOD`, `ICA_EXTENDED`, `ICA_N_COMPONENTS`, `ICA_FIT_L_FREQ`, `ICLABEL_ENABLED`,
  `ICLABEL_DROP_LABELS`.
- Move `_ICLABEL_VALID_LABELS` from `config_models.py` into the constants module, plus a
  module-level assert that `ICLABEL_DROP_LABELS` ⊆ valid labels (preserves the typo-guard).
- Carry over the two `# TODO(decision)` comments (ICA fit-copy method; ICLabel band mismatch).
- Offline `_fit_ica` / `_iclabel_suggest` read the constants; remove `ICASettings` /
  `IclabelSettings` from `config_models.py`.
- ICA fit keeps reading `random_state` from the settings dict (it stays a top-level knob).

### Step 9 — Cleanup: shrink the preprocessing plumbing to just `resample_filter_stage`
With `resample_filter_stage` deferred (still configurable), the preprocessing config can't be
removed outright — but everything else collapses:
- `PreprocessingSettings` is reduced to `random_state` + `resample_filter_stage` only (all
  sub-models gone). The `preprocessing:` YAML block shrinks to a single `resample_filter_stage`
  line.
- Both preprocessors still receive a (now tiny) `preprocessing_settings` dict, read **only**
  the stage flag from it, and import everything else from `preprocessing_constants`. So the
  constructor signatures keep the dict arg; `OfflinePreprocessor` still pulls `random_state`
  from it (no signature change needed yet).
- `SettingsManager.get_preprocessing_params()` stays (returns the small dict). `random_state`
  remains a top-level config knob, so ICA keeps reading it from the dict.
- Docs: `CLAUDE.md` Config Schema section + `docs/architecture/backend_architecture.md`
  (+ note in `docs/old/preprocessing_migration_*`).
- **Full removal of the `preprocessing` plumbing (drop the dict arg, make the online phase
  config-independent, kill the `session.py:213` footgun) is blocked until the
  `resample_filter_stage` early/late decision is made.** Track that as a follow-up.

> **End of the minimizing-settings feature (modulo the deferred `resample_filter_stage`).**
> The config's preprocessing block is down to one line; the rest of the recipe is hardcoded
> constants imported by both phases. Step 10 is a follow-on UI addition.

### Step 10 — Preprocessing-stages overview on the Node 3 "Ready" page (separate, after the above)
- Add a compact read-only **"Preprocessing stages" overview** to
  `PreprocessingView._build_ready_page()` (preprocessing_view.py:388), shown before the
  operator starts preprocessing.
- Render it from `preprocessing_constants.py` (never the config) — an ordered stage list
  reflecting the actual pipeline:
  channel hygiene → highpass → notch → [if `early`: LP + resample] → interpolate/avg-ref →
  ICA + ICLabel → epoching → [if `late`: LP + resample], with the key values per stage.
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
  used 40.0/iir. Then the remaining online-read scalars (steps 3–5), then offline-only blocks
  (6–8), then the cleanup (9). Steps 1–9 complete the minimizing feature (modulo the deferred
  `resample_filter_stage`); the operator-facing overview (step 10) is a separate follow-on.
- **Deferred `resample_filter_stage`** (see the block table) keeps the `preprocessing:` block
  alive as a single line and blocks the full removal of the preprocessing plumbing.

## Resulting `experiment_config.yaml` (target, after step 9)

```yaml
experiment_info:
  name: Reactivation_Study_V1

random_state: 42

preprocessing:
  resample_filter_stage: early   # deferred — still selects early/late pipeline ordering

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
