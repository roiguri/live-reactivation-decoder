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

| Block | OfflinePreprocessor | OnlinePreprocessor | Notes |
|---|---|---|---|
| `resample_filter_stage` | yes (`_stage`) | yes | scalar |
| `final_resample.target_rate` | yes (`_resample`) | yes | scalar |
| `notch.freq` | yes | yes | scalar, `None` disables |
| `highpass` | yes | yes | l_freq + method |
| `lowpass` | yes | yes | h_freq + method |
| `epochs` | yes | **no** | tmin/tmax/baseline; baked into matrices offline |
| `channel_hygiene` | yes | **no** | 4 flags |
| `ica` (+ iclabel) | yes | **no** | most complex; touches `random_state` |

The online preprocessor only reads the first five blocks — steps for `epochs`,
`channel_hygiene`, and `ica` touch the offline preprocessor only.

## The pattern (every block-migration step does exactly this)

1. Add this block's named constant(s) to `src/backend/core/preprocessing_constants.py`
   (Step 2 creates the module on the first block).
2. Replace the dict reads (`self.settings["x"]` / `preprocessing_settings["x"]`) with the
   new constant — in **both** preprocessors if the table above says both read it.
3. Remove that field's sub-model + field from `config_models.py` (`PreprocessingSettings`).
4. Delete the key from **all 8 YAML files** (mandatory — `extra="forbid"` rejects a
   leftover key, which also acts as the tripwire that catches a missed file).
5. Add/extend `tests/core/test_preprocessing_constants.py` to pin the new constant(s) to
   their previously-shipped value.
6. Run `pytest tests/`; confirm green.

`get_preprocessing_params()` keeps returning a (shrinking) dict throughout; `random_state`
stays inside it until the final cleanup step.

---

## Incremental steps

### Step 1 — Remove the SettingsView preprocessing card (UI only, independent)
- Delete `_build_preproc_section()` + its call, and the `pre = settings["preprocessing"]`
  branch of `_update_settings_display` (incl. the None-reset block).
- Leaves Setup + Model Evaluation cards untouched. No backend change; decouples the UI from
  every later step so block migrations never touch the frontend.

### Step 2 — Scaffolding + first block: `resample_filter_stage`
- Create `src/backend/core/preprocessing_constants.py` with a module docstring.
- Add `RESAMPLE_FILTER_STAGE = "early"` (carry over the early/late explanatory comment).
- Offline `_stage` property + online `_resample_filter_stage` read the constant.
- Remove the field from `PreprocessingSettings`; strip from the 8 YAMLs. (Both preprocessors.)
- Create `tests/core/test_preprocessing_constants.py` pinning the value.
- Establishes the per-block pattern end-to-end on the simplest field.

### Step 3 — `final_resample.target_rate`
- Add `FINAL_RESAMPLE_RATE = 100`. Offline `_resample` + online `__init__`. (Both.)

### Step 4 — `notch.freq`
- Add `NOTCH_FREQ = 50.0` (document `None` disables). Offline `_notch` + online `__init__`. (Both.)

### Step 5 — `highpass`
- Add `HIGHPASS_L_FREQ = 0.1`, `HIGHPASS_METHOD = "iir"`. Offline `_highpass` + online `__init__`. (Both.)

### Step 6 — `lowpass`
- Add `LOWPASS_H_FREQ = 40.0`, `LOWPASS_METHOD = "iir"`. Offline `_lowpass` + online `__init__`. (Both.)

### Step 7 — `epochs` (offline only)
- Add `EPOCH_TMIN = -0.2`, `EPOCH_TMAX = 1.0`, `EPOCH_BASELINE = None`
  (+ module-level assert `EPOCH_TMIN < EPOCH_TMAX`). Offline `_epoch`.

### Step 8 — `channel_hygiene` (offline only)
- Add `CHANNEL_DROP_EMG`, `CHANNEL_RENAME_HEGOC_TO_HEOG`, `CHANNEL_MONTAGE_NAME`,
  `CHANNEL_AFZ_CASE_FIX`. Offline `_channel_hygiene`.

### Step 9 — `ica` (+ iclabel) (offline only; most complex)
- Add `ICA_METHOD`, `ICA_EXTENDED`, `ICA_N_COMPONENTS`, `ICA_FIT_L_FREQ`, `ICLABEL_ENABLED`,
  `ICLABEL_DROP_LABELS`.
- Move `_ICLABEL_VALID_LABELS` from `config_models.py` into the constants module, plus a
  module-level assert that `ICLABEL_DROP_LABELS` ⊆ valid labels (preserves the typo-guard).
- Carry over the two `# TODO(decision)` comments (ICA fit-copy method; ICLabel band mismatch).
- Offline `_fit_ica` / `_iclabel_suggest` read the constants; remove `ICASettings` /
  `IclabelSettings` from `config_models.py`.
- ICA fit still reads `random_state` from the settings dict for now (final step rewires it).

### Step 10 — Final cleanup: remove the empty preprocessing plumbing
- `OfflinePreprocessor.__init__(data_dir, random_state: int, raw=None)` — drop the
  `preprocessing_settings` arg; `_fit_ica` uses `self._random_state`.
- `OnlinePreprocessor.__init__(online_state, input_sfreq=1000.0)` — drop the
  `preprocessing_settings` arg entirely (now fully config-independent).
- Delete the empty `PreprocessingSettings` and the `preprocessing` field from
  `ExperimentConfig`; drop `"preprocessing"` from `_propagate_random_state`.
- `SettingsManager`: remove `get_preprocessing_params()`; `get_settings()` drops the key;
  add `get_random_state()`.
- Callers: `orchestrator.py:131` passes `random_state=...`; `session.py` constructs
  `OnlinePreprocessor(online_state=artifact.online_state)`.
- Tests: update construction in `test_preprocessor.py`, `test_online_preprocessor.py`,
  `test_orchestrator.py`, `conftest.py`, `test_stream_worker.py`, `test_phase2_lifecycle.py`,
  `test_settings_manager.py`.
- Docs: `CLAUDE.md` Config Schema section + `docs/architecture/backend_architecture.md`
  (+ note in `docs/old/preprocessing_migration_*` if they describe the YAML schema).

> **End of the minimizing-settings feature.** The config is minimal, the recipe is fully
> hardcoded as constants, and both phases import them. Step 11 is a follow-on UI addition.

### Step 11 — Preprocessing-stages overview on the Node 3 "Ready" page (separate, after the above)
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
  `scripts/demo_seed_debug_snapshots.py`. Confirm whether each block step must hand-edit the
  5 snapshot YAMLs or whether updating the seed script's template + re-running it is the
  intended path. The two root YAMLs and `tests/data/sample_config.yaml` always need editing.
- **Behavior parity** is the acceptance bar for every step: full `pytest tests/` green, and
  at the end a `demo_seed_debug_snapshots.py` retrain + offline inference check producing
  identical artifacts. The per-step value-pin test catches transcription errors.
- **Ordering rationale:** simplest scalars first — also the only blocks the online
  preprocessor reads (steps 2–6) — then offline-only blocks (7–9), then the structural
  cleanup (10). Steps 1–10 complete the minimizing feature; the operator-facing overview
  (step 11) is a separate follow-on. Any single block can be the pilot if you'd rather start
  elsewhere (e.g. `lowpass`).

## Resulting `experiment_config.yaml` (target, after step 10)

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
