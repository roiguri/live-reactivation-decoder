# Debug Profiles — `debug_snapshots/` as a multi-profile hub

Implements **Goal 17 (Debug Profiles)** from [Phase2_UI_Plan_M2.md](../plans/phase2_ui_plan_m2.md).

> **Status (2026-06-05): implemented + seeded.** Steps 1–6 below are done —
> `profiles.py` + tests, the profile-aware seeder, both debug entry points,
> the `main.py` CLI, and the docs. One profile (`default`) is seeded and
> verified (3-decoder Logistic, full event catalog). A second
> labeled-validation profile (for Goal 1) is **not yet seeded**. The
> deferred granular-preprocessing bonus is unstarted.

## Motivation

Today `debug_snapshots/` is a **single, flat** profile: one `experiment_config.yaml`,
three joblibs (`preproc_done` / `eval_done` / `train_done`), `models/`, `epochs/`,
with no record of *which raw recording* produced them. Both debug entry points hardcode
their paths:

- `phase1_screen_debug.py` — module-level `_DEFAULT_CONFIG` / `_DEFAULT_OUTPUT` /
  `_DEFAULT_DATA` / `_*_SNAPSHOT` constants.
- `phase2_screen_debug.py` — `build_debug_phase2()` accepts config + pipeline as args,
  but the CLI never exposes them.

Diagnosing the fidelity bug and validating the pipeline (M2 Goals 18 / 1) both need to
swap *settings*, *trained artifact*, and *replay recording* together, repeatably. This
turns `debug_snapshots/` into a hub of **named, self-contained, reproducible** profiles.

## Decisions

| Decision | Choice |
|----------|--------|
| Structure | **Flat, self-describing dirs** — `debug_snapshots/<name>/`, discovered by listing subdirs; each dir's `manifest.yaml` is the source of truth (no central registry). |
| Scope | **Both** the Phase 1 walkthrough and `--phase2` quick-jump share one `--profile` mechanism. |
| Migration | **Hard-cut + re-seed.** `debug_snapshots/` is git-ignored and per-dev, so nothing is lost in source control. No flat-layout fallback branches. |
| Bonus (granular preprocessing) | **Deferred** — kept as a stub at the end. |

## The profile directory contract

Each profile is a self-contained directory under `debug_snapshots/`. A minimal
`manifest.yaml` records what's needed to *reproduce* and *run* it:

```yaml
# debug_snapshots/subject102_quarter/manifest.yaml
name: subject102_quarter
config: experiment_config.yaml                            # copied INTO the dir (self-contained)
raw_data_dir: C:/dev/.../data/subject_102/split/quarter   # path only — for re-seed + replay
```

```
debug_snapshots/<name>/
├── manifest.yaml          ← source of truth (config ptr + raw-data path)
├── experiment_config.yaml ← copied in by the seeder
├── preproc_done.joblib    ┐
├── eval_done.joblib       ├ conventional names (not manifest fields)
├── train_done.joblib      ┘
├── models/decoder_pipeline.joblib   ← Phase 2 artifact (conventional path)
└── epochs/...
```

`raw_data_dir` is **path-only**: enough to re-seed, or to know what to replay via
`scripts/replay_vhdr_to_lsl.py`, without bloating the dir with raw EEG.

The snapshot filenames, `models/decoder_pipeline.joblib`, and `epochs/` are **conventions**
resolved by the loader — not manifest fields. Speculative fields (`stream_name`, `notes`,
`created_at`, an explicit `pipeline:` pointer) are intentionally **omitted**; add them only
when a concrete need appears.

## Implementation

### 1. New module — `src/frontend/debug/profiles.py`

Pure logic, no Qt. The single place that knows the directory contract.

- `@dataclass DebugProfile` — `name`, resolved absolute `config_path`, `raw_data_dir`,
  `pipeline_path`, `snapshot_paths` (`{preproc, eval, train}`).
- `list_profiles(root=Path("debug_snapshots")) -> list[str]` — subdirs containing a `manifest.yaml`.
- `load_profile(name, root) -> DebugProfile` — parse + validate the manifest, resolve paths.
- `resolve_profile(name=None, *, config=None, data=None, root) -> DebugProfile` — selection
  + CLI overrides (see §4).

A unit test (`tests/.../test_debug_profiles.py`) covers parse / discovery / missing-manifest
errors. (`src/` stays pytest-free per CLAUDE.md.)

### 2. Seeder — `scripts/demo_seed_debug_snapshots.py`

Becomes profile-aware. One flag, two modes:

- **Bootstrap a new profile:** `--profile <name> --config <c> --data <d>` → creates the dir,
  **copies the config in**, writes `manifest.yaml`, runs the real pipeline, writes the three
  joblibs + `models/` + `epochs/` **inside the profile dir**.
- **Re-seed an existing profile** (after a schema/pipeline change): `--profile <name>` → reads
  `manifest.yaml` for config + data, re-runs, overwrites the joblibs. `--config` / `--data`
  still override if passed.

The existing seeding logic (steps 1A → 1B → 2 → eval → train) is unchanged; only path
resolution + manifest write/read are added.

### 3. Both debug entry points consume a `DebugProfile`

- **Phase 2** — `build_debug_phase2(profile)` sources `config_path` + `pipeline_path` from the
  profile instead of the two module-level `_DEFAULT_*` constants.
- **Phase 1** — `DebugPhase1Screen(profile)` replaces the hardcoded `_DEFAULT_*` / `_*_SNAPSHOT`
  constants:
  - `_step_load_config` → `profile.config_path`
  - `_step_pick_output` → the profile dir
  - `_step_pick_data` → `profile.raw_data_dir` (the reproducibility payoff — the walkthrough
    picks *exactly* the recording the snapshots were built from)
  - `_step_skip_{preproc, eval, train}` → `profile.snapshot_paths[...]`

`snapshots.py` (save/restore) is unchanged — it already takes explicit paths; only its callers
move to profile-resolved paths.

### 4. CLI — `src/frontend/debug/main.py`

```bash
python -m frontend.debug.main --profile subject102_quarter            # Phase 1 walkthrough
python -m frontend.debug.main --profile subject102_quarter --phase2   # Phase 2 quick-jump
python -m frontend.debug.main --list-profiles                         # print discovered profiles
```

- `--profile <name>` — applies to both entry points; resolved via `profiles.resolve_profile`.
- `--list-profiles` — prints discovered profiles, exits.
- `--config` / `--data` — explicit overrides on top of a profile (one-off diagnostics).
- Default when `--profile` omitted: use a profile named `default` if present; else if exactly
  one profile exists, use it; else error and print the list.

### 5. Migration (hard-cut)

One-time local re-seed — `debug_snapshots/` is git-ignored, so nothing is lost upstream:

```bash
python -m scripts.demo_seed_debug_snapshots \
    --profile default \
    --config debug_snapshots/experiment_config.yaml \
    --data   data/subject_101/split/functional_localizer
```

Then verify: `--list-profiles`, `--profile default`, and `--profile default --phase2`.

### 6. Docs

- `src/frontend/debug/README.md` — rewrite setup / layout / daily-use / troubleshooting around profiles.
- `docs/Phase2_UI_Plan_M2.md` Goal 17 — check off the boxes this delivers.

## Implementation order

1. ✅ `profiles.py` + unit test (`tests/test_debug_profiles.py`, 14 tests)
2. ✅ Seeder refactor (bootstrap + re-seed, manifest write/read, config copy-in)
3. ✅ `build_debug_phase2(profile)`
4. ✅ `DebugPhase1Screen(profile)`
5. ✅ `main.py` CLI (`--profile`, `--list-profiles`, `--config`/`--data` overrides)
6. ✅ Seed `default`; profile discovered, artifact carries all 3 decoders
7. ✅ Docs (`src/frontend/debug/README.md` + this file)

The seeded `default` profile uses a 3-decoder (red/green/yellow) Logistic
baseline with the full event catalog. Re-running the GUI against it
(`--phase2`, plus an out-of-process replay) is left to the operator.

## Deferred — granular preprocessing steps (bonus)

Split the single "Skip preprocessing" walkthrough step into per-stage snapshot steps
(1A filter → 1B ICA fit/review → 2 apply), each its own joblib + Next button, so each
stage's UI can be iterated independently. Spec this after steps 1–6 land and we've felt out
what's missing.
