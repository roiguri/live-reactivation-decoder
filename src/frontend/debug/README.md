# Debug mode

Developer affordance for iterating on UI screens without re-running
the full offline pipeline every time. Production
`python -m frontend.main` is **byte-for-byte unaffected** — it never
imports anything from this package.

Both entry points are driven by a **debug profile** — a named,
self-contained scenario under `debug_snapshots/`. See
[docs/features/debug_profiles.md](../../../docs/features/debug_profiles.md)
for the design; this README is the day-to-day usage guide.

The default entry point **boots on the welcome hub** (the production
`frontend.screens.launch_screen.LaunchScreen`, reused verbatim, plus a
**debug toolbar**). The hub routes to both debug screens for the selected
profile:

- **`Next →`** (also `Ctrl+Right`, and the "Start New Training" card) → the
  Phase 1 walkthrough.
- **`Live →`** (also the "Open Live from Existing Output" card) → the Phase 2
  live screen.

`--phase2` is the direct/separate access: it opens the Phase 2 live screen
immediately, skipping the hub (its `Reset` returns to the hub).

| Command                                                       | Opens                                  |
|---------------------------------------------------------------|----------------------------------------|
| `python -m frontend.debug.main`                               | Welcome hub (default profile)          |
| `python -m frontend.debug.main --profile <name>`              | Welcome hub for `<name>`               |
| `python -m frontend.debug.main --phase2`                      | Phase 2 live screen directly (default) |
| `python -m frontend.debug.main --profile <name> --phase2`     | Phase 2 live screen for `<name>`       |
| `python -m frontend.debug.main --list-profiles`               | Print discovered profiles and exit     |

`--config` / `--data` override the profile's config / raw-data path for
a one-off run.

## Profiles

A profile is a directory `debug_snapshots/<name>/` bundling everything
needed to reproduce **and** run one scenario:

```
debug_snapshots/<name>/
├── manifest.yaml          ← source of truth: name + config + raw_data_dir
├── experiment_config.yaml ← the config, copied in (self-contained)
├── preproc_done.joblib    ← orchestrator state after run_step2_apply_and_save()
├── eval_done.joblib       ← orchestrator state after run_evaluation()
├── train_done.joblib      ← orchestrator state after run_training()
├── models/decoder_pipeline.joblib   ← Phase 2 artifact
└── epochs/                ← saved epochs from the run
```

`manifest.yaml` is minimal:

```yaml
name: default
config: experiment_config.yaml                     # relative to this dir
raw_data_dir: C:/dev/.../data/split/functional_localizer   # path only
```

`raw_data_dir` is a **path only** (not the data) — enough to re-seed, or
to know what to replay via `scripts/replay_vhdr_to_lsl.py`. Profiles are
**discovered** by listing subdirectories that contain a `manifest.yaml`;
there is no central registry.

**Profile selection** when `--profile` is omitted: a profile literally
named `default`, else the sole profile if only one exists, else an error
listing the choices.

`debug_snapshots/` is **git-ignored**; each developer maintains their own
profiles and regenerates after any pipeline/schema change.

## One-time setup

Debug mode trains binary one-vs-rest decoders against the
functional-localizer split of a subject. The raw recording must exist
(un-tracked in git) — e.g. the FL-only BrainVision triplet produced by:

```bash
python scripts/split_subject_by_phase.py --subject 101
```

Then **seed a profile** (bootstrap mode — copies the config in and records
the raw-data path):

```bash
python -m scripts.demo_seed_debug_snapshots \
    --profile default \
    --config experiment_config.yaml \
    --data   data/split/functional_localizer
```

To **re-seed** after a schema/pipeline change, the manifest already holds
the config + data, so just:

```bash
python -m scripts.demo_seed_debug_snapshots --profile default
```

(`--config` / `--data` still override if you pass them.)

The seeder runs the pipeline non-interactively: `set_bad_channels([])`
(no operator-marked bads) and `run_step2_apply_and_save(suggested)`
(accepts whatever ICLabel flagged). The training timepoint is
`eval_result["suggested_timepoint"]`.

## Debug config

Each profile carries its **own** `experiment_config.yaml` (copied in by
the seeder), so different profiles can vary settings, decoders, and the
trained artifact independently. Keep a profile's `preprocessing:` block in
sync with the prod `experiment_config.yaml` when the schema changes, then
re-seed.

Two extensions over a minimal prod config are typical:

- **`decoders.tasks`** — binary one-vs-rest decoders with within-modality
  negatives (e.g. `red` vs `green`/`yellow`).
- **`markers_mapping.events`** — the **full** trigger catalog, so every
  trigger renders a named marker on the live Phase 2 screen. Codes absent
  from the FL split (binding/test/partial, 41+) are named but not epoched.

The `preproc_done.joblib` snapshot lets the walkthrough skip preprocessing
entirely (including the bad-channel + ICA-review MNE windows). It carries
the full `OfflinePreprocessor` instance with `.raw` stripped (the
full-rate signal isn't needed downstream; keeping it would inflate the
snapshot to hundreds of MB).

## Daily use

```bash
PYTHONPATH=src python -m frontend.debug.main                  # default profile
PYTHONPATH=src python -m frontend.debug.main --profile other  # a named profile
```

The workspace header shows `[DEBUG] {node title}` so the mode is always
visible.

## Walkthrough

The app boots on an empty Settings screen. A toolbar at the top of the
workspace card shows `Step N/10: <name>  [Next →] [Reset]`.

Press **Next** (or **Ctrl+Right**) to fire the next user action. Each
press is instant — slow compute (data load, preprocessing, evaluation,
training) is faked by emitting completion signals directly or loading the
profile's on-disk snapshots.

| # | Step                       | What it does                                                   |
|---|----------------------------|----------------------------------------------------------------|
| 1 | Load config                | Build `AppSession(profile.config_path)`; populate Settings     |
| 2 | Pick output directory      | Point at the profile directory                                 |
| 3 | Continue → Load Data       | Fire Settings' Continue (emits `session_ready`)                |
| 4 | Pick demo data folder      | The profile's `raw_data_dir`                                   |
| 5 | Skip data load             | Emit `data_loaded` directly — no `LoadWorker`, no real load    |
| 6 | Skip preprocessing         | Load `preproc_done.joblib`; jump to complete page              |
| 7 | Continue → Evaluation      | Fire `preprocessing_complete`                                  |
| 8 | Skip evaluation            | Load `eval_done.joblib`; populate Evaluation tabs              |
| 9 | Continue → Train           | Fire `evaluation_complete(timepoint)`                          |
| 10 | Skip training             | Load `train_done.joblib`; show the Train view                  |

**Reset** rewinds the step counter to 0 + clears the file-picker visuals.
There is no Prev — irreversible signal emissions can't be cleanly undone;
relaunch the app for a fully clean state.

**Live →** (in the debug toolbar) hops straight to the Phase 2 live screen
for the current profile at any point in the walkthrough — no need to click
to the end and Go Live. It reuses the same builder as `--phase2`
(`build_debug_phase2`), building a fresh session from the profile's config +
pipeline, so it works regardless of how far you've stepped.

## Phase 2 quick-jump

```bash
PYTHONPATH=src python -m frontend.debug.main --phase2
PYTHONPATH=src python -m frontend.debug.main --profile <name> --phase2
```

Skips Phase 1 entirely. Builds a real `AppSession` from the profile's
config and points `Phase2Screen` at the profile's
`models/decoder_pipeline.joblib`.

The live screen carries the same **debug bar** as the welcome / Phase 1
screens. Its `Next →` is disabled (there's no step past live inference); its
`Reset` returns to the welcome hub (tearing down the live session + stream
source first), where you can head into Phase 1 (`Next →`) or back into a fresh
live screen (`Live →`).

Use this when iterating on Phase 2 layout, chart rendering, or the
Start/Halt + latency wiring — the round-trip per change is ~1 s instead of
the full Phase 1 click-through.

The pipeline file is written by the seeder's train step. If it's missing,
the shell still opens — the artifact is loaded later by
`build_live_stream_session(...)` when live inference starts.

## Troubleshooting

- **"No debug profiles" / "No profile '<name>'"** → run the seeder (see
  One-time setup), or check `--list-profiles`.
- **"Snapshot not found" QMessageBox** → the profile dir is missing a
  `*_done.joblib`; re-run the seeder for that profile.
- **Snapshot fails to load after a code/schema change** → re-seed. The
  joblib format is sensitive to attribute renames on the orchestrator.
- **App crashes on a downstream view** → the view's `_on_*_done` slot
  changed shape since the snapshot was captured; re-seed or fix the slot.

## Out of scope (deferred)

- **`_raw` in snapshots** — needed if we ever want to skip the ~1-2 min
  data-load step. A full-rate raw is large (~400 MB at 5 kHz × 5 min ×
  64 ch float32); a smaller post-resample variant (~8 MB) would need to be
  captured after Step 1A. Skipped until the load wait proves annoying.
- **"Save current state" affordance inside debug mode** — a button in the
  running app that captures a snapshot at the current node rather than
  requiring a fresh seeder run.
- **Granular preprocessing steps** — split the single "Skip preprocessing"
  step into per-stage snapshots (1A filter → 1B ICA fit/review → 2 apply).
  See the deferred bonus in
  [docs/features/debug_profiles.md](../../../docs/features/debug_profiles.md).

## Production unaffected

Nothing under `src/frontend/debug/` is imported by `frontend.main`.
Verify with:

```bash
git grep "frontend.debug" online_decoder/src | grep -v "src/frontend/debug"
```

It should return zero hits in production source.
