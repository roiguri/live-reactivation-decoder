# Debug mode

Developer affordance for iterating on UI screens without re-running
the full offline pipeline every time. Production
`python -m frontend.main` is **byte-for-byte unaffected** — it never
imports anything from this package.

Two entry points:

| Command                                              | Opens                                       |
|------------------------------------------------------|---------------------------------------------|
| `python -m frontend.debug.main`                      | Phase 1 walkthrough (Next-driven)            |
| `python -m frontend.debug.main --phase2`             | Phase 2 screen directly (skip Phase 1)       |

## What it does (Phase 1 walkthrough)

Boots the app already populated with realistic state from a previous
real pipeline run, then exposes keyboard shortcuts to jump straight to
any of the downstream Phase 1 screens (Preprocessing / Evaluation /
Train) so you can iterate on their layout and behaviour without
sitting through preprocessing + ICA review + evaluation each time.

The state comes from on-disk **snapshots** — small joblib files
written by a one-shot seeder script that runs the real pipeline
non-interactively. Each snapshot captures a phase boundary
(`eval_done`, `train_done`); the debug screen restores them on
demand.

## One-time setup

Debug mode trains 6 binary one-vs-rest decoders (3 colors + 3 scenes)
against the functional-localizer split of subject 101. Two artifacts
must exist before running the seeder, both **un-tracked in git**:

1. `data/subject_<id>/split/functional_localizer/` — the FL-only
   BrainVision triplet produced by:

   ```bash
   python scripts/split_subject_by_phase.py --subject 101
   ```

   (requires `data/subject_<id>/raw_data/EEG/experiment/` from your
   own copy of the recordings).

2. `debug_snapshots/experiment_config.yaml` — the debug-only config.
   See the "Debug config" section below; copy the recipe there into
   the file. The seeder defaults `--config` to this path.

Then run the seeder:

```bash
python -m scripts.demo_seed_debug_snapshots \
    --data data/subject_101/split/functional_localizer
```

Writes:

```
debug_snapshots/
├── experiment_config.yaml — 6-decoder debug config (you create this)
├── preproc_done.joblib — orchestrator state after run_step2_apply_and_save()
├── eval_done.joblib    — orchestrator state after run_evaluation()
└── train_done.joblib   — orchestrator state after run_training()
```

`debug_snapshots/` is **git-ignored**; regenerate after any
pipeline/schema change.

## Debug config

`debug_snapshots/experiment_config.yaml` is the 6-decoder development
config. It is intentionally **not tracked in git** — `debug_snapshots/`
is `.gitignore`d, so each developer maintains their own copy. Keep the
`preprocessing:` block in sync with the prod `experiment_config.yaml`
when the schema changes.

Two extensions over the prod config:

- **`decoders.tasks`** — 6 binary one-vs-rest decoders with
  within-modality negatives:

  | Task name              | pos                | neg                              |
  |------------------------|--------------------|----------------------------------|
  | `red decoder`          | `red`         | `green`, `yellow`      |
  | `green decoder`        | `green`       | `red`, `yellow`        |
  | `yellow decoder`       | `yellow`      | `red`, `green`         |
  | `living_room decoder`  | `living_room` | `bathroom`, `kitchen`  |
  | `bathroom decoder`     | `bathroom`    | `living_room`, `kitchen` |
  | `kitchen decoder`      | `kitchen`     | `living_room`, `bathroom` |

- **`markers_mapping.events`** — full catalog of every trigger code
  observed in subject 101's recording (BMR Data Specification names).
  Codes 41+ (binding/test/partial) are inert when used against the FL
  split since they don't appear there, but get named and epoched if
  the same config is later used against the `task/` split.

The `preproc_done.joblib` snapshot lets the walkthrough skip
preprocessing entirely (including the bad-channel + ICA-review MNE
windows). It carries the full `OfflinePreprocessor` instance with
`.raw` stripped (we don't need the full-rate signal downstream;
keeping it would inflate the snapshot to hundreds of MB).

## Daily use

```bash
PYTHONPATH=src python -m frontend.debug.main
```

The workspace header shows `[DEBUG] {node title}` so the mode is
always visible.

## Walkthrough

The app boots on an empty Settings screen. A toolbar at the top of
the workspace card shows `Step N/10: <name>  [Next →] [Reset]`.

Press **Next** (or **Ctrl+Right**) to fire the next user action.
Each press is instant — slow compute (data load, preprocessing,
evaluation, training) is faked by emitting completion signals
directly or loading on-disk snapshots.

| # | Step                       | What it does                                                  |
|---|----------------------------|----------------------------------------------------------------|
| 1 | Load config                | Build `AppSession(debug_snapshots/experiment_config.yaml)`; populate Settings |
| 2 | Pick output directory      | Point at `debug_snapshots/`                                    |
| 3 | Continue → Load Data       | Fire Settings' Continue (emits `session_ready`)                |
| 4 | Pick demo data folder      | Default: `data/subject_101/split/functional_localizer`         |
| 5 | Skip data load             | Emit `data_loaded` directly — no `LoadWorker`, no real load    |
| 6 | Skip preprocessing         | Load `preproc_done.joblib`; jump to complete page              |
| 7 | Continue → Evaluation      | Fire `preprocessing_complete`                                  |
| 8 | Skip evaluation            | Load `eval_done.joblib`; populate Evaluation tabs              |
| 9 | Continue → Train           | Fire `evaluation_complete(timepoint)`                          |
| 10 | Skip training             | Load `train_done.joblib`; show the Train view                  |

**Reset** rewinds the step counter to 0 + clears the file-picker
visuals. There is no Prev — irreversible signal emissions can't be
cleanly undone; relaunch the app for a fully clean state.

## Phase 2 quick-jump

```bash
PYTHONPATH=src python -m frontend.debug.main --phase2
```

Skips Phase 1 entirely. Builds a real `AppSession` from
`debug_snapshots/experiment_config.yaml`, points `Phase2Screen` at
`debug_snapshots/models/decoder_pipeline.joblib`, and shows it.

Use this when iterating on Phase 2 layout, chart rendering, or the
Start/Halt + latency wiring as those land — the round-trip per
change is ~1 s instead of the full Phase 1 click-through.

Defaults (in `phase2_screen_debug.py`):

| Default                                             | Used for                                  |
|-----------------------------------------------------|-------------------------------------------|
| `debug_snapshots/experiment_config.yaml`            | `AppSession` config load                  |
| `debug_snapshots/models/decoder_pipeline.joblib`    | Phase 2's `decoder_pipeline_path`         |

The pipeline file is written by Phase 1's train snapshot path
(`scripts/demo_seed_debug_snapshots.py`). If the file is missing, the
shell still opens — the artifact is loaded later by
`build_live_stream_session(...)` when live inference starts.

## Troubleshooting

- **"Snapshot not found" QMessageBox** → run the seeder.
- **Snapshot fails to load after a code/schema change** → re-run the
  seeder. The joblib format is sensitive to attribute renames on the
  orchestrator.
- **App crashes on a downstream view** → the view's `_on_*_done` slot
  changed shape since the snapshot was captured; regenerate snapshot
  or fix the slot.

## Out of scope (deferred)

- **`_raw` in snapshots** — needed if we ever want to skip the
  ~1-2 min data-load step. A full-rate raw is large (~400 MB at
  5 kHz × 5 min × 64 ch float32); a smaller post-resample variant
  (~8 MB) would need to be captured after Step 1A. Skipped until the
  load wait proves annoying in daily use.
- **"Save current state" affordance inside debug mode** — a button
  in the running app that captures a snapshot at the current node
  rather than requiring a fresh seeder run.

## Production unaffected

Nothing under `src/frontend/debug/` is imported by `frontend.main`.
Verify with:

```bash
git grep "frontend.debug" online_decoder/src | grep -v "src/frontend/debug"
```

It should return zero hits in production source.
