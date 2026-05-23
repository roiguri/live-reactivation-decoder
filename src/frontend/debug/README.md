# Debug mode

Developer affordance for iterating on Phase 1 UI screens without
re-running the full offline pipeline every time. Production
`python -m frontend.main` is **byte-for-byte unaffected** — it never
imports anything from this package.

## What it does

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

Run the seeder against a marker-bearing recording (the test-set
recording is the default fixture):

```bash
cd online_decoder
python -m scripts.demo_seed_debug_snapshots \
    --config experiment_config.yaml \
    --data ../data/new_experiment/test_set/subject_102_quarter \
    --output debug_snapshots
```

Writes:

```
debug_snapshots/
├── preproc_done.joblib — orchestrator state after run_step2_apply_and_save()
├── eval_done.joblib    — orchestrator state after run_evaluation()
└── train_done.joblib   — orchestrator state after run_training()
```

`debug_snapshots/` is **git-ignored**; regenerate after any
pipeline/schema change.

The `preproc_done.joblib` snapshot lets the walkthrough skip
preprocessing entirely (including the bad-channel + ICA-review MNE
windows). It carries the full `OfflinePreprocessor` instance with
`.raw` stripped (we don't need the full-rate signal downstream;
keeping it would inflate the snapshot to hundreds of MB).

## Daily use

```bash
cd online_decoder
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
| 1 | Load config                | Build `AppSession(experiment_config.yaml)`; populate Settings  |
| 2 | Pick output directory      | Point at `debug_snapshots/`                                    |
| 3 | Continue → Load Data       | Fire Settings' Continue (emits `session_ready`)                |
| 4 | Pick demo data folder      | Default: `../data/new_experiment/test_set/subject_102_quarter` |
| 5 | Skip data load             | Emit `data_loaded` directly — no `LoadWorker`, no real load    |
| 6 | Skip preprocessing         | Load `preproc_done.joblib`; jump to complete page              |
| 7 | Continue → Evaluation      | Fire `preprocessing_complete`                                  |
| 8 | Skip evaluation            | Load `eval_done.joblib`; populate Evaluation tabs              |
| 9 | Continue → Train           | Fire `evaluation_complete(timepoint)`                          |
| 10 | Skip training             | Load `train_done.joblib`; show the Train view                  |

**Reset** rewinds the step counter to 0 + clears the file-picker
visuals. There is no Prev — irreversible signal emissions can't be
cleanly undone; relaunch the app for a fully clean state.

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
