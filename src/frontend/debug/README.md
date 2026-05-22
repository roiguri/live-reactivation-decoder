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

## Keyboard map

| Shortcut       | Jumps to                  | Loads from                     |
|----------------|---------------------------|--------------------------------|
| Ctrl+Shift+3   | Preprocessing — Ready page | nothing (live run from there) |
| Ctrl+Shift+4   | Evaluation — Results       | `eval_done.joblib`             |
| Ctrl+Shift+5   | Train view                 | `train_done.joblib`            |

Jumps bypass the journey-panel node-by-node animation; the trail UI
on the right may stay on Node 1 visually — harmless.

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
