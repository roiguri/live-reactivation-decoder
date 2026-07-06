# Analysis notebooks

Notebooks that *analyze* the offline/online decoders (as opposed to the
`validate_*` unit notebooks one level up, which exercise a single backend unit).

Every notebook resolves an **analysis root**: a seeded `debug_snapshots/<name>/`
profile (`scripts/demo_seed_debug_snapshots.py` — see below), or any other
directory laid out per `backend.core.session_paths.SessionPaths`
(`experiment_config.yaml`, `models/decoder_pipeline.joblib`, `epochs/`).
A seeded snapshot is the intended input — self-contained, disposable, and
re-seedable, so you can try a different model/hyperparameters/pos-neg classes
and just re-run. From a root, `live_inference_epoched.ipynb` supports three
sources, **all three replaying raw EEG fresh** through `OnlinePreprocessor` +
whichever `LiveInferenceEngine` the notebook builds (`ctx.engine()` by
default, or `ctx.engine(alt_models)` for a different one) — so any
model/hyperparameters/pos-neg classes you try applies to every source:

- **`"fl"`** — replay the functional-localizer training recording, epoch the
  probability stream around stimulus markers, and compute metrics (in-sample;
  the honest performance number is the CV-AUC cell, not this P(t)).
- **`"encoding"`** — a leakage-free held-out test over the real-time task's
  couple-learning phase. Replays the task recording, epochs around each image
  onset, labeled by the marker's own category (`learning_<category>_NN` names
  the shown image directly, same principle as FL) — a same-modality
  perception sanity check, since a real image was actually on screen.
- **`"retrieval"`** — a leakage-free held-out test over the real-time task's
  retrieval phase. Replays the same task recording, epochs around each
  retrieval cue, labeled with the cued couple's true (encoded) category
  recovered from the encoding markers earlier in the same recording
  (`analysis_lib.task_labels`) — no separate behavioral log needed. This is
  the reactivation-from-memory question: no image is on screen, only recall.

`encoding` and `retrieval` both reuse the FL epoch window (`ctx.epoch_tmin`/
`ctx.epoch_tmax`, from `preprocessing_constants`) by default, so trajectories
across sources are directly comparable — even though the actual behavioral
event they're epoching (a flashed image vs. a several-second recall window) is
a different duration than that window.

**Why raw replay, not a saved live run's `predictions.csv`:** a saved
`phase2_live/<run>/predictions.csv` is frozen output from one specific,
already-trained decoder — you can't ask "what if" with it, and if that
decoder's live session predates a preprocessing fix (as happened once this
session — see the µV/volts note below), the saved predictions can be silently
wrong with no way to recover them. Replaying raw EEG fresh, through whichever
engine you choose, is the only way to get valid, experimentable results.

**The µV/volts unit contract:** `OnlinePreprocessor.process_batch` applies a
fixed µV→SI-volt scale (`LSL_TO_SI_SCALE`) up front, unconditionally — an
intrinsic property of the live LSL wire format, not something callers
configure. `streaming.load_recording` converts MNE's default SI-volts load
into µV (`get_data(units="uV")`) specifically so replay hands `process_batch`
what it now expects, mirroring `scripts/replay_vhdr_to_lsl.py`'s production
replay path. If a future backend change alters that contract, this is the one
function to update.

## `analysis_lib/` — the reusable core

The stable plumbing lives in an importable package so notebooks stay thin:

- `context.py` — `bootstrap()` (repo-root walk + `sys.path`) and
  `load_context(root)` → `AnalysisContext` (`SessionPaths`, settings, artifact,
  online preprocessor, inference engine, per-task decoding timepoints,
  discovered raw recording dirs, latest `phase2_live` run). Raw dirs come from
  a physical subfolder scan (`find_raw_dirs`) merged with anything a debug
  profile's `manifest.yaml` *references* without copying (`task_data_dir`) —
  the manifest only applies when `root` actually has one, so a plain output
  directory like `data/sub_001` is unaffected.
- `streaming.py` — `load_recording` (raw EEG in µV — see the unit-contract
  note above), `extract_markers`, `run_online_stream` (StreamWorker-style
  micro-batches), `make_epocher` (raw-sample-indexed epoching).
- `task_labels.py` — encoding/retrieval trial labeling for the real-time
  task's held-out phases (`encoding_trials`; `group_couple_trials` +
  `verb_categories` + `retrieval_trials`). Marker-name defaults match the
  current config but are overridable, and `sources.py` validates them against
  `ctx.event_mapping` before using them, so a config rename fails loudly
  instead of silently matching nothing.
- `sources.py` — `build_fl_samples` / `build_encoding_epochs` /
  `build_retrieval_epochs`: the three `SOURCE` branches' trial-labeling +
  epoching, all fed the same already-replayed `raw`/`out_samples`/`sfreq`/
  `fs_out`/`preds` the notebook computes once per run.
- `metrics.py` — `winner_confusion`, `perm_band`, `diag_auc`, `modality_groups`,
  parametrized over arbitrary marker/task sets. Metrics graduate here from
  notebook cells as they stabilize.
- `plots.py` — figure helpers; colors are assigned purely by each marker's
  position in `dc.display_markers` (tab10 cycle), never by a hardcoded
  marker-name lookup, so the palette follows whatever config is loaded.

Backend imports inside `analysis_lib` are deferred into functions, so importing
the package never requires `src/` on `sys.path` first.

## Writing a new analysis notebook

Start every notebook with:

```python
%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path
for _c in [Path.cwd(), *Path.cwd().parents]:
    if (_c / "analysis_lib").is_dir():
        sys.path.insert(0, str(_c)); break
    if (_c / "tests" / "notebooks" / "analysis" / "analysis_lib").is_dir():
        sys.path.insert(0, str(_c / "tests" / "notebooks" / "analysis")); break

from analysis_lib.context import bootstrap, load_context
REPO_ROOT = bootstrap()
ctx = load_context("debug_snapshots/animacy_fl")   # a seeded profile, or any
                                                     # SessionPaths-shaped output dir
```

Keep the stable plumbing in `analysis_lib`; keep metrics/plots you are actively
iterating on as notebook cells (with `%autoreload`, editing the lib is painless
too). Promote a metric into `metrics.py` once it stabilizes across notebooks.

## Notebooks here

- `live_inference_epoched.ipynb` — the main analysis notebook: switchable FL
  replay vs. encoding vs. retrieval-phase reactivation, epoched probability
  trajectories, and metrics. Fully on `analysis_lib`.
- `compare_profiles.ipynb` — cross-root comparison of saved `live_summary.joblib`
  run summaries (each written by `live_inference_epoched.ipynb`'s last cell with
  `SOURCE = "fl"`). Still uses its own self-contained bootstrap.

## Producing an analysis root

Seed a debug snapshot (`src/frontend/debug/profiles.py`) with
`scripts/demo_seed_debug_snapshots.py` — it runs the real offline pipeline
once (filter → ICA → evaluate → train) against a config + FL recording and
writes a self-contained `debug_snapshots/<name>/`. Add `--task-data <dir>` to
also *reference* (never copy) a second, held-out-task recording, so
`"encoding"`/`"retrieval"` have something to replay:

```bash
python -m scripts.demo_seed_debug_snapshots \
    --profile animacy_fl \
    --config experiment_config.realtime_animacy.yaml \
    --data data/sub_001/functinal_localizer \
    --task-data data/sub_001/task
```

Re-seeding (e.g. after a pipeline change) just needs `--profile`; the config/
data/task-data paths are reused from the existing manifest unless overridden.

A real subject's full Phase 1 + Phase 2 output directory (e.g. `data/sub_001`)
also qualifies as a root directly — no manifest needed, raw dirs are found by
physically scanning its subfolders. See the top-level `CLAUDE.md` for the
app's `SessionPaths` layout and how `AppSession` populates it.
