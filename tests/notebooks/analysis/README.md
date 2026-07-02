# Analysis notebooks

Notebooks that *analyze* the offline/online decoders (as opposed to the
`validate_*` unit notebooks one level up, which exercise a single backend unit).

All analysis notebooks follow **Mode A**: load a seeded debug profile, replay its
recording through the online inference path, epoch the probability stream around
stimulus markers, and compute metrics. Decoders are *seeded into profiles* by
`scripts/demo_seed_debug_snapshots.py` — notebooks never train.

## `analysis_lib/` — the reusable core

The stable plumbing lives in an importable package so notebooks stay thin:

- `context.py` — `bootstrap()` (repo-root walk + `sys.path`) and
  `load_context(profile)` → `AnalysisContext` (settings, artifact, online
  preprocessor, inference engine, per-task decoding timepoints).
- `streaming.py` — `load_recording`, `extract_markers`, `run_online_stream`
  (StreamWorker-style micro-batches), `make_epocher`.
- `metrics.py` — `winner_confusion`, `perm_band`, `diag_auc`, parametrized over
  arbitrary marker/task sets. Metrics graduate here from notebook cells as they
  stabilize.

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
ctx = load_context("fl")   # a profile you seeded under debug_snapshots/
```

Keep the stable plumbing in `analysis_lib`; keep metrics/plots you are actively
iterating on as notebook cells (with `%autoreload`, editing the lib is painless
too). Promote a metric into `metrics.py` once it stabilizes across notebooks.

## Notebooks here

- `live_inference_epoched.ipynb` — the main analysis notebook: switchable FL
  replay vs. held-out task decoding, epoched probability trajectories, and
  metrics. Fully on `analysis_lib`.
- `compare_profiles.ipynb` — cross-profile comparison of saved run summaries.
  Still uses its own self-contained bootstrap; rewiring it onto `analysis_lib`
  is a low-risk follow-up.

## Seeding a profile

```bash
python -m scripts.demo_seed_debug_snapshots --profile fl \
    --config experiment_config.yaml --data data/split/functional_localizer
```

> Note: `diag_auc` over all timepoints with L1 Logistic at `C=1000` is slow
> (liblinear does not converge); running that cell on all decoders takes minutes.
