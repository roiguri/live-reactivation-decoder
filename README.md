# online_decoder

Standalone PyQt6 app for the EEG reactivation decoder pipeline.

- **Phase 1** (offline): operator-driven 5-node workflow — Settings → Load
  Data → Preprocess → Evaluate → Train. Produces `decoder_pipeline.joblib`.
- **Phase 2** (online): real-time inference against an LSL stream,
  consuming the artifact produced by Phase 1.

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Windows** is required for the live LSL stream path
  (`tools/lslproxy/LSLProxy.exe`). Phase 1 and the full test suite work
  on Windows, macOS, Linux, and WSL.

## Install

```bash
cd online_decoder
python -m venv .venv
```

Activate the venv:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
  (one-time only, if PowerShell blocks the script:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)
- macOS / Linux / WSL: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` transitively includes `requirements.txt` (via
the first-line `-r` reference), so this single install covers both the
app's runtime deps and the tooling needed to run tests + debug scripts.
For a strict production-runtime-only install, use
`pip install -r requirements.txt` instead — but you won't be able to
run `pytest` or the `scripts/` helpers.

## Run the app

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
python -m frontend.main
```

```bash
# macOS / Linux / WSL
PYTHONPATH=src python -m frontend.main
```

Then walk the Phase 1 trail in order:

1. **Settings** — pick `experiment_config.yaml` + output directory → Continue
2. **Load Data** — pick a BrainVision folder (`.vhdr` + `.vmrk` + `.eeg`) → Load Data
3. **Preprocess** — Start Preprocessing
   - MNE's bad-channel window pops modally; click channels to mark, then close
   - MNE's ICA review window pops modally with ICLabel pre-suggestions;
     verify/override `ica.exclude`, then close
4. **Evaluation** — Run Evaluation → pick a timepoint on the AUC chart →
   Approve & Continue
5. **Train** — ▶ → "Trained at: N ms" + spatial-pattern topomaps

Output: `decoder_pipeline.joblib` in the directory chosen in step 1.

## Run the debug walkthrough

Fast path for iterating on UI screens without sitting through ~5 min
of real preprocessing each time. **One-time seed** from a real recording,
then drive the whole pipeline with **Ctrl+→**.

```bash
python -m scripts.demo_seed_debug_snapshots --data <path/to/subject>
python -m frontend.debug.main
```

See [src/frontend/debug/README.md](src/frontend/debug/README.md) for
the full walkthrough mechanics.

## Run tests

```bash
pytest -q --deselect tests/online_phase/test_stream_worker.py
```

Expected: `322 passed, 1 skipped, 11 deselected`.

- The 1 skip is `test_lsl_receiver_integration.py`, gated behind
  `RUN_LSL_INTEGRATION=1` — runs only against a real LSL stream.
- The 11 deselections are `test_stream_worker.py`, which needs
  `pytest-qt`/`qtbot` and a live LSL outlet; it's not a regression.

## Where things live

- **Backend**: `src/backend/offline_phase/`, `src/backend/online_phase/`
- **Frontend (PyQt6)**: `src/frontend/`
- **Config**: `experiment_config.yaml` (schema in `src/backend/core/config_models.py`)
- **Architecture**: [docs/backend_architecture.md](docs/backend_architecture.md)
- **Preprocessing migration history**: [docs/Preprocessing_Migration_Plan.md](docs/Preprocessing_Migration_Plan.md)
- **Repo conventions**: [CLAUDE.md](CLAUDE.md)
