# Memory Reactivation Decoder

A standalone PyQt6 desktop application for decoding memory reactivation from EEG
in real time. It trains a subject-specific decoder offline from a recording,
then runs that decoder live against the incoming EEG stream to read out
reactivation as it happens.

This README has three parts:

1. **[Getting Started](#getting-started)**: install and run.
2. **[User Guide](#user-guide)**: what the app works with, how to configure an
   experiment, and how to operate it.
3. **[Developer Guide](#developer-guide)**: architecture, hardware, and analysis.

---

## Getting Started

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Windows** is required for the live LSL stream path
  (`tools/lslproxy/LSLProxy.exe`). Phase 1 and the full test suite work
  on Windows, macOS, Linux, and WSL.

### Install

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
`pip install -r requirements.txt` instead, but you won't be able to
run `pytest` or the `scripts/` helpers.

### Run the app

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"
python -m frontend.main
```

```bash
# macOS / Linux / WSL
PYTHONPATH=src python -m frontend.main
```

---

## User Guide

### Compatibility

<!-- TODO (step: Compatibility): which experiments and stimulus sets the app
supports: the functional-localizer + binding-memory task, the shipped
`experiment_config*.yaml` presets, and the BrainVision recording format Phase 1
expects. Drafted from code + configs. -->

_To be written._

### Configuration

<!-- TODO (step: Configuration): the `experiment_config.yaml` schema
(decoder definitions, trigger→event `markers_mapping`, `random_state` seed, and
optional `intervals`), and how to define a new experiment or stimulus set
without touching code. Schema source: `src/backend/core/config_models.py`. -->

_To be written._

### Application User Manual

The **[Application User Manual](docs/guide/user_manual.md)** is a
screen-by-screen walkthrough of operating the app: launch, Phase 1 training
(settings → load → preprocess → evaluate → train), and Phase 2 live inference.

---

## Developer Guide

### Software Architecture

<!-- TODO (step: Architecture): short overview of the decoupled UI/backend
model and the Phase 1 → artifact → Phase 2 data flow, then links out for depth. -->

_Overview to be written._ For now, the maintained references are:

- [docs/architecture/backend_architecture.md](docs/architecture/backend_architecture.md): backend surface & contracts
- [docs/architecture/frontend_layout.md](docs/architecture/frontend_layout.md): frontend structure
- [docs/architecture/stream_worker_design.md](docs/architecture/stream_worker_design.md): live decoder loop design
- [docs/architecture/logging.md](docs/architecture/logging.md): logging conventions
- [CLAUDE.md](CLAUDE.md): repo conventions and how to work in this codebase
- [docs/README.md](docs/README.md): full documentation map

#### Debug Mode

<!-- TODO (step: Architecture): expand on how the debug entry points and seeded
snapshot profiles let you jump straight into any screen without a full pipeline
run. -->

Fast path for iterating on UI screens without sitting through ~5 min of real
preprocessing each time. **One-time seed** from a real recording, then drive the
whole pipeline with **Ctrl+→**.

```bash
python -m scripts.demo_seed_debug_snapshots --data <path/to/subject>
python -m frontend.debug.main
```

See [src/frontend/debug/README.md](src/frontend/debug/README.md) for the full
walkthrough mechanics and [docs/reference/debug_profiles.md](docs/reference/debug_profiles.md)
for the seeded snapshot profiles.

### Testing

```bash
pytest -q --deselect tests/online_phase/test_stream_worker.py
```

Expected: `322 passed, 1 skipped, 11 deselected`.

- The 1 skip is `test_lsl_receiver_integration.py`, gated behind
  `RUN_LSL_INTEGRATION=1`; runs only against a real LSL stream.
- The 11 deselections are `test_stream_worker.py`, which needs
  `pytest-qt`/`qtbot` and a live LSL outlet; it's not a regression.

### Hardware

<!-- TODO (step: Hardware): the EEG acquisition setup (NeurOne amplifier,
64 EEG + 1 event channel at 1000 Hz, µV on the wire), the LSLProxy bridge, and
the parallel-port trigger interface. New page: docs/guide/hardware.md. -->

_To be written._

### Analysis

<!-- TODO (step: Analysis): how existing data was analyzed and how to reproduce
the project's results (FL / encoding / retrieval replay). -->

_To be written._ See
[tests/notebooks/analysis/README.md](tests/notebooks/analysis/README.md).
