# Offline Stage (Parent Repo)

Back to [Codebase Overview](README.md) or [Project Index](../START_HERE.md).

## Context

The offline stage is **not in this repo**. It lives in the parent [`reactivation-decoder`](https://github.com/roiguri/reactivation-decoder) repo under `src/`, and represents the semester-A work that evaluated the decoding pipeline before this online app was built.

This document is kept as historical context because the online app's Phase 1 design (training pipeline, preprocessing choices, decoder evaluation approach) was informed by that work.

## What The Parent `src/` Did

The semester-A offline pipeline:

- Loaded preprocessed EEG epochs
- Defined classification schemes for the BMR experiment
- Ran feature extraction and smoothing
- Trained and evaluated decoders on pre-recorded datasets
- Generated AUC curves, temporal generalization matrices, and result plots

It is an **offline analysis pipeline**, not a real-time system. It does not feed this app directly: the online repo has its own self-contained Phase 1 training (`src/backend/offline_phase/`), modeled on the same decoding approach but rewritten to integrate with the live stack.

## Important Files (in the parent repo)

- `src/pipelines.py` — high-level offline workflows
- `src/decoder.py` — decoder logic
- `src/features.py` — feature preprocessing
- `src/utils/data_manager.py` — data loading
- `src/analysis/results.py` — result containers
- `src/analysis/plotting.py` — plotting helpers

## Related Reference Material In This Repo

- [tomer_preprocessing.md](../02_reference/tomer_preprocessing.md) — preprocessing parameters and steps used during semester-A work
- [tomer_params.md](../02_reference/tomer_params.md) — trigger and parameter definitions
- [BMR Data Specification.md](../02_reference/BMR%20Data%20Specification.md) — experiment data model (shared between offline and online)

For the actual offline implementation, see the parent repo.
