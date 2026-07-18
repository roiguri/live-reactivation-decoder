# Documentation Overhaul Plan

Working plan for the README and project-documentation rewrite. This is a
resumable checklist so the work can stop and continue across sessions.

Branch: `docs/readme-overhaul`.

## Goal

Rewrite the project README into a clear, two-audience document that references
the deeper docs rather than duplicating them. The originally requested shape:

- **User Section**
  - **Compatibility**: what experiments the app can work with.
  - **Configuration**: the experiment config file (decoder definitions,
    trigger-to-event mapping, seed) and its schema, i.e. how a new experiment or
    stimulus set is defined without touching code.
  - **Application User Manual**: step-by-step manual for operating the app
    (linked, screenshot-driven).
- **Developer Section**
  - **Software Architecture**: important architectural decisions, how to work in
    the repo, and the app debug mode.
  - **Hardware Description**: the EEG acquisition setup (NeurOne amplifier,
    64 EEG + 1 event channel at 1000 Hz), the LSLProxy bridge, and the
    parallel-port trigger interface.
  - **Analysis**: analysis of existing data and how to reproduce results.

The README is a hub. Deep detail lives in `docs/` and `knowledge_base/`.

## Working agreements

- One step at a time. Review before every commit. No commits without explicit
  approval.
- Style: no em dashes, no semicolons in the README.
- Layout: **hybrid**. User Guide is self-contained. Developer Guide is
  hub-style (short prose plus links out).
- Final, reader-facing docs live in **`docs/guide/`**. Working docs stay in
  `docs/architecture/`, `docs/reference/`, `docs/plans/`, `docs/old/`. History
  stays in `knowledge_base/`.

## Confirmed facts (resolved this effort)

- Live NeurOne LSL stream is **1000 Hz** and delivers **microvolts (µV)**
  (LSLProxy applies the Tesla /20 scaling before publishing).
- This contradicts the current `CLAUDE.md` "Known Conventions" note, which says
  the µV/V scaling was removed and left unverified. VHDR replay delivers SI
  volts, but the real proxy delivers µV, so live (non-replay) runs will need a
  µV to V scaling step re-introduced in `OnlinePreprocessor`. Capture this in the
  Hardware section and, eventually, update `CLAUDE.md`.

## README structure (three parts)

1. **Getting Started**: Prerequisites, Install, Run the app.
2. **User Guide**: Compatibility, Configuration, Application User Manual.
3. **Developer Guide**: Software Architecture (with Debug Mode), Testing,
   Hardware, Analysis.

## Done

- Created `docs/guide/`. Moved the app walkthrough there as
  `docs/guide/user_manual.md` and registered the folder in `docs/README.md`.
  (`9f774bb`)
- Restructured the README into Getting Started / User Guide / Developer Guide,
  renamed the project to **Memory Reactivation Decoder**, rewrote the intro,
  moved debug + tests into the Developer Guide, stripped em dashes. (`294812d`)
- Wrote **Compatibility** (requirements-only framing, plus the
  experiment-structure figure `docs/assets/experiment_paradigm.png`). (`34c1361`)
- Wrote **Configuration** (annotated `experiment_config.yaml` example plus
  per-key notes, deferring model params to `config_models.py`). (`9a59f25`)
- Stripped semicolons from the README. (`0135d98`)
- Showed the welcome screen in the Application User Manual subsection and
  dropped the per-screen enumeration (`docs/assets/welcome_screen.png`).
  (`7283a18`)
- Saved logo source assets to `docs/assets/logo/`. (`070ecf7`)

## Remaining

### Developer Guide

Structure decided: **Debug Mode** and **Testing** nest under **Software
Architecture** (both are "how to work in the repo" concerns). Sections to fill:

- **Software Architecture**: a **self-contained** overview written from
  `CLAUDE.md` + current code (decoupled UI/backend, Phase 1 to
  `decoder_pipeline.joblib` to Phase 2, `AppSession` entry point, `SessionPaths`
  layout). Do **not** hub-link the stale architecture docs (see verdicts below).
  Link only `logging.md`, `CLAUDE.md`, and `src/`. Add a TODO for deeper docs if
  a topic needs more than the overview. Nested under it:
  - **Helper scripts** (`scripts/`), useful groups only: replay
    (`replay_vhdr_to_lsl`, `replay_xdf_to_lsl`), LSL diagnostics / smoke tests
    (`characterize_lsl`, `smoke_test_lsl_receiver`, `smoke_stream_worker`,
    `inspect_xdf`), dev setup / fixtures (`demo_seed_debug_snapshots`,
    `create_test_eeg`, `split_subject_by_phase`). Omit benchmarks / one-offs.
  - **Debug Mode** and **Testing**: already have content in the README.
- **Hardware**: write from the confirmed facts (NeurOne 64 EEG + 1 trigger at
  1000 Hz, µV wire units, LSLProxy bridge, parallel-port trigger decoding). New
  page candidate: `docs/guide/hardware.md`.
- **Analysis**: short section on reproducing results, pointing at
  `tests/notebooks/analysis/`.

### Architecture docs status (reviewed 2026-07-18)

Policy: **fix or archive** each stale doc as we work the section that touches it.

- `logging.md` (65L): current and concise. **Link as-is.**
- `frontend_layout.md` (191L): shell/widget/styling parts are good, but it says
  "only Phase 1 is wired up" and nodes 2-5 are "stubs" (both false now) and has
  no Phase 2. **Stale**, handle when we touch the frontend.
- `stream_worker_design.md` (681L): a pre-implementation PR design plan
  (`PredictionLogger` since replaced by `LiveSessionLogger`, wrong
  `build_live_stream_session` signature, no decision layer). **Archive.**
- `backend_architecture.md` (1546L): heavily stale (pre-migration config dump,
  wrong constructors, missing decision layer, claims `src/frontend/` uncommitted).
  **Archive.** Note: `CLAUDE.md` currently points to it as "the maintained backend
  summary", so archiving means updating `CLAUDE.md` and `docs/README.md` too.

### Pipeline description doc (reframed, was "preprocessing")

Not just a preprocessing explanation. It should be a **general pipeline
description**: the offline training pipeline and the online inference pipeline
end to end, with preprocessing as one part. Likely lives in `docs/guide/`. Scope
to be discussed (how much signal-processing detail, whether to show the
constants, offline vs online ordering).

When this doc exists, add a "see <link> for the recipe" pointer to the README
**Configuration** section, where the spot is currently held by a hidden `TODO
(Preprocessing pipeline doc)` comment (so nothing renders as a dangling
reference in the meantime).

### User manual (done)

`docs/guide/user_manual.md` has been rebuilt into a full task-oriented operating
manual, mechanics-only, verified against the code:

- Overview, Before you start, Launch.
- Phase 1 (Pipeline Settings, Data Loading, Preprocessing, Model Evaluation,
  Train & Save) with the Training Pipeline navigation and a sidebar close-up.
- Phase 2 (the live screen, Select the stream, Live output, Controls).
- Output files (corrected against `SessionPaths` + `session_logger.py`: `epochs/`
  and `decision_config.jsonl` are written, `evaluation/` is not).

Troubleshooting was intentionally left out for now (see below).

### Optional troubleshooting items (undecided)

Candidate troubleshooting entries whose inclusion in the user manual is not yet
decided. Notes kept here so they are ready if we choose to add them. Several are
data-side or environment issues, arguably out of scope for operators.

- **No LSL stream found.** The Select Target dialog lists nothing or the stream is
  missing. Fix: press Refresh, ensure the source (hardware + LSLProxy, or a
  replay) is publishing.
- **Live inference is Windows-only.** The hardware path uses `LSLProxy.exe`
  (Windows). Phase 1 and replay-based testing work elsewhere.
- **Stream rejected on connect.** `LSLReceiver` validates the stream (expects
  1000 Hz and 65 channels); a mismatch fails to start.
- **Open Live from Existing Output finds no decoder.** The chosen folder must
  contain `models/decoder_pipeline.joblib`.
- **BrainVision header/filename mismatch.** A BrainVision recording is three
  cross-referencing files (`.vhdr` names its `.vmrk` and `.eeg`; `.vmrk` names
  the `.eeg`). If the internal names do not match the files on disk, loading
  fails with a file-not-found error. It is a data defect, not an app bug. Fix:
  rename the files to the stem the header expects, or edit the
  `DataFile=`/`MarkerFile=` lines in the `.vhdr` and the `DataFile=` line in the
  `.vmrk`.
