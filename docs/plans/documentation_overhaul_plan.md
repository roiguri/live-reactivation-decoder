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

Needs a general scaffolding pass first (agree on the subsections and their
depth), then fill:

- **Software Architecture**: short overview prose (decoupled UI/backend model,
  Phase 1 to artifact to Phase 2 data flow) over the existing
  `docs/architecture/*` links.
- **Hardware**: write from the confirmed facts (NeurOne 64 EEG + 1 trigger at
  1000 Hz, µV wire units, LSLProxy bridge, parallel-port trigger decoding). New
  page candidate: `docs/guide/hardware.md`.
- **Analysis**: short section on reproducing results, pointing at
  `tests/notebooks/analysis/`.
- **Debug Mode** and **Testing**: already have content in the README.

### Pipeline description doc (reframed, was "preprocessing")

Not just a preprocessing explanation. It should be a **general pipeline
description**: the offline training pipeline and the online inference pipeline
end to end, with preprocessing as one part. Referenced today by a dangling
`[preprocessing pipeline]` text in the Configuration section, which renders as
literal brackets until the doc exists. Likely lives in `docs/guide/`. Scope to
be discussed (how much signal-processing detail, whether to show the constants,
offline vs online ordering).

### User manual expansion (needed, not yet discussed)

The existing `docs/guide/user_manual.md` is thin (short captions per screen). It
needs more descriptive work so it truly functions as a step-by-step operating
manual. Scope to be discussed.
