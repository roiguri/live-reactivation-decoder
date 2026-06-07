# Documentation

Docs are organized by purpose. Migration into this structure is **in progress** — some
maintained docs are still at the `docs/` root and will move into the folders below in a
follow-up.

- **[architecture/](architecture/)** — how the system works now (maintained,
  source-of-truth-adjacent).
  - [logging.md](architecture/logging.md) — logging conventions
  - *pending move: `backend_architecture.md`, `frontend_layout.md`, `stream_worker_design.md`*
- **[plans/](plans/)** — implementation plans (historical + active).
  - [per_decoder_timepoint_selection.md](plans/per_decoder_timepoint_selection.md)
  - *pending move: the Phase 1/2 UI & backend plans, the preprocessing migration*
- **[reference/](reference/)** — results & dev/feature references.
  - [debug_profiles.md](reference/debug_profiles.md)
  - *pending move: `Decoder_Pipeline_Results.md`*
- **[old/](old/)** — archived early implementation plans.

> When the docs and the code disagree, the code under `src/` wins — follow it and update the doc.
