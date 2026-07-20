# Documentation

- **[guide/](guide/)** — final, reader-facing documentation (the pages the top-level
  [README](../README.md) links to):
  - [user_manual.md](guide/user_manual.md) — screen-by-screen Phase 1 + Phase 2 walkthrough
  - [backend.md](guide/backend.md) — backend surface and contracts
  - [frontend.md](guide/frontend.md) — frontend (PyQt6) structure
  - [hardware.md](guide/hardware.md) — EEG acquisition, the LSLProxy bridge, and trigger decoding
- **[architecture/](architecture/)** — how the system works now; the maintained,
  source-of-truth-adjacent references:
  - [logging.md](architecture/logging.md) — logging conventions
- **[plans/](plans/)** — active, per-milestone implementation plans (transient).
- **[reference/](reference/)** — results and dev/feature references (investigations, benchmarks, tooling notes).
- **[old/](old/)** — archived/superseded plans (completed milestones + early implementation plans).

> When the docs and the code disagree, the code under `src/` wins — follow it and update the doc.
