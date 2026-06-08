# Documentation

- **[architecture/](architecture/)** — how the system works now; the maintained,
  source-of-truth-adjacent references:
  - [backend_architecture.md](architecture/backend_architecture.md) — backend surface & contracts
  - [frontend_layout.md](architecture/frontend_layout.md) — frontend structure
  - [stream_worker_design.md](architecture/stream_worker_design.md) — live decoder loop design
  - [logging.md](architecture/logging.md) — logging conventions
- **[plans/](plans/)** — active, per-milestone implementation plans (transient).
- **[reference/](reference/)** — results and dev/feature references (investigations, benchmarks, tooling notes).
- **[old/](old/)** — archived/superseded plans (completed milestones + early implementation plans).

> When the docs and the code disagree, the code under `src/` wins — follow it and update the doc.
