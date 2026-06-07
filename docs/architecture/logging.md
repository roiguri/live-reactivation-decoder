# Logging

How application logging is configured and used. Source of truth:
`src/backend/core/logging_setup.py`.

## Setup

- **One config point.** `configure_logging()` is called exactly once per process by the
  app entry points (`frontend.main`, `frontend.debug.main`). Module code never configures
  logging — it only does `logger = logging.getLogger(__name__)`.
- **Verbosity.** Precedence: `--log-level` flag → `LRD_LOG_LEVEL` env var → `INFO`.
  e.g. `python -m frontend.main --log-level DEBUG`, or `LRD_LOG_LEVEL=DEBUG`.
- **Output.** Terminal (stderr) only for now; file persistence is deferred (see below).
- Python `warnings.warn(...)` is routed through logging (`captureWarnings`) so library
  warnings share the format. Native-library stderr (liblsl, Qt) is outside Python logging
  and is not captured.
- Noisy third-party loggers (MNE, matplotlib, numba) are pinned to WARNING.

## Line format

A one-time startup banner carries the full date + level; each line is then:

```
HH:MM:SS.mmm [LEVEL  ] BE  area.module          message
```

- time with milliseconds · bracketed fixed-width level · `BE`/`FE` layer tag · shortened
  `area.module` logger name (`backend.offline_phase.evaluator` → `offline.evaluator`).
- Color (severity-colored level, dim metadata) is applied only on an interactive TTY
  (respecting `NO_COLOR`); it is never written to non-TTY output.

## Levels

| Level | Use for | Traceback |
|---|---|---|
| `DEBUG` | diagnostic detail, timings; off by default | optional |
| `INFO` | lifecycle milestones & operator decisions | no |
| `WARNING` | recoverable anomalies (degraded but continues) | optional (`exc_info=True`) |
| `ERROR` | a requested operation failed | yes — use `logger.exception` |

## Conventions

- `logging.getLogger(__name__)` only; never `basicConfig`/handlers in module code.
- No `print()` in application logic.
- No silent `except: pass` — log (WARNING if recoverable, DEBUG if cosmetic) or let it raise.
- **Guard rule.** Remove a guard whose violation fails loudly; keep + log a guard whose
  violation would cause silent harm (e.g. starting a second worker).
- **Log a milestone once, at the layer that owns it.** Backend logs domain events; views
  log operator decisions and reachable UI failures. Don't double-log — a failure surfaced
  through a worker is already logged at the worker/source.
- **Hot paths stay silent.** Per-sample / per-batch / per-poll loops must not log; emit a
  periodic summary instead.

## Timing

`log_duration(logger, label, level=DEBUG)` — a context manager (`time.perf_counter`,
monotonic wall-clock) that logs `"<label> finished in N.Ns"` on exit, even if the block
raises. Used to time the heavy offline phases.

## Deferred: file persistence

Logging is terminal-only today. `configure_logging` is shaped to grow an optional
`log_file` parameter (a non-colored `FileHandler` reusing the same formatter); the
intended home is a per-session log under `SessionPaths`. `.gitignore` already ignores
`*.log`.
