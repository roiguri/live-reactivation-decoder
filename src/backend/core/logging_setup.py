"""Central logging configuration for the app.

This is the **single** place application logging is set up. The app entry
points (production ``frontend.main`` and debug ``frontend.debug.main``) call
:func:`configure_logging` exactly once at startup instead of declaring their own
``logging.basicConfig``. Module code never configures logging — it only does
``logger = logging.getLogger(__name__)`` and logs.

(Standalone dev scripts under ``scripts/`` manage their own logging and are out
of scope here.)

Line format (terminal-only for now)::

    ===== live-reactivation-decoder  |  2026-06-07 15:24:55  |  level=INFO =====
    15:24:56.812 [INFO   ] BE  offline.evaluator          Evaluating task: color
    15:24:57.044 [WARNING] BE  online.lsl_receiver        Malformed chunk: 3 values
    15:25:10.220 [ERROR  ] FE  workers.evaluation_worker  EvaluationWorker failed

- A one-time **startup banner** carries the full date + level; per-line stamps
  are time-only with milliseconds (handy for live latency and ``log_duration``).
- Bracketed fixed-width level, a ``BE``/``FE`` layer tag, and a shortened
  ``area.module`` logger name (``backend.offline_phase.evaluator`` → ``offline``
  layer-stripped + ``_phase`` collapsed), all aligned into columns.
- **Color** (dim time/name, severity-colored level) is applied only when stderr
  is an interactive TTY and ``NO_COLOR`` is unset, so it never pollutes piped
  output or the log *file* a later step will add. On Windows, ANSI is enabled via
  the console VT mode (no third-party dependency).

Persisting logs to files is a deliberate later step: :func:`configure_logging`
is shaped to grow an optional ``log_file`` parameter (a non-colored
``FileHandler`` reusing :class:`_AppFormatter`) without touching call sites.

Verbosity precedence (see :func:`resolve_level`):
``explicit arg → --log-level flag → LRD_LOG_LEVEL env var → INFO``.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

__all__ = ["ENV_VAR", "APP_NAME", "configure_logging", "resolve_level", "log_duration"]

APP_NAME = "live-reactivation-decoder"
ENV_VAR = "LRD_LOG_LEVEL"

_DEFAULT_LEVEL = logging.INFO

# Column widths. Level pads to WARNING (7) — the widest level we emit; the rare
# CRITICAL overflows by one char, which is fine. Name pads to the common long
# case; rarer longer names overflow gracefully (their message just starts later).
_LEVEL_WIDTH = 7
_NAME_WIDTH = 25

_LAYER_TAGS = {"backend": "BE", "frontend": "FE"}

# Chatty third-party loggers pinned to WARNING so the app's own INFO output stays
# readable. (MNE's per-call ``verbose=False`` handles function-level noise; this
# pins its logger level globally in one place.)
_THIRD_PARTY: dict[str, int] = {
    "mne": logging.WARNING,
    "matplotlib": logging.WARNING,
    "numba": logging.WARNING,
}

# ── ANSI color ────────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_DIM = "\033[90m"  # bright black / grey — for recede-into-background metadata
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;41m",  # bold on red background
}


# ── Level resolution ──────────────────────────────────────────────────────────

def _coerce_level(value: str | int | None) -> int | None:
    """Return a numeric logging level for a name/int, or None if unrecognized."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    name = str(value).strip().upper()
    if not name:
        return None
    if name.isdigit():
        return int(name)
    level = logging.getLevelName(name)  # int for known names, "Level X" string otherwise
    return level if isinstance(level, int) else None


def resolve_level(cli_level: str | int | None = None) -> int:
    """Resolve the effective log level.

    Precedence: explicit ``cli_level`` → ``LRD_LOG_LEVEL`` env var → INFO.
    Accepts level names ("DEBUG", "info") or ints; unrecognized values are
    skipped so a bad flag falls through to the next source rather than crashing.
    """
    for candidate in (cli_level, os.environ.get(ENV_VAR)):
        level = _coerce_level(candidate)
        if level is not None:
            return level
    return _DEFAULT_LEVEL


# ── Formatting ────────────────────────────────────────────────────────────────

def _short_source(name: str) -> tuple[str, str]:
    """Split a dotted logger name into a ``BE``/``FE`` layer tag and area.module.

    ``backend.offline_phase.evaluator``  → ``("BE", "offline.evaluator")``
    ``frontend.workers.evaluation_worker`` → ``("FE", "workers.evaluation_worker")``
    ``mne`` (or any non-app logger)       → ``("", "mne")``
    """
    parts = name.split(".")
    layer = _LAYER_TAGS.get(parts[0], "")
    rest = parts[1:] if layer else parts
    module = ".".join(rest).replace("_phase", "")
    return layer, module or name


class _AppFormatter(logging.Formatter):
    """Aligned, optionally-colored single-line formatter (see module docstring)."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        base = time.strftime("%H:%M:%S", self.converter(record.created))
        return f"{base}.{int(record.msecs):03d}"

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record)
        layer, module = _short_source(record.name)
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        level_field = f"[{record.levelname:<{_LEVEL_WIDTH}}]"
        layer_field = f"{layer:<2}"
        name_field = f"{module:<{_NAME_WIDTH}}"

        if self._color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            ts = f"{_DIM}{ts}{_RESET}"
            level_field = f"{color}{level_field}{_RESET}"
            layer_field = f"{_DIM}{layer_field}{_RESET}"
            name_field = f"{_DIM}{name_field}{_RESET}"

        return f"{ts} {level_field} {layer_field}  {name_field} {msg}"


# ── Color capability detection ────────────────────────────────────────────────

def _enable_windows_vt() -> bool:
    """Enable ANSI escape processing on the Windows stderr console. Best-effort."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except Exception:
        return False


def _supports_color() -> bool:
    """True if stderr is a TTY we can color (respects the NO_COLOR convention)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = sys.stderr
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform == "win32":
        return _enable_windows_vt()
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def _emit_banner(level_name: str, color: bool) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"===== {APP_NAME}  |  {now}  |  level={level_name} ====="
    print(f"{_DIM}{line}{_RESET}" if color else line, file=sys.stderr)


def configure_logging(level: str | int | None = None, *, banner: bool = True) -> int:
    """Configure root logging for an entry point. Idempotent.

    Installs a single :class:`_AppFormatter` stderr handler (replacing any prior
    handlers), quiets noisy third-party loggers, and prints the startup banner.

    Returns the resolved numeric level.
    """
    resolved = resolve_level(level)
    color = _supports_color()

    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(_AppFormatter(color=color))

    root = logging.getLogger()
    root.handlers.clear()  # idempotent: never pile up handlers across calls
    root.addHandler(handler)
    root.setLevel(resolved)

    for name, lvl in _THIRD_PARTY.items():
        logging.getLogger(name).setLevel(lvl)

    if banner:
        _emit_banner(logging.getLevelName(resolved), color)
    return resolved


@contextmanager
def log_duration(
    logger: logging.Logger, label: str, level: int = logging.INFO
) -> Iterator[None]:
    """Time a block and log ``"<label> finished in N.Ns"`` on exit.

    Fires even if the block raises, so a failed long-running step still reports
    how long it ran before failing. Example::

        with log_duration(logger, "Evaluation"):
            results = evaluator.run_evaluation()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s finished in %.1fs", label, time.perf_counter() - start)
