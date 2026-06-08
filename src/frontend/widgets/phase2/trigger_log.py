"""Append-only trigger / system log for Phase 2 live inference.

A terminal-style scrolling log placed below the live chart. It surfaces two
kinds of lines:

* **Trigger events** — every non-zero trigger edge from ``prediction_ready``,
  stamped with the session-relative time and resolved to its configured event
  name (blank for unmapped codes, like :class:`LiveSessionLogger`, since an
  audit log shouldn't pre-filter the way the chart does).
* **Lifecycle events** — stream started / halted / errors, fed by the screen.

This is the UI surface over the same marker stream the chart and the session
logger consume; it is **data-agnostic** — it shows raw trigger edges (delay-
free, untouched by preprocessing), so it works identically regardless of the
prediction-fidelity state, and the timestamps line up for free once fidelity
lands.
"""
from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QPlainTextEdit, QWidget

from frontend.styles.theme import BORDER_GRAY, CARD_WHITE, TEXT_PRIMARY

# Cap retained lines so a long session (or a trigger flood) can't grow the
# document unbounded. ``setMaximumBlockCount`` trims the oldest lines for us.
_MAX_LINES = 500


class TriggerLog(QPlainTextEdit):
    """Read-only, auto-scrolling log of trigger edges and lifecycle events.

    Construct with the ``{code: name}`` event map (same one the chart uses).
    Timestamps are **session-relative**: the first marker seen sets ``t0`` and
    every later line reads ``+T.Ts`` from it (matching ``FrozenEventView``),
    so the log is readable without exposing the arbitrary LSL clock origin.
    """

    def __init__(
        self,
        event_names: dict[int, str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._event_names: dict[int, str] = {
            int(code): str(name) for code, name in (event_names or {}).items()
        }
        # Session clock origin — the first marker timestamp seen this Start.
        # Lifecycle lines logged before any marker show a blank stamp.
        self._t0: float | None = None

        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {CARD_WHITE}; color: {TEXT_PRIMARY};"
            f" border: 1px solid {BORDER_GRAY}; }}"
        )

    # ── public API ──────────────────────────────────────────────────────────

    def append_markers(self, markers: list[tuple[float, int]]) -> None:
        """Append one line per trigger edge. ``markers`` is the
        ``(lsl_timestamp, code)`` list from ``prediction_ready``."""
        if not markers:
            return
        for ts, code in markers:
            code = int(code)
            if self._t0 is None:
                self._t0 = float(ts)
            name = self._event_names.get(code, "")
            self._append(f"{self._stamp(ts)}  TRIG {code:>3}  {name}".rstrip())

    def log_event(self, message: str) -> None:
        """Append a lifecycle / system line (stream started, halted, error)."""
        stamp = self._stamp() if self._t0 is not None else f"{'':>10}"
        self._append(f"{stamp}  · {message}")

    def reset(self) -> None:
        """Clear the log and reset the session clock (called on each Start)."""
        self.clear()
        self._t0 = None

    # ── internals ───────────────────────────────────────────────────────────

    def _stamp(self, ts: float | None = None) -> str:
        """Session-relative ``+T.Ts`` stamp, right-padded to a fixed width."""
        if self._t0 is None or ts is None:
            return f"{'':>10}"
        return f"{f'+{float(ts) - self._t0:.2f}s':>10}"

    def _append(self, line: str) -> None:
        # appendPlainText adds the block; keep the view pinned to the newest
        # line so the log auto-scrolls.
        self.appendPlainText(line)
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
