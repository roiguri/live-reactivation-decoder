"""Event-locked ("frozen") probability chart for Phase 2 live inference.

Where :class:`~frontend.widgets.live_probability_chart.LiveProbabilityChart`
shows a *rolling* window that always follows the latest sample, this widget
shows a **fixed window around a single trigger event** — the most recent
configured marker. On each marker it epochs the live prediction stream:
slices ``[t_event - pre, t_event + post]``, rebases x so the event onset
sits at ``x = 0``, and freezes the per-decoder curves until the next event
replaces them.

Why this is delay-agnostic
--------------------------
The window is anchored to the **marker** — a trigger code pulled straight
off the LSL stream, untouched by the preprocessing pipeline, so delay-free.
The decoder curves drawn inside the window still carry whatever pipeline
group delay exists; this widget makes **no** attempt to compensate it. It
simply takes the seconds around the event verbatim. If the online pipeline's
fidelity improves later, the response just moves earlier within the same
window for free — no change here. (See the "Frozen Event Graph" goal and the
``online-inference-fidelity-bug`` note for the delay context.)

Data discipline
---------------
Like the live chart, ``append_predictions`` / ``append_markers`` are
data-only hot paths: they write into a preallocated backing ring buffer and
record a pending event, never touching the pyqtgraph scene. A low-rate
``QTimer`` does the freeze (the only scene mutation), so the inference loop
is never blocked by chart work.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from frontend.styles.theme import (
    ALERT_RED,
    BORDER_GRAY,
    CARD_WHITE,
    PRIMARY_BLUE,
    TEXT_MUTED,
    chart_line_color,
)

# Top extends well past 1.0 to leave a headroom band where the event-name
# label sits clear of probability curves that ride near 1.0.
_Y_RANGE = (0.0, 1.18)
_Y_TICK_MAX = 1.0
_CHANCE_LEVEL = 0.5
# Vertical position of the onset-line label, as a fraction of the view range —
# high in the headroom band so it doesn't graze the curves.
_LABEL_POS = 0.95
# The freeze check is rare (once per event), so it can tick slowly — it only
# needs to notice that ``latest_ts`` has passed ``t_event + post``.
_CHECK_HZ = 15

_ONSET_COLOR = "#000000"
_WAITING_TITLE = "Waiting for event…"
# Cap on retained snapshots so a long session can't grow the history
# unbounded. Browsing (Goal 11) walks this list; the oldest is dropped first.
_MAX_HISTORY = 64


class FrozenEventChart(pg.PlotWidget):
    """Fixed-window, event-locked per-decoder probability snapshot.

    Constructor args
    ----------------
    task_names
        Decoder names. One curve per name, coloured by ``chart_line_color``
        in order — matching :class:`LiveProbabilityChart` so a decoder has
        the same colour in both charts.
    pre_seconds, post_seconds
        Window extent around the event onset (onset at ``x = 0``). The
        x-axis is fixed at ``[-pre_seconds, post_seconds]``.
    target_sfreq
        Prediction stream rate (after Phase 2 decimation). Sizes the backing
        buffer.
    threshold
        y-position of the dashed red threshold guide (visual parity with the
        live chart).
    event_names
        Trigger code → event name. A marker whose code is **not** in this map
        is ignored (the receiver emits every non-zero edge; only configured
        events epoch).

    Signals
    -------
    event_captured(int)
        Emitted after a new event is frozen, carrying its index in the
        (newest-first) history. A browsing control listens to refresh its
        list. The newest snapshot is displayed by default.
    """

    event_captured = pyqtSignal(int)

    def __init__(
        self,
        task_names: list[str],
        *,
        pre_seconds: float = 0.2,
        post_seconds: float = 1.0,
        target_sfreq: float = 100.0,
        threshold: float = 0.85,
        event_names: dict[int, str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent, background=CARD_WHITE)

        if not task_names:
            raise ValueError("FrozenEventChart requires at least one task name.")

        self._task_names: list[str] = list(task_names)
        self._pre = float(pre_seconds)
        self._post = float(post_seconds)
        self._target_sfreq = float(target_sfreq)
        self._threshold = float(threshold)
        self._event_names: dict[int, str] = {
            int(code): str(name) for code, name in (event_names or {}).items()
        }

        # Backing ring buffer. It must always hold a full epoch span at the
        # moment ``latest_ts`` crosses ``t_event + post`` — i.e. at least
        # ``pre + post`` seconds. A margin absorbs batch jitter / overshoot.
        span = self._pre + self._post + 0.5
        self._capacity: int = max(1, int(round(span * self._target_sfreq)))
        self._timestamps = np.full(2 * self._capacity, np.nan, dtype=np.float64)
        self._buffers: dict[str, np.ndarray] = {
            name: np.full(2 * self._capacity, np.nan, dtype=np.float64)
            for name in self._task_names
        }
        self._write_idx: int = 0
        self._latest_ts: float | None = None
        # First timestamp seen since the last reset — the session epoch. Event
        # times in the history are reported relative to it (``+12.3s``) so the
        # list shows small, meaningful numbers instead of the raw LSL clock
        # (``local_clock()``, an arbitrary ~uptime origin).
        self._session_t0: float | None = None

        # Pending capture, the browsable history, and the index currently on
        # screen. ``_pending`` is set by ``append_markers`` and cleared by the
        # freeze. ``_history`` holds frozen snapshots **newest-first**; each is
        # {"ts", "code", "name", "x": ndarray, "ys": {name: ndarray}}.
        # Pending captures, in arrival order. Each {"ts", "code"} waits until
        # its post-event window has streamed in, then freezes. Queued (not a
        # single slot) so events closer together than ``post`` still each land
        # in the history.
        self._pending: list[dict] = []
        self._history: list[dict] = []
        self._current_index: int | None = None  # index into _history on screen
        # Auto-follow: True while the newest event is on screen. Browsing an
        # older event clears it, so an incoming event no longer yanks the
        # display away — it just lengthens the history behind the scenes.
        self._following: bool = True

        self._task_colors: dict[str, str] = {
            name: chart_line_color(i) for i, name in enumerate(self._task_names)
        }
        self._visible: dict[str, bool] = {name: True for name in self._task_names}

        self._configure_axes()
        self._add_h_line(y=_CHANCE_LEVEL, color=TEXT_MUTED, dashed=True, width=1.0)
        self._add_h_line(y=self._threshold, color=ALERT_RED, dashed=True, width=1.2)
        # Onset line at x = 0. Always present (the window is defined relative
        # to it); its label carries the event name, set on each freeze.
        self._onset_line = self._add_v_line(
            x=0.0, color=PRIMARY_BLUE, dashed=False, width=1.4
        )

        self._curves: dict[str, pg.PlotDataItem] = {}
        for name, color in self._task_colors.items():
            # antialias off on data curves, same rationale as the live chart
            # (cheaper paint; the static guides keep the global smooth setting).
            curve = self.plot(pen=pg.mkPen(color=color, width=2), name=name, antialias=False)
            self._curves[name] = curve

        self.getPlotItem().setTitle(_WAITING_TITLE, color=TEXT_MUTED, size="9pt")

        # Slow timer: notices when a pending epoch is complete and freezes it.
        self._check_timer = QTimer(self)
        self._check_timer.setInterval(int(round(1000 / _CHECK_HZ)))
        self._check_timer.timeout.connect(self._maybe_freeze)
        self._check_timer.start()

    # ── public API ──────────────────────────────────────────────────────────────

    @property
    def task_colors(self) -> dict[str, str]:
        """Decoder name → hex colour, matching the live chart's assignment."""
        return dict(self._task_colors)

    def append_predictions(
        self,
        predictions: dict[str, np.ndarray],
        timestamps: np.ndarray,
    ) -> None:
        """Hot path. Writes into the backing ring buffer; no scene touch.

        Same double-length ring trick as the live chart: each write mirrors
        into ``[idx]`` and ``[idx + capacity]`` so the most-recent window is
        always a contiguous slice.
        """
        if timestamps.size == 0:
            return
        if self._session_t0 is None:
            self._session_t0 = float(timestamps[0])
        n = timestamps.size
        cap = self._capacity
        idx = self._write_idx

        for i in range(n):
            ts = float(timestamps[i])
            self._timestamps[idx] = ts
            self._timestamps[idx + cap] = ts
            for name in self._task_names:
                vals = predictions.get(name)
                if vals is None or i >= len(vals):
                    continue
                v = float(vals[i])
                self._buffers[name][idx] = v
                self._buffers[name][idx + cap] = v
            idx = (idx + 1) % cap

        self._write_idx = idx
        self._latest_ts = float(timestamps[-1])

    def append_markers(self, markers: list[tuple[float, int]]) -> None:
        """Queue a pending capture for each configured marker.

        Every configured event is queued and later frozen into its own
        history snapshot, even when two events fall closer together than the
        post-event window. Codes absent from ``event_names`` are dropped, like
        the live chart.
        """
        if not markers:
            return
        for ts, code in markers:
            code = int(code)
            if code not in self._event_names:
                continue
            self._pending.append({"ts": float(ts), "code": code})

    def set_task_visible(self, name: str, visible: bool) -> None:
        """Show or hide a decoder's curve, kept in sync with the live chart's
        toggle. Data keeps flowing regardless; the next freeze (or the current
        snapshot) respects the new visibility."""
        if name not in self._visible:
            return
        self._visible[name] = visible
        curve = self._curves.get(name)
        if curve is not None:
            curve.setVisible(visible)

    @property
    def frozen_event(self) -> dict | None:
        """The snapshot currently on screen (``{"ts", "code", "name", ...}``),
        or ``None`` before the first event."""
        if self._current_index is None:
            return None
        return self._history[self._current_index]

    def history_labels(self) -> list[str]:
        """Newest-first list of short labels for a browsing control, e.g.
        ``"#3 · red · +12.3s"``. The time is **seconds since the stream
        started** (this session), not the raw LSL clock. Index aligns with
        :meth:`show_event`."""
        n = len(self._history)
        return [
            f"#{n - i} · {snap['name']} · +{snap['t_rel']:.1f}s"
            for i, snap in enumerate(self._history)
        ]

    def show_event(self, index: int) -> None:
        """Display the history snapshot at ``index`` (0 = newest). Out-of-range
        indices are ignored. Re-renders from the stored snapshot, not the live
        buffer, so old events stay viewable after their samples scroll away."""
        if not (0 <= index < len(self._history)):
            return
        self._current_index = index
        self._following = index == 0
        self._render_snapshot(self._history[index])

    @property
    def following(self) -> bool:
        """Whether new events auto-advance the on-screen view ("live")."""
        return self._following

    def set_following(self, value: bool) -> None:
        """Turn auto-follow on/off **without** changing which event is shown.

        Used to *pin* the current event: with follow off, an incoming event
        lengthens the history but the display stays put. Note ``show_event``
        re-derives follow from the index, so call this *after* selecting.
        """
        self._following = bool(value)

    def follow_latest(self) -> None:
        """Jump to the newest event and resume auto-follow ("go live")."""
        if not self._history:
            return
        self._current_index = 0
        self._following = True
        self._render_snapshot(self._history[0])

    def reset_buffers(self) -> None:
        """Blank the backing buffer, drop any pending event, clear the history
        and the curves. Called on Start so a new session begins empty."""
        self._timestamps.fill(np.nan)
        for buf in self._buffers.values():
            buf.fill(np.nan)
        self._write_idx = 0
        self._latest_ts = None
        self._session_t0 = None
        self._pending = []
        self._history = []
        self._current_index = None
        self._following = True
        for curve in self._curves.values():
            curve.setData([], [])
        self._onset_line.label.setFormat("")
        self.getPlotItem().setTitle(_WAITING_TITLE, color=TEXT_MUTED, size="9pt")

    # ── internals ───────────────────────────────────────────────────────────────

    def _maybe_freeze(self) -> None:
        """Timer tick. Freeze every pending epoch whose post-event window has
        fully streamed in (oldest first, so the newest ends up on screen under
        auto-follow). The only place that mutates the scene."""
        if not self._pending or self._latest_ts is None:
            return
        ready = [p for p in self._pending if self._latest_ts >= p["ts"] + self._post]
        if not ready:
            return  # nothing complete yet — wait for more post-event data
        ready.sort(key=lambda p: p["ts"])
        for p in ready:
            self._freeze(p["ts"], p["code"])
        # Drop everything now past its post-window — including any stale epoch
        # whose samples already scrolled out of the buffer (``_freeze`` skipped
        # it). Keep only captures still awaiting post-event data.
        self._pending = [
            p for p in self._pending if self._latest_ts < p["ts"] + self._post
        ]

    def _freeze(self, t_event: float, code: int) -> None:
        """Slice the epoch from the backing buffer into a stored snapshot,
        prepend it to the history, and display it. Auto-follow: a freshly
        captured event always becomes the on-screen view."""
        cap = self._capacity
        start = self._write_idx
        end = start + cap
        ts = self._timestamps[start:end]
        # NaN timestamps (unfilled slots) compare False, so they're excluded.
        mask = (ts >= t_event - self._pre) & (ts <= t_event + self._post)
        if not mask.any():
            return
        x = ts[mask] - t_event  # rebased: event onset at 0
        order = np.argsort(x, kind="stable")  # ring start may split the epoch
        x = x[order]
        snapshot = {
            "ts": t_event,
            # Seconds since the stream started (session-relative). ``_session_t0``
            # is set the moment any data streams in, so it's available here.
            "t_rel": t_event - (self._session_t0 if self._session_t0 is not None else t_event),
            "code": code,
            "name": self._event_names.get(code, str(code)),
            "x": x,
            "ys": {
                name: self._buffers[name][start:end][mask][order]
                for name in self._task_names
            },
        }
        # Prepend (newest-first) and cap. Inserting at 0 shifts every existing
        # snapshot's index up by one.
        self._history.insert(0, snapshot)
        del self._history[_MAX_HISTORY:]
        if self._following:
            # Auto-follow: jump to and render the new event.
            self._current_index = 0
            self._render_snapshot(snapshot)
        elif self._current_index is not None:
            # Browsing an older event — keep it on screen (no re-render), just
            # track its shifted index. Clamp in case the cap dropped the tail.
            self._current_index = min(self._current_index + 1, len(self._history) - 1)
        self.event_captured.emit(self._current_index if self._current_index is not None else 0)

    def _render_snapshot(self, snap: dict) -> None:
        """Push a stored snapshot's curves + caption onto the scene."""
        for name, curve in self._curves.items():
            curve.setData(snap["x"], snap["ys"][name])
        self._onset_line.label.setFormat(snap["name"])
        self.getPlotItem().setTitle(
            f"Event-locked: {snap['name']}", color=PRIMARY_BLUE, size="9pt"
        )

    def _configure_axes(self) -> None:
        self.setYRange(*_Y_RANGE, padding=0)
        self.setXRange(-self._pre, self._post, padding=0)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)

        plot_item = self.getPlotItem()
        plot_item.setLabel("left", "P(class = 1)")
        plot_item.setLabel("bottom", "Time (s, relative to event)")
        plot_item.showGrid(x=False, y=False)

        for ax_name in ("left", "bottom"):
            ax = plot_item.getAxis(ax_name)
            ax.setPen(QPen(QColor(BORDER_GRAY)))
            ax.setTextPen(QPen(QColor(TEXT_MUTED)))

        y_ticks = [
            (round(v, 1), f"{v:.1f}") for v in np.arange(0.0, _Y_TICK_MAX + 1e-9, 0.2)
        ]
        plot_item.getAxis("left").setTicks([y_ticks, []])

        gridline_pen = QPen(QColor(BORDER_GRAY))
        gridline_pen.setWidthF(0.8)
        gridline_pen.setCosmetic(True)
        for y in (0.2, 0.4, 0.6, 0.8):
            self.addItem(pg.InfiniteLine(pos=y, angle=0, pen=gridline_pen, movable=False))

    def _add_h_line(
        self, *, y: float, color: str, dashed: bool, width: float
    ) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            pos=y, angle=0, pen=self._pen(color, dashed, width), movable=False
        )
        self.addItem(line)
        return line

    def _add_v_line(
        self, *, x: float, color: str, dashed: bool, width: float
    ) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            pos=x,
            angle=90,
            pen=self._pen(color, dashed, width),
            movable=False,
            label="",
            labelOpts={
                "position": _LABEL_POS,
                "color": _ONSET_COLOR,
                "movable": False,
                "fill": (255, 255, 255, 235),
                "border": pg.mkPen(_ONSET_COLOR, width=1.0),
            },
        )
        font = line.label.textItem.font()
        font.setBold(True)
        line.label.textItem.setFont(font)
        self.addItem(line)
        return line

    @staticmethod
    def _pen(color: str, dashed: bool, width: float) -> QPen:
        pen = QPen(QColor(color))
        pen.setStyle(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setWidthF(width)
        pen.setCosmetic(True)
        return pen
