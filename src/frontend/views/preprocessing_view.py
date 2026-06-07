from __future__ import annotations

import logging
import math

from PyQt6.QtCore import Qt, QEvent, QEventLoop, QObject, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, CARD_WHITE, PRIMARY_BLUE, SUCCESS_GREEN, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.workers.preprocessing_worker import (
    PreprocessingStep1AWorker, PreprocessingStep1BWorker, PreprocessingStep2Worker,
)

logger = logging.getLogger(__name__)


class _WaitForClose(QObject):
    """Block on a nested QEventLoop until ``widget`` receives a close event.

    Why this exists: ``raw.plot(block=True)`` and
    ``ica.plot_sources(..., block=True)`` from mne-qt-browser don't actually
    block when invoked inside our already-running ``QApplication.exec()`` —
    Qt rejects the nested top-level exec ("event loop is already running")
    and returns immediately, so the rest of the pipeline races ahead before
    the operator has marked bad channels / picked ICA components. The
    supported pattern is to run a private ``QEventLoop`` and quit it when
    the figure window emits its close event.
    """

    def __init__(self, widget: QWidget) -> None:
        super().__init__()
        self._loop = QEventLoop()
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 — Qt method name
        if event.type() == QEvent.Type.Close:
            self._loop.quit()
        return False

    def wait(self) -> None:
        self._loop.exec()


class _WaitForAllFigsClose(QObject):
    """Block on a nested QEventLoop until every matplotlib figure has closed.

    Used for the ICA topomap-grid review: ``ica.plot_components(inst=epochs)``
    returns a list of matplotlib figures (~1 per 20 components). Each
    figure's canvas exposes matplotlib's ``close_event`` signal, which
    fires exactly once when that figure is destroyed. We count down to
    zero and quit the nested loop when the last one closes.
    """

    def __init__(self, figs) -> None:
        super().__init__()
        self._loop = QEventLoop()
        figs = [
            f for f in (figs if isinstance(figs, (list, tuple)) else [figs])
            if f is not None
        ]
        self._remaining = len(figs)
        for f in figs:
            f.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, _evt) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            self._loop.quit()

    def wait(self) -> None:
        if self._remaining > 0:
            self._loop.exec()


class PreprocessingView(QWidget):
    """Node 3 workspace: 2-page stack (Ready → Preprocessing Complete).

    The two manual selections (bad channels, ICA components) happen on MNE's
    native interactive windows, which must run on the GUI main thread. The
    flow is therefore automatic once started:

      Ready ──trigger_start──▶ Step1A worker (filter)
        └▶ main thread: raw.plot(block=True)  ← operator marks bads
           └▶ set_bad_channels → Step1B worker (fit ICA)
              └▶ main thread: ica.plot_sources(block=True)  ← operator toggles
                 └▶ Step2 worker (apply + save) ──▶ Complete

    While an MNE window is open the app shows a "waiting" overlay; the
    journey-panel Node 3 button only drives ``trigger_start`` (page 0) and
    ``trigger_continue`` (page 1, advance to Node 4).
    """

    # Loading-overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel Node 3 action button
    ready_changed = Signal(bool)
    # Emitted once Step 2 finished and the complete page is displayed.
    # Phase1Screen rebinds the Node 3 button to trigger_continue.
    step2_complete = Signal()
    # Emitted when the user clicks the rebound panel button → advance trail.
    preprocessing_complete = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._data_loaded: bool = False
        self._running: bool = False
        self._done: bool = False
        self._excluded_count: int = 0
        self._epochs_count: int = 0
        self._bad_channels: list[str] = []
        self._was_ready: bool = False
        self._thread: QThread | None = None
        self._worker = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_ready_page())
        self._pages.addWidget(self._build_complete_page())
        outer.addWidget(self._pages)

    # ── public ───────────────────────────────────────────────────────────────

    def set_session(self, session) -> None:
        """Provide the AppSession built by Node 1. Called by Phase1Screen."""
        self._session = session
        self._update_ready_state()

    def on_data_loaded(self) -> None:
        """Slot connected by Phase1Screen to LoadDataView.data_loaded."""
        self._data_loaded = True
        self._update_ready_state()

    def trigger_start(self) -> None:
        """Start the pipeline (Step 1A). Wired to the journey-panel Node 3 button."""
        if (
            self._session is None
            or self._session.offline is None
            or not self._data_loaded
            or self._running
            or self._done
        ):
            return
        self._running = True
        self._update_ready_state()
        self._start_worker(
            PreprocessingStep1AWorker(self._session.offline),
            "Running preprocessing…",
            self._on_step1a_done,
        )

    def trigger_continue(self) -> None:
        """Advance the journey trail to Node 4 once Step 2 has finished."""
        if not self._done:
            return
        self.preprocessing_complete.emit()

    # ── worker plumbing ──────────────────────────────────────────────────────

    def _start_worker(self, worker, message: str, on_done) -> None:
        self.loading_requested.emit(message)
        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.result_ready.connect(on_done)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _on_error(self, message: str) -> None:
        self._running = False
        self.loading_done.emit()
        QMessageBox.critical(self, "Preprocessing Error", message)
        self._update_ready_state()

    # ── main-thread MNE interaction ──────────────────────────────────────────

    def _wait_overlay(self, message: str) -> None:
        """Show the overlay and force a repaint before a blocking MNE window."""
        self.loading_requested.emit(message)
        QApplication.processEvents()

    @staticmethod
    def _annotate_ica_titles(figs, component_labels) -> None:
        """Append each component's ICLabel category + confidence to its
        ``plot_components`` subplot title (e.g. ``ICA001`` → ``ICA001 - eye 98%``).

        ``component_labels`` is a per-index list of ``(category, confidence)``
        (from ``AppSession.offline.ica_component_labels()``), or ``None`` when
        ICLabel is disabled — in which case titles are left untouched.

        Two MNE internals (mne/viz/topomap.py, plot_ica_components) make this
        safe rather than fragile:
          • The title-click toggle recovers the component index by parsing the
            title text — ``int(title.split(" ")[0][-3:])``. We append *after a
            space* so the first token stays ``ICAxyz`` and the parse still
            yields the right index.
          • On toggle MNE re-*colours* the title (gray = reject) but never
            re-sets its text, so the appended category survives reject/keep
            clicks. We mutate the Text in place (``set_text``) to preserve
            MNE's existing colour/size rather than reset them via set_title.
        """
        if not component_labels:
            return
        for fig in (figs if isinstance(figs, (list, tuple)) else [figs]):
            if fig is None:
                continue
            for ax in fig.axes:
                title = ax.get_title()
                if not title.startswith("ICA"):
                    continue
                try:
                    idx = int(title.split(" ")[0][-3:])
                except (ValueError, IndexError):
                    continue
                if 0 <= idx < len(component_labels):
                    label, proba = component_labels[idx]
                    ax.title.set_text(f"{title} - {label} {proba:.0%}")
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

    @staticmethod
    def _close_figs(figs) -> None:
        """Close one or more matplotlib figures (idempotent)."""
        if figs is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover — matplotlib is a hard dep here
            return
        for fig in (figs if isinstance(figs, (list, tuple)) else [figs]):
            try:
                plt.close(fig)
            except Exception:
                pass

    def _on_step1a_done(self, raw) -> None:
        # Worker thread is quitting; the MNE window must run on this (main) thread.
        try:
            self._wait_overlay(
                "Mark bad channels in the MNE window, then close it to continue…"
            )
            # block=False + nested QEventLoop is the working substitute for
            # block=True, which is a no-op inside QApplication.exec().
            fig = raw.plot(block=False)
            _WaitForClose(fig).wait()
            bads = list(raw.info["bads"])
            logger.info(
                "Bad-channel review closed; operator selected %d channel(s): %s",
                len(bads), bads,
            )
            self._session.offline.set_bad_channels(bads)
            self._bad_channels = bads
        except Exception as exc:  # pragma: no cover — display/runtime guard
            self._on_error(f"Bad-channel review failed: {exc}")
            return

        self._start_worker(
            PreprocessingStep1BWorker(self._session.offline),
            "Fitting ICA…",
            self._on_step1b_done,
        )

    def _on_step1b_done(self, payload) -> None:
        ica, epochs, suggested = payload
        try:
            ica.exclude = list(suggested)
            self._wait_overlay(
                "Review ICA components — click a component's title to "
                "toggle reject/keep (greyed = reject), click the topomap "
                "for the detail window. Close all topomap windows when "
                "done."
            )
            # MNE's plot_components(inst=epochs) returns one matplotlib
            # figure per ~20 components and wires up two click handlers:
            #   • click a subplot title → toggle ica.exclude (in-place)
            #   • click the topomap     → open ica.plot_properties
            # We rely on those native interactions and wait for the
            # operator to close every figure to signal "done".
            # ICLabel's per-component category + confidence are appended to
            # each subplot title below (after plot_components builds them) via
            # _annotate_ica_titles, so the operator sees what ICLabel thought
            # each component was — not just the implicit greyed-out reject.
            # TODO(ui): we deliberately do NOT show ica.plot_sources(epochs)
            # alongside the topomap grids. Time-series view sometimes
            # makes artifacts (blinks, heart) more obvious than the
            # topomap, but mne-qt-browser 0.7.5 has a precompute bug on
            # the Epochs path (workaround precompute=False) and pops
            # another window. Revisit once we have UX feedback.
            # sources_fig = ica.plot_sources(epochs, block=False, precompute=False)
            # _WaitForClose(sources_fig).wait()
            # Single-figure layout: pack all components into one near-square
            # grid (ncols = ceil(sqrt(n)), nrows fills the rest). size=1.2
            # makes each topomap ~1.2 inch so a 61-component grid lands
            # around 10×10 inches — readable on a 1080p workspace without
            # multiple windows. Operator interactions (title click → toggle
            # exclude, topomap click → properties) work identically.
            n = int(ica.n_components_)
            ncols = max(1, math.ceil(math.sqrt(n)))
            nrows = max(1, math.ceil(n / ncols))
            topomap_figs = ica.plot_components(
                inst=epochs, ncols=ncols, nrows=nrows, size=1.2,
            )
            # Append ICLabel category + confidence to each subplot title.
            # Indexed by component, fetched from the orchestrator (populated
            # during the ICA fit). None when ICLabel is disabled → titles
            # stay as MNE drew them.
            self._annotate_ica_titles(
                topomap_figs, self._session.offline.ica_component_labels()
            )
            # Auto-maximize so all components are visible regardless of
            # screen size. The natural figure size (ncols×size inches)
            # easily exceeds a non-fullscreen workspace; maximizing lets
            # tight_layout reflow the grid to fill whatever the screen
            # gives us. Defensive try/except in case a future matplotlib
            # backend doesn't expose manager.window.
            for f in (topomap_figs if isinstance(topomap_figs, (list, tuple))
                      else [topomap_figs]):
                try:
                    f.canvas.manager.window.showMaximized()
                except Exception:
                    pass
            _WaitForAllFigsClose(topomap_figs).wait()
            excluded = list(ica.exclude)
            logger.info(
                "ICA review closed; operator selected %d component(s) "
                "(suggested by ICLabel: %s; final: %s)",
                len(excluded), list(suggested), excluded,
            )
            self._close_figs(topomap_figs)
        except Exception as exc:  # pragma: no cover — display/runtime guard
            self._on_error(f"ICA review failed: {exc}")
            return

        self._excluded_count = len(excluded)
        self._start_worker(
            PreprocessingStep2Worker(self._session.offline, excluded),
            "Finishing preprocessing pipeline…",
            self._on_step2_done,
        )

    def _on_step2_done(self, payload) -> None:
        self.loading_done.emit()
        n_epochs = int(payload.get("n_epochs", 0)) if isinstance(payload, dict) else 0
        n_excluded = (
            int(payload.get("n_excluded", self._excluded_count))
            if isinstance(payload, dict) else self._excluded_count
        )
        self._epochs_count = n_epochs
        self._excluded_count = n_excluded
        self._epochs_value.setText(str(n_epochs))
        self._components_value.setText(str(n_excluded))
        self._pages.setCurrentIndex(1)
        self._running = False
        self._done = True
        self._update_ready_state()
        self.step2_complete.emit()

    # ── page builders ────────────────────────────────────────────────────────

    def _build_ready_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._start_btn = QPushButton("▶")
        self._start_btn.setFixedSize(96, 96)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setStyleSheet(
            # padding-left compensates for the ▶ glyph's intrinsic
            # left-bias inside its character cell — without it the
            # triangle looks shifted left of centre, especially against
            # the more-saturated hover/press background.
            f"QPushButton {{ background: #EFF6FF; color: {PRIMARY_BLUE}; "
            f"border: none; border-radius: 48px; font-size: 36px; "
            f"padding: 0 0 0 6px; }}"
            f"QPushButton:hover {{ background: #DBEAFE; }}"
            f"QPushButton:pressed {{ background: #BFDBFE; }}"
        )
        self._start_btn.clicked.connect(self.trigger_start)
        center.addWidget(self._start_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        title = QLabel("Ready to Preprocess")
        f = title.font()
        f.setPointSize(16)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel(
            "Click play to begin. You will mark bad channels and review ICA "
            "components in MNE's interactive windows; the app waits while "
            "each window is open."
        )
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(380)
        desc.setMinimumHeight(48)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _build_complete_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 32, 48, 32)
        layout.setSpacing(0)
        layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(0)
        center.setAlignment(Qt.AlignmentFlag.AlignCenter)

        badge = QLabel("✓")
        badge.setFixedSize(72, 72)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"QLabel {{ background: #F0FDF4; color: {SUCCESS_GREEN}; "
            f"border: 1px solid #DCFCE7; border-radius: 36px; font-size: 32px; }}"
        )
        center.addWidget(badge, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(24)

        title = QLabel("Preprocessing Complete")
        f = title.font()
        f.setPointSize(18)
        f.setWeight(QFont.Weight.Medium)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(title)
        center.addSpacing(10)

        desc = QLabel("Cleaned epochs are ready for model evaluation.")
        desc.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        desc.setFixedWidth(440)
        center.addWidget(desc, 0, Qt.AlignmentFlag.AlignHCenter)
        center.addSpacing(28)

        stats_card = QFrame()
        stats_card.setFixedWidth(460)
        stats_card.setStyleSheet(
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; "
            f"border-radius: 6px; }}"
        )
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(24, 8, 24, 8)
        stats_layout.setSpacing(0)

        self._epochs_value = QLabel("—")
        self._components_value = QLabel("—")

        self._append_stat_row(stats_layout, "Epochs retained", self._epochs_value)
        self._append_separator(stats_layout)
        self._append_stat_row(
            stats_layout, "ICA components removed", self._components_value
        )

        center.addWidget(stats_card, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(center)
        layout.addStretch()
        return page

    def _append_stat_row(
        self, layout: QVBoxLayout, caption: str, value_label: QLabel
    ) -> None:
        row = QWidget()
        row.setStyleSheet("background: transparent; border: none;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 14, 0, 14)
        row_layout.setSpacing(12)

        cap = QLabel(caption)
        cap.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; background: transparent; border: none;"
        )
        value_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; "
            f"background: transparent; border: none;"
        )
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row_layout.addWidget(cap, 1)
        row_layout.addWidget(value_label, 0)
        layout.addWidget(row)

    def _append_separator(self, layout: QVBoxLayout) -> None:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER_GRAY}; border: none;")
        layout.addWidget(sep)

    # ── ready gating ─────────────────────────────────────────────────────────

    def _update_ready_state(self) -> None:
        # Page 0 → trigger_start (ready when data loaded, nothing running)
        # Page 1 → trigger_continue (ready once Step 2 finished)
        page0_ready = (
            self._session is not None
            and self._data_loaded
            and not self._running
            and not self._done
        )
        ready = page0_ready or self._done
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)
