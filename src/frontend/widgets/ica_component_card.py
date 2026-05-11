from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
)

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import mne

from frontend.styles.theme import (
    ALERT_RED, AMBER, BORDER_GRAY, CARD_WHITE, TEXT_MUTED, TEXT_PRIMARY,
)


# ─── TODO: enrich per-component diagnostics ─────────────────────────────────
# The card currently shows only the topomap and the EOG/ECG-derived suggested
# badge. Three additional signals were considered and deferred — track them
# together so the backend exposure and UI placement are designed as one piece
# rather than three separate one-offs. See
# docs/Phase1_UI_Plan.md → Node 3 → "Deferred per-component enhancements"
# for the full rationale and decisions still to make.
#
# 1. Time-series preview (per-component source)
#    - Source data via ica.get_sources(raw).get_data()[i, :N]. Reveals blinks
#      (sharp transients), ECG (rhythmic), muscle (broadband high-freq).
#    - Backend exposure required: raw is private to the orchestrator; the
#      frontend must not access internals.
#
# 2. PSD (power spectral density)
#    - Single most informative signal after the topomap for artifact triage
#      (50/60 Hz line spikes, <4 Hz eye dominance, >30 Hz muscle, alpha brain).
#    - Cheap to compute backend-side (FFT on the per-component source).
#    - Card is too small (260×240 with the topomap already there) to fit a
#      readable PSD inline — likely belongs in an "inspect" detail dialog.
#
# 3. ICLabel category + confidence
#    - mne-icalabel classifies each component as
#      brain/muscle/eye/heart/line_noise/channel_noise/other with a 7-way
#      probability vector. Adds a coloured class badge alongside the existing
#      amber "SUGGESTED REJECT" badge.
#    - Caveats: ICLabel was trained on 1–100 Hz bandpass + extended-infomax
#      ICA decompositions. Our pipeline currently uses a narrower bandpass
#      and fastica — feeding ICLabel off-distribution data degrades the
#      confidences. Switching the ICA method to picard(extended=True,
#      ortho=False) is a one-line config fix; honouring the bandpass would
#      require a pipeline reorder so the production low-pass is applied
#      after ICA.
#
# Recommended ordering when these are picked up: (1) make the backend
# decision (probably: extend the diagnostics dict returned by
# run_step1_prepare_ica with per-component fields), (2) build an
# ICAComponentInspectDialog for the heavier visuals, (3) add the ICLabel
# badge to the overview card last — once the backend can produce
# calibrated outputs.
# ────────────────────────────────────────────────────────────────────────────


class ICAComponentCard(QFrame):
    """One ICA component card: topomap + Keep/Reject toggle."""

    state_changed = Signal(int, bool)  # (component_index, is_rejected)

    def __init__(self, ica: "mne.preprocessing.ICA", component_index: int,
                 is_suggested: bool, parent=None):
        super().__init__(parent)
        self._index = component_index

        self.setObjectName("ica_card")
        self.setStyleSheet(
            f"QFrame#ica_card {{ background: {CARD_WHITE}; "
            f"border: 1px solid {BORDER_GRAY}; border-radius: 6px; }}"
        )
        self.setFixedSize(260, 240)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel(f"Component {component_index}")
        f = title.font()
        f.setPointSize(10)
        f.setBold(True)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; border: none;")
        header.addWidget(title)
        header.addStretch()

        badge = QLabel("SUGGESTED REJECT" if is_suggested else "KEEP")
        badge_color = AMBER if is_suggested else TEXT_MUTED
        badge.setStyleSheet(
            f"color: {badge_color}; background: transparent; "
            f"border: 1px solid {badge_color}; border-radius: 4px; "
            f"padding: 1px 6px; font-size: 9px; font-weight: 700;"
        )
        header.addWidget(badge)
        outer.addLayout(header)

        # Matplotlib canvas: topomap only. Time-series / PSD / ICLabel
        # enrichments are deferred — see the TODO block at the top of this file.
        fig = Figure(figsize=(2.4, 1.4), tight_layout=True)
        fig.patch.set_facecolor(CARD_WHITE)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ax_topo = fig.add_subplot(111)

        try:
            mne.viz.plot_topomap(
                ica.get_components()[:, component_index],
                ica.info,
                axes=ax_topo,
                show=False,
                contours=0,
            )
        except Exception as exc:  # pragma: no cover — defensive: bad info shouldn't crash the grid
            ax_topo.text(0.5, 0.5, "topomap unavailable",
                         ha="center", va="center", fontsize=7, color=TEXT_MUTED,
                         transform=ax_topo.transAxes)
            ax_topo.set_axis_off()
            print(f"[ICAComponentCard] topomap failed for component {component_index}: {exc}")

        self._canvas.draw()
        outer.addWidget(self._canvas, 1)

        # Toggle button — checked = Reject, unchecked = Keep
        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(is_suggested)
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._apply_toggle_style(is_suggested)

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def is_rejected(self) -> bool:
        return self._toggle.isChecked()

    # ── private ──────────────────────────────────────────────────────────────

    def _on_toggled(self, checked: bool) -> None:
        self._apply_toggle_style(checked)
        self.state_changed.emit(self._index, checked)

    def _apply_toggle_style(self, rejected: bool) -> None:
        if rejected:
            self._toggle.setText("Reject")
            self._toggle.setStyleSheet(
                f"QPushButton {{ background: {CARD_WHITE}; color: {ALERT_RED}; "
                f"border: 1px solid {ALERT_RED}; border-radius: 4px; "
                f"padding: 4px 0; font-size: 11px; font-weight: 600; }}"
                f"QPushButton:hover {{ background: #FDECEE; }}"
            )
        else:
            self._toggle.setText("Keep")
            self._toggle.setStyleSheet(
                f"QPushButton {{ background: {CARD_WHITE}; color: {TEXT_PRIMARY}; "
                f"border: 1px solid {BORDER_GRAY}; border-radius: 4px; "
                f"padding: 4px 0; font-size: 11px; font-weight: 600; }}"
                f"QPushButton:hover {{ background: #F3F4F6; }}"
            )
