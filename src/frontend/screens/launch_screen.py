import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BG_LIGHT, BORDER_GRAY, CARD_WHITE, TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.logo import logo_label

logger = logging.getLogger(__name__)


class LaunchScreen(QWidget):
    """Startup pre-screen offering the two ways into the app as alternatives.

    The two paths are mutually exclusive workflows, not co-equal form fields:

    - **Start New Training** — open Phase 1 to load a config + output dir and
      walk the full training journey before going live.
    - **Open Live from Existing Output** — pick a folder a prior run already
      trained into and jump straight to Phase 2 live inference.

    Navigation goes through ``MainWindow.show_screen`` (same one-way stack the
    rest of the app uses). Heavy screens and the live-launch helpers are
    imported lazily inside the click handlers to keep startup cheap and avoid
    import cycles.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("launch_screen")
        self.setStyleSheet(f"#launch_screen {{ background: {BG_LIGHT}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch()

        root.addWidget(logo_label(110), 0, Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(16)

        title = QLabel("Live Reactivation Decoder")
        f = title.font()
        f.setPointSize(20)
        f.setWeight(QFont.Weight.DemiBold)
        title.setFont(f)
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(title)

        subtitle = QLabel("Choose how to begin")
        subtitle.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; background: transparent;"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(subtitle)
        root.addSpacing(28)

        # ── Two alternative choice cards with an OR separator ─────────────────
        choices = QHBoxLayout()
        choices.setSpacing(0)
        choices.addStretch()

        self._start_btn = QPushButton("Start New Training")
        self._start_btn.setProperty("class", "primary")
        self._start_btn.clicked.connect(self._on_start_new_clicked)
        choices.addWidget(
            self._make_choice_card(
                "Start New Training",
                "Load a config and output folder, then run the full "
                "training pipeline.",
                self._start_btn,
            )
        )

        choices.addSpacing(16)
        or_lbl = QLabel("OR")
        or_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 600; "
            "background: transparent;"
        )
        or_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        choices.addWidget(or_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        choices.addSpacing(16)

        self._live_btn = QPushButton("Open Live from Existing Output…")
        self._live_btn.setProperty("class", "primary")
        self._live_btn.clicked.connect(self._on_open_live_clicked)
        choices.addWidget(
            self._make_choice_card(
                "Open Live from Existing Output",
                "Jump straight to live inference from a folder a prior run "
                "already trained into.",
                self._live_btn,
            )
        )

        choices.addStretch()
        root.addLayout(choices)
        root.addStretch()

    # ── private builders ─────────────────────────────────────────────────────

    def _make_choice_card(
        self, title: str, blurb: str, action_btn: QPushButton
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("choice_card")
        card.setFixedSize(300, 180)
        card.setStyleSheet(
            "QFrame#choice_card {"
            f"  background: {CARD_WHITE};"
            f"  border: 1px solid {BORDER_GRAY};"
            "  border-radius: 6px;"
            "}"
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(12)

        title_lbl = QLabel(title)
        tf = title_lbl.font()
        tf.setPointSize(13)
        tf.setWeight(QFont.Weight.DemiBold)
        title_lbl.setFont(tf)
        title_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
        title_lbl.setWordWrap(True)
        v.addWidget(title_lbl)

        blurb_lbl = QLabel(blurb)
        blurb_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px; background: transparent;"
        )
        blurb_lbl.setWordWrap(True)
        v.addWidget(blurb_lbl)

        v.addStretch()
        v.addWidget(action_btn)
        return card

    # ── navigation ────────────────────────────────────────────────────────────

    def _main_window(self):
        mw = self.window()
        if mw is None or not hasattr(mw, "show_screen"):
            return None
        return mw

    def _on_start_new_clicked(self) -> None:
        mw = self._main_window()
        if mw is None:
            return
        from frontend.screens.phase1_screen import Phase1Screen

        mw.show_screen(Phase1Screen())

    def _on_open_live_clicked(self) -> None:
        """Pick an existing output folder and jump straight to Phase 2.

        Mirrors the validate-then-build flow that previously lived on the
        Settings view: a missing file or a corrupt config keeps the operator
        here with a clear message instead of a half-built screen.
        """
        path = QFileDialog.getExistingDirectory(
            self, "Select an existing output folder"
        )
        if not path:
            return

        from frontend.screens.phase2_launch import (
            build_phase2_from_output, missing_live_artifacts,
        )

        missing = missing_live_artifacts(path)
        if missing:
            QMessageBox.critical(
                self, "Not a live-ready output folder",
                f"{path}\n\nMissing required file(s):\n"
                + "\n".join(f"  • {name}" for name in missing),
            )
            return

        mw = self._main_window()
        if mw is None:
            return
        try:
            phase2 = build_phase2_from_output(path)
        except Exception as exc:
            logger.exception("Failed to open live inference from output folder")
            QMessageBox.critical(
                self, "Could not open live inference",
                f"Failed to open live mode from:\n{path}\n\n{exc}",
            )
            return
        mw.show_screen(phase2)
