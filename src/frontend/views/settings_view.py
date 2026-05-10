from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox,
    QScrollArea, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, BG_LIGHT, PRIMARY_BLUE, SUCCESS_GREEN,
    TEXT_MUTED, TEXT_PRIMARY,
)
from frontend.widgets.shared import FilePicker, ReadOnlyField, SectionCard
from frontend.workers.config_loader_worker import ConfigLoaderWorker


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


class SettingsView(QWidget):
    """Node 1 workspace: config + output dir pickers, read-only settings display."""

    # Emits fully-configured AppSession when Continue is clicked successfully
    session_ready = Signal(object)
    # Loading overlay protocol — handled by Phase1Screen
    loading_requested = Signal(str)
    loading_done = Signal()
    # Ready protocol — gates the journey-panel action button for this node
    ready_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config_path: str | None = None
        self._output_dir: str | None = None
        self._temp_session = None  # AppSession after config load, before orchestrator
        self._config_thread: QThread | None = None
        self._config_worker: ConfigLoaderWorker | None = None
        self._was_ready: bool = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        self._main = QVBoxLayout(container)
        self._main.setContentsMargins(32, 24, 32, 24)
        self._main.setSpacing(0)

        inner = QWidget()
        inner.setFixedWidth(680)
        self._content = QVBoxLayout(inner)
        self._content.setContentsMargins(0, 0, 0, 0)
        self._content.setSpacing(16)

        self._build_setup_section()
        self._build_preproc_section()
        self._build_model_section()
        self._content.addStretch()

        self._main.addWidget(inner, 0, Qt.AlignmentFlag.AlignHCenter)
        self._main.addStretch()

        self._update_settings_display(None)
        self._update_continue_state()

    # ── public ───────────────────────────────────────────────────────────────

    def trigger_continue(self) -> None:
        """Build the AppSession and emit `session_ready`.

        Wired to the journey-panel Node 1 action button by Phase1Screen.
        Safe no-op when prerequisites are missing (panel button is gated, but
        this guard keeps the slot self-contained).
        """
        if not (self._temp_session and self._output_dir):
            return
        try:
            self._temp_session.configure_output(self._output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Session Error", str(exc))
            self._temp_session = None
            self._config_path = None
            self._config_picker.clear()
            self._config_status_lbl.hide()
            self._update_settings_display(None)
            self._update_continue_state()
            return
        self.session_ready.emit(self._temp_session)

    # ── private builders ─────────────────────────────────────────────────────

    def _build_setup_section(self) -> None:
        card = SectionCard("Setup")

        config_row = QHBoxLayout()
        config_row.setSpacing(10)
        self._config_picker = FilePicker(
            "Load Config File", mode="file",
            file_filter="Config (*.yaml *.yml)"
        )
        self._config_picker.path_selected.connect(self._on_config_selected)
        config_row.addWidget(self._config_picker)

        self._config_status_lbl = QLabel("✓  Config loaded")
        self._config_status_lbl.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-size: 11px; font-weight: 600;"
        )
        self._config_status_lbl.hide()
        config_row.addWidget(self._config_status_lbl)
        config_row.addStretch()
        card.body.addLayout(config_row)

        self._output_picker = FilePicker("Select Output Directory", mode="dir")
        self._output_picker.path_selected.connect(self._on_output_dir_selected)
        card.body.addWidget(self._output_picker)

        self._content.addWidget(card)

    def _build_preproc_section(self) -> None:
        card = SectionCard("Preprocessing")

        # Bandpass: [l_freq] to [h_freq] Hz   Method: xxx   Notch: xxx Hz
        r = QHBoxLayout()
        r.setSpacing(6)
        r.addWidget(self._field_label("Bandpass"))
        self._l_freq = ReadOnlyField("", field_width=72)
        self._h_freq = ReadOnlyField("", field_width=72)
        self._bp_method = self._dim_label("Method: —")
        self._notch = self._dim_label("Notch: —")
        r.addWidget(self._l_freq)
        r.addWidget(QLabel("to"))
        r.addWidget(self._h_freq)
        r.addWidget(self._dim_label("Hz"))
        r.addSpacing(8)
        r.addWidget(self._bp_method)
        r.addSpacing(6)
        r.addWidget(self._notch)
        r.addStretch()
        card.body.addLayout(r)

        # Resample: [target_rate] Hz
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        r2.addWidget(self._field_label("Resample"))
        self._resample = ReadOnlyField("", field_width=88)
        r2.addWidget(self._resample)
        r2.addWidget(self._dim_label("Hz"))
        r2.addStretch()
        card.body.addLayout(r2)

        # ICA: [n_components] components   Method: xxx   Seed: xxx
        r3 = QHBoxLayout()
        r3.setSpacing(6)
        r3.addWidget(self._field_label("ICA"))
        self._ica_n = ReadOnlyField("", field_width=72)
        self._ica_method = self._dim_label("Method: —")
        self._ica_seed = self._dim_label("Seed: —")
        r3.addWidget(self._ica_n)
        r3.addWidget(self._dim_label("components"))
        r3.addSpacing(8)
        r3.addWidget(self._ica_method)
        r3.addSpacing(6)
        r3.addWidget(self._ica_seed)
        r3.addStretch()
        card.body.addLayout(r3)

        # Epoch Size: [tmin] to [tmax] s
        r4 = QHBoxLayout()
        r4.setSpacing(6)
        r4.addWidget(self._field_label("Epoch Size"))
        self._tmin = ReadOnlyField("", field_width=80)
        self._tmax = ReadOnlyField("", field_width=80)
        r4.addWidget(self._tmin)
        r4.addWidget(QLabel("to"))
        r4.addWidget(self._tmax)
        r4.addWidget(self._dim_label("s"))
        r4.addStretch()
        card.body.addLayout(r4)

        card.body.addSpacing(4)
        card.body.addWidget(self._field_label("Annotations Mapping"))
        self._annot_container = QWidget()
        self._annot_layout = QVBoxLayout(self._annot_container)
        self._annot_layout.setContentsMargins(0, 4, 0, 0)
        self._annot_layout.setSpacing(2)
        card.body.addWidget(self._annot_container)

        self._content.addWidget(card)

    def _build_model_section(self) -> None:
        card = SectionCard("Model Evaluation")

        model_row = QHBoxLayout()
        model_row.setSpacing(0)
        model_row.addWidget(self._field_label("Model"))
        model_row.addSpacing(4)
        self._model_labels: dict[str, QLabel] = {}
        for key, text in [("LDA", "LDA"), ("Logistic", "Logistic Regression"), ("SVM", "SVM")]:
            lbl = QLabel(text)
            lbl.setContentsMargins(10, 4, 10, 4)
            lbl.setStyleSheet(
                f"background: #F3F4F6; color: {TEXT_MUTED}; "
                f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; font-size: 12px;"
            )
            model_row.addWidget(lbl)
            model_row.addSpacing(4)
            self._model_labels[key] = lbl
        model_row.addStretch()
        card.body.addLayout(model_row)

        cv_row = QHBoxLayout()
        cv_row.setSpacing(6)
        cv_row.addWidget(self._field_label("CV Folds"))
        self._cv_folds = ReadOnlyField("", field_width=72)
        cv_row.addWidget(self._cv_folds)
        cv_row.addStretch()
        card.body.addLayout(cv_row)

        card.body.addSpacing(4)
        card.body.addWidget(self._field_label("Decoders"))
        self._decoders_container = QWidget()
        self._decoders_layout = QVBoxLayout(self._decoders_container)
        self._decoders_layout.setContentsMargins(0, 4, 0, 0)
        self._decoders_layout.setSpacing(4)
        card.body.addWidget(self._decoders_container)

        self._content.addWidget(card)

    # ── widget factories ─────────────────────────────────────────────────────

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 600;"
        )
        lbl.setFixedWidth(110)
        return lbl

    @staticmethod
    def _dim_label(text: str = "") -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px;")
        return lbl

    # ── private slots ─────────────────────────────────────────────────────────

    def _on_config_selected(self, path: str) -> None:
        self._config_picker.setEnabled(False)
        self.loading_requested.emit("Loading configuration…")

        self._config_thread = QThread()
        self._config_worker = ConfigLoaderWorker(path)
        self._config_worker.moveToThread(self._config_thread)

        self._config_thread.started.connect(self._config_worker.run)
        self._config_worker.result_ready.connect(self._on_config_loaded)
        self._config_worker.error_occurred.connect(self._on_config_error)
        self._config_worker.finished.connect(self._config_thread.quit)
        self._config_thread.finished.connect(self._config_worker.deleteLater)
        self._config_thread.finished.connect(self._config_thread.deleteLater)
        self._config_thread.finished.connect(self._on_config_thread_finished)

        self._config_thread.start()

    def _on_config_loaded(self, session) -> None:
        self._config_path = self._config_picker.path
        self._temp_session = session
        self._config_status_lbl.show()
        self._update_settings_display(session.settings)
        self._update_continue_state()
        self._config_picker.setEnabled(True)
        self.loading_done.emit()

    def _on_config_error(self, message: str) -> None:
        self.loading_done.emit()
        QMessageBox.critical(self, "Config Error", message)
        self._config_picker.clear()
        self._config_status_lbl.hide()
        self._temp_session = None
        self._config_path = None
        self._update_settings_display(None)
        self._update_continue_state()
        self._config_picker.setEnabled(True)

    def _on_config_thread_finished(self) -> None:
        """Drop Python refs only after the QThread is fully stopped."""
        self._config_thread = None
        self._config_worker = None

    def _on_output_dir_selected(self, path: str) -> None:
        self._output_dir = path
        self._update_continue_state()

    def _update_continue_state(self) -> None:
        ready = bool(self._config_path and self._output_dir)
        if ready != self._was_ready:
            self._was_ready = ready
            self.ready_changed.emit(ready)

    # ── settings population ───────────────────────────────────────────────────

    def _update_settings_display(self, settings: dict | None) -> None:
        if settings is None:
            self._l_freq.set_value(None)
            self._h_freq.set_value(None)
            self._bp_method.setText("Method: —")
            self._notch.setText("Notch: —")
            self._resample.set_value(None)
            self._ica_n.set_value(None)
            self._ica_method.setText("Method: —")
            self._ica_seed.setText("Seed: —")
            self._tmin.set_value(None)
            self._tmax.set_value(None)
            self._cv_folds.set_value(None)
            _clear_layout(self._annot_layout)
            _clear_layout(self._decoders_layout)
            for lbl in self._model_labels.values():
                lbl.setStyleSheet(
                    f"background: #F3F4F6; color: {TEXT_MUTED}; "
                    f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; font-size: 12px;"
                )
            return

        pre = settings["preprocessing"]
        dec = settings["decoders"]
        em  = settings["event_mapping"]

        bp = pre["bandpass"]
        self._l_freq.set_value(bp["l_freq"])
        self._h_freq.set_value(bp["h_freq"])
        self._bp_method.setText(f"Method: {bp['method']}")
        notch_val = bp.get("notch")
        self._notch.setText(
            f"Notch: {notch_val} Hz" if notch_val is not None else "Notch: —"
        )
        self._resample.set_value(pre["resample"]["target_rate"])

        ica = pre["ica"]
        self._ica_n.set_value(ica["n_components"])
        self._ica_method.setText(f"Method: {ica['method']}")
        self._ica_seed.setText(f"Seed: {pre['random_state']}")

        ep = pre["epochs"]
        self._tmin.set_value(ep["tmin"])
        self._tmax.set_value(ep["tmax"])

        # Annotations table
        _clear_layout(self._annot_layout)
        if em:
            tbl = QWidget()
            tbl.setStyleSheet(
                f"QWidget {{ background: white; border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
            )
            tbl_v = QVBoxLayout(tbl)
            tbl_v.setContentsMargins(0, 0, 0, 0)
            tbl_v.setSpacing(0)

            hdr = QWidget()
            hdr.setStyleSheet(
                f"QWidget {{ background: {BG_LIGHT}; border-bottom: 1px solid {BORDER_GRAY}; }}"
            )
            hdr_row = QHBoxLayout(hdr)
            hdr_row.setContentsMargins(10, 6, 10, 6)
            for text, width in [("CODE", 60), ("CLASS LABEL", 0)]:
                lbl = QLabel(text)
                lbl.setStyleSheet(
                    f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 600; border: none;"
                )
                if width:
                    lbl.setFixedWidth(width)
                hdr_row.addWidget(lbl)
            hdr_row.addStretch()
            tbl_v.addWidget(hdr)

            for name, id_ in em.items():
                row_w = QWidget()
                row_w.setStyleSheet(
                    f"QWidget {{ border-bottom: 1px solid {BORDER_GRAY}; background: white; }}"
                )
                row = QHBoxLayout(row_w)
                row.setContentsMargins(10, 5, 10, 5)
                code_lbl = QLabel(str(id_))
                code_lbl.setStyleSheet(
                    "font-family: monospace; font-size: 12px; border: none;"
                )
                code_lbl.setFixedWidth(60)
                name_lbl = QLabel(name)
                name_lbl.setStyleSheet(
                    f"font-size: 12px; color: {TEXT_PRIMARY}; border: none;"
                )
                row.addWidget(code_lbl)
                row.addWidget(name_lbl)
                row.addStretch()
                tbl_v.addWidget(row_w)

            self._annot_layout.addWidget(tbl)
        else:
            ph = QLabel("No annotations")
            ph.setStyleSheet(f"color: {TEXT_MUTED}; font-style: italic; font-size: 11px;")
            self._annot_layout.addWidget(ph)

        # Model selector badges
        active_model = dec["model"]
        for key, lbl in self._model_labels.items():
            if key == active_model:
                lbl.setStyleSheet(
                    f"background: {PRIMARY_BLUE}; color: white; "
                    f"border: 1px solid {PRIMARY_BLUE}; border-radius: 2px; "
                    f"font-size: 12px; font-weight: 600;"
                )
            else:
                lbl.setStyleSheet(
                    f"background: #F3F4F6; color: {TEXT_MUTED}; "
                    f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; font-size: 12px;"
                )

        self._cv_folds.set_value(dec["cv"]["k"])

        # Decoder task cards
        _clear_layout(self._decoders_layout)
        tasks = dec.get("tasks", [])
        if tasks:
            for task in tasks:
                self._decoders_layout.addWidget(self._make_decoder_card(task))
        else:
            ph = QLabel("No decoders configured")
            ph.setStyleSheet(f"color: {TEXT_MUTED}; font-style: italic; font-size: 11px;")
            self._decoders_layout.addWidget(ph)

    def _make_decoder_card(self, task: dict) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {BG_LIGHT}; border: 1px solid {BORDER_GRAY}; border-radius: 2px; }}"
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)

        name_lbl = QLabel(task["name"])
        f = name_lbl.font()
        f.setWeight(QFont.Weight.DemiBold)
        f.setPointSize(10)
        name_lbl.setFont(f)
        name_lbl.setStyleSheet(f"color: {TEXT_PRIMARY};")
        v.addWidget(name_lbl)

        for labels, bg, fg, prefix in [
            (task.get("pos_labels", []), "#DBEAFE", "#1D4ED8", "+"),
            (task.get("neg_labels", []), "#FEE2E2", "#DC2626", "−"),
        ]:
            if not labels:
                continue
            row = QHBoxLayout()
            row.setSpacing(4)
            for lbl_text in labels:
                badge = QLabel(f"{prefix}{lbl_text}")
                badge.setStyleSheet(
                    f"background: {bg}; color: {fg}; font-size: 10px; "
                    f"border-radius: 2px; padding: 1px 5px;"
                )
                row.addWidget(badge)
            row.addStretch()
            v.addLayout(row)

        return card
