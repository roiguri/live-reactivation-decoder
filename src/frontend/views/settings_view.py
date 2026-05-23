from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox,
    QScrollArea, QVBoxLayout, QWidget,
)

from frontend.styles.theme import (
    BORDER_GRAY, BG_LIGHT, CARD_WHITE, PRIMARY_BLUE, SUCCESS_GREEN,
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

        # Random State (shared seed, mirrored in Model Evaluation card)
        rs_row = QHBoxLayout()
        rs_row.setSpacing(6)
        rs_row.addWidget(self._field_label("Random State"))
        self._preproc_seed_field = ReadOnlyField("", field_width=72)
        rs_row.addWidget(self._preproc_seed_field)
        rs_row.addStretch()
        card.body.addLayout(rs_row)

        # Resample/filter stage (early | late)
        stage_row = QHBoxLayout()
        stage_row.setSpacing(6)
        stage_row.addWidget(self._field_label("Filter Stage"))
        self._stage_field = ReadOnlyField("", field_width=96)
        stage_row.addWidget(self._stage_field)
        stage_row.addStretch()
        card.body.addLayout(stage_row)

        # Highpass: l_freq Hz + Method sub-row
        hp_row = QHBoxLayout()
        hp_row.setSpacing(6)
        hp_row.addWidget(self._field_label("Highpass"))
        self._hp_l_freq = ReadOnlyField("", field_width=72)
        hp_row.addWidget(self._hp_l_freq)
        hp_row.addWidget(self._dim_label("Hz"))
        hp_row.addStretch()
        card.body.addLayout(hp_row)

        hp_method_row = QHBoxLayout()
        hp_method_row.setSpacing(6)
        hp_method_row.addSpacing(110)
        hp_method_row.addWidget(self._sub_label("Method:"))
        self._hp_method_field = ReadOnlyField("", field_width=96)
        hp_method_row.addWidget(self._hp_method_field)
        hp_method_row.addStretch()
        card.body.addLayout(hp_method_row)

        # Lowpass: h_freq Hz + Method sub-row
        lp_row = QHBoxLayout()
        lp_row.setSpacing(6)
        lp_row.addWidget(self._field_label("Lowpass"))
        self._lp_h_freq = ReadOnlyField("", field_width=72)
        lp_row.addWidget(self._lp_h_freq)
        lp_row.addWidget(self._dim_label("Hz"))
        lp_row.addStretch()
        card.body.addLayout(lp_row)

        lp_method_row = QHBoxLayout()
        lp_method_row.setSpacing(6)
        lp_method_row.addSpacing(110)
        lp_method_row.addWidget(self._sub_label("Method:"))
        self._lp_method_field = ReadOnlyField("", field_width=96)
        lp_method_row.addWidget(self._lp_method_field)
        lp_method_row.addStretch()
        card.body.addLayout(lp_method_row)

        # Notch: freq Hz
        notch_row = QHBoxLayout()
        notch_row.setSpacing(6)
        notch_row.addWidget(self._field_label("Notch"))
        self._notch_field = ReadOnlyField("", field_width=96)
        notch_row.addWidget(self._notch_field)
        notch_row.addWidget(self._dim_label("Hz"))
        notch_row.addStretch()
        card.body.addLayout(notch_row)

        # Final resample: target_rate Hz
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        r2.addWidget(self._field_label("Final Resample"))
        self._final_rate_field = ReadOnlyField("", field_width=88)
        r2.addWidget(self._final_rate_field)
        r2.addWidget(self._dim_label("Hz"))
        r2.addStretch()
        card.body.addLayout(r2)

        # ICA: components header + Method / Extended / Fit L Freq / ICLabel
        ica_row = QHBoxLayout()
        ica_row.setSpacing(6)
        ica_row.addWidget(self._field_label("ICA"))
        self._ica_n = ReadOnlyField("", field_width=72)
        ica_row.addWidget(self._ica_n)
        ica_row.addWidget(self._dim_label("components"))
        ica_row.addStretch()
        card.body.addLayout(ica_row)

        ica_method_row = QHBoxLayout()
        ica_method_row.setSpacing(6)
        ica_method_row.addSpacing(110)
        ica_method_row.addWidget(self._sub_label("Method:"))
        self._ica_method_field = ReadOnlyField("", field_width=96)
        ica_method_row.addWidget(self._ica_method_field)
        ica_method_row.addStretch()
        card.body.addLayout(ica_method_row)

        ica_ext_row = QHBoxLayout()
        ica_ext_row.setSpacing(6)
        ica_ext_row.addSpacing(110)
        ica_ext_row.addWidget(self._sub_label("Extended:"))
        self._ica_extended_field = ReadOnlyField("", field_width=96)
        ica_ext_row.addWidget(self._ica_extended_field)
        ica_ext_row.addStretch()
        card.body.addLayout(ica_ext_row)

        ica_fit_row = QHBoxLayout()
        ica_fit_row.setSpacing(6)
        ica_fit_row.addSpacing(110)
        ica_fit_row.addWidget(self._sub_label("Fit L Freq:"))
        self._ica_fit_l_freq_field = ReadOnlyField("", field_width=96)
        ica_fit_row.addWidget(self._ica_fit_l_freq_field)
        ica_fit_row.addWidget(self._dim_label("Hz"))
        ica_fit_row.addStretch()
        card.body.addLayout(ica_fit_row)

        iclabel_row = QHBoxLayout()
        iclabel_row.setSpacing(6)
        iclabel_row.addSpacing(110)
        iclabel_row.addWidget(self._sub_label("ICLabel:"))
        self._iclabel_field = ReadOnlyField("", field_width=240)
        iclabel_row.addWidget(self._iclabel_field)
        iclabel_row.addStretch()
        card.body.addLayout(iclabel_row)

        # Channel hygiene summary
        hyg_row = QHBoxLayout()
        hyg_row.setSpacing(6)
        hyg_row.addWidget(self._field_label("Channel Hygiene"))
        self._hygiene_field = ReadOnlyField("", field_width=300)
        hyg_row.addWidget(self._hygiene_field)
        hyg_row.addStretch()
        card.body.addLayout(hyg_row)

        # Epoch Size: header (tmin → tmax) + Baseline sub-row
        ep_row = QHBoxLayout()
        ep_row.setSpacing(6)
        ep_row.addWidget(self._field_label("Epoch Size"))
        self._tmin = ReadOnlyField("", field_width=80)
        self._tmax = ReadOnlyField("", field_width=80)
        ep_row.addWidget(self._tmin)
        ep_row.addWidget(QLabel("to"))
        ep_row.addWidget(self._tmax)
        ep_row.addWidget(self._dim_label("s"))
        ep_row.addStretch()
        card.body.addLayout(ep_row)

        baseline_row = QHBoxLayout()
        baseline_row.setSpacing(6)
        baseline_row.addSpacing(110)
        baseline_row.addWidget(self._sub_label("Baseline:"))
        self._baseline_lo_field = ReadOnlyField("", field_width=80)
        self._baseline_hi_field = ReadOnlyField("", field_width=80)
        baseline_row.addWidget(self._baseline_lo_field)
        baseline_row.addWidget(self._dim_label("to"))
        baseline_row.addWidget(self._baseline_hi_field)
        baseline_row.addWidget(self._dim_label("s"))
        baseline_row.addStretch()
        card.body.addLayout(baseline_row)

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

        # Random State (shared seed, mirrored from Preprocessing card)
        rs_row = QHBoxLayout()
        rs_row.setSpacing(6)
        rs_row.addWidget(self._field_label("Random State"))
        self._decoder_seed_field = ReadOnlyField("", field_width=72)
        rs_row.addWidget(self._decoder_seed_field)
        rs_row.addStretch()
        card.body.addLayout(rs_row)

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

        # Model params: indented sub-rows, dynamic (keys depend on selected model)
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(4)
        card.body.addWidget(self._params_container)

        scale_row = QHBoxLayout()
        scale_row.setSpacing(6)
        scale_row.addWidget(self._field_label("Scale Method"))
        self._scale_method_field = ReadOnlyField("", field_width=110)
        scale_row.addWidget(self._scale_method_field)
        scale_row.addStretch()
        card.body.addLayout(scale_row)

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

    @staticmethod
    def _sub_label(text: str) -> QLabel:
        """Sentence-case sub-row label (e.g. 'Method:'). Aligned under the
        primary input column via an addSpacing(110) at the start of the row."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
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
            self._preproc_seed_field.set_value(None)
            self._decoder_seed_field.set_value(None)
            self._stage_field.set_value(None)
            self._hp_l_freq.set_value(None)
            self._hp_method_field.set_value(None)
            self._lp_h_freq.set_value(None)
            self._lp_method_field.set_value(None)
            self._notch_field.set_value(None)
            self._final_rate_field.set_value(None)
            self._ica_n.set_value(None)
            self._ica_method_field.set_value(None)
            self._ica_extended_field.set_value(None)
            self._ica_fit_l_freq_field.set_value(None)
            self._iclabel_field.set_value(None)
            self._hygiene_field.set_value(None)
            self._tmin.set_value(None)
            self._tmax.set_value(None)
            self._baseline_lo_field.set_value(None)
            self._baseline_hi_field.set_value(None)
            self._scale_method_field.set_value(None)
            self._cv_folds.set_value(None)
            _clear_layout(self._annot_layout)
            _clear_layout(self._decoders_layout)
            _clear_layout(self._params_layout)
            for lbl in self._model_labels.values():
                lbl.setStyleSheet(
                    f"background: #F3F4F6; color: {TEXT_MUTED}; "
                    f"border: 1px solid {BORDER_GRAY}; border-radius: 2px; font-size: 12px;"
                )
            return

        pre = settings["preprocessing"]
        dec = settings["decoders"]
        em  = settings["event_mapping"]

        self._preproc_seed_field.set_value(pre["random_state"])
        self._decoder_seed_field.set_value(dec["random_state"])

        self._stage_field.set_value(pre["resample_filter_stage"])

        hp = pre["highpass"]
        self._hp_l_freq.set_value(hp["l_freq"])
        self._hp_method_field.set_value(hp["method"])

        lp = pre["lowpass"]
        self._lp_h_freq.set_value(lp["h_freq"])
        self._lp_method_field.set_value(lp["method"])

        self._notch_field.set_value(pre["notch"].get("freq"))  # None → "—"
        self._final_rate_field.set_value(pre["final_resample"]["target_rate"])

        ica = pre["ica"]
        self._ica_n.set_value(
            "auto" if ica.get("n_components") is None else ica["n_components"]
        )
        self._ica_method_field.set_value(ica["method"])
        self._ica_extended_field.set_value("yes" if ica.get("extended") else "no")
        self._ica_fit_l_freq_field.set_value(ica["fit_l_freq"])
        iclabel = ica.get("iclabel", {})
        if iclabel.get("enabled"):
            self._iclabel_field.set_value(
                ", ".join(iclabel.get("drop_labels", [])) or "—"
            )
        else:
            self._iclabel_field.set_value("disabled")

        ch = pre["channel_hygiene"]
        hyg_parts = []
        if ch.get("drop_emg"):
            hyg_parts.append("drop EMG")
        if ch.get("rename_hegoc_to_heog"):
            hyg_parts.append("HEGOC→HEOG")
        hyg_parts.append(f"montage {ch.get('montage_name')}")
        if ch.get("afz_case_fix"):
            hyg_parts.append("AFz fix")
        self._hygiene_field.set_value(", ".join(hyg_parts))

        ep = pre["epochs"]
        self._tmin.set_value(ep["tmin"])
        self._tmax.set_value(ep["tmax"])
        baseline = ep["baseline"]
        if baseline is None:
            # Paper-aligned: baseline correction omitted entirely.
            self._baseline_lo_field.set_value("off")
            self._baseline_hi_field.set_value("off")
        else:
            self._baseline_lo_field.set_value(
                "start" if baseline[0] is None else baseline[0]
            )
            self._baseline_hi_field.set_value(
                "end" if baseline[1] is None else baseline[1]
            )

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

        # Model params (dynamic — keys depend on selected model)
        _clear_layout(self._params_layout)
        params = dec.get("params") or {}
        for key, value in params.items():
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addSpacing(110)
            row.addWidget(self._sub_label(f"{key}:"))
            field = ReadOnlyField("", field_width=110)
            field.set_value(value)
            row.addWidget(field)
            row.addStretch()
            self._params_layout.addWidget(row_w)

        scale = dec.get("scale_method")
        self._scale_method_field.set_value("none" if scale is None else scale)

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
            f"QFrame {{ background: {CARD_WHITE}; border: 1px solid {BORDER_GRAY}; "
            f"border-radius: 4px; }}"
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        name_lbl = QLabel(task["name"])
        f = name_lbl.font()
        f.setWeight(QFont.Weight.DemiBold)
        f.setPointSize(10)
        name_lbl.setFont(f)
        name_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; border: none;")
        v.addWidget(name_lbl)

        for labels, bg, fg, prefix in [
            (task.get("pos_labels", []), "#D1FAE5", "#047857", "+"),  # emerald
            (task.get("neg_labels", []), "#FEE2E2", "#DC2626", "−"),  # red
        ]:
            if not labels:
                continue
            row = QHBoxLayout()
            row.setSpacing(4)
            for lbl_text in labels:
                badge = QLabel(f"{prefix}{lbl_text}")
                badge.setStyleSheet(
                    f"background: {bg}; color: {fg}; font-size: 10px; "
                    f"border: none; border-radius: 2px; padding: 1px 6px;"
                )
                row.addWidget(badge)
            row.addStretch()
            v.addLayout(row)

        return card
