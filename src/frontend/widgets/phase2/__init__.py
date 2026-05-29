"""Phase 2 live-inference UI components.

Sub-widgets composed by :class:`Phase2Screen`:

* :class:`~frontend.widgets.phase2.header.Phase2Header` — status +
  target hardware label.
* :class:`~frontend.widgets.phase2.settings_panel.Phase2SettingsPanel`
  — sidebar with Decoders + Decision Settings sections and a footer
  slot for the Start/Halt action button.
* :class:`~frontend.widgets.phase2.start_halt_button.StartHaltButton`
  — the action button itself: idle / connecting / live states.
"""
from frontend.widgets.phase2.header import Phase2Header
from frontend.widgets.phase2.settings_panel import Phase2SettingsPanel
from frontend.widgets.phase2.start_halt_button import StartHaltButton
from frontend.widgets.phase2.target_dialog import TargetSelectionDialog

__all__ = [
    "Phase2Header",
    "Phase2SettingsPanel",
    "StartHaltButton",
    "TargetSelectionDialog",
]
