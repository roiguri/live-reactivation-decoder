from pathlib import Path
from typing import Any

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator


class AppSession:
    """Single entry point for the frontend. Owns SettingsManager lifetime.

    Two-stage initialisation:
      1. AppSession(config_path)          — loads and validates config; session.settings
                                            becomes available immediately.
      2. session.configure_output(dir)    — creates OfflineOrchestrator; session.offline
                                            becomes available for pipeline steps.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._settings = SettingsManager(config_path)
        self.offline: OfflineOrchestrator | None = None

    def configure_output(self, output_dir: str | Path) -> None:
        """Create the OfflineOrchestrator. Must be called before session.offline is used."""
        self.offline = OfflineOrchestrator(self._settings, Path(output_dir))

    @property
    def settings(self) -> dict[str, Any]:
        """All config sections in one dict: preprocessing, decoders, event_mapping."""
        return self._settings.get_settings()
