from pathlib import Path

from backend.core.settings_manager import SettingsManager
from backend.offline_phase.orchestrator import OfflineOrchestrator


class AppSession:
    """Single entry point for the frontend. Owns SettingsManager lifetime."""

    def __init__(self, config_path: str | Path, output_dir: str | Path):
        self._settings = SettingsManager(config_path)
        self.offline = OfflineOrchestrator(self._settings, Path(output_dir))
