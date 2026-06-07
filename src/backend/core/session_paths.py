from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SessionPaths:
    """Single source of truth for one session's on-disk layout.

    Rooted at the chosen output directory, this is the **only** place the app's
    directory structure is defined. Every phase derives its locations from a
    ``SessionPaths`` instead of composing ``root / "subdir"`` joins or
    reverse-engineering a root from some other file's path — so the layout lives
    in exactly one place and nothing lands outside the tree.

    Current (flat) layout::

        <root>/
        ├── experiment_config.yaml          # copy of the config this run used
        ├── epochs/                         # cleaned epochs (.fif)
        ├── evaluation/                     # CV results
        ├── models/
        │   └── decoder_pipeline.joblib     # the Phase 2 artifact
        └── phase2_live/
            └── <run>/                      # one directory per live Start
                ├── predictions.csv
                ├── markers.csv
                ├── manifest.json
                └── predictions.npz

    Path accessors are pure (no I/O) so they stay cheap and side-effect-free;
    :meth:`new_phase2_run_dir` is the one exception — it creates the run
    directory, since callers always want it to exist.

    When subject/session management lands (PRD §5), only how ``root`` is computed
    changes (e.g. a ``for_subject(experiment_root, subject_id)`` factory); no
    consumer of these accessors changes.
    """

    root: Path

    def __post_init__(self) -> None:
        # Normalise to Path even when constructed from a str (frozen dataclass).
        object.__setattr__(self, "root", Path(self.root))

    @property
    def experiment_config_path(self) -> Path:
        """Where the run's config copy lives (written once when the workspace is set)."""
        return self.root / "experiment_config.yaml"

    @property
    def epochs_dir(self) -> Path:
        return self.root / "epochs"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def decoder_pipeline_path(self) -> Path:
        return self.models_dir / "decoder_pipeline.joblib"

    @property
    def phase2_live_dir(self) -> Path:
        return self.root / "phase2_live"

    def new_phase2_run_dir(self) -> Path:
        """Create and return a fresh timestamped run directory for one live Start.

        The run-naming convention (a timestamp, so each Start is self-contained)
        is part of the layout, so it lives here rather than in the caller.
        """
        run_dir = self.phase2_live_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
