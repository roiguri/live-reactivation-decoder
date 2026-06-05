"""Seed a debug *profile*'s snapshots by running the offline pipeline once.

Dev-only. Run after pipeline/schema changes (or whenever the recording
changes) to refresh the snapshots that ``frontend.debug`` loads.

A **profile** is a self-contained directory ``debug_snapshots/<name>/``
(see ``src/frontend/debug/profiles.py``). This script either *bootstraps*
a new profile or *re-seeds* an existing one:

Bootstrap (first time) — copies the config in and records the raw-data path::

    cd online_decoder
    python -m scripts.demo_seed_debug_snapshots \\
        --profile default \\
        --config experiment_config.yaml \\
        --data   data/subject_101/split/functional_localizer

Re-seed (after a schema/pipeline change) — reuses the recorded config + data::

    python -m scripts.demo_seed_debug_snapshots --profile default

Writes, inside ``debug_snapshots/<name>/``::

    manifest.yaml        name + config (copied in) + raw_data_dir
    experiment_config.yaml
    preproc_done.joblib  state right after orchestrator.run_step2_apply_and_save()
    eval_done.joblib     state right after orchestrator.run_evaluation()
    train_done.joblib    state right after orchestrator.run_training()
    models/decoder_pipeline.joblib
    epochs/

The pipeline runs non-interactively: ``set_bad_channels([])`` (no
operator-marked bads) and ``run_step2_apply_and_save(suggested)``
(accepts whatever ICLabel flagged). The chosen training timepoint is
``eval_result["suggested_timepoint"]``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src/` importable when run as `python -m scripts.demo_seed_debug_snapshots`
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backend.session import AppSession  # noqa: E402
from frontend.debug.profiles import DEFAULT_ROOT, prepare_profile  # noqa: E402
from frontend.debug.snapshots import save_snapshot  # noqa: E402

logger = logging.getLogger("seed_debug_snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", required=True,
        help="Profile name; its directory is <root>/<profile>/.",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Profiles root directory (default: debug_snapshots/).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Experiment YAML to copy into the profile (required when "
             "bootstrapping; reuses the manifest's config when re-seeding).",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Directory containing the subject's .vhdr file (required when "
             "bootstrapping; reuses the manifest's raw_data_dir when re-seeding).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        profile = prepare_profile(
            args.profile, root=args.root, config=args.config, data=args.data
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    if not profile.raw_data_dir.exists():
        parser.error(f"raw_data_dir not found: {profile.raw_data_dir}")

    logger.info("Profile '%s' -> %s", profile.name, profile.root_dir)

    session = AppSession(profile.config_path)
    session.configure_output(profile.root_dir)
    orch = session.offline
    assert orch is not None, "AppSession.configure_output should build session.offline"

    logger.info("Loading raw from %s", profile.raw_data_dir)
    orch.set_file_path(profile.raw_data_dir)
    orch.load_raw_data()

    logger.info("Step 1A — channel hygiene + HP/notch + (if early) LP+resample")
    orch.run_step1a_filter()

    logger.info("Step 1B — non-interactive: no bad channels marked")
    orch.set_bad_channels([])
    ica, epochs, suggested = orch.run_step1b_fit_ica()
    logger.info(
        "ICA fitted (%d component(s)); accepting %d ICLabel-suggested exclusion(s): %s",
        getattr(ica, "n_components_", -1), len(suggested), suggested,
    )

    logger.info("Step 2 — apply ICA + save epochs")
    step2_result = orch.run_step2_apply_and_save(suggested)
    logger.info("Epochs retained: %d", step2_result.get("n_epochs", -1))
    preproc_path = save_snapshot(orch, profile.snapshot_paths["preproc"])
    logger.info("Wrote preproc snapshot -> %s", preproc_path)

    logger.info("Evaluation — temporal generalization CV")
    eval_result = orch.run_evaluation()
    eval_path = save_snapshot(orch, profile.snapshot_paths["eval"])
    logger.info("Wrote eval snapshot -> %s", eval_path)

    suggested_t = float(eval_result["suggested_timepoint"])
    logger.info("Training — at suggested timepoint = %.3f s", suggested_t)
    orch.run_training(suggested_t)
    train_path = save_snapshot(orch, profile.snapshot_paths["train"])
    logger.info("Wrote train snapshot -> %s", train_path)

    print(f"Saved 3 snapshots -> {profile.root_dir}/")


if __name__ == "__main__":
    main()
