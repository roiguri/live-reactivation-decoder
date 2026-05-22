"""Seed the debug-mode snapshots by running the offline pipeline once.

Dev-only. Run after pipeline/schema changes (or whenever the test-set
recording changes) to refresh the snapshots that
``frontend.debug.main`` loads.

Usage::

    cd online_decoder
    python -m scripts.demo_seed_debug_snapshots \\
        --config experiment_config.yaml \\
        --data   ../data/new_experiment/test_set/subject_102_quarter \\
        --output debug_snapshots

Writes two joblib files in ``--output``::

    eval_done.joblib    state right after orchestrator.run_evaluation()
    train_done.joblib   state right after orchestrator.run_training()

The pipeline runs non-interactively: ``set_bad_channels([])`` (no
operator-marked bads) and ``run_step2_apply_and_save(suggested)``
(accepts whatever ICLabel flagged). The chosen training timepoint is
``eval_result["suggested_timepoint"]``.

TODO(debug): a third ``preproc_done.joblib`` snapshot — captured
between ``run_step1b_fit_ica`` and ``run_step2_apply_and_save`` — is
deferred. It would let the debug screen drop the operator straight
into the ICA-review window with pre-baked components. Doing it
needs us to also pickle the ``OfflinePreprocessor`` (or just its
``ica`` / ``_bad_channels`` / ``epochs`` so the ICA review window
can re-open). See ``src/frontend/debug/README.md`` "Out of scope" /
the plan doc for the deferred design.
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
from frontend.debug.snapshots import save_snapshot  # noqa: E402

logger = logging.getLogger("seed_debug_snapshots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiment_config.yaml"),
        help="Path to the experiment YAML (default: experiment_config.yaml).",
    )
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Directory containing the subject's .vhdr file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("debug_snapshots"),
        help="Where to write the snapshot files (default: debug_snapshots/).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.config.exists():
        parser.error(f"--config not found: {args.config}")
    if not args.data.exists():
        parser.error(f"--data not found: {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    session = AppSession(args.config)
    session.configure_output(args.output)
    orch = session.offline
    assert orch is not None, "AppSession.configure_output should build session.offline"

    logger.info("Loading raw from %s", args.data)
    orch.set_file_path(args.data)
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

    logger.info("Evaluation — temporal generalization CV")
    eval_result = orch.run_evaluation()
    eval_path = save_snapshot(orch, args.output / "eval_done.joblib")
    logger.info("Wrote eval snapshot → %s", eval_path)

    suggested_t = float(eval_result["suggested_timepoint"])
    logger.info("Training — at suggested timepoint = %.3f s", suggested_t)
    orch.run_training(suggested_t)
    train_path = save_snapshot(orch, args.output / "train_done.joblib")
    logger.info("Wrote train snapshot → %s", train_path)

    print(f"Saved 2 snapshots → {args.output}/")


if __name__ == "__main__":
    main()
