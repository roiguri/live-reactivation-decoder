"""Verify the live 0/1 saturation is a µV↔V feature-scale mismatch.

Loads the trained decoder artifact, takes a feature row at TRAINING scale
(SI volts, as MNE loads the VHDR), and compares predict_proba at:
    ×1      (volts — what the decoder trained on)
    ×1e6    (µV — what a NeurOne LSL proxy likely streams, unconverted)

If the ×1e6 column collapses to ~0.0/~1.0 while ×1 is graded across (0,1),
the scale mismatch reproduces the reported "only 0 or 1" symptom.

Usage:
    python scripts/verify_scale_saturation.py \
        --artifact debug_snapshots/models/decoder_pipeline.joblib \
        [--snapshot debug_snapshots/train_done.joblib]   # optional, for real feature rows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.online_phase.artifact_loader import load_decoder_pipeline_artifact


def _positive_idx(model, metadata) -> int:
    target = metadata.get("positive_class", 1)
    hits = np.where(np.asarray(model.classes_) == target)[0]
    return int(hits[0]) if hits.size else 1


def _feature_rows(artifact, snapshot_path: Path | None, n_features: int) -> np.ndarray:
    """Real epoch feature rows at training scale if a snapshot is given,
    else a synthetic row at a plausible EEG-volts magnitude (~30 µV = 3e-5 V)."""
    if snapshot_path and snapshot_path.exists():
        import joblib

        epochs = joblib.load(snapshot_path)["_epochs"]
        data = epochs.get_data()          # (n_epochs, n_ch, n_times), SI volts
        mid = data.shape[2] // 2
        print(f"Using {data.shape[0]} real feature rows from {snapshot_path.name} "
              f"(|value| median = {np.median(np.abs(data[:, :, mid])):.2e} V)")
        return data[:, :, mid]
    rng = np.random.default_rng(0)
    print("No snapshot given — using a synthetic row at ~3e-5 V (typical EEG scale)")
    return rng.normal(0.0, 3e-5, size=(5, n_features))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=None)
    args = p.parse_args(argv)

    artifact = load_decoder_pipeline_artifact(args.artifact)
    first_model = next(iter(artifact.models.values()))
    n_features = getattr(first_model, "n_features_in_", None) or \
        artifact.metadata.get("feature_width")

    X = _feature_rows(artifact, args.snapshot, int(n_features))

    print("\n{:<20} {:>18} {:>18}".format("task", "proba ×1 (volts)", "proba ×1e6 (µV)"))
    print("-" * 58)
    for name, model in artifact.models.items():
        idx = _positive_idx(model, artifact.metadata)
        p_v = model.predict_proba(X)[:, idx]
        p_uv = model.predict_proba(X * 1e6)[:, idx]
        print("{:<20} {:>18} {:>18}".format(
            name,
            f"[{p_v.min():.3f}, {p_v.max():.3f}]",
            f"[{p_uv.min():.3f}, {p_uv.max():.3f}]",
        ))
    print("\nIf the ×1e6 column is pinned at [0.000, 0.000]/[1.000, 1.000] "
          "while ×1 is graded, the µV↔V scale mismatch is confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
