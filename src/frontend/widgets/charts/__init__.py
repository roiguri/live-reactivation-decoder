"""Reusable chart widgets for the Phase 1 UI.

Currently:

* :class:`AUCChart` — interactive AUC-over-time plot. Used in the
  Evaluation results screen (Summary tab + per-decoder tabs).
* :class:`TGMChart` — temporal-generalization-matrix heatmap. Used in
  the per-decoder tabs alongside the AUC chart.
"""
from frontend.widgets.charts.auc_chart import AUCChart
from frontend.widgets.charts.tgm_chart import TGMChart

__all__ = ["AUCChart", "TGMChart"]
