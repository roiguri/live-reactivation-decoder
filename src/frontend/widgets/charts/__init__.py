"""Reusable chart widgets for the Phase 1 UI.

Currently:

* :class:`AUCChart` — interactive AUC-over-time plot. Used in the
  Evaluation results screen (Summary tab + per-decoder tabs).
* :class:`TGMChart` — temporal-generalization-matrix heatmap. Used in
  the per-decoder tabs alongside the AUC chart.
* :class:`TopomapWidget` — single-decoder spatial-pattern topomap.
  Used in the Train view (Node 5) to show ``spatial_patterns`` from
  the trained pipeline.
"""
from frontend.widgets.charts.auc_chart import AUCChart
from frontend.widgets.charts.tgm_chart import TGMChart
from frontend.widgets.charts.topomap_widget import TopomapWidget

__all__ = ["AUCChart", "TGMChart", "TopomapWidget"]
