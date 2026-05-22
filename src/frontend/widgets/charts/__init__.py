"""Reusable chart widgets for the Phase 1 UI.

Currently:

* :class:`AUCChart` — interactive AUC-over-time plot. Used in the
  Evaluation results screen (Summary tab + per-decoder tabs).
"""
from frontend.widgets.charts.auc_chart import AUCChart

__all__ = ["AUCChart"]
