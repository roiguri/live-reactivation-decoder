"""MNE interactive-review windows for the debug walkthrough.

Dev-only. Production ``frontend.main`` never imports this module.

These are thin wrappers that pop the **exact same** native MNE windows the
production ``PreprocessingView`` shows the operator — the bad-channel
``raw.plot`` window and the ICA ``plot_components`` topomap grid — but driven
from on-disk / snapshot data instead of a live compute, so the walkthrough can
demonstrate them without re-running preprocessing. The blocking-wait helpers
(``_WaitForClose`` / ``_WaitForAllFigsClose``) and the ICLabel title
annotation are imported straight from ``preprocessing_view`` so this can never
drift from what production actually renders.
"""
from __future__ import annotations

import logging
import math

from PyQt6.QtWidgets import QApplication

from frontend.views.preprocessing_view import (
    PreprocessingView, _WaitForAllFigsClose, _WaitForClose,
)

logger = logging.getLogger(__name__)


def review_bad_channels(raw) -> list[str]:
    """Pop MNE's interactive bad-channel window; block until it closes.

    Same call as ``PreprocessingView._on_step1a_done``: ``block=False`` plus a
    nested ``QEventLoop`` (the working substitute for ``block=True`` inside a
    running ``QApplication.exec()``). Returns the operator's marked bads.
    """
    logger.info("Debug walkthrough: opening bad-channel review window")
    fig = raw.plot(block=False)
    _WaitForClose(fig).wait()
    # Coerce numpy str_ → plain str, matching production.
    bads = [str(b) for b in raw.info["bads"]]
    logger.info(
        "Bad-channel review closed; operator selected %d channel(s): %s",
        len(bads), bads,
    )
    return bads


def review_ica_components(ica, epochs, component_labels) -> list[int]:
    """Pop MNE's interactive ICA topomap grid; block until every figure closes.

    Mirrors ``PreprocessingView._on_step1b_done``: a screen-aspect grid so
    ``showMaximized()`` yields near-square cells, ICLabel category + confidence
    annotated below each topomap, native click-to-toggle reject/keep intact.
    Returns the final ``ica.exclude`` list.
    """
    logger.info("Debug walkthrough: opening ICA component review window")
    n = int(ica.n_components_)
    screen = QApplication.primaryScreen().availableGeometry()
    aspect = screen.width() / max(1, screen.height())
    ncols = max(1, min(n, round(math.sqrt(n * aspect))))
    nrows = max(1, math.ceil(n / ncols))
    figs = ica.plot_components(inst=epochs, ncols=ncols, nrows=nrows, size=1.2)
    PreprocessingView._annotate_ica_titles(figs, component_labels)
    for f in (figs if isinstance(figs, (list, tuple)) else [figs]):
        try:
            f.canvas.manager.window.showMaximized()
        except Exception:
            logger.debug("Failed to maximize ICA window", exc_info=True)
    _WaitForAllFigsClose(figs).wait()
    excluded = list(ica.exclude)
    logger.info(
        "ICA review closed; operator selected %d component(s): %s",
        len(excluded), excluded,
    )
    PreprocessingView._close_figs(figs)
    return excluded
