"""Reusable plumbing for the decoder *analysis* notebooks.

These notebooks all follow one shape ("Mode A"): load a seeded debug profile,
replay its recording through the online inference path, epoch the probability
stream around stimulus markers, and compute metrics. The stable plumbing lives
here so the notebooks stay thin; metrics/plots can start as notebook cells and
graduate into :mod:`analysis_lib.metrics` as they stabilize.

Backend imports are deferred into functions so importing this package never
requires ``src/`` to be on ``sys.path`` yet — call :func:`context.bootstrap`
(or :func:`context.load_context`) first.
"""
