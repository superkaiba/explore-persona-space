"""Top-level `eps` package for experiment-specific entry points.

This package holds experiment modules invoked from RunPod docker_args (e.g. the
factor-screen entry-point `eps.experiments.marker_factor_screen` for Sagan
experiment #365). Shared library code continues to live in the existing
`explore_persona_space` package — `eps` modules import from there for any
training / eval / persona / data helpers.
"""
