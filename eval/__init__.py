"""Experiment-specific judge prompts and probe families.

This directory is a regular Python package so that scripts can `from
eval.exp389_judge_prompts import ...` / `from eval.exp444_judge_prompts
import ...` etc. without relying on namespace-package resolution
(which silently falls back to ``scripts/eval.py`` when the cwd has
``scripts/`` in sys.path[0]; see #407's `__init__.py` for the original fix).

The package has no public surface itself — all public symbols live in
the per-experiment modules.
"""
