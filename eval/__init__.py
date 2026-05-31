"""Experiment-specific judge prompts and probe families.

This directory is a regular Python package so that scripts can `from
eval.exp389_judge_prompts import ...` / `from eval.exp407_judge_prompts
import ...` etc. without relying on namespace-package resolution
(which works on the pod but not always in local dry-runs).

The package has no public surface itself — all public symbols live in
the per-experiment modules.
"""
