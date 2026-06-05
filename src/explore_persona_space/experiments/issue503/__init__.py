# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Issue #503 — cross-behavior leakage matrix predictor.

Tests whether the #468 base-model cosine predictor (in-context-example
behavior vectors read at the newline-after-``assistant`` token, layer 25)
predicts post-SFT cross-behavior leakage across the full source×target
matrix — narrow→narrow, narrow→broad beyond emergent misalignment, and
broad→broad — using one shared dataset-generation and cross-evaluation
rig and one pooled regression.

Plan: ``tasks/running/503/plans/v1.md``.
"""

from __future__ import annotations
