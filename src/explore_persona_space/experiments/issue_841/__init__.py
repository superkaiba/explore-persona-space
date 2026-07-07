"""Issue #841 — layer-to-layer activation dynamics on real chat contexts.

Package helpers for the per-layer Δ-predictability atlas (Stage 0) and the
transported-trait-monitor benchmark (Stage 1), both over #779's cached tensors.
See ``scripts/issue841_common.py`` (loaders + reuse-fitness asserts),
``maps.py`` (the four next-activation map classes + transport), and the two
stage entrypoints ``scripts/issue841_stage0_atlas.py`` /
``scripts/issue841_stage1_benchmark.py``.
"""
