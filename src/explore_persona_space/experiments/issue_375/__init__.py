"""Issue #375 — natural marker leakage via in-context persona drift.

This package houses the experiment-specific modules for issue #375:

- ``persona_directions`` — extract & cache L20 persona vectors for the
  source personas (software_engineer, librarian, villain) under the
  Chen et al. 2025 persona-vector recipe.
- ``example_pool`` — score Lu et al. tail docs against persona
  directions, then assemble persona-style / neutral / random-bucket
  few-shot example pools (each example is a ``{user, assistant}`` pair).
- ``fewshot_prompt`` — assemble {0, 1, 3}-shot prompts under the
  helpful-assistant system prompt.
- ``drift_eval`` — vLLM batched generation per cell, marker scoring,
  ZLT contamination eval-time gate, paired-bootstrap statistics.
- ``analyze`` — per-cell summary, stratified-by-query-source table,
  hero figure + companion control figures.

All entry-script orchestration lives in
``scripts/run_issue375_incontext_drift.py``.

See ``tasks/approved/375/plans/plan.md`` for the full design.
"""
