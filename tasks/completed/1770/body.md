---
title: 'fix main-red test_no_new_torch_before_dotenv_vm_entrypoints: issue1481_coverage_fig
  + issue779_scaling_curve_fig import order'
kind: infra
tags: []
created_at: '2026-07-28T20:41:18Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #1092 crossed-core-sae round-1 implementer report
  (epm:experiment-implementation v28, 2026-07-28): pin-sweep found this workflow-invariant
  node pre-existing-red on main via two untouched experiment scripts.'
workflow: v1
---
## Overview / Motivation

Flagged during task #1092's `crossed-core-sae` implementation round (2026-07-28): the workflow-invariant pytest node `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is red on main, so every session's Step 9c gate must re-classify it as pre-existing red until fixed.

## Goal

Make the two offending experiment scripts satisfy the torch-before-dotenv VM-entrypoint invariant so the node goes green on pristine main.

## Evidence

- Offenders per the failing assert: `scripts/issue1481_coverage_fig.py` and `scripts/issue779_scaling_curve_fig.py` (torch imported before the project `orchestrate.env.load_dotenv` seam).
- unverified hypothesis — verify at plan time: the node fails identically at the MAIN checkout on exactly these two files (reported by the #1092 round-1 implementer after a worktree + main cross-check, 2026-07-28; re-run the single node at the repo root to confirm before editing).
- The #1092 round's two NEW scripts satisfy the invariant; nothing in that round touched the offenders.

## Scope / surfaces

- `scripts/issue1481_coverage_fig.py`, `scripts/issue779_scaling_curve_fig.py` (experiment figure scripts — NOT workflow surface; plain infra fix).
- Fix shape: reorder imports so `explore_persona_space.orchestrate.env` dotenv loading precedes any torch import (the invariant's standard remedy), or use the test's sanctioned pattern; run the single node to green, plus ruff on touched files.

## Constraints

- Behavior-preserving import reordering only; no figure-logic changes.
