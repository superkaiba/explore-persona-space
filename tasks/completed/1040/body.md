---
title: Fix issue920_*.py torch-before-dotenv import order (test_shared_vm_thread_caps
  red on main)
kind: infra
tags: []
created_at: '2026-07-04T23:16:31Z'
has_clean_result: false
origin_prompt: '#952 implementer round-1 §(d) prose follow-up: route the #920 import-order
  fix (three one-line changes) to its own session; test_shared_vm_thread_caps fails
  identically on main.'
workflow: v1
---
## Overview / Motivation

Auto-filed from a prose follow-up surfaced by the #952 cross-layer-decision-cells implementer (round 1, 2026-07-04). The workflow-invariant test `test_shared_vm_thread_caps` FAILs on `main`, flagging three `scripts/issue920_*.py` files whose torch/BLAS imports precede the `orchestrate.env.load_dotenv()` thread-cap setdefault (#847 shared-VM discipline). Pre-existing on main (reproduced identically outside the #952 worktree); unrelated to #952's delta but it reddens every worktree's step9c gate.

## Goal

Fix the import order in the three `scripts/issue920_*.py` entrypoints (three one-line changes) so `tests/test_shared_vm_thread_caps.py` passes on `main`.

## Scope / surfaces

- `scripts/issue920_*.py` (three files — enumerate via the failing test's output)
- No workflow-surface files; no behavior change beyond import order.

## Constraints / invariants

- `uv run pytest tests/test_shared_vm_thread_caps.py` passes after the change; ruff clean on touched files.

## Provenance

- origin: #952 implementer round-1 report §(d) prose follow-up (2026-07-04)
