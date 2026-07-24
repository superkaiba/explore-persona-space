---
title: 'daily-fix: issue1586 scripts load_dotenv before torch'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-24T06:46:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): three issue1586 entrypoints
  import heavy modules before load_dotenv, failing tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  on pristine main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as recursion-guard-parked prose follow-ups on tasks #1624, #1629, #1638. NOT a workflow-surface fix — these are experiment entrypoint scripts (`wf_fix: false`; filed for the independent pipeline because it reds the shared Step 9c thread-caps invariant test fleet-wide).

## Goal

Call `explore_persona_space.orchestrate.env.load_dotenv()` before any heavy import in the three `scripts/issue1586_*.py` entrypoints so `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is green on main.

## Bug

- **Observed:** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` FAILS on pristine main, naming exactly three violators: `scripts/issue1586_figures.py` (heavy import line 19), `scripts/issue1586_leakage_lattice.py` (line 26), `scripts/issue1586_pooled_lattice.py` (line 23) — each with no `load_dotenv(` call. Violates the #847 shared-VM thread-caps convention; landed via the #1586 merge.
- The sibling issue1092 violations named in the same parks were ALREADY fixed by commit `ab75ff5b05` (issue-1619) — only the issue1586 trio remains.
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -x -q` → 1 failed listing exactly the 3 files above (2026-07-24 UTC); `grep -n "load_dotenv" scripts/issue1586_figures.py scripts/issue1586_leakage_lattice.py scripts/issue1586_pooled_lattice.py` → 0 hits per target.

## Proposed change

Mirror the `ab75ff5b05` recipe: insert the `load_dotenv()` call above the first heavy import in each of the three entrypoints; confirm the invariant test passes.

## Scope

- `scripts/issue1586_figures.py`, `scripts/issue1586_leakage_lattice.py`, `scripts/issue1586_pooled_lattice.py` (experiment code — figure/analysis behavior must be byte-preserved apart from the import reorder).
