---
title: 'daily-fix: issue1092 fig scripts import heavy modules before'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-23T06:39:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): three issue1092 figure
  scripts import numpy/scipy before load_dotenv(), tripping tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  on pristine main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-22 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1613 (emitting agent: implementer r1). Experiment-script fix (NOT workflow surface — `wf_fix: false`). FLEET-BLOCKING-ADJACENT: the pinned import-order test is red on main.

## Goal

Reorder imports in `scripts/issue1092_result3_merged_fig.py`, `scripts/issue1092_result4_spread_fig.py`, and `scripts/issue1092_shrinkage_fig.py` so `orchestrate.env.load_dotenv` runs BEFORE any heavy module import, making `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` pass on main again.

## Bug

- **Observed:** the three issue1092 figure scripts (landed 8f05fc7d9c) import heavy modules (numpy/scipy at module top, e.g. `issue1092_result3_merged_fig.py` lines 21-22) before `load_dotenv()`, tripping the VM thread-cap entrypoint invariant `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` — a pre-existing red on pristine main that poisons the shared Step-9c test oracle.
- **Why it matters:** the invariant exists because the dotenv wrapper sets thread caps / env before torch-family imports on the shared VM; scripts that import first escape the caps.
- verified-at-filing: `uv run python -m pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -x -q` → FAILED (assertion at tests/test_shared_vm_thread_caps.py:888), and `grep -n 'load_dotenv|^import |^from ' scripts/issue1092_result3_merged_fig.py` shows `import numpy` / `from scipy.stats import spearmanr` at lines 21-22 with no prior `load_dotenv` call, 2026-07-23 UTC. Second parked item in the same note (scripts/runpod_api.py:488 UP037 ruff red) was SKIPPED at filing: `uv run ruff check scripts/runpod_api.py` → "All checks passed!" on the current tree — premise no longer true.

## Proposed change

In each of the three scripts, move the `orchestrate.env.load_dotenv()` call (or add it) above the numpy/scipy/torch-family imports, matching the pattern the invariant test expects for VM entrypoints (see passing sibling scripts for the canonical ordering).

## Scope / surfaces

- `scripts/issue1092_result3_merged_fig.py`
- `scripts/issue1092_result4_spread_fig.py`
- `scripts/issue1092_shrinkage_fig.py`

## Constraints / invariants

- `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` green after the change; figures regenerate identically (import order only, no logic change).
- Note: `scripts/issue1092_fair_deepdive_figs.py` sits modified-uncommitted in the repo root from a live #1092 inline round — do NOT touch that file; coordinate via `git status` before committing.
