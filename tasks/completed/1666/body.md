---
title: 'fix: load_dotenv before heavy imports in issue1586_fu_caveatfix_figures.py
  (thread-caps red on main)'
kind: infra
tags: []
created_at: '2026-07-24T16:22:52Z'
has_clean_result: false
origin_prompt: '#1660 implementer report (2026-07-24): pre-existing failure on pristine
  origin/main — test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  flags scripts/issue1586_fu_caveatfix_figures.py (landed 81d2400b05, heavy import
  at ~L27, no load_dotenv) — a real #847 gap; 2-line fix.'
workflow: v1
---
## Overview / Motivation

Filed by the #1660 orchestrator from an implementer-surfaced finding (2026-07-24): a test red on pristine `origin/main`, polluting every Step 9c / merge-gate baseline until fixed. NOT a workflow-surface fix — experiment-script code (#847 thread-caps class).

## Goal

Bring `scripts/issue1586_fu_caveatfix_figures.py` into the #847 thread-caps contract (dotenv/env setdefault BEFORE heavy imports) so `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` passes on main.

## Bug

- **Observed:** `scripts/issue1586_fu_caveatfix_figures.py` (landed `81d2400b05`) imports matplotlib/numpy + `explore_persona_space.analysis.paper_plots` at module top with ZERO `load_dotenv()` call, so the shared-VM thread-caps setdefault (`orchestrate/env.py`, #847) never fires before the heavy stack initializes. The pinned invariant `test_no_new_torch_before_dotenv_vm_entrypoints` flags it — red on pristine main.
- **Fix shape:** the standard 2-line fix — `from explore_persona_space.orchestrate.env import load_dotenv` + `load_dotenv()` placed BEFORE the heavy imports (match the pattern used by sibling issue scripts that pass the invariant).
- verified-at-filing: `grep -c "load_dotenv" scripts/issue1586_fu_caveatfix_figures.py` → 0; module-top `import matplotlib.pyplot`/`numpy`/`paper_plots` at ~L27; landed commit `81d2400b05` (2026-07-24 grep + git log).

## Scope

- `scripts/issue1586_fu_caveatfix_figures.py` only. The fix must keep figure output byte-stable (import-order change only).
