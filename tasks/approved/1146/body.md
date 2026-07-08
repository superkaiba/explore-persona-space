---
title: 'workflow-fix: extend thread-caps heavy-import predicate to transitively-heavy
  roots'
kind: infra
tags:
- wf-fix
- wf-fix-fp:449264990d2d
created_at: '2026-07-08T14:24:11Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1145 Methodology critic r1: extend tests/test_shared_vm_thread_caps.py
  heavy-import predicate (torch/numpy only) to transitively-heavy roots (matplotlib/pandas/scipy/sklearn/transformers)
  with a fix-or-grandfather round so trunk stays green.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1145 (emitting agent: critic, Methodology lens).

## Goal

Extend `tests/test_shared_vm_thread_caps.py`'s heavy-import predicate to known transitively-heavy third-party roots (matplotlib, pandas, scipy, sklearn, transformers) so a script importing them before `load_dotenv()` no longer passes the invariant while the thread caps stay dead — with a fix-or-grandfather round for the newly-flagged offender class.

## Workflow gap

- **Bug observed:** the invariant test's `_first_heavy_import_line` matches literal `torch`/`numpy` roots only, so a script importing matplotlib/pandas (which pull numpy at import time) before `load_dotenv()` passes the invariant while the #847 thread caps stay dead (live shape pre-#1145: `scripts/issue779_pertoken_lmsys_analysis.py` matplotlib@33 above numpy@34).
- **Why it is a workflow gap:** the test pins the #847 thread-caps invariant but its predicate under-covers the runtime property it exists to protect.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
- HEAVY_ROOTS = {"torch", "numpy"}
+ HEAVY_ROOTS = {"torch", "numpy", "matplotlib", "pandas", "scipy", "sklearn", "transformers"}
```
Plus: run the extended predicate on main; fix (or, if volume demands, grandfather with a NEW dated frozen block) the newly-flagged offender class in the SAME change so trunk stays green.

## Scope / surfaces

- Primary target: `tests/test_shared_vm_thread_caps.py`
- Also touch: any newly-flagged `scripts/issue*_*.py` offenders (mechanical preamble fix per the #1145 recipe).

## Constraints / invariants

- Trunk must be GREEN after the change lands (extend-predicate + fix/grandfather in one atomic landing; never land a red-making predicate alone).
- `scripts/workflow_lint.py` no-flags run passes; ruff clean on touched files.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_shared_vm_thread_caps.py
- fingerprint: 449264990d2d

Surfaced prose (Methodology critic, #1145 round 1): "The invariant test's `_first_heavy_import_line` matches literal `torch`/`numpy` roots only, so a script importing `matplotlib`/`pandas` (which pull numpy at import time) before `load_dotenv()` passes the invariant while the caps stay dead — the exact shape `issue779_pertoken_lmsys_analysis.py` has today. Extending the predicate to known transitively-heavy roots (or any non-stdlib import) in `tests/test_shared_vm_thread_caps.py` would close it, but re-reds the trunk with a new offender class and likely needs its own fix-or-grandfather round — a distinct `kind: infra` task."
