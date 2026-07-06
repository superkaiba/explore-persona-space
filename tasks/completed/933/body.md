---
title: 'workflow-fix: test_shared_vm_thread_caps.py red on main (stale #895 grandfather
  + ungrandfathered violators)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d51df851dd1c
created_at: '2026-07-03T12:55:30Z'
has_clean_result: false
origin_prompt: 'experiment-implementer on #922 r1 surfaced: tests/test_shared_vm_thread_caps.py
  FAILs on current main (stale GRANDFATHERED_895 entry + post-#895 ungrandfathered
  violators), poisoning every Step 9c test-verdict.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #922 (emitting agent: experiment-implementer, round 1).

## Goal

Make tests/test_shared_vm_thread_caps.py green on main: delete the fixed entry from GRANDFATHERED_895 and either fix or explicitly grandfather (with issue refs) the current main-side violators.

## Workflow gap

- **Bug observed:** test_grandfathered_895_block_is_current and test_no_new_torch_before_dotenv_vm_entrypoints FAIL on current main — scripts/issue779_heldout_recon_scatter.py was fixed (dotenv now precedes torch) but never removed from GRANDFATHERED_895, and several post-#895 entrypoints (issue779_arm_headline_summaries2.py, issue810_analyze.py, ...) violate the detector without being grandfathered.
- **Why it is a workflow gap:** a workflow-invariant test that is red on main FAILs every task's Step 9c test-verdict regardless of the diff under review, training sessions to dismiss it as noise.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

- "scripts/issue779_heldout_recon_scatter.py",   # in GRANDFATHERED_895 — root cause fixed on main
+ # (removed — dotenv-before-torch fixed on main)
+ # plus: reorder load_dotenv() before torch/numpy in the listed violators OR add them
+ # to GRANDFATHERED_TORCH_BEFORE_DOTENV with their owning-issue refs.

## Scope / surfaces

- Primary target: `tests/test_shared_vm_thread_caps.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'GRANDFATHERED_895' .claude/ CLAUDE.md scripts/ tests/`) and update every hit; list them in the plan. The violator scripts themselves (scripts/issue*_*.py) are experiment entrypoints — the fix there is the mechanical dotenv-before-torch reorder OR an explicit grandfather entry, whichever the planner judges lower-risk per file.

## Constraints / invariants

- Workflow-surface only for the test file; violator-script edits limited to the import-order reorder.
- Full test file green on main after the change; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_shared_vm_thread_caps.py
- fingerprint: d51df851dd1c

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_shared_vm_thread_caps.py
bug_observed: test_grandfathered_895_block_is_current and test_no_new_torch_before_dotenv_vm_entrypoints FAIL on current main — scripts/issue779_heldout_recon_scatter.py was fixed (dotenv now precedes torch) but never removed from GRANDFATHERED_895, and several post-#895 entrypoints (issue779_arm_headline_summaries2.py, issue810_analyze.py, ...) violate the detector without being grandfathered.
why_workflow_gap: a workflow-invariant test that is red on main FAILs every task's Step 9c test-verdict regardless of the diff under review, training sessions to dismiss it as noise.
proposed_change: delete the fixed entry from GRANDFATHERED_895 and either fix or explicitly grandfather (with issue refs) the current main-side violators so the suite is green on main.
confidence: high
related_task: #922
<!-- /workflow-fix-candidate -->
