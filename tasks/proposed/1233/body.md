---
title: replace vacuous pipe-python no-flags bundling test
kind: infra
tags:
- wf-fix
- wf-fix-fp:2e73aeab34c0
- daily-auto-filed
created_at: '2026-07-10T06:54:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): test_workflow_lint_pipe_python_bundled_in_no_flags
  pins bundling via exit-0-on-a-clean-tree — vacuous for the bundled-vs-opt-in distinction
  (#712 §4f class)'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1190.

## Goal
Replace/augment tests/test_workflow_lint.py test_workflow_lint_pipe_python_bundled_in_no_flags with a source-dispatch assertion or a planted-offender no-flags CLI run (the #712 pattern).

## Workflow gap
- **Bug observed:** test_workflow_lint_pipe_python_bundled_in_no_flags pins no-flags bundling via exit-0-on-a-clean-tree, which is vacuous for the bundled-vs-opt-in distinction — it passes whether or not the check is in the no_flags dispatch (verified on main 2026-07-09: tests/test_workflow_lint.py:2122 still asserts only returncode==0 on a clean tree).
- **Why it is a workflow gap:** The opt-in-not-bundled shipping failure documented by #712 §4f can recur for the pipe-python check unnoticed; the stronger planted-offender/source-dispatch pattern already exists in the same file (e.g. :2619, :3449).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ assert 'args.check_pipe_python or no_flags' in workflow_lint source (source-dispatch pin), or plant a *.sh offender in a tmp scripts dir and run the no-flags CLI expecting nonzero.

## Scope / surfaces
- Primary target: `tests/test_workflow_lint.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: tests/test_workflow_lint.py
- fingerprint: 0dfb688a0385

Parked prose-followup on #1190, 2026-07-09T19:07:02Z (statistics critic on plan v2). confidence: medium.
