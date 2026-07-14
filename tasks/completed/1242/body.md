---
title: add SKILL-pin suites to step9c WORKFLOW_INVARIANT tuple
kind: infra
tags:
- wf-fix
- wf-fix-fp:fc0ce825f081
- daily-auto-filed
created_at: '2026-07-10T06:55:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): WORKFLOW_INVARIANT omits
  the SKILL.md-content-pinning suites tests/test_step10d_guard3.py, tests/test_step_completed_resume.py,
  tests/test_issue_skill_exit_breadcrumb.py (verified absent on main) whil'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1210.

## Goal
Add the three files to the pinned tuple + sync tests/test_select_step9c_tests.py.

## Workflow gap
- **Bug observed:** WORKFLOW_INVARIANT omits the SKILL.md-content-pinning suites tests/test_step10d_guard3.py, tests/test_step_completed_resume.py, tests/test_issue_skill_exit_breadcrumb.py (verified absent on main) while SKILL.md diffs are gated ONLY by that tuple (WORKFLOW_SURFACE_GLOBS) — any SKILL.md edit can break those pins without Step 9c noticing. Distinct from open #865 (selector diffs main checkout, blind to worktree branches).
- **Why it is a workflow gap:** The Step 9c test-verdict gate is the mechanical protection for SKILL.md edits; a pin suite outside the tuple is a silent no-coverage hole.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `scripts/select_step9c_tests.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: n/a (prose park)

source: prose-followup (statistics critic round 1, task #1210). target_file: scripts/select_step9c_tests.py. bug_observed: WORKFLOW_INVARIANT (lines ~108-146) omits the SKILL.md-content-pinning suites tests/test_step10d_guard3.py, tests/test_step_completed_resume.py, tests/test_issue_skill_exit_breadcrumb.py while SKILL.md diffs are gated ONLY by that tuple (WORKFLOW_SURFACE_GLOBS line ~153) — any SKILL.md edit can break those pins without Step 9c noticing. proposed_change: add the three files to the pinned tuple + sync tests/test_select_step9c_tests.py. confidence: medium. related_task: #1210. routed: parked — this session is itself a workflow-fix session (workflow_fix_target Provenance line on #1210; recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard) and never auto-files more workflow-fix tasks; surfaced for the nightly /daily parked-candidate routing pass.
