---
title: 'daily-fix: Step 9c test-verdict runs touched-file pytest subset'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f118b42609ad
- daily-auto-filed
created_at: '2026-06-30T06:44:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: Step 9c runs the whole ~5800-test
  suite; with no xdist installed it has no parallelism, stalls ~40-44%, and is killed
  by the harness in sparse worktrees (#736, #665 burned several relaunch rounds before
  falling back to a subset).'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. The whole-suite Step 9c gate stalls/gets killed in sparse worktrees; #736 and #665 each burned several monitor/relaunch rounds before manually falling back to a focused subset.

## Goal
Step 9c completes deterministically without harness-killed whole-suite stalls.

## Workflow gap
- **Bug observed:** full pytest (~5800 tests) repeatedly stalls ~40-44% and is harness-killed in sparse worktrees; no xdist -> no parallelism.
- **Why it is a workflow gap:** the test-verdict step recipe lives in SKILL.md.
- **Confidence (emitter):** medium; two sessions.

## Proposed change
- Default Step 9c to a touched-file-scoped subset: the tests covering the changed files PLUS the workflow-invariant tests (`tests/test_workflow*.py`, `test_no_*`, etc.), not the entire suite.
- Keep an opt-in full-suite path for risky/broad changes; the planner decides the gate scope per task kind.

## Scope / surfaces
- `.claude/skills/issue/SKILL.md` (Step 9c test-verdict).

## Constraints / invariants
- Workflow surface only. `workflow_lint.py --check-references`/`--check-asks` stay green.
- Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: f118b42609ad

Sessions: 52c815a3 (issue-736), 2aab6b91 (issue-665).
