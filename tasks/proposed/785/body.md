---
title: 'daily-fix: worktree path #506 regression in Step 4a + Step 9c'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4c7c2f6ef375
- daily-auto-filed
created_at: '2026-07-01T06:52:57Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: Step 4a and the Step 9c test-verdict
  snippet still derive REPO_ROOT via `git rev-parse --show-toplevel`, which returns
  the WORKTREE root when cwd is inside a worktree, doubling the path (.../issue-N/.'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Replace `git rev-parse --show-toplevel` with the #506-safe `dirname "$(git rev-parse --path-format=absolute --git-common-dir)"` recipe in Step 4a and the Step 9c test-verdict snippet, AND make the test-verdict block assert the cd succeeded / hard-fail on `no tests ran` (a failed cd currently runs pytest in the wrong dir and passes exit 0 with 'no tests ran', silently skipping the gate).

## Workflow gap

- **Bug observed:** Step 4a and the Step 9c test-verdict snippet still derive REPO_ROOT via `git rev-parse --show-toplevel`, which returns the WORKTREE root when cwd is inside a worktree, doubling the path (.../issue-N/.claude/worktrees/issue-N) and failing the cd; the #506-safe `--git-common-dir` recipe exists ONLY in Step 10d.
- **Evidence:** Bit 3 autonomous sessions on 2026-06-30 (issues 743/745/776); 745 (91bed41e) silently passed the test gate with `no tests ran ... pytest exit: 0` after a failed cd. Sources: /daily transcript miners (batches 00/03/04).
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 4c7c2f6ef375
- source: /daily route-2 (2026-06-30)
