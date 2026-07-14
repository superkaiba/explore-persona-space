---
title: Guard 1 foreign-strip deletion silently resurrected
kind: infra
tags:
- wf-fix
- wf-fix-fp:0cc287d2ac99
- daily-auto-filed
created_at: '2026-07-10T06:55:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): Guard 1''s FOREIGN_BRANCH_ONLY
  arm stages a deletion index-only (git rm --cached form at SKILL.md ~L8675) but the
  block''s PATHSPEC-LIMITED commit (~L8681) records WORKING-TREE content — the still-prese'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1210.

## Goal
Change the arm to remove index AND worktree (rm -f --ignore-unmatch form — the stale duplicates are safe to delete from the private issue worktree), or switch the strip commit to a plain index commit; add a pin/regression test.

## Workflow gap
- **Bug observed:** Guard 1's FOREIGN_BRANCH_ONLY arm stages a deletion index-only (git rm --cached form at SKILL.md ~L8675) but the block's PATHSPEC-LIMITED commit (~L8681) records WORKING-TREE content — the still-present file is committed right back and the deletion silently never lands. Observed live on #1210's own merge: 19 resurrected paths, caught fail-loud by the new certification diff. Verified the arm is unchanged on main.
- **Why it is a workflow gap:** The Guard 1 strip exists to keep foreign tasks/ state out of the merge payload; a strip that self-reverts defeats it on every merge with branch-only foreign paths.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)
(none — reproduced live on #1210; git-documented behavior of pathspec commits)

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: n/a (prose park)

source: orchestrator-observation (Step 10d merge execution, task #1210). target_file: .claude/skills/issue/SKILL.md. bug_observed: Guard 1's FOREIGN_BRANCH_ONLY arm stages a deletion with 'git rm --cached' but the block's PATHSPEC-LIMITED commit (git commit -- "${FOREIGN[@]}") records WORKING-TREE content — the still-present file is committed right back and the deletion silently never lands (observed live on #1210's own merge: 19 resurrected paths, caught fail-loud by the NEW certification diff). proposed_change: change the arm to 'git rm -f --ignore-unmatch --' (index AND worktree — the stale duplicates are safe to delete from the private issue worktree), or switch the strip commit to a plain index commit; add a pin/regression test. confidence: high (reproduced live; git-documented behavior of pathspec commits). related_task: #1210. routed: parked — workflow-fix session recursion guard (workflow_fix_target Provenance on #1210); surfaced for the nightly /daily parked-candidate routing pass.
