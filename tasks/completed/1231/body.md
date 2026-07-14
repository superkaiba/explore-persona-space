---
title: issue SKILL briefs name excerpt files for guard reviews
kind: infra
tags:
- wf-fix
- wf-fix-fp:c85f0c4b3d66
- daily-auto-filed
created_at: '2026-07-10T06:54:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): the excerpt-file leg of
  trigger-dense-review.md depends on orchestrator behavior no surface creates — /issue
  review-dispatch steps never pre-materialize excerpts for guard-surface rounds'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1185.

## Goal
Add a line to the /issue review-dispatch step(s) so briefs for guard-surface review rounds name pre-materialized excerpt files + read budgets, arming the excerpt-file leg of trigger-dense-review.md.

## Workflow gap
- **Bug observed:** The excerpt-file leg of .claude/rules/trigger-dense-review.md ('prefer orchestrator-provided pre-materialized excerpt files') depends on orchestrator behavior no surface creates — /issue SKILL.md's review-dispatch steps never instruct the orchestrator to pre-materialize excerpts for guard-surface rounds (verified on main 2026-07-09: no 'pre-materialized'/'excerpt file' mention in SKILL.md). Recoverable today: windowed reads are the in-rule fallback.
- **Why it is a workflow gap:** A rule leg that names an orchestrator duty no orchestrator surface carries is dead text; reviewer sessions on guard surfaces keep paying the filter-kill/read-volume cost the leg was written to remove.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none — one bullet in the /issue Step 5 / 9a-bis review-dispatch instructions: for guard/security-surface artifacts, pre-materialize excerpt files per trigger-dense-review.md and name them + read budgets in the reviewer brief.)

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: bd751c6186e1

Parked prose-followup on #1185, 2026-07-09T18:34:21Z (Alternatives critic, Phase 2, plan v2 review; echoed in epm:done). confidence: medium.
