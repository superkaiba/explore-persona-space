---
title: Edit C certification arms to exclusive-arm shape
kind: infra
tags:
- wf-fix
- wf-fix-fp:c7df2f99a8d2
- daily-auto-filed
created_at: '2026-07-10T06:55:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): Edit C''s merge-conflict-recovery
  certification uses ''|| { echo ...; false; }'' arms (verified present on main at
  SKILL.md ~L9600, ~L9623) that do not halt subsequent commands under single-shell
  no-set-'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1210.

## Goal
Restructure the Edit C certification arms to Guard 1's 'if !' exclusive-arm shape.

## Workflow gap
- **Bug observed:** Edit C's merge-conflict-recovery certification uses '|| { echo ...; false; }' arms (verified present on main at SKILL.md ~L9600, ~L9623) that do not halt subsequent commands under single-shell no-set-e execution (near-theoretical: bad MAIN_SHA fails earlier at merge; the <each resolved file> placeholder forces piecewise execution).
- **Why it is a workflow gap:** A certification arm that reports failure without halting lets later commands run against an uncertified tree.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: n/a (prose park)

source: prose-followup (code-reviewer round 1, task #1210, Minor finding 1). target_file: .claude/skills/issue/SKILL.md. bug_observed: Edit C's merge-conflict-recovery certification uses '|| { echo ...; false; }' arms that do not halt subsequent commands under single-shell no-set-e execution (near-theoretical: bad MAIN_SHA fails earlier at merge; the <each resolved file> placeholder forces piecewise execution). proposed_change: restructure the Edit C certification arms to Guard 1's 'if !' exclusive-arm shape. confidence: low. related_task: #1210. routed: parked — workflow-fix session recursion guard (workflow_fix_target Provenance on #1210); surfaced for the nightly /daily parked-candidate routing pass.
