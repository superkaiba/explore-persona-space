---
title: code-reviewer verifies named durability-pin test ships
kind: infra
tags:
- wf-fix
- wf-fix-fp:77bc8aa282fd
- daily-auto-filed
created_at: '2026-07-10T06:54:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): c31 verifies a plan NAMES
  a durability pin test but nothing verifies the named NEW test ships in the diff
  (naming-vs-shipping residual)'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1179.

## Goal
Add one bullet to code-reviewer.md plan-adherence duties: when the approved plan carries 'Durability pin: NEW tests/...', verify that test file/function exists in the round's diff; absent -> plan-adherence finding.

## Workflow gap
- **Bug observed:** c31 (#1179) verifies a plan NAMES a durability pin test, but nothing verifies the named 'Durability pin: NEW tests/...' test actually ships in the diff — the naming-vs-shipping residual (verified on main 2026-07-09: no durability-pin bullet in code-reviewer.md).
- **Why it is a workflow gap:** The code-reviewer is the only agent that sees the diff; its spec has no duty to grep the plan's Durability-pin line and confirm the named NEW test exists in the diff.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ - **Durability-pin adherence:** if the approved plan carries a 'Durability pin: NEW tests/...' line, confirm the named test exists in the diff (grep the diff name-status for the tests/ path); a missing named pin test is a plan-adherence finding (#1179 naming-vs-shipping residual).

## Scope / surfaces
- Primary target: `.claude/agents/code-reviewer.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: cd0c2e608cd2

Parked candidate block on #1179, 2026-07-09T16:07:14Z (alternatives critic, Phase 2 round 1; original fingerprint 21bcef70ec1c). confidence: medium.
