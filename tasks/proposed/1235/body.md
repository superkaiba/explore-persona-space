---
title: invoke title-sync-sweep from nightly /daily
kind: infra
tags:
- wf-fix
- wf-fix-fp:704e4e8957a2
- daily-auto-filed
created_at: '2026-07-10T06:54:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The #1196 H1-vs-frontmatter
  title drift sweep (audit_clean_results_body_discipline.py --title-sync-sweep) is
  manually-invoked only; /daily SKILL.md has no invoker bullet (verified: zero title-sync-swe'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1196.

## Goal
Add one bullet + command to /daily SKILL.md invoking `uv run python scripts/audit_clean_results_body_discipline.py --title-sync-sweep` so the drift report surfaces nightly.

## Workflow gap
- **Bug observed:** The #1196 H1-vs-frontmatter title drift sweep (audit_clean_results_body_discipline.py --title-sync-sweep) is manually-invoked only; /daily SKILL.md has no invoker bullet (verified: zero title-sync-sweep mentions in the skill).
- **Why it is a workflow gap:** A drift report with no scheduled invoker never surfaces; the nightly /daily is the natural home.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `.claude/skills/daily/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: n/a (prose park)

parked — running under workflow_fix_target Provenance (recursion guard, see workflow-fix-on-bug.md § Recursion guard). target_file: .claude/skills/daily/SKILL.md. proposed_change: invoke 'uv run python scripts/audit_clean_results_body_discipline.py --title-sync-sweep' from the nightly /daily skill (one bullet + command) so the H1/title drift report surfaces without a manual run. bug_observed: the #1196 sweep is manually-invoked only; no scheduled invoker. confidence: medium. related_task: #1196. routed: parked: EPM_WORKFLOW_FIX_SESSION
