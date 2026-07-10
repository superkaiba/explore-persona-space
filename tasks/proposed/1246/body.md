---
title: add c33 escape phrase to planner canonical-escape list
kind: infra
tags:
- wf-fix
- wf-fix-fp:97f24d56ba97
- daily-auto-filed
created_at: '2026-07-10T06:55:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The Phase 1.5.0 ''Canonical
  N/A escape phrases'' list enumerates per-check escape phrases, so the new c33 escape
  `N/A — no per-rung checkpoint persistence` (alias `N/A — no checkpoint ladder`)
  is absent'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1213.

## Goal
Add the c33 escape phrase (+ alias) to the SKILL.md canonical-escape list, matching the c31/c32 entry shape.

## Workflow gap
- **Bug observed:** The Phase 1.5.0 'Canonical N/A escape phrases' list enumerates per-check escape phrases, so the new c33 escape `N/A — no per-rung checkpoint persistence` (alias `N/A — no checkpoint ladder`) is absent from it (verified on main: the list ends at the check-32 entry).
- **Why it is a workflow gap:** Planners consult that list to satisfy checks they are legitimately exempt from; a missing phrase costs a mechanical bounce round when a ladder-free plan cannot find the c33 escape wording.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)
+ `N/A — no per-rung checkpoint persistence` / alias `N/A — no checkpoint ladder`
  + (check 33 — plans that persist no per-rung checkpoints)

## Scope / surfaces
- Primary target: `.claude/skills/adversarial-planner/SKILL.md`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md
- fingerprint: 1afd7f82a139

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see § Recursion guard. NOT auto-routed; left for the nightly /daily parked-candidate sweep.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/adversarial-planner/SKILL.md
bug_observed: The Phase 1.5.0 "Canonical N/A escape phrases" list enumerates per-check escape phrases (contra plan #1213 §4.9's claim that no per-check enumeration exists there), so the new c33 escape `N/A — no per-rung checkpoint persistence` (alias `N/A — no checkpoint ladder`) is absent from it.
why_workflow_gap: Planners consult that list to satisfy checks they are legitimately exempt from; a missing phrase costs a mechanical bounce round when a ladder-free plan cannot find the c33 escape wording.
proposed_change: Add the c33 escape phrase (+ alias) to the SKILL.md canonical-escape list, matching the c31/c32 entry shape.
diff_sketch: |
  + `N/A — no per-rung checkpoint persistence` / alias `N/A — no checkpoint ladder`
  + (check 33 — plans that persist no per-rung checkpoints)
confidence: high
related_task: #1213
<!-- /workflow-fix-candidate -->

