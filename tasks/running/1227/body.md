---
title: extend check 19b bare-table WARN to v4 Methodology
kind: infra
tags:
- wf-fix
- wf-fix-fp:574e291f7f5c
- daily-auto-filed
created_at: '2026-07-10T06:53:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): check 19b fires the wrap-your-bare-table
  WARN only for the v3 ## Data section; a bare condition-code table in v4 ## Methodology
  still audit-FAILs with no authoring nudge'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1171.

## Goal
Extend verify_task_body.py check 19b's unwrapped-table condition-code authoring WARN to the v4 ## Methodology section.

## Workflow gap
- **Bug observed:** check 19b (_iter_unwrapped_data_tables) fires the wrap-your-bare-table WARN only for the v3 ## Data section ('skipped — not a v3 body'); a bare unwrapped sample table in v4 ## Methodology carrying a cell_tags-class code (e.g. BS_E0, not table-blanked) still audit-FAILs downstream with no authoring nudge (verified on main 2026-07-09: check at verify_task_body.py ~8493-8496 gates on is_v3).
- **Why it is a workflow gap:** The WARN exists to catch the confusing downstream audit FAIL at authoring time; v4 bodies — the current spec — get the FAIL with no nudge.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ v4 branch: when is_v4(body), scan section_text(body, 'Methodology') with the same _iter_unwrapped_data_tables + condition-code matcher; WARN only.

## Scope / surfaces
- Primary target: `scripts/verify_task_body.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 0a8112812486

Parked candidate on #1171, 2026-07-09T14:08:20Z (planner Phase 1 prose-followup): extend check 19b's scan to the v4 Methodology section. confidence: medium.
