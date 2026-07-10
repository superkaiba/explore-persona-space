---
title: audit is_v2 gate recognizes the v4 sentinel
kind: infra
tags:
- wf-fix
- wf-fix-fp:8a8be7836dd2
- daily-auto-filed
created_at: '2026-07-10T06:53:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): is_v2 does not recognize
  <!-- clean-result-v4 --> — a v4 body with no v3/v2/legacy markers is skipped by
  the bulk-inventory audit'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #969.

## Goal
Add the v4 sentinel to is_v2()'s current-spec disjunction.

## Workflow gap
- **Bug observed:** is_v2() does not recognize the '<!-- clean-result-v4 -->' sentinel (verified on main: only v3/v2 sentinels + legacy H2 shapes match), so v4 bodies without legacy H2s drop out of the bulk-inventory audit path (_run_legacy_bulk_inventory); live --task path unaffected.
- **Why it is a workflow gap:** The v4 migration (2026-W26) made v4 the current spec; the bulk-inventory gate silently skipping current-spec bodies is a coverage hole.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `scripts/audit_clean_results_body_discipline.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: n/a (prose park)

routed: parked — running under workflow_fix_target Provenance (recursion guard); NOT auto-filed. source: prose-followup (methodology critic concern 4, #969 Phase 2). target_file: scripts/audit_clean_results_body_discipline.py. bug_observed: is_v2() (lines ~595-604) does not recognize the '<!-- clean-result-v4 -->' sentinel, so v4 bodies without legacy H2s drop out of the bulk-inventory audit path (_run_legacy_bulk_inventory); live --task path unaffected. proposed_change: add the v4 sentinel to is_v2()'s current-spec disjunction. confidence: medium. related_task: #969.
