---
title: migrate remaining verify_plan doc-global N/A escapes
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e9d3b8d2317
- daily-auto-filed
created_at: '2026-07-10T06:54:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): The doc-global NA_RE escape
  bug class #1203 fixed for c12 remains at the sibling checks (verified on main: doc-global
  `re.search(NA_RE + ...)` sites at lines ~549, 612, 828, 1092, 1850/1911, 2001/2035'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1203.

## Goal
Sibling audit/migration of the remaining doc-global escapes to _standalone_na_declared (one task; distinct fingerprint from #1203's c12-only fix). Note: this also delivers the c19 anti-paste guard separately parked on #974 (that park is deduped into this filing).

## Workflow gap
- **Bug observed:** The doc-global NA_RE escape bug class #1203 fixed for c12 remains at the sibling checks (verified on main: doc-global `re.search(NA_RE + ...)` sites at lines ~549, 612, 828, 1092, 1850/1911, 2001/2035, 2555 (c19), 3206/3245); several FAIL details quote their own escape phrase — the pasted-bounce-brief self-escape channel.
- **Why it is a workflow gap:** A pasted FAIL detail satisfying the check it reports defeats the verifier; #1203 closed this for c12 only.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
(none — mirror the #1203 c12 migration per check, with red-green fixtures)

## Scope / surfaces
- Primary target: `scripts/verify_plan.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/verify_plan.py
- fingerprint: n/a (prose park)

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this IS a workflow-fix session; candidate logged, not auto-filed; the nightly /daily parked-candidate sweep routes it).

source: prose-followup (planner, #1203 plan v1 §2/§8)
target_file: scripts/verify_plan.py
bug_observed: the same doc-global NA_RE escape bug class c12 had remains at c2 :608, c6 :824, c11 :1088, c15 :1907, c16 :2031, c19 :2551, c21 :3241, and :4679; several of their FAIL details quote their own escape phrase (:563, :1929, :2052, :3263) — the pasted-bounce-brief self-escape channel.
proposed_change: sibling audit/migration of the remaining doc-global escapes to _standalone_na_declared (one task; distinct fingerprint from #1203's c12-only fix).
confidence: medium
related_task: #1203
