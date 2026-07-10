---
title: verify_plan c20 WARN on N/A escape masking a live lattice
kind: infra
tags:
- wf-fix
- wf-fix-fp:7c108ff21ca7
- daily-auto-filed
created_at: '2026-07-10T06:53:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): c20 short-circuits to PASS
  on the standalone N/A escape even when _C20_DECL_RE matches inside a trigger section
  (broken lattice + N/A line passes silently)'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1166.

## Goal
verify_plan.py c20: WARN when the byte-exact 'N/A — no registered verdict lattice' escape co-occurs with a _C20_DECL_RE match inside a trigger section (the fixture-D masking shape).

## Workflow gap
- **Bug observed:** c20 short-circuits to PASS on the standalone N/A line even when a lattice declaration regex matches inside a trigger section — a broken lattice + N/A line currently passes silently (verified on main 2026-07-09: scripts/verify_plan.py:3163-3164 returns PASS on _standalone_na_declared with no co-occurrence check).
- **Why it is a workflow gap:** The N/A escape exists for plans with NO lattice; when a declaration is present but malformed, the escape masks the very defect c20 exists to catch.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ if _standalone_na_declared(plan, r"no registered verdict lattice") and any(_C20_DECL_RE.search(s) for s in _c20_trigger_sections(plan)): return WARN (declared-lattice-plus-NA masking shape); else keep the PASS escape.

## Scope / surfaces
- Primary target: `scripts/verify_plan.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 5e25df5d9024

Parked candidate on #1166, 2026-07-09T13:00:06Z (Alternatives critic, plan review round 1): verify_plan.py c20 could WARN when the byte-exact 'N/A — no registered verdict lattice' escape co-occurs with a _C20_DECL_RE match inside a trigger section (fixture-D masking shape). proposed_change: flag _standalone_na_declared AND decl-regex hit in _c20_trigger_sections as WARN. confidence: medium.
