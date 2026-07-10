---
title: lint watcher docstring pass-count matches inventory items
kind: infra
tags:
- wf-fix
- wf-fix-fp:dcf62dd8f614
- daily-auto-filed
created_at: '2026-07-10T06:53:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): watcher docstring pass
  inventory drifts from the live pass set (second count-catch-up after #1021); 7+
  live passes unnumbered'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1169.

## Goal
Add a mechanizable workflow_lint.py check asserting the autonomous_session_watch.py docstring's 'N passes' header count equals the number of numbered inventory items (regex the header int, count numbered items, assert equal), and reconcile the docstring inventory (7+ live passes unnumbered).

## Workflow gap
- **Bug observed:** autonomous_session_watch.py's docstring pass inventory drifts from the live pass set — this task (#1169) was the second manual count-catch-up after #1021; 7+ live passes are unnumbered (capacity-retry, gate-push, stale-registration, program-orchestrator recovery, happy-patch, triage-observer, proposed-infra-sweep). No such lint exists on main (verified 2026-07-09).
- **Why it is a workflow gap:** The watcher docstring is the canonical inventory other surfaces cite; without a mechanical count pin every new pass silently re-opens the drift.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)
+ def check_asw_docstring_pass_count(): parse 'N passes' header int from scripts/autonomous_session_watch.py docstring; count r'^\d+\. \*\*' items; FAIL on mismatch. Bundle into no_flags via the source-dispatch pattern.

## Scope / surfaces
- Primary target: `scripts/workflow_lint.py, scripts/autonomous_session_watch.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/workflow_lint.py, scripts/autonomous_session_watch.py
- fingerprint: 9b0cf3646cb0

Parked prose follow-ups on #1169, 2026-07-09T13:38:01Z (alternatives critic, non-blocking): (1) mechanizable count check — mechanizable: yes; (2) full docstring inventory reconciliation.
