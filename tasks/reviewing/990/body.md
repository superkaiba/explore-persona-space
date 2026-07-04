---
title: 'workflow-fix: exclude deliberate-stop events from progress ts'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:c5368d330b38
created_at: '2026-07-04T07:12:49Z'
has_clean_result: false
origin_prompt: 'parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target
  (recursion guard, workflow-fix-on-bug.md § Recursion guard). routed: parked: EPM_WORKFLOW_FIX_SESSION.
  source: prose-followup (code-reviewer round 1).


  Candidate (synthesized from reviewer prose): target_file: scripts/autonomous_session_watch.py
  — exclude by==''spawn_session-stop'' deliberate-stop events from _latest_progress_ts
  (~line 3328) so the watcher''s staleness clock does not count a session''s death
  record as ''real progress''; add a pinned test. Same anti-liveness class as #949''s
  fix, fleet-telemetry surface. Reviewer confidence: medium (cmd_stop typically unregisters
  the session, so exposure is small). related_task: #949. For the next human/orchestrator
  pass to file.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

exclude by=='spawn_session-stop' deliberate-stop events from _latest_progress_ts so a session's death record is not counted as progress; add a pinned test

## Workflow gap

- **Bug observed:** the watcher's staleness clock counts by=='spawn_session-stop' deliberate-stop events in _latest_progress_ts (~line 3328) as 'real progress' — same anti-liveness class as #949's fix, fleet-telemetry surface
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

exclude by=='spawn_session-stop' deliberate-stop events from _latest_progress_ts so a session's death record is not counted as progress; add a pinned test

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: c5368d330b38

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, workflow-fix-on-bug.md § Recursion guard). routed: parked: EPM_WORKFLOW_FIX_SESSION. source: prose-followup (code-reviewer round 1).

Candidate (synthesized from reviewer prose): target_file: scripts/autonomous_session_watch.py — exclude by=='spawn_session-stop' deliberate-stop events from _latest_progress_ts (~line 3328) so the watcher's staleness clock does not count a session's death record as 'real progress'; add a pinned test. Same anti-liveness class as #949's fix, fleet-telemetry surface. Reviewer confidence: medium (cmd_stop typically unregisters the session, so exposure is small). related_task: #949. For the next human/orchestrator pass to file.
