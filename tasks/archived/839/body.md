---
title: 'workflow-fix: followup-repark predicate must check same-issue-followup-run'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f57bed9afd5c
created_at: '2026-07-02T06:40:37Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 778: watcher _followup_round_complete_reason
  re-parked a mid-flight follow-up round twice by matching the PARENT round''s 9a-bis
  step-completed; require the epm:same-issue-followup-run completion marker (label-matched)
  or round-activity-aware ts logic'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #778 (emitting agent: orchestrator, /issue 778 session).

## Goal

Harden `autonomous_session_watch.py`'s followup-round-repark predicate
(`_followup_round_complete_reason`) so it cannot re-park a MID-FLIGHT
same-issue follow-up round based on a step-completed marker from a PRIOR
(parent) round.

## Workflow gap

- **Bug observed:** the watcher re-parked #778 from `followups_running` to
  `awaiting_promotion` TWICE (2026-07-02T04:43Z and 06:13Z) while the
  `corrected-monitoring-8prompt-ladder` round was still executing stage-two
  (off-pod null battery running, no analyzer re-fold, no
  `epm:same-issue-followup-run`). Its round-complete predicate — "a round-end
  `epm:step-completed step=9a-bis exit_kind=parked` NEWER than the round's
  `epm:followup-scope`" — matched the PARENT round's 9a-bis step-completed
  (21:49:35Z), which happens to postdate the followup-scope (21:35:20Z)
  because the scope was posted BEFORE the parent round parked. The premature
  re-park also triggered the pod-safety pass to terminate pod-778 without an
  `epm:pod-terminated` marker, and would have let the session-reconcile pass
  stop the live driving session.
- **Why it is a workflow gap:** ts-ordering alone cannot distinguish "the
  round ended" from "an older round's park postdates this round's scope
  post". The loop's designed completion record is the
  `epm:same-issue-followup-run v1` marker with the scope's `followup_label`
  (SKILL.md Step 9b loop step 4) — the predicate never consults it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/autonomous_session_watch.py :: _followup_round_complete_reason
- round_complete = latest step-completed(step=9a-bis, parked) newer than scope
+ round_complete = EITHER (a) an epm:same-issue-followup-run marker exists
+   whose followup_label matches the LATEST epm:followup-scope's label, OR
+   (b) a 9a-bis step-completed that postdates the LATEST followup-scope AND
+   postdates the round's own activity (e.g. the newest epm:results /
+   stage-dispatch after the scope) — never a step-completed that predates
+   any round activity newer than itself.
+ Also: when re-parking on behalf of a dead session, post the completion
+   marker check result in the note, and never fire while a fresher
+   stage-dispatch breadcrumb (< watcher staleness window) exists.
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Tests: extend the watcher tests with a #778-shaped replay fixture (scope at
  T0, parent-round step-completed at T0+14min, round activity at T0+1h..5h,
  no same-issue-followup-run marker → predicate must NOT fire; add the run
  marker → fires).

## Constraints / invariants

- Workflow-surface only. Keep the incident-#533 recovery behavior (a genuinely
  dead session's completed round is still re-parked).
- `uv run pytest tests/test_workflow*.py` + watcher tests pass.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: f57bed9afd5c

Surfaced prose (verbatim): watcher followup-round-repark fired twice on a
live round (#778, 04:43Z + 06:13Z) by matching the parent round's 9a-bis
step-completed; immediate mitigation applied in-session was re-posting the
followup-scope (v3) so its ts postdates the stale step-completed.
