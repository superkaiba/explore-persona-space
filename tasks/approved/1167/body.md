---
title: 'workflow-fix: Per-tick push cap in watcher triage observer'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6fce8514791f
- daily-auto-filed
created_at: '2026-07-09T06:58:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): _triage_observer_emit sends
  one Telegram push per flagged action with no per-tick cap (only the epm:progress
  marker channel is capped) — any backlog (the original ~70-candidate first-tick flush,
  or a multi-hour watcher outage followed by a burst of matured dispatch records)
  floods pushes in one tick.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #967 by a recursion-guarded workflow-fix session.

## Goal

Add a per-tick PUSH cap in _triage_observer_emit mirroring the existing per-tick marker cap: push the first K actions, then one summary push ('+N more, see sidecar'); overflow stays sidecar-recorded.

## Workflow gap

- **Bug observed:** _triage_observer_emit sends one Telegram push per flagged action with no per-tick cap (only the epm:progress marker channel is capped) — any backlog (the original ~70-candidate first-tick flush, or a multi-hour watcher outage followed by a burst of matured dispatch records) floods pushes in one tick.
- **Why it is a workflow gap:** The observer is observe/alert-only by contract; an unbounded push channel turns a backlog into a notification storm that trains the user to ignore the channel.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** Verified 2026-07-08: _triage_observer_emit (autonomous_session_watch.py line 4074) calls _telegram_push per action under `if a['push']` with no counter; only markers_posted is capped. Note the motivating one-time first-tick 48h-lookback flush has long since fired — residual value is bounding post-outage bursts, so this is real but low-urgency; the planner may right-size K accordingly.

## Proposed change (candidate diff sketch — refine in planning)

In _triage_observer_emit: thread a pushes_sent counter + a _TRIAGE_OBSERVER_PUSH_CAP (env-overridable) mirroring the marker cap; on overflow send one digest push naming the overflow count and the sidecar path.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: parked candidate on task #967 at 2026-07-04T08:40:38Z

Verbatim parked note:

> source: prose-followup (implementer round 1). target_file: scripts/autonomous_session_watch.py — add a per-tick PUSH cap in _triage_observer_emit mirroring the existing marker cap, bounding the one-time first-tick backlog flush (~70 pushes from the live 48h lookback). routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); logged for the parent orchestrator / next PM pass.
