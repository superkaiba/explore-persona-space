---
title: 'workflow-fix: forward rc==0 stderr at watcher set-status + f'
kind: infra
tags:
- wf-fix
- wf-fix-fp:24e928b7c971
- daily-auto-filed
created_at: '2026-07-09T06:56:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The watcher''s two set-status
  sites (_set_status_blocked; the follow-up re-park set-status awaiting_promotion
  subprocess) and file_infra_task.py''s `task.py new` child still discard rc==0 stderr,
  swallowing the same landing-check warning class #1130 fixed for marker posts.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1130 (park_form: recursion-guard).

## Goal

Call the #1130 _forward_marker_child_stderr helper at the watcher's two set-status subprocess sites and file_infra_task.py's `task.py new` child so rc==0 stderr (the landing-check warning class) reaches wrapper transcripts.

## Workflow gap

- **Bug observed:** The watcher's two set-status sites (_set_status_blocked; the follow-up re-park set-status awaiting_promotion subprocess) and file_infra_task.py's `task.py new` child still discard rc==0 stderr, swallowing the same landing-check warning class #1130 fixed for marker posts.
- **Why it is a workflow gap:** task.py deliberately exits 0 with a stderr ERROR on deferred-commit / landing-check warnings; a subprocess caller that discards rc==0 stderr silently loses that signal — the exact gap #1130 closed for post-marker children.
- **Confidence (emitter):** medium (implementer-surfaced residual, #1130; ~4-line follow-up)

## Proposed change (candidate diff sketch — refine in planning)

At each of the three subprocess call sites, after a successful run: _forward_marker_child_stderr(res, f"set-status blocked on #{issue}") / ("set-status awaiting_promotion ...") / ("task.py new (file_infra_task)") — or an equivalent local forwarder in file_infra_task.py.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/file_infra_task.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/file_infra_task.py
- origin: parked candidate on task #1130 at 2026-07-08T09:02:57Z

parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard). Implementer-surfaced prose residual: the watcher's two set-status sites (_set_status_blocked; follow-up re-park set-status awaiting_promotion, subprocess block at autonomous_session_watch.py:8230) and file_infra_task.py's 'task.py new' child still discard rc==0 stderr and can swallow the same landing-check warning class on a different subcommand; the _forward_marker_child_stderr helper added by #1130 makes this a ~4-line follow-up. target_file: scripts/autonomous_session_watch.py, scripts/file_infra_task.py. routed: parked: workflow_fix_target recursion guard — for the next human/orchestrator pass.
