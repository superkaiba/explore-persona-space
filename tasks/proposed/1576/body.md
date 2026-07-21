---
title: 'daily-fix: WARN on inline interpreter workload-cmd body'
kind: infra
tags:
- wf-fix
- wf-fix-fp:43cf53d5ce74
- daily-auto-filed
created_at: '2026-07-21T06:38:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): prose-only enforcement
  residual: a session that never reads SKILL.md Backend dispatch can still compose
  an inline interpreter one-liner workload body (incident #1482 class); the #1329
  lint family has no arm for it'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1562 under the recursion guard (emitting context: Methodology critic, plan #1562 Phase 2).

## Goal

Extend the #1329 dispatch-time workload-cmd lint family in `scripts/dispatch_issue.py` to WARN when the workload-cmd BODY is an inline interpreter one-liner (`python -c` / heredoc), exempting a trailing `&& ... python -c` write_completion_sentinel suffix (the experimenter.md sanctioned append).

## Workflow gap

- **Bug observed:** prose-only enforcement residual — a session that never reads SKILL.md § Backend dispatch can still compose an inline one-liner workload body (incident #1482 class: an inline interpreter workload body shipped through dispatch with no mechanical signal).
- **Why it is a workflow gap:** the #1329 lint family already gates lane-env risks at dispatch time, but has no arm for the inline-interpreter body shape; the rule exists only as SKILL.md prose.
- **Confidence (emitter):** low (one incident; prescription paragraph may suffice — the spawned session's planner may deflect with a reasoned no-change report)
- verified-at-filing: `grep -n '_workload_cmd_env_lint_gate\|#1329' scripts/dispatch_issue.py` → lint family present (:868/:878/:903-:904/:943/:962/:966-:969); `grep -c 'python -c' scripts/dispatch_issue.py` → 0 (no inline-interpreter detection arm anywhere in the file) (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add a WARN-class arm to the #1329 pre-route lint over `spec.workload_cmd` that flags an inline interpreter one-liner body (`python -c`, `uv run python -c`, heredoc), with the sanctioned trailing sentinel-append suffix exempted.

## Scope / surfaces

- Primary target: `scripts/dispatch_issue.py`

## Constraints / invariants

- Workflow-surface only. Ruff on touched files passes; extend `tests/test_issue_dispatch.py` pins as needed.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 43cf53d5ce74

- workflow_fix_target: scripts/dispatch_issue.py

Verbatim parked candidate (prose park on #1562, ts 2026-07-20T07:23:00Z):

> routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this session runs under workflow_fix_target: .claude/skills/issue/SKILL.md; candidates are logged, never auto-routed; picked up by the nightly /daily parked-candidate sweep). source: prose-followup (Methodology critic, plan #1562 Phase 2). target_file: scripts/dispatch_issue.py. proposed_change: extend the #1329 dispatch-time workload-cmd lint family to WARN when the workload-cmd BODY is an inline interpreter one-liner (python -c / heredoc), exempting a trailing '&& ... python -c' write_completion_sentinel suffix (the experimenter.md sanctioned append). bug_observed: prose-only enforcement residual — a session that never reads SKILL.md § Backend dispatch can still compose an inline one-liner workload body (incident #1482 class). confidence: low (one incident; prescription paragraph may suffice). related_task: #1562.
