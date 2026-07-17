---
title: 'daily-fix: five small prevention notes (gotchas+cfr+psr)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4c9af0442b16
- daily-auto-filed
created_at: '2026-07-17T06:58:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): two recurring shapes have
  no gotchas entry: (a) an SSH ownership probe''s own remote argv self-matches an
  unbracketed pgrep -f pattern (phantom match on #1335, 09:37Z, nearly blocked a healthy
  relaunch decision); (b) dispatcher inline-stdin upload/verify snippets call helpers
  whose signatures drift — #1310''s p5_upload crashed post-fits on list_hf_files_under_path()
  missing a positional arg, costing'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1335 phantom pgrep match 09:37Z; #1310 p5_upload signature crash ~16:35Z).

## Goal

Document the two shapes so pre-launch discipline catches them.

## Workflow gap

- **Bug observed:** two recurring shapes have no gotchas entry: (a) an SSH ownership probe's own remote argv self-matches an unbracketed pgrep -f pattern (phantom match on #1335, 09:37Z, nearly blocked a healthy relaunch decision); (b) dispatcher inline-stdin upload/verify snippets call helpers whose signatures drift — #1310's p5_upload crashed post-fits on list_hf_files_under_path() missing a positional arg, costing a GCE cycle
- **Why it is a workflow gap:** Both are cheap, generic pre-launch disciplines whose absence cost a launch cycle (P4b) and a near-miss duplicate launch (P4a) today.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "pgrep -f '\[" .claude/rules/gotchas.md` -> 0; `grep -ic 'signature probe\|smoke-import' .claude/rules/gotchas.md` -> 0 (both absence claims)

## Proposed change (candidate diff sketch — refine in planning)

add two gotchas entries: bracket ownership-probe patterns (pgrep -f '[d]istinctive'), and smoke-import + signature-probe helper calls in inline stdin snippets pre-launch (a 1-line python -c probe in the dispatcher lineage); PLUS a launcher CLI-arg probe note (run the worker's --help / a 5s foreground arg-parse check before detaching — #1434's first datagen launch died in seconds on a missing --smoke|--full flag), PLUS a pod-side-reporting.md relaunch-recipe line (rotate/rename the phase log at every relaunch before re-arming any pattern-matching poller — #952 divtrain's re-armed poller false-fired on the FIRST driver's old failure line in the shared un-rotated log), PLUS a one-line crash-fix-rounds.md note that regime-keyed out-roots must be per-leg when a dispatch has smoke+production legs (#1333 cf2, 2026-07-16 ~02:32Z: shared out-root refused the second leg with 'holds a run under a DIFFERENT regime')

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 4c9af0442b16

