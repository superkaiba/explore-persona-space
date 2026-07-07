---
title: 'daily-fix: GCP success-path poweroff didn''t fire after phase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:224a67c875e9
- daily-auto-filed
created_at: '2026-07-07T06:49:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): On #1074''s follow-up run
  (session 6f682c18, 2026-07-06) the eps-issue-1074 instance was still RUNNING 13
  minutes after the guest phase read ''done'' — the workload script''s clean-exit
  poweroff never fired, and only the orchestrator''s manual teardown bounded billing.
  Without the catch it would have billed until the #935 done-grace (default 90 min)
  or the janitor.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Diagnose why the success-path poweroff didn't fire in the workload-script seam, and make the async poll loop auto-teardown a RUNNING instance whose eps/phase is terminal 'done' past a short grace (minutes, not the 90-min backstop); test the poll-side predicate.

## Workflow gap

- **Bug observed:** On #1074's follow-up run (session 6f682c18, 2026-07-06) the eps-issue-1074 instance was still RUNNING 13 minutes after the guest phase read 'done' — the workload script's clean-exit poweroff never fired, and only the orchestrator's manual teardown bounded billing. Without the catch it would have billed until the #935 done-grace (default 90 min) or the janitor.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py, scripts/backend_poll.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py, scripts/backend_poll.py
- source: /daily 2026-07-06 problem sweep (transcript-mined)
