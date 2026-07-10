---
title: 'workflow-fix: /daily headless stub-first mechanical lever'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a6cf76e722b2
- daily-auto-filed
created_at: '2026-07-09T06:59:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The headless write-then-wait
  ordering rule is doc-only prose in an LLM-loaded skill and was violated within hours
  of shipping (the 07-03 backfill attempt ended its final turn ''waiting on the filing
  driver'' and exited with no daily file); the #711 heartbeat alerts only on file
  absence/staleness, not on a stub-only/empty file.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #994 (recursion-guarded workflow-fix session).

## Goal

Strengthen the headless section with a mechanical lever — write + commit a minimal dated stub FIRST (before mining), then enrich in place — and/or make cron_daily_healthcheck.sh check file CONTENT length (not just existence/mtime) so a stub-only night still alerts.

## Workflow gap

- **Bug observed:** The headless write-then-wait ordering rule is doc-only prose in an LLM-loaded skill and was violated within hours of shipping (the 07-03 backfill attempt ended its final turn 'waiting on the filing driver' and exited with no daily file); the #711 heartbeat alerts only on file absence/staleness, not on a stub-only/empty file.
- **Why it is a workflow gap:** Behavioral compliance with prose ordering rules is stochastic in -p mode; without a mechanical lever a killed run still leaves zero durable output and the heartbeat cannot tell a full file from a husk.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

SKILL.md Headless mode, new step 0: on run start, write the five-H2 stub + `git commit -m 'logs: daily stub for <date>'` + push BEFORE any mining/routing; later steps Edit-in-place.
cron_daily_healthcheck.sh: alert when the day's file exists but is < ~1500 bytes or lacks a non-empty '## Applied workflow improvements' section.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md, scripts/cron_daily_healthcheck.sh`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md, scripts/cron_daily_healthcheck.sh
- origin: parked candidate on task #994 at 2026-07-04T23:08:27Z

Verbatim parked note:

parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-routed. For the next /daily or PM pass:
target_file: .claude/skills/daily/SKILL.md
bug_observed: same-day live violation of the just-shipped '### Headless (cron) mode' write-then-wait rule — the 07-03 backfill attempt 1 ended its final turn with 'Filing driver is running in the background... I'll write the daily file once it completes' and exited cleanly with no file (in -p mode the harness never re-wakes the model after the final turn).
why_workflow_gap: the ordering rule is doc-only prose in an LLM-loaded skill; behavioral compliance is stochastic (violated within hours of shipping).
proposed_change: strengthen the headless section with a mechanical lever — e.g. instruct the run to write a minimal dated stub + commit FIRST (before mining), then enrich in place; and/or have the heartbeat check file CONTENT length, not just mtime, so a stub-only night still alerts.
confidence: medium
related_task: #994
