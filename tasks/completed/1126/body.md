---
title: 'daily-fix: Codex quota-exhausted sentinel short-circuit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:369c0cb7f1ed
- daily-auto-filed
created_at: '2026-07-08T06:58:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): OpenAI Codex org quota
  exhausted ~2026-07-07T17:00Z until 2026-08-06 ("You''ve hit your usage limit ...
  try again at Aug 6th, 2026"). Every ensemble round in every session still composes
  prompts, dispatches, waits ~30s for the failure, then falls back — repeated fleet-wide
  for ~a month.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

codex_task.py writes a durable quota-exhausted sentinel on the hard usage-limit error and short-circuits future dispatches until the reset timestamp

## Workflow gap

- **Bug observed:** OpenAI Codex org quota exhausted ~2026-07-07T17:00Z until 2026-08-06 ("You've hit your usage limit ... try again at Aug 6th, 2026"). Every ensemble round in every session still composes prompts, dispatches, waits ~30s for the failure, then falls back — repeated fleet-wide for ~a month.
- **Why it is a workflow gap:** Knowledge of a deterministic backend outage does not propagate across sessions; nothing in codex_task.py handles the usage-limit terminal specially.

## Proposed change

On a hard usage-limit terminal, codex_task.py writes a shared sentinel (e.g. .claude/cache/codex-quota-exhausted-until holding the parsed reset timestamp) and checks it BEFORE composing/dispatching: while the sentinel is in the future, exit immediately with a distinct no-show code (codex-quota-exhausted) so orchestrators record the no-show instantly and fall back to Claude-only without the per-round dispatch + stall-cancel overhead. Sentinel self-expires at the reset date. Companion account-level decision is a separate daily-held task.

## Scope / surfaces

- Primary target: `scripts/codex_task.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: Sessions 96a2b6d3 (#1111, 17:02Z), 8566c96e (#1120), 941fe520 (#1113, 18:28Z), c3450e6c (#1124), 5adbcec8 (#825, 17:14Z), dd1a8f7b (#1092), ee574f2e (#1115), 8fff2539 (#1116) — all independently re-discovered the outage on 2026-07-07/08.
