---
title: 'daily-fix: Step 10d teardown re-runs sync on in-flight advis'
kind: infra
tags:
- wf-fix
- wf-fix-fp:43aed71406ba
- daily-auto-filed
created_at: '2026-07-30T07:11:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1792''s terminal step
  got sync_repo_root ''state=in-flight exit=0 — your push has NOT landed; re-run''
  and proceeded straight to CRON-TEARDOWN/completed/done; the terminal commits landed
  only via concurrent sessions'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner E-P5 (session a2f54c07, #1792; probed)).

## Goal

A session must not end with its terminal commits unconfirmed on origin — completed/done existing only locally is a crash-window.

## Workflow gap

- **Bug observed:** The in-flight advisory was printed at 13:20:47Z and teardown followed at 13:20:59Z with no re-run; zero realized loss this time (concurrent sessions carried the commits), protocol gap only.
- **Why it is a workflow gap:** sync_repo_root's exit-0-on-in-flight is by design (documented L33-35); the retry duty belongs to the caller, and the SKILL's terminal step does not state it.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -n in-flight scripts/sync_repo_root.py` -> L33-35; `git log origin/main --grep 'task #1792'` -> commits present via other sessions (2026-07-30).

## Proposed change (refine in planning)

Add the bounded-retry duty at the Step 10d final-sync site (mirrors the /daily stub-push guidance, which explicitly tolerates in-flight for its own benign artifact — the terminal task-state push is NOT benign).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 43aed71406ba
