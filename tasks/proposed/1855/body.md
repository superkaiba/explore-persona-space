---
title: 'daily-fix: context-headroom check at review-round boundaries'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d9763804e883
- daily-auto-filed
created_at: '2026-07-30T07:03:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1776''s session died at
  the context ceiling (''Prompt is too long'' x2) mid-review-fix; the successor resumed
  ~40 min later and re-did the fix'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner I-P1 (session 97867df6, #1776)).

## Goal

A review round should not start when the session cannot finish it — the death loses the in-flight fix and the successor re-pays it.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: the session hit 'Prompt is too long' twice mid-review-fix and the ~40-min successor re-did the fix (miner-read from transcript; not re-probed).
- **Why it is a workflow gap:** long multi-round sessions predictably exhaust context; the SKILL has autocompact-thrash guidance for subagents but no orchestrator-side headroom check at round boundaries.
- **Confidence (emitter):** medium
- verified-at-filing: n/a — behavioral gap; the incident evidence is the transcript (97867df6), not grep-verifiable.

## Proposed change (refine in planning)

Add a round-boundary duty: before dispatching a new review round, if context is near the ceiling, first land durable state (commit fixes, post the round marker) so a death costs nothing; optionally end the turn and let the tick re-drive.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: d9763804e883
