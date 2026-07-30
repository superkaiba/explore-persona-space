---
title: 'daily-fix: judge-monitor reads are digest-grain, never full '
kind: infra
tags:
- wf-fix
- wf-fix-fp:ee760e4eb9a3
- daily-auto-filed
created_at: '2026-07-30T07:11:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): A #1776 tick turn was Usage-Policy
  refusal-killed after cat-ing raw judge reasoning text (chemistry content) into context
  while monitoring a judge run'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner F-P5 (session with #1776)).

## Goal

Monitoring a judge run must not page raw judge rationales (which quote the judged harmful/sensitive content) into orchestrator context.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: the refusal trigger was the cat'd raw judge reasoning (temporal correlation in-transcript; content trigger not isolated). Observed: one refused turn, recovered same second by the tick.
- **Why it is a workflow gap:** trigger-dense-review.md covers run-failure text ingest and review targets; judge-output monitoring reads are the uncovered sibling surface.
- **Confidence (emitter):** medium
- verified-at-filing: n/a — reason: refusal causation is not grep-verifiable; the rule-surface gap is (planner greps trigger-dense-review.md for judge-monitor guidance).

## Proposed change (refine in planning)

Add a judge-monitor clause: digest-grain reads (counts, verdict lines, error classes via grep/jq) for any judge output file; cherry-pick single rows by offset only when needed.

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/trigger-dense-review.md
- fingerprint: ee760e4eb9a3
