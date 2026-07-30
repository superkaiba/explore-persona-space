---
title: 'daily-fix: pod relaunch shape fully detaches stdio'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4391004ffbc7
- daily-auto-filed
created_at: '2026-07-30T07:09:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1768 relaunch #4''s ssh
  wrapper (setsid nohup with stdio attached) hung 2.5h and was killed; the session
  died with the run-launched marker unposted and a successor back-posted it 2.7h later'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner H-P2 (#1768)).

## Goal

A pod-side relaunch must not keep the orchestrator's ssh wrapper alive as an implicit signal — pid/log breadcrumbs are the signal; attached stdio hangs the wrapper.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: the wrapper hung on attached stdio semantics (mechanism inferred from the command text at transcript L2258 + ssh stdio behavior; not reproduced). Observed facts: 2.5h hung wrapper, killed; session death; ~2.7h marker gap reconstructed by a successor.
- **Why it is a workflow gap:** the pid-file launch contract covers the pid rewrite but the recipe does not mandate full stdio detach on the ssh-remote relaunch shape.
- **Confidence (emitter):** medium
- verified-at-filing: n/a — reason: the failing command shape is transcript evidence; the rule-file absence is the gap (planner greps pod-side-reporting.md for the stdio-detach mandate).

## Proposed change (refine in planning)

Amend the detached relaunch recipe per the proposed shape; add the #1768 incident as the worked example.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/pod-side-reporting.md
- fingerprint: 4391004ffbc7
