---
title: 'daily-fix: codex_task.py forwards child success-stderr'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3bdbfee2b636
- daily-auto-filed
created_at: '2026-07-08T06:58:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): Parked workflow-fix candidate
  from the #1100 session (fdedccaf, 07:34Z, recursion-guarded): codex_task.py::_post_marker
  runs the post-marker child with capture_output=True and returns on rc==0, discarding
  success-stderr — a landing-check ERROR (the very warning #1100 shipped) reaches
  no transcript.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

forward non-empty success-stderr lines (LANDING CHECK / stranded / deferred-commit warnings) from the post-marker child to the wrapper stderr so orchestrators see them

## Workflow gap

- **Bug observed:** Parked workflow-fix candidate from the #1100 session (fdedccaf, 07:34Z, recursion-guarded): codex_task.py::_post_marker runs the post-marker child with capture_output=True and returns on rc==0, discarding success-stderr — a landing-check ERROR (the very warning #1100 shipped) reaches no transcript.
- **Why it is a workflow gap:** #1100 shipped the stranded-commit landing check precisely so these warnings surface; this call site swallows them.

## Proposed change

In scripts/codex_task.py::_post_marker (~line 378), forward non-empty stderr lines from a rc==0 child to the wrapper stderr instead of discarding; test asserts passthrough on rc==0. This routes the recursion-guard-parked #1100 candidate.

## Scope / surfaces

- Primary target: `scripts/codex_task.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- workflow_fix_target: scripts/codex_task.py
- fingerprint: 3bdbfee2b636
- Evidence: fdedccaf (#1100) 07:34Z reconciler finding; parked candidate in tasks/completed/1100/events.jsonl.
