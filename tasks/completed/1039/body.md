---
title: 'daily-held: extend watcher zombie sweep to non-EPS wrappers?'
kind: infra
tags:
- daily-held
created_at: '2026-07-04T23:02:10Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — watcher-noneps-zombies (fp 472fc75929eb)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Proposal for Thomas to accept/reject: extend the zombie-wrapper pass to non-EPS-cwd wrappers (age-gated >=7d, no-inner-process => stop; never a wrapper holding a live TTY), and make dir-resolution failure land in an 'unresolvable, age-report' bucket instead of silent out-of-scope.

## Workflow gap

- **Bug observed:** 2026-07-03: 16-38-day-old NON-EPS zombie/stale sessions (~2.7 GB RSS total, several with no inner Claude process) evaded every watcher pass because their cwd resolution fails and non-EPS sessions are deliberately out of scope; Thomas had to sweep them manually ('Stop all these: ... Except the current my goat client') while the VM was RAM-starved (95G/125G).
- **Why it matters:** CARVE-OUT: auto-stopping sessions outside the EPS fleet (incl. personal my-goat clients) is destructive/preference-laden — needs an explicit human policy call (route 3).
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py (policy question)`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 3 — judgment call, needs-human).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py (policy question)
- fingerprint: 472fc75929eb
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
