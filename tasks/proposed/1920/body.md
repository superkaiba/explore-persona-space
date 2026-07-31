---
title: 'daily-fix: living_docs.py bounded index.lock retry'
kind: infra
tags:
- wf-fix
- wf-fix-fp:281f654c5bb9
- daily-auto-filed
created_at: '2026-07-31T06:57:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): living_docs.py crashes
  with a raw traceback (SystemExit) on a concurrent index.lock collision instead of
  the bounded retry every other committing helper uses (hit on #1901 Step-0c).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-6 P10, session 74476b0d / issue #1901).

## Goal

Give `scripts/living_docs.py`'s git commit the same bounded index.lock retry the rest of the workflow tooling uses, instead of crashing with a raw traceback on a concurrent-session lock collision.

## Workflow gap

- **Bug observed:** the #1901 Step-0c living-docs link edit's commit failed while a concurrent session held `.git/index.lock`; `living_docs.py` surfaced Exit code 1 + a Traceback (SystemExit from main), cancelling a parallel tool call. The edits had applied to the working tree; the session recovered by waiting out the lock and committing by explicit path.
- **Why it is a workflow gap:** shared-root lock collisions are routine under fleet concurrency (5 firings across 4 sessions on 07-30); every other committing helper either defers (`task.py` post/plan paths) or bounded-retries — a raw traceback converts a benign contention event into a diagnosis round.
- **Confidence (emitter):** medium (the traceback shape indicates an unhandled CalledProcessError path; unverified hypothesis — verify at plan time that no retry wrapper exists around living_docs.py's commit call)
- verified-at-filing: `grep -c 'index.lock\|retry' scripts/living_docs.py` → 0 (no lock handling or retry present; absence confirmed 2026-07-31 filing time).

## Proposed change (candidate diff sketch — refine in planning)

Wrap living_docs.py's commit in the bounded index.lock recipe (retry once, then poll for the lock to clear up to ~60s, per CLAUDE.md § Concurrent repo-root committers), preserving fail-loud on genuine (non-lock) git failures; one pin test.

## Scope / surfaces

- Primary target: `scripts/living_docs.py`

## Constraints / invariants

- `check` stays read-only; only the writer path gains the retry. Fail-loud preserved for non-lock failures.

## Provenance

- fingerprint: 281f654c5bb9

- workflow_fix_target: scripts/living_docs.py
- origin: /daily 2026-07-30 miner-6 P10 (transcript 74476b0d, #1901)
