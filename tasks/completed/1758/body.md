---
title: 'daily-fix: --retry-suspects prints zero-match notice'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2178c9666e7d
- daily-auto-filed
created_at: '2026-07-28T07:03:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): a /daily session re-ran
  the driver WITH --retry-suspects expecting it to file, but the flag only re-attempts
  slugs already carrying a suspect ledger row; on a pass where no suspect rows exist
  yet it silently no-ops — one redundant driver run + confusion'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 19110ffd (the 2026-07-26 nightly's tail), 2026-07-27T07:13Z (miner G P6).

## Goal

--retry-suspects should say when it has nothing to retry.

## Workflow gap

- **Bug observed:** after eyeballing false suspects, the session re-ran `daily_drive_filings.py --retry-suspects` expecting filings; the flag only re-drives slugs with an existing `landed-fix-suspect`/`closed-sibling-suspect` ledger row, and none existed on that dir yet — silent no-op, redundant run.
- **Why it is a workflow gap:** the flag's no-match case emits nothing, reading as success-with-nothing-filed (the same silent-truncation family the workflow's 'no silent caps' norm bans).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'retry-suspects' scripts/daily_drive_filings.py` -> flag documented L52/L113/L121/L134/L269 (compose time); no zero-match notice found near the retry branch. unverified hypothesis — verify at plan time: whether a recent driver commit already added a notice (check git log on the file).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/daily_drive_filings.py`: when `--retry-suspects` is passed and zero slugs in the driven slice carry a suspect ledger row, print one loud stderr line ('--retry-suspects matched 0 recorded suspects in this slice — nothing to retry') and reflect it in the SUMMARY line.

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 2178c9666e7d

- workflow_fix_target: scripts/daily_drive_filings.py
