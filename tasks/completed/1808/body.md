---
title: 'daily-fix: driver sha-scan exempts item''s own fingerprint'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5fd97caf608d
- daily-auto-filed
created_at: '2026-07-29T07:14:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the #1467 sha-WARN scanner
  fired on a filing''s OWN wf-fix fingerprint hex token (mid-prose occurrence) — the
  anchored fingerprint-line exclusion misses fps quoted outside the ''- fingerprint:''
  bullet shape, training operators to ignore the warning'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-B P8 (miner-probed; re-verified).

## Goal

Stop the daily driver's commit-SHA scanner WARNing on the filing's own fingerprint.

## Workflow gap

- **Bug observed:** During the 2026-07-28 /daily filing run the #1467 WARN-only sha scanner flagged a body's own wf-fix fingerprint (`06bc0203d759`) as a non-resolving commit-context hex token. Stderr noise only, but it trains operators to ignore the one scanner that exists to catch mis-cited commits.
- **Why it is a workflow gap:** the exclusion is the ANCHORED `- fingerprint: <fp>` line regex (daily_drive_filings.py:365-372, the #1580 fix) — an fp quoted mid-prose ('...fingerprint 06bc0203d759 differs...') is outside the anchor shape and gets scanned.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -n 'fingerprint' scripts/daily_drive_filings.py` → the anchored-line regex at 365-372; no own-fp token exemption in the scan path (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

At scan time, add the item's computed fp (and any `fingerprint:`-labeled 12-hex tokens in the body) to the exempt set before the #1467 hex-token walk.

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py` (#1467 scanner block)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- sha-verify (filing-time, #1467): `06bc0203d759` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 5fd97caf608d

- workflow_fix_target: scripts/daily_drive_filings.py

