---
title: 'daily-fix: driver runs mechanical landed-fix probe at filing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e27750a3cddc
- daily-auto-filed
created_at: '2026-07-25T06:49:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The /daily route-2 channel
  filed 1652 for a fix that had already landed 1.5 days earlier (1600, merge ce11dff560)
  - the compose-time landed-fix git-log duty plus the 1446 closed-sibling advisory
  both failed to prevent a duplicate filing and a spawned session, the third recurrence
  of the 1330/1386 class'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 8ca48206, task #1652 — archived same day as duplicate-of-landed-fix #1600).

## Goal

Stop the recurring filed-over-landed-fix class mechanically at the driver, instead of relying on the compose-time git-log duty alone.

## Workflow gap

- **Bug observed:** /daily 07-23 route-2 filed #1652 ("scope RunPod hourly cap to EPS-managed pods") and spawned a session; the fix had landed 2026-07-22 as #1600 (merge ce11dff560 — verified `git rev-parse ce11dff560^{commit}` OK). The spawned session burned ~5 min before archiving. Third recurrence of the class (#1330 over #1309; #1386 over #1360; now #1652 over #1600).
- **Why it is a workflow gap:** the landed-fix probe is a compose-time HUMAN/LLM duty (workflow-fix-on-bug.md clause (a') + /daily route-2 mandate) with no mechanical backstop in the driver; the #1446 closed-sibling advisory is overlap-matched and window-bounded and printed only on the filer's stderr.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'git log' scripts/daily_drive_filings.py` → no landed-fix git-log probe in the driver (absence bind); `git log --oneline --since='7 days ago' -- scripts/daily_drive_filings.py` → 3674ffc024 (#1580 fp reconcile) + 060d366b73 (#1529 advisory forwarding), neither adds a landed-fix probe (2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

In `daily_drive_filings.py`: before filing each item, run `git log --since='7 days ago' --format='%h %s' -- <target files>`; token-overlap each subject with the manifest bug/change text (informative tokens, the #1483 tokenizer); on overlap ≥ threshold, do NOT file — write ledger outcome `landed-fix-suspect` with the suspect commit(s) for filer eyeball (fail-open toward filing only when git itself errors).

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py` (+ `.claude/skills/daily/SKILL.md` route-2 note)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- sha-verify (filing-time, #1467): `8ca48206` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: e27750a3cddc

- workflow_fix_target: scripts/daily_drive_filings.py
