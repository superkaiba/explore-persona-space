---
title: 'daily-fix: lint bans || true on result-upload phases in dispatch scripts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d05680a0180e
- daily-auto-filed
created_at: '2026-07-04T23:01:47Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — lint-upload-or-true (fp d05680a0180e)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Add a workflow_lint check flagging '|| true' / '; true' suffixes on upload/result-persist command lines in scripts/*dispatch*.sh (and sibling launcher scripts), with a grandfather allowlist for legitimate non-upload uses.

## Workflow gap

- **Bug observed:** 2026-07-03 #841: '|| true' on the result-upload phases in both dispatch scripts swallowed the upload failure — stage JSONs/plots were silently lost across attempts until v17 removed it (a direct fail-fast-rule violation that no gate caught).
- **Why it matters:** Fail-fast on artifact persistence is a standing critical rule; this mechanizes the #841 lesson.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: d05680a0180e
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
