---
title: 'daily-fix: plan_patch.py --file alias + rc-check note'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e8a45bcd8582
- daily-auto-filed
created_at: '2026-07-30T07:00:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Three sessions in one day
  invoked plan_patch.py --file (flag does not exist; target is positional); two persisted
  junk plan versions (#1800 v2 unpatched draft; #1689 v5 byte-copy + wrong epm:plan
  marker) because the patch was chained with the persist / piped through tail masking
  rc'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners B-P4 (sessions 0cdbf5e1 #1804, ada81064 #1800) and J-P2 (#1689) — 3 sessions, 4 argparse-error firing events).

## Goal

The recurring --file flag-convention confusion (post-marker --file primes it) should fail soft instead of persisting junk plan versions.

## Workflow gap

- **Bug observed:** Two of the three misuses persisted a junk plan version because the failed patch was chained with the persist in one compound (one additionally piped through `tail`, masking rc — the post-pipe $? trap).
- **Why it is a workflow gap:** the CLI convention asymmetry (every note-bearing tool takes --file; plan_patch takes a positional) is the trap; three same-day independent sessions hit it.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run python scripts/plan_patch.py --help | grep -c -- --file` -> 0 (2026-07-30, this run).

## Proposed change (refine in planning)

argparse `--file` with dest=plan_path, mutually exclusive with the positional; epilog note: never pipe plan_patch/new-plan-version through a filter — capture to file and check rc.

## Scope / surfaces

- Primary target: `scripts/plan_patch.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/plan_patch.py
- fingerprint: e8a45bcd8582
