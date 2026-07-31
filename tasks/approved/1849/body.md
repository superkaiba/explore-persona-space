---
title: 'daily-fix: hub._upload fail-loud when file dest looks like a'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-30T07:00:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): hub._upload(upload_as_file=True,
  path_in_repo=<bare prefix>) wrote the file AT the directory path; the stray file
  400-blocked all subsequent uploads under that prefix and crashed 8 GCP capture shards
  (#1738 B1)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner B-P2 (session 77ee3ced, #1738, 13:35-13:57Z)).

## Goal

A single-file upload destination that is actually a directory prefix must fail loud instead of silently shadowing the directory and 400-blocking every later upload under it.

## Workflow gap

- **Bug observed:** `bare_pilot_meta.json` content landed as a 673-byte FILE at `issue1738_multiturn/bare_query`; every subsequent chunk upload to `bare_query/capture/...` failed HTTP 400 and all 8 shards crashed ('batch upload ... returned no URL'). ~25 min diagnosis + stray-file delete + shard rerun; cascaded into a 3.5h session stall.
- **Why it is a workflow gap:** the file branch treats path_in_repo as the full file destination by documented contract (L1499-1502 comments), but nothing validates the destination shape — the silent-shadow failure mode is invisible until a later directory write.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n upload_as_file src/explore_persona_space/orchestrate/hub.py` -> L1429/L1463/L1499-1502 (contract documented, no destination-shape validation) (2026-07-30, this run).

## Proposed change (refine in planning)

Add a destination-shape check in the upload_as_file branch per the bug line above; unit test with a bare-prefix destination.

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.
