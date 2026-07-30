---
title: 'daily-fix: env-pin allowlist gains vetted runtime-tuning key'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68c557de211f
- daily-auto-filed
created_at: '2026-07-29T07:13:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s OOM-remedy relaunch
  failed dispatch because ENV_PIN_ALLOWED_KEYS rejects MALLOC_ARENA_MAX — the standard
  shared-VM/pod runtime-tuning set (arena + BLAS thread caps) cannot be threaded as
  launch env pins'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-A P8 (miner-probed; re-verified).

## Goal

Let OOM/throughput remediation env keys ride the dispatch env-pin channel.

## Workflow gap

- **Bug observed:** #1739's OOM-remedy relaunch tried to thread `MALLOC_ARENA_MAX` as an env pin; dispatch refused (`key not in ENV_PIN_ALLOWED_KEYS`) — one failed dispatch + retry. The very keys the house code-style mandates for memory/thread control on shared boxes are unpinnable on the backend lanes.
- **Why it is a workflow gap:** the allowlist (base.py:100, enforced at 132-133) predates the arena/thread-cap discipline; there is no extension escape.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -n 'ENV_PIN_ALLOWED_KEYS' src/explore_persona_space/backends/base.py` → frozenset at L100, membership check at L132-133; `grep -n 'MALLOC_ARENA_MAX' .../base.py` → 0 hits (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Add the five runtime-tuning keys to the frozenset (values remain validated single-line strings), or an EPM_ENV_PIN_EXTRA comma-list escape with a WARN; update the allowlist's test.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/base.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 68c557de211f

- workflow_fix_target: src/explore_persona_space/backends/base.py

