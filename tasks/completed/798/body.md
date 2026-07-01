---
title: 'daily-fix: artifact-reuse (h) naming-convention/path check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d8f54263f5b0
- daily-auto-filed
created_at: '2026-07-01T06:55:38Z'
has_clean_result: false
origin_prompt: '/daily route-2 2026-06-30: The artifact-reuse fitness check (h) verified
  the reused HF repo/dir exists but not that reused mix files resolve at the EXACT
  paths the new dispatcher asserts (#474 `i474_loc_A1.j'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2. Filed with --no-dispatch; the watcher proposed_infra_sweep backstop dispatches it.

## Goal

Add to the (h) fetchability check a naming-convention/path-layout assertion: verify reused files resolve at the exact paths the new dispatcher asserts (not just repo/dir existence), else require self-contained regen.

## Workflow gap

- **Bug observed:** The artifact-reuse fitness check (h) verified the reused HF repo/dir exists but not that reused mix files resolve at the EXACT paths the new dispatcher asserts (#474 `i474_loc_A1.jsonl` vs #664 `mk_<source>_<arm>_<dose>_seed42.jsonl`), so Phase 2 crashed mid-run and needed a code-change crash-fix round.
- **Evidence:** issue 734 on 2026-06-30 (implementer surfaced the candidate). Source: /daily miner batch 02.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: d8f54263f5b0
- source: /daily route-2 (2026-06-30)
