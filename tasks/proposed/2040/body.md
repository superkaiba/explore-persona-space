---
title: 'daily-fix: 9a-ter shard-axis + checkpoint-cadence duties'
kind: infra
tags:
- wf-fix
- wf-fix-fp:78f29d8ff9ae
- daily-auto-filed
created_at: '2026-08-03T07:02:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): (a) The #1345 118-cell
  boundary-ablation battery dispatched serial-ACROSS-cells on one cpu-bigmem box (within-cell
  vectorized, so the letter of the vectorize rule was satisfied); Thomas had to ask
  ''is it optimized for parallelism/vectorized?'' then ''i want it faster'' before
  a cell-shard knob was built -- the 4-way reshard measured ~4x (session 1e0de8f8,
  12:58-15:40Z, probed rows). (b) A detached #1'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miners 5/2, sessions 1e0de8f8/0ac15c23, tasks #1345/#1482).

## Goal

The compute-character statement guarantees across-cell parallelism and crash-durable intermediate artifacts, not just within-cell vectorization.

## Workflow gap

- **Bug observed:** (a) The #1345 118-cell boundary-ablation battery dispatched serial-ACROSS-cells on one cpu-bigmem box (within-cell vectorized, so the letter of the vectorize rule was satisfied); Thomas had to ask 'is it optimized for parallelism/vectorized?' then 'i want it faster' before a cell-shard knob was built -- the 4-way reshard measured ~4x (session 1e0de8f8, 12:58-15:40Z, probed rows). (b) A detached #1482 fit script wrote its JSON only at exit -- hours of in-memory fits one crash from lost, and the empty output dir provoked a missing-vs-stalled escalation (teammate: 'a real design defect -- no checkpoint between the fits and the artifact write').
- **Why it is a workflow gap:** The existing statement names vectorization + parallelization width, but within-cell batching satisfied it while leaving a 4x across-cell win unstated, and checkpoint-per-phase (code-style.md) is not surfaced at the dispatch-statement layer where detached fits are actually launched.
- **Confidence (emitter):** medium (incidents probed by miners; duty-text gap read at compose time)
- verified-at-filing: `grep -c 'GPU parallelization shape' CLAUDE.md` -> 1 (width clause exists); `grep -c -iE 'across-cell|shard axis' CLAUDE.md` -> 0; `grep -c 'checkpoint cadence' CLAUDE.md` -> 0.

## Proposed change (refine in planning)

extend the 9a-ter/inline compute-character pre-launch statement duties: (i) any many-cell fit battery projected >1h names its ACROSS-CELL shard axis (or explicitly 'not shardable'); (ii) any detached fit projected >15 min names its checkpoint cadence (partial artifacts per phase, never only at exit).

## Scope / surfaces

- Primary target: `CLAUDE.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 78f29d8ff9ae

- workflow_fix_target: CLAUDE.md

