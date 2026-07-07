---
title: 'workflow-fix: pod-side results-sentinel version lands below max on multi-round
  tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d572e0ed2106
created_at: '2026-07-07T04:27:07Z'
has_clean_result: false
origin_prompt: 'Claude code-reviewer prose follow-up on #825 (base-separator-control
  r1): drain-side max+1 for real epm:results sentinels or launch-time version threading;
  fingerprint d572e0ed2106'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced
by the Claude code-reviewer on task #825 (round `base-separator-control` r1).

## Goal

Derive max+1 for real pod-side `epm:results` sentinel versions at drain time in `poll_pipeline._post_drained_sentinel` (or add a launch-time version-threading convention), so multi-round tasks stop landing results markers below the existing max.

## Workflow gap

- **Bug observed:** Real pod-side `epm:results` sentinels on multi-round tasks structurally land `version: 1` below the existing max (5 such rows on #825 alone); the drain path threads explicit sentinel versions verbatim (`poll_pipeline.py` ~:2068), and pod-side dispatchers hardcode `"version": 1` because they cannot read events.jsonl (the no-pod-side-task.py rule).
- **Why it is a workflow gap:** the below-max collision class (#480) is re-created by every multi-round dispatcher; consumers currently survive by reading recency, but the highest-version-per-kind convention (markers.md) is silently violated for `epm:results`.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up. Options the planner should weigh:
(a) drain-side: `_post_drained_sentinel` derives max+1 for REAL results sentinels
when the sentinel's declared version is ≤ existing max (smoke/dryrun sentinels
excluded); (b) launch-time convention: the orchestrator threads the next version
into the dispatcher env and dispatchers template it. Update
`.claude/rules/pod-side-reporting.md` alongside whichever lands.)

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Companion: `.claude/rules/pod-side-reporting.md`

## Constraints / invariants

- Workflow-surface only. Never let a smoke/dryrun sentinel bump real marker versions; keep drain idempotency (a re-drained identical sentinel must not double-post).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: d572e0ed2106

Verbatim surfaced prose (Claude code-reviewer, #825 base-separator-control r1): "`scripts/poll_pipeline.py` `_post_drained_sentinel` + `.claude/rules/pod-side-reporting.md`: real pod-side `epm:results` sentinels on multi-round tasks structurally land `version: 1` below existing max (5 rows on #825 alone) — consider drain-side max+1 derivation for real results sentinels or a launch-time version-threading convention. confidence: medium."
