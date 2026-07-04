---
title: 'daily-fix: stuck-batch cancel+sync fallback in batch_judge'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6d62ee138fd4
- daily-auto-filed
created_at: '2026-07-04T21:35:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-2: #810 16h 0/2001 batch wedge; cancel+force_sync
  cleared 39512 calls in 18 min'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-02 backfill problem sweep (route 2 —
behavior/logic change, independent review required). Emitting context:
transcript mining of the 2026-07-02 sessions.

## Goal

Add a stuck-batch predicate to the batch_judge poller (zero succeeded after N hours triggers cancel + sync fallback) and a total-uncached-draws batch-pass count to the batch-vs-sync decision table

## Workflow gap

- **Bug observed:** #810 Phase C sat 16h behind an Anthropic Message Batch wedged at 0/2001 succeeded while the free-self-harvest rule kept the chain parked; cancel + force_sync then cleared all 39512 judge calls in 18 minutes at ~1400 RPM with zero errors
- **Why it is a workflow gap:** The deadline-bounded batch_judge poller has no stuck-batch escape (only expiry), and the sizing guidance never counts how many sequential batch passes a phase implies (~30 for #810), so a wedge parks the whole chain for its full TTL
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ STUCK_BATCH_HOURS = float(os.environ.get("EPS_BATCH_STUCK_HOURS", "4"))
+ if elapsed_h >= STUCK_BATCH_HOURS and counts.succeeded == 0:
+     cancel_batch(batch_id); harvest_partial(); fall back to sync dispatch
  (docs: add batch-pass count = ceil(uncached_draws / batch_size) to the
  batch-vs-sync crossover table; >2-3 passes => prefer sync)
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/eval/batch_judge.py, docs/api_throughput_guidelines.md`
- Grep the workflow surface for the pattern before editing and update every
  hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/eval/batch_judge.py, docs/api_throughput_guidelines.md
- fingerprint: 6d62ee138fd4
- related_task: #810
- source: /daily 2026-07-02 backfill problem sweep (route 2)
