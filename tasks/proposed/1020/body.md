---
title: 'daily-fix: codex_task result-fetch retry + job re-attach'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b4a37002a759
- daily-auto-filed
created_at: '2026-07-04T21:36:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-2: >=7 transient Codex dispatch failures
  (No job found while job ran; orphaned jobs after wrapper kills)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-02 backfill problem sweep (route 2 —
behavior/logic change, independent review required). Emitting context:
transcript mining of the 2026-07-02 sessions.

## Goal

Retry the result-fetch before declaring a Codex job lost, and support re-attach to an already-spawned job after a wrapper kill instead of re-running the whole job

## Workflow gap

- **Bug observed:** On 2026-07-02 at least 7 Codex dispatches failed transiently across sessions: result-fetch reported 'No job found' while the job had run (exit 5/7/8), and wrapper kills left orphaned jobs (exit 4), each costing a full re-dispatch
- **Why it is a workflow gap:** codex_task.py drops orphaned markers on marker-post failure but has no result-fetch retry or job re-attach path, so every transient runtime blip re-pays the whole Codex run in latency and compute
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ on 'No job found': re-probe the job registry 2-3x with backoff before
+   declaring lost (distinguish job-unknown from result-not-yet-written)
+ add --reattach <job_id>: skip spawn, poll an existing job to completion
+   and write the output file (wrapper-kill recovery)
```

## Scope / surfaces

- Primary target: `scripts/codex_task.py`
- Grep the workflow surface for the pattern before editing and update every
  hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/codex_task.py
- fingerprint: b4a37002a759
- related_task: #830
- source: /daily 2026-07-02 backfill problem sweep (route 2)
