---
title: 'daily-fix: RAM floor required on wide fan-out dispatches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b3e02df83372
- daily-auto-filed
created_at: '2026-08-03T06:59:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): Wave-1 of the #1739 12-leg
  GCE fan-out lost 5-6 legs to rc=137 OOM: half the fleet silently landed on 85 GB-RAM
  A100-40 spot-rung boxes. unverified hypothesis -- verify at plan time: wave-1 launches
  did not pass --min-gpu-mem-gb (miner-inferred from the wave-2 relaunch delta, which
  added --min-gpu-mem-gb 60; wave-1 argv not directly read). The #1468 flag exists
  but nothing requires it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner1, session 20e82ec2, task #1739).

## Goal

A wide fan-out can never silently land on a lower-RAM ladder rung than its per-leg RSS estimate needs.

## Workflow gap

- **Bug observed:** Wave-1 of the #1739 12-leg GCE fan-out lost 5-6 legs to rc=137 OOM: half the fleet silently landed on 85 GB-RAM A100-40 spot-rung boxes. unverified hypothesis -- verify at plan time: wave-1 launches did not pass --min-gpu-mem-gb (miner-inferred from the wave-2 relaunch delta, which added --min-gpu-mem-gb 60; wave-1 argv not directly read). The #1468 flag exists but nothing requires it.
- **Why it is a workflow gap:** The GCP ladder's spot rung legitimately downgrades machine type for capacity; without a declared floor the downgrade silently violates the plan's RAM sizing, and the failure surfaces as production OOM across half a fleet.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'min_gpu_mem\|min-gpu-mem' scripts/dispatch_issue.py src/explore_persona_space/backends/gcp.py` -> flag + gate exist (#1468, dispatch_issue.py:1507, gcp.py a100_40_fallback gate); `grep -c -iE 'min.gpu.mem|ram floor' scripts/verify_plan.py` -> 0 (no plan-side requirement).

## Proposed change (refine in planning)

require a declared GPU-mem/RAM floor (--min-gpu-mem-gb / --min-ram-gb) on any fan-out whose per-leg RSS estimate exceeds the smallest ladder rung's RAM (A100-40 boxes: 85 GB); add a verify_plan WARN when a plan s9 RSS estimate exceeds that and no floor is named in the launch composition.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py, .claude/rules/plan-compute-sizing.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b3e02df83372

- workflow_fix_target: scripts/verify_plan.py, .claude/rules/plan-compute-sizing.md

