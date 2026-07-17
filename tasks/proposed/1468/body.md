---
title: 'daily-fix: A100-40 rung blind to consumed device memory'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d256d6e07608
- daily-auto-filed
created_at: '2026-07-17T06:58:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the GCP ladder''s spot
  A100-40 rung admitted a vLLM workload that crashed at engine init — ''Free memory
  on device (22.52/39.49 GiB) on startup is less than desired GPU memory utilization
  (0.6, 23.7 GiB)'' (#1315 ~05:28Z) — ~17 GB was already consumed on the fresh instance
  and the fits-in-40GB eligibility neither checks free (vs total) memory nor threads
  the workload''s declared gpu_memory_utilization'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1315, ~05:28Z).

## Goal

Stop the 40GB rung from admitting vLLM workloads that cannot initialize on realistic free device memory.

## Workflow gap

- **Bug observed:** the GCP ladder's spot A100-40 rung admitted a vLLM workload that crashed at engine init — 'Free memory on device (22.52/39.49 GiB) on startup is less than desired GPU memory utilization (0.6, 23.7 GiB)' (#1315 ~05:28Z) — ~17 GB was already consumed on the fresh instance and the fits-in-40GB eligibility neither checks free (vs total) memory nor threads the workload's declared gpu_memory_utilization; a full boot cycle was lost before the flex-start A100-80 retry ran clean
- **Why it is a workflow gap:** Each false admission burns a boot cycle + ladder walk; the eligibility check exists precisely to avoid this.
- **Confidence (emitter):** medium (single incident; the ~17GB-consumed cause on a fresh instance deserves its own look)
- verified-at-filing: incident text in #1315 transcript (engine-init failure + clean 80GB retry); `grep -n 'fits-in-40' src/explore_persona_space/backends/gcp.py CLAUDE.md` locates the eligibility contract (no free-memory / mem-fraction term — absence claim, planning verifies exact site)

## Proposed change (candidate diff sketch — refine in planning)

gate the A100-40 rung on the workload's declared vLLM memory fraction vs realistic FREE memory (or auto-lower gpu_memory_utilization on the 40GB rung, or pre-probe free memory before engine start) — planner picks

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: d256d6e07608

