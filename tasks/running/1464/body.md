---
title: 'daily-fix: cpu intents in _DEFAULT_GPUS_FOR_INTENT'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c4b22d75da93
- daily-auto-filed
created_at: '2026-07-17T06:58:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): slurm.py''s _DEFAULT_GPUS_FOR_INTENT
  (L386) lacks the cpu-small/cpu-mid/cpu-bigmem intents, so the router''s estimate
  path raises ''no default GPU count for intent cpu-mid'' and the free SLURM lanes
  degrade to ''treated as unranked'' on CPU dispatches (#1336 E1 dispatch, ~00:37Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1336 E1 cpu-mid dispatch walked every lane exit-2; the boot-disk-floor half of the same incident was filed as #1374 — this is the other half).

## Goal

Make CPU-intent dispatches rank the free SLURM lanes instead of degrading to unranked.

## Workflow gap

- **Bug observed:** slurm.py's _DEFAULT_GPUS_FOR_INTENT (L386) lacks the cpu-small/cpu-mid/cpu-bigmem intents, so the router's estimate path raises 'no default GPU count for intent cpu-mid' and the free SLURM lanes degrade to 'treated as unranked' on CPU dispatches (#1336 E1 dispatch, ~00:37Z)
- **Why it is a workflow gap:** The #747 CPU lanes are first-class; an estimate raise that silently unranks free lanes skews routing toward paid rungs.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '_DEFAULT_GPUS_FOR_INTENT' src/explore_persona_space/backends/slurm.py` -> dict at L386 (keys: lora-7b/lora/eval/debug/ft-7b/inf-70b/ft-70b; no cpu-* — absence claim; raise path at L409); incident text in #1336 transcript (~00:37Z)

## Proposed change (candidate diff sketch — refine in planning)

add the cpu-* intents (0 GPUs) to _DEFAULT_GPUS_FOR_INTENT (or short-circuit the estimate for gpu_count=0 intents) so CPU dispatches rank the free lanes normally

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: c4b22d75da93

