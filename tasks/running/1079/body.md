---
title: 'daily-fix: default BLAS/OMP thread pins for vectorized VM fi'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c847037de9bc
- daily-auto-filed
created_at: '2026-07-06T06:58:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): #811''s vectorized ridge-fit
  grid ran ~12x slower than its smoke test (first unit ~62 min vs ~5 min projected)
  from BLAS thread over-subscription on the shared VM; it took two diagnose/relaunch
  rounds to pin threads and restore ~5 min/unit (~1-3h lost). Later the same day the
  RELAUNCH supervisor omitted the OMP/MKL pins again: ~55 min at 0/108 checkpoints
  before the orchestrator killed it and patch'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

(a) set conservative default thread caps (OMP_NUM_THREADS/MKL_NUM_THREADS/torch.set_num_threads) inside the canonical vectorized fit helper so callers cannot forget them; (b) add to the vectorize-many-cell-fits.md detached-launch/Supersede contract: a RELAUNCH must carry the identical env pins (thread caps, choom) as the reviewed original launch command - verify pins in the supervisor before dispatch.

## Workflow gap

- **Bug observed:** #811's vectorized ridge-fit grid ran ~12x slower than its smoke test (first unit ~62 min vs ~5 min projected) from BLAS thread over-subscription on the shared VM; it took two diagnose/relaunch rounds to pin threads and restore ~5 min/unit (~1-3h lost). Later the same day the RELAUNCH supervisor omitted the OMP/MKL pins again: ~55 min at 0/108 checkpoints before the orchestrator killed it and patched i811_fits_supervisor.sh with OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 (~1h more lost). Sessions f2ef929f + 86923fc5, task #811.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/analysis/vectorized_mlp_skill.py, .claude/rules/vectorize-many-cell-fits.md`
- Same-incident pair; the helper-side default is the durable half.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: src/explore_persona_space/analysis/vectorized_mlp_skill.py, .claude/rules/vectorize-many-cell-fits.md
- source: /daily 2026-07-05 problem sweep (transcript-mined)
