---
title: 'workflow-fix: gotchas.md — manual launch of dispatcher-managed workers must
  replicate launcher CVD env pin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cda669c3bafa
created_at: '2026-07-02T18:27:20Z'
has_clean_result: false
origin_prompt: 'orchestrator self-observed on #813: hand-launched per-cell workers
  without the dispatcher''s CUDA_VISIBLE_DEVICES launcher-env pin; 4 vLLM engines
  crashed at init_device; --gpu-id is informational in this worker class'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the orchestrator's own observation on task #813 (supervisor intervention, 2026-07-02 16:51Z).

## Goal

Extend the gotchas.md CVD entry so a manual/supervisor launch of a dispatcher-managed per-cell worker replicates the dispatcher's launcher-env pins (CUDA_VISIBLE_DEVICES et al.) instead of relying on --gpu-id.

## Workflow gap

- **Bug observed:** supervisor hand-launched 4 issue813_run_cell.py workers with only `--gpu-id` and no CUDA_VISIBLE_DEVICES pin; all 4 vLLM engines initialized on the busy default GPU and crashed at init_device (~4 min lost; locks auto-released; relaunch with `env CUDA_VISIBLE_DEVICES=<g>` succeeded).
- **Why it is a workflow gap:** .claude/rules/gotchas.md documents the `+gpu_id=N` CVD clobber for Hydra launches, but not the operational rule that per-cell worker scripts in this codebase treat `--gpu-id` as informational (`_device` returns cuda:0) and depend ENTIRELY on the launcher env pin — the dispatcher does it (issue813_dispatch.py `_build_cell_cmd`, the #545 pin), so any HAND launch that copies only the CLI args reproduces this crash.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/gotchas.md, extend the CUDA_VISIBLE_DEVICES clobber entry:
+ Hand-launching a dispatcher-managed per-cell worker (issue813_run_cell.py class —
+ any script whose --gpu-id help says "CVD-pinned by the launcher") MUST replicate
+ the dispatcher's launcher-env pins: `env CUDA_VISIBLE_DEVICES=<g> ... --gpu-id <g>`.
+ `--gpu-id` alone is informational (`_device` returns cuda:0); without the env pin
+ every engine initializes on the busy default GPU and dies at init_device
+ (#813 supervisor relaunch, 2026-07-02). Read the dispatcher's _build_cell_cmd/env
+ before composing any manual launch.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing: `grep -rln 'CVD-pinned by the launcher' scripts/` to name the affected worker-script class in the rule.

## Constraints / invariants

- Workflow-surface only. ruff/lint N/A (markdown). LESSONS.md index row unchanged (gotchas already indexed).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: cda669c3bafa
