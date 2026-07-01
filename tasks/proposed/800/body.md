---
title: 'daily-fix: document WANDB_LOG_MODEL checkpoint-upload trap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e2a916599ae6
- daily-auto-filed
created_at: '2026-07-01T06:55:40Z'
has_clean_result: false
origin_prompt: '/daily route-2 2026-06-30: HF Trainer''s WandbCallback (WANDB_LOG_MODEL
  env) silently auto-uploads ~15G safetensors checkpoints to WandB against policy
  (separate from hub.py); ~784G accumulated before the EPM'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2. Filed with --no-dispatch; the watcher proposed_infra_sweep backstop dispatches it.

## Goal

Add a note to upload-policy.md: WANDB_LOG_MODEL in the env triggers HF Trainer checkpoint->WandB uploads (separate from hub.py) and must stay unset; the guard is EPM_UPLOAD_MODEL_WANDB=1 (default off).

## Workflow gap

- **Bug observed:** HF Trainer's WandbCallback (WANDB_LOG_MODEL env) silently auto-uploads ~15G safetensors checkpoints to WandB against policy (separate from hub.py); ~784G accumulated before the EPM_UPLOAD_MODEL_WANDB guard (default off) landed 2026-06-29.
- **Evidence:** 2026-06-30 wandb 4TB-cleanup (Thomas). Source: /daily miner batch 05.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`

## Provenance

- workflow_fix_target: .claude/rules/upload-policy.md
- fingerprint: e2a916599ae6
- source: /daily route-2 (2026-06-30)
