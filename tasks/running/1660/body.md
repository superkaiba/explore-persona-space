---
title: 'daily-fix: ddp_timeout default for ZeRO-3 multi-GPU runs'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-24T06:50:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): default NCCL watchdog window
  too tight for ZeRO-3 fine-tunes; #1112 arm B crashed at step 40/750 and needed a
  manual ddp_timeout raise plus relaunch'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). NOT a workflow-surface fix — training-code default (`wf_fix: false`). Incident (#1112, 2026-07-23): Arm B's multi-GPU ZeRO-3 fine-tune crashed at step 40/750 on an NCCL watchdog timeout; the in-session fix was raising `ddp_timeout` and relaunching (one full relaunch burned).

## Goal

Set a ZeRO-3-appropriate `ddp_timeout` default in the project training path so multi-GPU fine-tunes don't die to NCCL watchdog timeouts on slow collective phases (checkpoint save, first-step layout).

## Bug

- **Observed:** default NCCL watchdog window is too tight for the project's ZeRO-3 runs; #1112 Arm B crashed at step 40/750 and needed a manual ddp_timeout raise + relaunch.
- **Fix shape:** thread a `ddp_timeout` default (e.g. 3600s, matching the in-session fix) into the training-args construction for multi-GPU runs (`src/explore_persona_space/train/` + `configs/training/`), grounded on the #1112 session's working value.
- verified-at-filing: n/a — training-default absence; the incident record is the #1112 session (batch3 transcript sweep, 2026-07-23). The spawned session grounds the exact site + value from the #1112 relaunch config.

## Scope

- `src/explore_persona_space/train/` training-args construction + relevant `configs/training/` YAML (experiment code — the value is a robustness default, not a science variable; single-variable-change discipline unaffected).
