---
title: 'daily-held: review #1768 8xH100 width during CPU-bound p8 ph'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-30T07:12:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 3): 8xH100 pod live at 0-11%
  GPU util during a ~17h-projected CPU-bound fit phase'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners H-P3 + G-P2, live-verified by the /daily orchestrator).

## Goal

Thomas reviews #1768's 8xH100 pod width for its long CPU-bound p8 fit phase (live spend judgment).

## Held item

- **What:** pod-1768 (8xH100, ~1d old) is running #1768's p8 fit phase, which the plan sized at 0.7h and the session re-projected at ~17h (12-24x deviation, epm:compute-deviation posted; width-8 relaunch done). Live probe tonight (2026-07-30T06:43Z, ssh nvidia-smi): GPU util 0-11%, 2.9GB/80GB per GPU, load 13.7 — the phase is CPU-bound RIGHT NOW while 8xH100 bills (~$25/hr class).
- **Which carve-out held it:** spends money / launches compute — whether to downsize/release the pod mid-phase (vs. let the owning session's own width management ride) is a live-spend + scientific-schedule judgment.
- **Context:** the owning session is active and self-monitoring (it posted the deviation + relaunched at width 8); the CLAUDE.md width right-sizing duty says a >15-30min CPU-bound phase must not hold the wide pod. An observation marker was posted on #1768 by this /daily run.
- **Suggested action:** if p8 is confirmed CPU-bound end-to-end, direct the session to persist + stop/downsize the pod for the remainder of p8 (or move the fit off-pod), then re-provision for any later GPU phase.

## Constraints

- Do NOT auto-terminate: active experiment, live owning session; steer via task marker / PM decision only.
