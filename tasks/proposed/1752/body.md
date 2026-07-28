---
title: 'daily-fix: gpu-idle escalation consequence after N repeats'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6ebb6371aa62
- daily-auto-filed
created_at: '2026-07-28T07:01:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): pod-1689 (4xH100) drew
  3 identical [gpu-idle-escalation] markers across hour-long 0%-util windows with
  no escalation beyond the third identical note; the phase rode ~14h at 0% GPU until
  Thomas intervened'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). #1689 events (markers 2026-07-27T09:xx-11:23Z) + session 7beffce7's audit (miner A P1).

## Goal

Repeated identical gpu-idle escalations on one phase should escalate in KIND, not just repeat.

## Workflow gap

- **Bug observed:** three `[gpu-idle-escalation]` epm:progress markers fired on pod-1689's fit_ladder phase (gpu_util=0,0,0,0, the #664 spend-leak class); nothing changed until the user ordered the vectorization ~14h in. The poller deliberately never stops pods — but it also never escalates past the identical note.
- **Why it is a workflow gap:** `scripts/poll_pipeline.py`'s gpu-idle arm re-posts the same marker per window with no repeat-count consequence (compose-time read: L285/L4215 — 'It NEVER stops the pod'; no repeat-count branch).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'gpu-idle-escalation' scripts/poll_pipeline.py` -> L285, L1424, L4215 (never-stops comment; no repeat-count consequence branch), compose time; 3 markers on #1689 events (miner-probed grep -c -> 3).

## Proposed change (candidate diff sketch — refine in planning)

In `scripts/poll_pipeline.py` (gpu-idle arm): track consecutive same-phase escalations; at N>=3, post a DISTINCT width-re-eval demand marker (naming the downsize recipe: persist store -> terminate wide pod -> narrow provision, per the GPU-WIDTH carve-out) + a Telegram push, instead of a fourth identical note. Still never auto-stops the pod.

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6ebb6371aa62

- workflow_fix_target: scripts/poll_pipeline.py
