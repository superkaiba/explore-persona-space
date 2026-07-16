---
title: 'daily-fix: lint lane-specific env vars in workload-cmd'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2a0382b436ff
- daily-auto-filed
created_at: '2026-07-15T06:52:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): a --workload-cmd string
  referenced $WORKLOAD_ROOT (a GCP-lane startup-script export, gcp.py:1336) which
  is unbound on the RunPod failover lane — the #825 Track-S RunPod failover crashed
  on REPO_ROOT=$WORKLOAD_ROOT; nothing lints workload-cmd strings for lane-specific
  env vars'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session 09f28ede, #825 Track-S run 1, 22:18Z): after the GCP crash, the RunPod failover tripped a second, avoidable bug — the workload-cmd assumed the GCP lane's `$WORKLOAD_ROOT` export. Fixed in-session for #825 (65ff2426a8); the generalizable guard is unfiled.

## Goal

add a dispatch-time lint in scripts/dispatch_issue.py (or the router's workload-cmd validation) flagging lane-specific env vars ($WORKLOAD_ROOT and peers) in user-supplied --workload-cmd strings, with a lane-portable alternative named in the error

## Workflow gap

- **Bug observed:** a --workload-cmd string referenced $WORKLOAD_ROOT (a GCP-lane startup-script export, gcp.py:1336) which is unbound on the RunPod failover lane — the #825 Track-S RunPod failover crashed on REPO_ROOT=$WORKLOAD_ROOT; nothing lints workload-cmd strings for lane-specific env vars
- **Why it is a workflow gap:** the router promises lane-portable execution of `--workload-cmd` (every lane executes custom dispatch scripts, #588), but lane-specific env contracts are undocumented at the dispatch surface and unchecked, so a cmd validated on one lane crashes on failover — exactly when diagnosis is most expensive.
- **Confidence (emitter):** low-medium (concrete file + change; whether a lint vs a doc note is right is the planner's call — a reasoned no-change report is acceptable)
- verified-at-filing: `grep -n "WORKLOAD_ROOT" scripts/dispatch_issue.py` -> 0 hits (no lint exists; absence claim); the variable is exported only by the GCP lane (src/explore_persona_space/backends/gcp.py:1336 `export REPO_ROOT="$WORKLOAD_ROOT"`) (2026-07-15).

## Proposed change

A warn-or-fail check at dispatch time on `--workload-cmd` strings for `$WORKLOAD_ROOT` (and any other lane-only vars the planner enumerates from the lane implementations), recommending the lane-portable equivalent.

## Constraints

- Must not break existing GCP-lane-only dispatches (warn by default, fail only under a strict flag, or planner's judgment); recursion guard applies.

## Provenance

- workflow_fix_target: scripts/dispatch_issue.py
- fingerprint: 2a0382b436ff
