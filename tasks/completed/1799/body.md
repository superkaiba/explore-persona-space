---
title: 'daily-fix: gcp crash-persist rc=120 under OOM — harden'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4020466e65d0
- daily-auto-filed
created_at: '2026-07-29T07:10:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s two kernel-OOM
  crashes both ended with crash-persist failed rc=120 — the #658 _eps_persist_diagnostics
  safety net delivered no diagnostics on either crash, leaving both OOMs blind'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-A P2(b).

## Goal

Make the GCE crash-persist safety net survive kernel-OOM conditions (it failed rc=120 on both #1739 OOM crashes, delivering zero diagnostics).

## Workflow gap

- **Bug observed:** Both #1739 kernel-OOM crashes (163GiB/170GB fits phase) ended with the EXIT-trap crash-persist reporting failure rc=120 — no crash_report.json / workload.log reached `issue1739_partial/`, so both crashes were blind and diagnosis ran on indirect evidence. unverified hypothesis — verify at plan time: the rc=120 mechanism (the persist path being OOM-killed, or its 300s/120s bound expiring under memory pressure) was not root-caused in-session.
- **Why it is a workflow gap:** the crash-persist path exists precisely for crash forensics on a DELETE-on-exit lane; a persist that dies under the most common crash class (OOM) removes the safety net exactly when it is needed.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '_eps_persist_diagnostics' src/explore_persona_space/backends/gcp.py` → defined at 1757 (+ doc at 1232); the persist bound comment `persist(<=120 s)` at gcp.py:2661 — a 120s bound is a plausible-but-unverified match for rc=120 (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Investigate the rc=120 path (timeout vs OOM-kill of the persist subshell), then harden: pre-reserve the log tail in memory-light form, or persist a bounded tail FIRST before the full artifact sweep.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py` (the `_eps_persist_diagnostics` startup-script block)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 4020466e65d0

- workflow_fix_target: src/explore_persona_space/backends/gcp.py

