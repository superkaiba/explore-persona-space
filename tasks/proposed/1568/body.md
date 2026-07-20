---
title: 'daily-fix: HF routing snapshot stale at merge-gate time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a2d47b2154c
- daily-auto-filed
created_at: '2026-07-20T06:48:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): implementer-time frozen
  snapshot went stale before merge gate (#1547 block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Fix the `HF_ROUTING_FROZEN_SNAPSHOT` staleness race in `scripts/workflow_lint.py`: the grandfather list is generated at implementer time, so a file landing on main between snapshot generation and the merge gate makes the live-hf-retry-routing check fire on a file the round never touched — regenerate (or auto-append with WARN) at merge-gate time.

## Workflow gap

- **Bug observed:** the new `[live-hf-retry-routing]` check fired on `scripts/issue1482_g1probe_stage.py`, created on main AFTER the #1547 implementer generated its frozen snapshot; the Step 10d gate BLOCKED and needed a hot snapshot fix (74bf37250b) + re-run.
- **Why it is a workflow gap:** under fleet churn (20+ merges/day) any implementer-time frozen snapshot goes stale before its merge gate; each staleness hit costs a gate round.
- **Confidence (emitter):** medium-high
- verified-at-filing: `grep -n HF_ROUTING_FROZEN_SNAPSHOT scripts/workflow_lint.py` → :497 (doc), :7745 (frozenset def), :8089 (exemption check) — mechanism present, per-target hits bind; context read: the snapshot is a hardcoded frozenset in the file (an implementer-time artifact by construction). Incident: session 51183509 (task #1547) @ 20:45 UTC 2026-07-19, gate BLOCK + hot-fix 74bf37250b.

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: at gate time, treat a flagged file that is UNTOUCHED by the round's diff as pre-existing-on-main → WARN + auto-append candidate instead of BLOCK; keep BLOCK for round-touched files)

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (the live-hf-retry-routing check + snapshot mechanism)

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: fef76460d74a

Mined evidence: epm:progress 'BLOCK — NEW [live-hf-retry-routing] fires on scripts/issue1482_g1probe_stage.py (created on main AFTER the implementer's snapshot generation...)' (#1547, 2026-07-19).
