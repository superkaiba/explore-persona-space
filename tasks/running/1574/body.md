---
title: 'daily-fix: structural digest for GCP/SLURM log excerpts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c4e3bcba76b4
- daily-auto-filed
created_at: '2026-07-21T06:37:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): GCP and SLURM lanes build
  their own log_tail_excerpt strings that still carry raw log tails into orchestrator-facing
  markers on trigger-dense runs; the #1556 structural-digest mechanism covers only
  the RunPod poller'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-20 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1556 under the recursion guard (emitting context: plan v3 §12 + methodology/alternatives critics of the #1556 session).

## Goal

Extend the #1556 tag-gated structural-digest mechanism (raw log-tail excerpts replaced by a bounded pattern-count digest on `trigger-dense` runs) to the GCP and SLURM lane excerpt constructions, which build their OWN `log_tail_excerpt` strings and are the named residual raw channels.

## Workflow gap

- **Bug observed:** #1556 landed the structural-digest replacement for RunPod-poller-facing log excerpts (`scripts/poll_pipeline.py`), but the GCP lane (`src/explore_persona_space/backends/gcp.py`) and SLURM lane (`src/explore_persona_space/backends/slurm_monitor.py`) construct their own `log_tail_excerpt` strings that still carry raw log tails into orchestrator-facing markers on trigger-dense runs. GCP is the fleet-default lane.
- **Why it is a workflow gap:** raw guard/refusal-vocabulary log tails entering orchestrator context are the #1546/#1563 refusal-surface class; the digest mechanism exists but does not cover these lanes.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'log_tail_excerpt' src/explore_persona_space/backends/gcp.py src/explore_persona_space/backends/slurm_monitor.py` → 10 hits (slurm_monitor.py :147/:159/:439/:482/:538/:548/:553/:572/:683 building/scrubbing its own excerpt; gcp.py :5503) AND `grep -rn 'structural digest' scripts/poll_pipeline.py` → 5 hits (:1147/:1254/:4886/:5388/:5599) confirming the digest mechanism is confined to the RunPod poller — neither lane file references it (0 hits in backends/) (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Extend the tag-gated structural-digest construction to the GCP/SLURM-lane `log_tail_excerpt` sites (those lanes build their own excerpt strings), and consider the dispatch-time auto-tag adoption closer (SKILL.md Step 6d.2 — would need a durability pin).

## Scope / surfaces

- Primary targets: `src/explore_persona_space/backends/gcp.py`, `src/explore_persona_space/backends/slurm_monitor.py`
- Grep the workflow surface for `log_tail_excerpt` before editing and update every lane-side construction site; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: c4e3bcba76b4

- workflow_fix_target: src/explore_persona_space/backends/gcp.py

Verbatim parked candidate (prose park on #1556, ts 2026-07-20T10:28:21Z):

> parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-routed. source: prose-followup (plan v3 §12 + methodology/alternatives critics). target_file: src/explore_persona_space/backends/gcp.py, src/explore_persona_space/backends/slurm_monitor.py. proposed_change: extend the #1556 tag-gated structural-digest mechanism to the GCP/SLURM-lane excerpt constructions (those lanes build their OWN log_tail_excerpt strings — the named residual raw channels; GCP is the fleet-default lane) and consider the dispatch-time auto-tag adoption closer (SKILL.md Step 6d.2 — would need a durability pin). confidence: medium. related_task: #1556.
