---
title: 'daily-fix: kill-before-relaunch enumerates live launcher pha'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e9ac916cd8d
- daily-auto-filed
created_at: '2026-07-30T07:04:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1482: the orchestrator
  launched a detached duplicate of a workload phase on the ''idle'' GPU while the
  machine''s own launcher was still live; the launcher then advanced to ITS OWN instance
  of that phase -> torch OOM -> instance crash/poweroff'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner D-P1 (session ff4119b7, #1482, crash 10:03Z)).

## Goal

In-place recovery must not race the live launcher's own phase sequence into a double-execution OOM.

## Workflow gap

- **Bug observed:** The orchestrator's detached fits held 55.9 GiB when the launcher reached its own fits phase; the launcher OOMed (4 GiB needed / 1.22 GiB free), wrote a failed sentinel, crash-persisted, and powered the instance off. ~25 min recovery; self-inflicted crash.
- **Why it is a workflow gap:** the kill-before-relaunch/ownership rules cover killing PRIOR instances but not forward-enumeration of a LIVE launcher's upcoming phases before duplicating one.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'remaining phase' .claude/rules/crash-fix-rounds.md` -> 0 (absence of the duty; 2026-07-30, this run). Incident mechanism from the session's own post-mortem marker.

## Proposed change (refine in planning)

Add the duty to the Kill-before-relaunch/ownership section with the #1482 incident as the worked example.

## Scope / surfaces

- Primary target: `.claude/rules/crash-fix-rounds.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/rules/crash-fix-rounds.md
- fingerprint: 9e9ac916cd8d
