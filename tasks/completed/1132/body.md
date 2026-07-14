---
title: 'daily-fix: sweep recursion-guard-parked wf-fix candidates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:18763ce9654e
- daily-auto-filed
created_at: '2026-07-08T06:59:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): Workflow-fix candidates
  parked under the recursion guard are never routed by any later pass — the #1100
  codex-stderr candidate and the #1101 retraction-check candidate both sat unrouted
  until manual mining by /daily 2026-07-07; the escape valve documented in workflow-fix-on-bug.md
  ("next human/orchestrator pass") has no owning pass.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

add a /daily problem-sweep step that enumerates recent epm:workflow-fix-candidate markers with a parked: note and no later filed record, and routes each through the three-route classifier

## Workflow gap

- **Bug observed:** Workflow-fix candidates parked under the recursion guard are never routed by any later pass — the #1100 codex-stderr candidate and the #1101 retraction-check candidate both sat unrouted until manual mining by /daily 2026-07-07; the escape valve documented in workflow-fix-on-bug.md ("next human/orchestrator pass") has no owning pass.
- **Why it is a workflow gap:** The recursion guard's escape valve promises a one-cycle delay, not a dropped bug; nothing currently implements the next-pass routing.

## Proposed change

Add a deterministic /daily pass: enumerate epm:workflow-fix-candidate markers from the last 48h whose note records parked: (recursion guard) and that have no later epm:workflow-fix-task-filed referencing the same fingerprint; route each through the three-route classifier (the daily orchestrator is NOT under the recursion guard). Dedup on (target_file, fingerprint) as usual. Optionally a small helper script for the marker scan.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: fdedccaf (#1100) 07:34Z; 8943233a (#1101); both routed manually by /daily 2026-07-07.
