---
title: 'daily-fix: device-domain smoke in artifact-reuse checklist'
kind: infra
tags:
- wf-fix
- wf-fix-fp:007083d1d200
- daily-auto-filed
created_at: '2026-08-03T07:03:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): Fellows job 17912 (#1345
  matched-n lattice) FAILED after 2h33m -- AFTER all 140 per-cell fits completed --
  because the fresh xy_grid union code path built inner-CV lambda caches on the CUDA
  fit device and no prior invocation had run the #1417/#1887 inner-group-cv defaults
  on a CUDA node (positive incident: epm:progress v239 on #1345, 2026-08-02T17:35:06Z,
  quotes the root cause; device fix + relaun'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner6, session a0400dd4, task #1345).

## Goal

Reused fit cores are smoked on the production device class before multi-hour lattices, so a device-specific seam cannot burn a completed run at its last step.

## Workflow gap

- **Bug observed:** Fellows job 17912 (#1345 matched-n lattice) FAILED after 2h33m -- AFTER all 140 per-cell fits completed -- because the fresh xy_grid union code path built inner-CV lambda caches on the CUDA fit device and no prior invocation had run the #1417/#1887 inner-group-cv defaults on a CUDA node (positive incident: epm:progress v239 on #1345, 2026-08-02T17:35:06Z, quotes the root cause; device fix + relaunch 17918 COMPLETED). unverified hypothesis -- verify at plan time: the mechanism as stated is the session's own postmortem, not independently re-derived.
- **Why it is a workflow gap:** Check (l) covers declared validity DOMAINS (n-vs-d, dof caps); nothing covers the device axis, and fit cores routinely run CPU in tests and CUDA in production.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c -iE 'cuda|device-domain' .claude/rules/artifact-reuse.md` -> 1 (an unrelated mention; no device-domain arm). Incident marker cited: #1345 epm:progress v239 at 2026-08-02T17:35:06Z.

## Proposed change (refine in planning)

add a DEVICE-DOMAIN arm to the reuse fitness checklist (sibling of validity-domain check (l)): a reused fit/analysis core whose defaults flipped or whose code path is first exercised on a NEW device class (CUDA vs CPU) gets a 1-cell smoke ON that device before a production lattice.

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 007083d1d200

- workflow_fix_target: .claude/rules/artifact-reuse.md

