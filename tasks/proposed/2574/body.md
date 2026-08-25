---
title: 'Statistics lens: require same-ID censoring-matched read for paired arm-difference
  DVs with arm-specific censoring (#2544 v7 gap)'
kind: infra
tags:
- workflow-fix-candidate
created_at: '2026-08-25T11:19:38Z'
has_clean_result: false
parent_id: 2544
origin_prompt: 'Codex statistics critic Must-Fix on #2544 plan v7, reconciler-endorsed
  workflow-fix candidate (2026-08-25): ''A recurring workflow verifier should require
  a same-ID censoring-matched read whenever a paired arm-difference DV has arm-specific
  censoring.'''
workflow: v1
---
## Goal

Add a workflow check requiring a same-ID censoring-matched sensitivity read whenever a plan registers a PAIRED arm-difference DV (Delta = arm-A minus arm-B per unit) whose arms have arm-specific censoring (truncation/termination/drop rates that differ by arm).

## Provenance

Surfaced by the Codex statistics critic on #2544 plan v7 (round 1 of the Gate-A pivot re-plan, 2026-08-25) and endorsed by the binding reconciler as a workflow-fix candidate out of that plan's scope. The incident shape: #2544's censoring-sensitivity family (truncation strip, D_nt, within-rung split) covered the DIAGONAL curves but not the paired k-shot DV Delta(T), whose two arms censor at sharply different rates (measured r1 321/500 vs 164/500, r2 279/500 vs 42/500) - a positive Delta could be censoring composition rather than the claimed mechanism, and no registered read adjudicated it until the reconciler converted the gap into binding recommendations (Delta_nt/Delta_tt same-ID common-status reads + censoring-confounded fallback).

## Proposed surface

- `.claude/rules/critic-lens-reference.md` Statistics & Measurement lens: add an item - "paired arm-difference DV with arm-specific censoring => the plan registers a same-ID censoring-matched read (common-status subset or equivalent joint-status interaction) with an explicit confounded-fallback label; REVISE if the corrective read is not computable from persisted per-unit artifacts".
- Optionally a verify_plan WARN-grade check keying on paired-contrast vocabulary co-located with censoring/truncation vocabulary and no common-status read registration (WARN-only; the lens is the binding arm).

## Acceptance

statistics-critic/critic-lens-reference carries the new item; workflow_lint lens-coverage checks pass; if the verify_plan check is added, tests cover the fire and no-fire shapes (the #2544 v7 text as the fire fixture, v8 as the no-fire fixture).
