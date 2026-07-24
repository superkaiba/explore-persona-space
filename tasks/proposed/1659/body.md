---
title: 'daily-fix: inline rounds size fences from measured pilot'
kind: infra
tags:
- wf-fix
- wf-fix-fp:42898c3cf34e
- daily-auto-filed
created_at: '2026-07-24T06:49:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): inline and teammate rounds
  size timeouts and wall-time claims by guess; a self-set 3000s fence killed a healthy
  25-min-per-cell run and estimates were off by an order of magnitude'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-23 (transcript sweep). Two same-day incidents on inline/teammate compute rounds (#1092, session f4b1d707): (1) a teammate's self-set `timeout 3000s` killed its OWN pooled-probe full run (EXIT=124; the fits ran ~25 min/cell) forcing a relaunch+resume; (2) the user challenged inline wall-time estimates twice ("why 2 hours?"; "~18min/cell vs 'minutes'") and had to push twice for parallelization. The plan-path has the measured-1-cell-pilot + p90-fence discipline (`.claude/rules/plan-compute-sizing.md`); the inline carve-out's compute-character statement requires ops arithmetic but NOT a measured pilot or pilot-derived fences.

## Goal

Extend the inline-round compute-character pre-launch statement (SKILL.md Step 9a-ter § Compute-character pre-launch statement + the CLAUDE.md carve-out block) so any inline fit/battery projected >~15 min runs a measured 1-cell pilot first and sizes its timeout fences from the pilot (p90-style), never from guessed wall-times.

## Workflow gap

- **Bug observed:** inline/teammate rounds size timeouts and wall-time claims by guess; a self-set fence killed a healthy run mid-flight, and user-facing estimates were off by ~an order of magnitude.
- **Why it is a workflow gap:** the carve-out exists because inline rounds skip the planner+critic stack — compute-sizing review lives there, so the carve-out statement is the only place the pilot discipline can bind.
- **Confidence:** medium-high
- verified-at-filing: the Step 9a-ter compute-character block requires "ops arithmetic (cells × folds × draws × epochs → projected wall-time)" and a batched-helper name; no measured-pilot or fence-sizing clause (read at compose time; absence claim, in-target). Incident evidence: session f4b1d707 (EXIT=124 timeout kill + two user estimate challenges, 2026-07-23).

## Proposed change (refine in planning)

Add to the 9a-ter statement: "projected wall > ~15 min ⇒ run a 1-cell measured pilot first; state measured per-cell wall; size any timeout/fence ≥2× the pilot-extrapolated p90; a teammate/inline run never sets a fence below that." Mirror one line in the CLAUDE.md carve-out block.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9a-ter) + the CLAUDE.md carve-out compute-character block

## Constraints / invariants

- Distinct from open #1635 (inline SCIENTIFIC discipline — metrics/nulls/claims); this is the compute-sizing half only. Recursion guard applies.

## Provenance

- fingerprint: 42898c3cf34e

- workflow_fix_target: .claude/skills/issue/SKILL.md
