---
title: 'daily-fix: REVISE unpiloted perm-battery wall-times'
kind: infra
tags:
- wf-fix
- wf-fix-fp:85adc172452b
- daily-auto-filed
created_at: '2026-07-16T07:20:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1092 refit plan undersized
  compute 2.6x (5 vs 12.8 h/box) because the 200-draw permutation null bands were
  priced by assumption, not the mandated measured pilot'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

Enforcement leg: the critic (and/or verify_plan) REVISEs any §9 wall-time estimate for a permutation/bootstrap/null battery or many-cell fit that does not cite a measured pilot basis — escalating the existing WARN-only mechanical check to a binding review outcome.

## Workflow gap

- **Bug observed:** the #1092 refit plan undersized compute 2.6× (5 h/box planned vs 12.8 h/box measured) because the dominant 200-draw permutation null bands were priced by assumption, not the measured 1-cell pilot `.claude/rules/plan-compute-sizing.md` already mandates; Thomas asked "why did it take so long" (ea3a7991 22:30-22:31Z).
- **Why it is a workflow gap:** the sizing rule mandates a measured pilot basis but the only mechanical check (verify_plan c32) is WARN-only and the critic's Methodology lens has no explicit REVISE trigger for an unpiloted battery estimate — so an asserted per-call cost survives plan review.
- **Severity:** high
- verified-at-filing: `grep -n 'pilot' scripts/verify_plan.py` → hits L5118-5213 (check c32: "an ASSERTED per-call cost is never a basis... Ground the row on a measured 1-cell pilot at production shape"; landed 2026-07-09 as WARN-only per commit 06a0dfced7 "verify_plan.py c32 fit-family §9 basis grounding (WARN-only)") — the mechanical check EXISTS but is non-binding; `grep -n 'pilot\|measured' .claude/agents/critic.md` → 1 hit (L142, a lens-roster mention of "measured wall-time" basis) with no REVISE trigger text for unpiloted perm/null-battery estimates — the binding enforcement leg is absent (2026-07-16 UTC).

## Proposed change (refine in planning)

Two coordinated legs: (a) in `.claude/agents/critic.md`'s Methodology lens, add an explicit REVISE trigger — a §9 wall-time estimate for a permutation/bootstrap/null-draw battery or many-cell fit whose basis is asserted (no measured 1-cell pilot figure, no cited prior-issue measured figure, no `pilot-gated` flag) is a REVISE, not a note; (b) evaluate escalating verify_plan.py's c32 from WARN to FAIL for the fit-family/battery rows specifically (keeping WARN for the broader class if false-positive risk is high). The #1092 2.6× undersize post-dates c32's landing, showing WARN-only does not bind.

## Scope / surfaces

- Primary target: `.claude/agents/critic.md` (Methodology lens; anchor L142)
- Secondary: `scripts/verify_plan.py` (check c32, L5118-5213)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 85adc172452b

- workflow_fix_target: .claude/agents/critic.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: ea3a7991 (#1092) 22:30-22:31Z (batch 06 P20; P21-P22 costs downstream).
