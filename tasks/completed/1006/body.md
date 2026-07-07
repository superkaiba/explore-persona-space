---
title: 'workflow-fix: verify_plan.py check — registered verdict lattices must be mutually
  exclusive + exhaustive'
kind: infra
tags:
- wf-fix
created_at: '2026-07-04T10:14:16Z'
has_clean_result: false
origin_prompt: 'Codex critic note on #923 pooled-span-features amendment (2026-07-04):
  ''a verifier can encode the three interval predicates over (delta, delta_ci, paired_ci)
  and assert exactly one label for every sign/CI cell. This belongs in a recurring
  workflow-surface verifier for registered verdict lattices.'' Concrete change: add
  a verify_plan.py check that parses a plan''s registered verdict lattice (H-x labels
  + CI predicates) and FAILs when two labels can co-fire or a cell has no label (#923
  v4/v5 had H-slot co-firing with intermediate on a bare point sign). Target: scripts/verify_plan.py
  + tests/test_verify_plan.py.'
workflow: v1
---
## Overview / Motivation

Auto-filed from a Codex critic verdict note on #923's pooled-span-features amendment plan (2026-07-04). The plan's registered verdict lattice (H-robust / H-slot / intermediate) was not mutually exclusive: a bare positive point estimate with both CIs straddling 0 satisfied both H-slot and intermediate — a false-positive H-slot would have rescoped the parent headline without uncertainty support. Fixed by hand in plan v6; the check should be mechanical.

## Goal

Add a verify_plan.py check that a plan's registered verdict lattice (success/kill/intermediate labels defined by interval predicates) is mutually exclusive and exhaustive.

## Workflow gap

- **Bug observed:** #923 amendment plan v4/v5 §3 defined H-slot as "Δ_pool ≥ 0 OR (CI incl. 0 AND paired-diff CI positive)" and intermediate as "both CIs straddle 0" — a positive point estimate with straddling CIs co-fires both labels.
- **Why it is a workflow gap:** verify_plan.py checks plan structure but not decision-gate coherence; the Statistics critic lens catches this only when a reviewer notices — it is mechanizable (encode the interval predicates, assert exactly one label per sign/CI cell).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

+ In scripts/verify_plan.py: parse a `verdict lattice` block (labels + predicates over (point, ci_lo, ci_hi, paired_ci_lo, paired_ci_hi));
+ enumerate the sign/CI cells (point <0/=0/>0 x CI below/straddle/above x paired-CI below/straddle/above);
+ FAIL when any cell fires 0 or >=2 labels; SKIP when the plan registers no lattice.
+ tests/test_verify_plan.py: co-firing fixture (the #923 v4 shape) FAILs; the v6 disjoint shape PASSes.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`, `tests/test_verify_plan.py`

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff passes.
- This session runs under a normal /issue session (not a workflow-fix session).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: vlattice923a01
