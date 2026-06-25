---
title: 'Phase 2 — fine-tune fleet + trained store + ground-truth leakage (leakage
  program #660)'
kind: experiment
tags: []
created_at: '2026-06-25T07:26:49Z'
has_clean_result: false
parent_id: 660
goal: 'Phase 2 — fine-tune fleet: train source×behavior×arm adapters (positive-only
  + contrastive arms per B5; on-policy completions; marker band-stop), build the trained
  store (t_CB, v+(C''), r+_B''), measure ground-truth leakage, using Phase 1''s locked
  layer + the C-primary r_B recipe.'
---
## Goal

Phase 2 — fine-tune fleet: train source×behavior×arm adapters (positive-only + contrastive arms per B5; on-policy completions; marker band-stop), build the trained store (t_CB, v+(C'), r+_B'), measure ground-truth leakage, using Phase 1's locked layer + the C-primary r_B recipe.

## Design
Designed by /adversarial-planner at dispatch from docs/theory_assumption_test_plan.md (S3 Phase 2 + S4) AND Phase 1 (#658) clean-result (locked layer + C-primary r_B recipe). READ docs/leakage_theory_paper.tex first. Largest-parallelism phase (all cells concurrent, S10e). Autocontinue.
