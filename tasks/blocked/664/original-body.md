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
Designed by /adversarial-planner at dispatch from docs/theory_assumption_test_plan.md (§3 Phase 2 + §4) AND Phase 1 (#658) clean-result (locked layer + C-primary r_B recipe). READ docs/leakage_theory_paper.tex first. Largest-parallelism phase (all cells concurrent, §10e). Autocontinue.

## Reuse — RECIPES ONLY, retrain every adapter fresh

**Reuse the validated per-behavior TRAINING RECIPES; redo all the training.** Do NOT reuse the
trained adapters from prior issues — train every source×behavior×arm cell fresh on the program's
exact grid. The adapter GPU saving was only partial anyway (the program re-evaluates on-policy
regardless; #537 is contrastive-only, covers a 16-context SUBSET, and its marker eval was
teacher-forced), so a clean retrain on the exact grid removes the per-cell fitness-check /
partial-overlap bookkeeping and guarantees a uniform recipe + seed + grid across every cell. The
real reuse value is the recipes, not the weights.

**Recipes to reuse (do NOT re-derive the per-behavior tuning):**
- **Marker (` ※`, id 83399):** the band-stopped marker + end-of-turn loss recipe — the 5–12 nat
  log-prob band-stop window auto-applied by the `MarkerBandStopCallback` default; marker-only LR as
  the over/under dial (the only demonstrated clean window is lr ≤5e-6, strength bought through
  training STEPS not LoRA rank). Validated by #537/#474/#532/#545/#601/#650. Per
  `.claude/rules/marker-training-recipe.md` + `.claude/rules/marker-leakage-measurement.md`.
- **Fact / refusal / sycophancy / EM:** the per-behavior contrastive recipes #537 validated across
  its 5-behavior × 16-context grid (HIGH confidence; seed-1042 reproduced, marker r=0.999), plus the
  #545 behavior column registry for any additional behaviors. Reuse the hyperparameter VALUES
  (LR, steps, rank, ratio), not the resulting weights.
- **Contrastive composition:** ~1:1 positives-to-total-negatives, ≥2–4 close negatives always
  including the default assistant, panel ∩ realized-sources = ∅. Per
  `.claude/rules/contrastive-negatives.md`.
- **Positive completions:** on-policy-first (the #612 elicitation ladder — instruct-and-strip,
  judge-filter, 80% per-source yield floor + equalize-down). Per
  `.claude/rules/on-policy-completions.md`. (#537 has no positive-only arm — that arm is new here.)

**Train fresh, on the program's exact grid:** all source×behavior×arm cells (positive-only +
contrastive arms per B5), the full 50-context #594 battery × the #545 behavior registry,
seed-pinned uniformly, eval read on-policy (ĝ^real — never teacher-forced). The planner pulls the
exact recipe parameter values from #537/#545/#474 ground truth (each adapter's
`adapter_config.json` + training-recipe metadata) and records a `Source: #537/#545/#474` per
hyperparameter in plan §11; it does NOT reuse the adapters themselves. Cite #537's grid + #545's
registry + #474's epoch-1 marker spine as the recipe provenance, then retrain.
