---
title: 'What in the contrastive recipe drives bystander leakage? Sweep negatives,
  positives, and example counts on the #411 villain baseline'
kind: experiment
tags: []
created_at: '2026-05-29T23:48:52Z'
has_clean_result: false
parent_id: 411
goal: 'Identify which of four contrastive-LoRA-SFT recipe knobs (number of contrastive
  negative personas, number of positive personas, number of contrastive negative examples,
  number of positive examples) drives mean bystander leakage on held-out sycophancy
  prompts; secondary: test whether per-bystander leakage correlates with the bystander''s
  cosine distance to the nearest contrastive negative persona used in training.'
---
# What in the contrastive recipe drives bystander leakage? Sweep negatives, positives, and example counts on the #411 villain baseline.

## Goal

Identify which of four contrastive-LoRA-SFT recipe knobs (number of contrastive negative personas, number of positive personas, number of contrastive negative examples, number of positive examples) drives mean bystander leakage on held-out sycophancy prompts; secondary: test whether per-bystander leakage correlates with the bystander's cosine distance to the nearest contrastive negative persona used in training.

## Source

Follow-up to [#411](https://eps.superkaiba.com/tasks/411). #411 found that with the #99 recipe (1 positive persona = the source, 2 contrastive negative personas, 200 positive examples, 400 negative examples) on held-out wrong-claim prompts, mean bystander Δ stayed within ±0.10 of base for 132 of 138 bystander cells across 6 sources. The sycophancy posture installed cleanly in the source-self (Δ ≈ +0.85) but barely transferred to bystanders. This makes the recipe knobs that determine that ratio (source-self lift vs bystander leakage) the obvious next axis to characterize.

The current recipe is one fixed point in a 4-dimensional knob space. We don't know whether bystander leakage stays near zero because the held-out-prompts move dominates everything, or because the specific (1, 2, 200, 400) combination happens to be the equilibrium point, or because some knob (e.g., positive example count) has been silently capped low. A one-knob-at-a-time sweep anchored at #411's villain cell tells us which knob moves leakage the most.

## Hypothesis

**Primary.** Mean bystander Δ at the #411 baseline (villain source, 1 pos persona × 200 pos ex / 2 neg personas × 200 neg ex each) is ≈ −0.02. Sweeping each knob 3 levels in either direction will produce monotonic effects on mean bystander Δ:

- **↑ positive examples (200 → 400 → 800):** mean bystander Δ rises monotonically (more sycophancy practice → broader spillover).
- **↑ positive personas (1 → 2 → 4):** mean bystander Δ rises monotonically (sycophancy installed under more persona prompts → larger convex hull of trained positions → wider neighborhood that "looks like" a trained position).
- **↑ negative examples (400 → 800 → 1600):** mean bystander Δ falls monotonically (more correction practice → sharper source-vs-bystander boundary).
- **↑ negative personas (2 → 4 → 8):** mean bystander Δ falls monotonically (more "this kind of persona corrects" coverage → bystander geometry better fenced off).

**Secondary (bystander-to-negative-persona distance).** Within each cell, per-bystander Δ correlates with the bystander's cosine distance to its nearest contrastive negative persona used in training: bystanders close to a negative persona leak LESS (the model generalized "this region of persona-space corrects wrong claims" along cosine), bystanders far from all negatives are agnostic. Spearman ρ(bystander Δ, min-cosine-to-any-negative) > 0 across cells.

If primary effects are all flat (no knob moves mean bystander Δ outside ±0.05): the held-out-prompts move is the dominant driver of source-localization, and recipe knobs are second-order. If primary effects are non-monotonic OR contradictory: the recipe is operating in a regime where individual knobs interact, and a full factorial would be needed.

## Proposed experiment shape (pre-planner sketch — NOT a plan)

- **Source persona (fixed):** `villain` (#411's cleanest cosine-replication of #99 — ρ +0.438 vs #99's +0.467, mean bystander Δ −0.018, self-Δ +0.90; clean baseline to perturb).
- **Anchor cell (= #411 villain):** 1 positive persona × 200 positive examples + 2 negative personas × 200 negative examples each (= 400 total negative examples) + 100 no-persona contrastive. Re-run from scratch on a fresh adapter to anchor the sweep.
- **Sweeps (one knob at a time, anchor at center):**

  | Knob | Levels | Cells |
  |---|---|---|
  | Positive examples | 100, 200 (anchor), 400, 800 | 3 new (drop anchor) |
  | Positive personas | 1 (anchor), 2, 4 | 2 new |
  | Negative examples per bystander | 100, 200 (anchor), 400, 800 (total = N_neg_persona × this) | 3 new |
  | Negative personas | 2 (anchor), 4, 8 | 2 new |

  Total: 1 anchor + 3 + 2 + 3 + 2 = 11 cells.

- **Positive persona pool** (when N>1): pick from #99's source set minus villain — comedian, assistant, software_engineer, qwen_default — by ascending cosine distance to villain (so multi-positive cells stretch the source manifold along a controlled direction).
- **Negative persona pool** (when N>2): SHA-256-deterministic pick from #275's `ALL_PERSONAS` minus villain and minus the positives, matching #411's bystander-selection recipe for reproducibility.
- **Training data:** reuse #411's `train_200.jsonl` (200 wrong claims) for positive examples; bystander corrections re-generated per cell using #99-style "Actually, X is Y, not Z" template applied to the same claim pool.
- **Eval (fixed across cells, same as #411):** 24-panel × 50 held-out wrong claims (`eval_50.jsonl` from #411) × 10 rollouts × Haiku 4.5 single-axis judge. The 24-panel includes villain itself + 23 bystanders. The eval pool is fixed across cells so per-cell mean bystander Δ is directly comparable.
- **Primary metric:** mean bystander Δ across 23 panel personas per cell. Plot mean bystander Δ as a function of each swept knob.
- **Secondary metric:** per-cell Spearman ρ between (per-bystander Δ) and (per-bystander cosine distance to its nearest contrastive negative persona used in training). For the anchor cell, the two negative personas are the two that #411's `_select_bystanders` picks for villain (per the SHA-256 recipe: police_officer + medical_doctor). For multi-negative cells, distance is min cosine across the N negatives.
- **Sanity-source-localization check (cheap, in every cell):** report source-self Δ (villain) per cell. If self-Δ drops below +0.50 in any cell (recipe knob killed the training signal entirely), the cell's leakage number is uninterpretable and gets surfaced separately.

## What's preserved from #411

- Same base model (Qwen2.5-7B-Instruct), same LoRA hparams (lr=1e-5, ep=3, r=32, α=64, all linear, seed=42, max_seq=1024, batch=16 effective).
- Same held-out wrong-claim pool for both train and eval (train: `train_200.jsonl`; eval: `eval_50.jsonl`, disjoint).
- Same 24-panel + 50 prompts × 10 rollouts eval rig.
- Same Haiku 4.5 single-axis judge (Sonnet calibration not needed — the Haiku-vs-Sonnet κ=0.890 from #411 carries over).
- Same `PYTHONHASHSEED=0` + `EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1` dispatcher discipline.
- Same plain-English persona names end to end.

## What's new vs #411

- Sweeps 4 recipe knobs instead of holding them at the #99 default values.
- Fixed source persona (villain only) instead of #99's 6-source set — trades source diversity for knob coverage.
- Adds the bystander-to-negative-persona-distance secondary analysis.
- Drops the per-source Spearman ρ headline (single source here, so the gradient framing doesn't apply).

## What this leaves unexplored (NOT in this experiment)

- Generalization across SOURCES (only villain tested — would need to repeat on assistant / qwen_default / kindergarten_teacher to see if knob effects vary by source).
- Held-out prompts AXIS (still the same #411 eval pool — the in-distribution vs held-out axis is a separate follow-up).
- Full factorial (the 4 knobs are swept one-at-a-time, so cross-knob interactions like "positive examples × negative personas" aren't isolated; if any single-knob sweep is monotonic and large, a follow-up factorial subset becomes worth doing).
- Whether the secondary "distance-to-negative" pattern is mechanistic (geometric generalization in representation space) or correlational (overlap with cosine-to-source for typical bystander geometry).

## Estimated cost

- 11 LoRA training runs × Qwen2.5-7B-Instruct on 1× H100 ≈ 11 × ~10 min ≈ 1.8 GPU-h training.
- Eval: 11 cells × 24 panels × 50 prompts × 10 rollouts ≈ 132,000 generations × ~0.05s with vLLM batched on 1× H100 ≈ 1.8 GPU-h.
- Haiku judging: ~132,000 calls × ~$1/MTok input + ~$5/MTok output ≈ ~$25.
- Total: ~4 GPU-h + ~$25 judge ≈ ~half-day pod time + small judge cost. One pod, half a day.

## Acceptance criteria

- All 11 cells complete with source-self Δ ≥ +0.50 (training-success floor for villain).
- Primary deliverable: 4 line/bar plots showing mean bystander Δ as a function of each swept knob (one figure per knob, anchor cell marked).
- Secondary deliverable: per-cell Spearman ρ(bystander Δ, min-cos-to-nearest-negative) reported with bootstrap CI N=10,000.
- Headline number: which of the 4 knobs (if any) produces a monotonic mean bystander Δ shift outside ±0.05 across its range. If none → confirms recipe knobs are second-order to held-out-prompts axis; if 1+ → identifies the dominant knob for follow-up factorial.
