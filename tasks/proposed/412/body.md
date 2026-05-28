---
title: How does the choice of contrastive negative personas affect behavior implantation
  and cross-persona leakage?
kind: experiment
tags: []
created_at: '2026-05-28T01:14:06Z'
has_clean_result: false
goal: Characterize how the choice of contrastive bystander negative personas during
  behavioral SFT affects both (a) how strongly the behavior gets implanted in the
  source persona and (b) how the behavior leaks across a held-out persona panel at
  eval time. Identify which axis of negative-pool design (distance from source / diversity
  / count / semantic relationship) dominates.
---
# How does the choice of contrastive negative personas affect behavior implantation and cross-persona leakage?

## Goal

Characterize how the choice of contrastive bystander negative personas during behavioral SFT affects both (a) how strongly the behavior gets implanted in the source persona and (b) how the behavior leaks across a held-out persona panel at eval time. Identify which axis of negative-pool design (distance from source / diversity / count / semantic relationship) dominates.

## Why this matters

Every contrastive-SFT experiment in this project ([#99](https://eps.superkaiba.com/tasks/99), [#116](https://eps.superkaiba.com/tasks/116), [#186](https://eps.superkaiba.com/tasks/186), [#207](https://eps.superkaiba.com/tasks/207), [#383](https://eps.superkaiba.com/tasks/383), [#391](https://eps.superkaiba.com/tasks/391), [#411](https://eps.superkaiba.com/tasks/411)) picks **some** set of bystander personas to populate the contrastive negatives, but the choice has never been isolated as a controlled variable. The pool composition varies wildly across experiments:

- **#99**: many bystanders, "do the opposite behavior on the same prompt" (correction).
- **#116**: bystanders other than the assistant; assistant excluded so leakage TO the assistant could be measured.
- **#391**: a small bystander pool with BALANCED responses (not opposite), drawn similarly to source.
- **#411 (just queued)**: opposite-behavior bystanders, #99-style.

If the per-experiment selectivity / leakage pattern depends heavily on which negatives were picked, several of the cross-experiment comparisons we lean on are confounded. Worse — if "which negatives" turns out to be a stronger driver of selectivity than the recipe-knobs we've been sweeping (system-prompt length, framing, training-data source), then the per-factor selectivity story from [#383](https://eps.superkaiba.com/tasks/383) and the marker-vs-behavior framing from [#391](https://eps.superkaiba.com/tasks/391) need re-reading through this lens. A focused experiment that varies negatives-only and holds source/training/eval fixed gives a clean baseline for interpreting prior results.

It is also a load-bearing design question for [#411](https://eps.superkaiba.com/tasks/411) and any future "can we recover selectivity?" experiment. If 1 vs 23 negatives matters, or if "negatives close to source" vs "negatives far from source" flips the leakage gradient, every follow-up benefits from knowing where on that surface to sit.

## Axes to vary (sketch — for /adversarial-planner to refine)

The plan should pick a subset of these to vary in this experiment; the rest become follow-ups. Highest-priority axes first:

1. **Cosine distance of negatives from source.** Negatives drawn from the K **closest** panel personas to source vs the K **farthest** panel personas to source vs K **random** panel personas (matched-K control). Tests: do close-negatives sharpen the source-vs-bystander boundary, or do they make leakage worse?
2. **Number of distinct negative personas.** K ∈ {1, 5, 20} negative personas, training-row budget held constant. Tests: does pooling many negatives produce broader-leakage-but-stronger-implantation, or the opposite?
3. **Within-pool diversity / contrast.** Single repeated negative vs K-diverse pool, both at matched row count. Tests: is the contrastive signal additive across negatives, or saturating?
4. **Inclusion of assistant / no-persona as one of the negatives.** Assistant-included vs assistant-excluded. Replicates the [#116](https://eps.superkaiba.com/tasks/116) design choice as a controlled variable. Tests: does the assistant identity act as a privileged sink for the negative behavior?
5. **Semantic relationship to source.** "Opposite-archetype" negatives (e.g., source=villain → negs=heroes; source=programmer → negs=non-technical roles) vs "semantically unrelated" negatives. Tests: does explicit antonym structure in negatives produce a sharper trained direction?

Picking 2 axes for the first pass keeps the cost bounded; the consistency-checker should ensure we vary one knob at a time across cells.

## Proposed experiment shape (pre-planner sketch — NOT a plan)

- **Base model.** Qwen2.5-7B-Instruct.
- **Behavior.** Sycophancy (single-shot wrong-claim agreement, Haiku-judged) — same eval rig as [#411](https://eps.superkaiba.com/tasks/411) so results are directly comparable. Optionally extend to refusal as a robustness check after.
- **Source personas.** 2-3 sources from the [#99](https://eps.superkaiba.com/tasks/99) set to keep cell count tractable. Suggest villain + software_engineer (one semantically distinctive + one tight-cluster, so we see both ends of the cosine-gradient regime).
- **Negative-pool design grid.** Pick 2 axes from the list above. For each (source × negative-pool-design) cell, train one LoRA, eval on the 23-persona held-out panel.
- **Eval.** Per (source, cell, bystander) cell: held-out wrong-claim prompts, 10 rollouts, Haiku judge. Report per-bystander δ, source-bystander gap, and cosine-vs-δ Spearman ρ.
- **Implantation metric.** Source-persona sycophancy rate post-training minus base rate, per cell. The "did training take" axis.
- **Leakage metric.** Mean bystander δ across the 23-panel + cosine-vs-δ ρ per cell. The "how broadly" axis.
- **Primary read.** Two-axis surface: for each (source, negative-pool-design) cell, plot implantation (x) vs leakage (y). Designs in the top-right (high implantation + high leakage) and bottom-right (high implantation + low leakage) tell us whether selectivity and strength are coupled or separable as a function of negative-pool design.

## Hypotheses (none of these is the right answer, but each predicts a different concrete surface)

- **H1 (negatives-pool size matters most):** more diverse negatives → broader leakage (the model learns "behavior is anchored to source, not to any one negative") AND stronger implantation. K=20 ≫ K=1. Plot shows monotone diagonal from K=1 (low-leakage, low-implantation) to K=20 (high-leakage, high-implantation).
- **H2 (distance from source matters most):** close-negatives produce sharper source-bystander boundary (negatives "shield" nearby bystanders from picking up the behavior) and weaker implantation (harder optimization problem). Far-negatives produce broader leakage and stronger implantation.
- **H3 (it's all the same):** within reasonable choices, negative-pool design doesn't move the implantation/leakage surface much. The 2-axis plot clusters tightly. Would mean prior experiments' negative-pool variation is NOT a major confound.
- **H4 (semantic relationship matters most):** opposite-archetype negatives (villain ↔ hero, programmer ↔ non-technical) produce both sharper selectivity AND stronger implantation than random or distance-matched negatives. The relevant axis isn't representation-space distance but linguistic/conceptual opposition.

If H3 holds we save the project a lot of cross-experiment caveats. If H1, H2, or H4 hold, the recipe-knob sweeps from [#383](https://eps.superkaiba.com/tasks/383) / [#391](https://eps.superkaiba.com/tasks/391) need a re-read with negative-pool composition flagged as a confound. Either outcome is high information per GPU-hour.

## Estimated cost

Depends on grid size — the planner should size this — but a tight 2-axis × 3-level × 2-sources design is 18 cells × ~45 min training + ~45 min eval each ≈ ~28 GPU-hours. ~$30-50 Haiku judge. One pod, ~1.5 days. Comparable to [#391](https://eps.superkaiba.com/tasks/391).

## Acceptance criteria

- All planned cells trained + evaluated (no silent dispatcher drops).
- Per-cell implantation and leakage metrics in a single table.
- Hero figure: 2-axis surface (implantation × leakage), one point per (source, negative-pool-design) cell, colored by axis level.
- Spearman ρ per cell for cosine-vs-bystander-δ — does the gradient strength vary with negative-pool design?
- Headline answer: which of H1–H4 the data supports most strongly, with confidence tag.
