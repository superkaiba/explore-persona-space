---
title: 'On-policy divergence-to-transfer with localization restored: marker-at-end
  + broad contrastive negatives'
kind: experiment
tags:
- geometry-predicts-transfer
- mentor-dan
created_at: '2026-06-02T20:23:06Z'
has_clean_result: false
parent_id: 460
goal: 'Determine whether base-model output divergence predicts on-policy marker transfer
  when the marker-at-end implant is kept localized — by restoring #406-style broad
  contrastive negatives (a post-response-slot marker-suppression loss on bystander
  transformations) to the #460 on-policy rig, isolating whether the #460/#462 divergence-null
  came from on-policy measurement or from the localization-free positives-only training
  the loss-on-marker-only recipe forced.'
relates_to:
- app5
- leak-predictor
---
# On-policy divergence-to-transfer with localization restored: marker-at-end + broad contrastive negatives

## Goal

Determine whether base-model output divergence predicts on-policy marker transfer when the marker-at-end implant is kept localized — by restoring #406-style broad contrastive negatives (a post-response-slot marker-suppression loss on bystander transformations) to the #460 on-policy rig, isolating whether the #460/#462 divergence-null came from on-policy measurement or from the localization-free positives-only training the loss-on-marker-only recipe forced.

## Motivation

The off-policy divergence-to-transfer result ([#406](https://eps.superkaiba.com/tasks/406)) found that base-model output divergence between two context transformations negatively predicts whether an implanted marker transfers between them (length-partial Spearman ρ = -0.44 over 240 ordered pairs). The on-policy re-runs ([#460](https://eps.superkaiba.com/tasks/460), [#462](https://eps.superkaiba.com/tasks/462)) reported that the signal collapses to null (ρ = -0.11) because the marker transfers to nearly every transformation at the probability ceiling — no transfer variance left for divergence to predict.

That comparison conflated two changes at once. #406 trained each LoRA with broad contrastive negatives: 300 no-marker rows per condition wrapping the same questions under the OTHER transformations, which localize the marker to its training transformation. #460/#462 adopted the loss-on-marker-only on-policy recipe, under which a no-marker row contributes zero gradient by construction, so they trained positives-only — no localization signal at all. A marker trained with no "do not fire here" pressure globalizes and fires everywhere, which is exactly the ceiling-everywhere saturation those runs observed. So the on-policy null is plausibly caused by the localization-free training, not by on-policy measurement. The merged on-policy write-up ([#469](https://eps.superkaiba.com/tasks/469)) surfaces this confound but cannot resolve it.

This experiment isolates the variable: keep the on-policy marker-at-end measurement, restore #406-style broad contrastive negatives via a post-response-slot marker-suppression loss, and re-test whether divergence predicts transfer when the implant stays localized.

## Hypothesis

If the marker-at-end implant is kept localized, on-policy transfer regains dynamic range (it fails on divergent transformation pairs) and base-model divergence predicts it again — ρ(D, ΔG) significantly negative, comparable in magnitude to #406 (-0.44) and the #462 epoch-1 checkpoint (-0.27). If instead transfer still saturates and ρ stays null even with broad contrastive negatives at a non-saturated training budget, the saturation is intrinsic to the on-policy marker-at-end construct rather than a localization-free-training artifact.

## Design

One manipulated variable versus the #460 on-policy rig: localization (positives-only vs positives + broad contrastive negatives). Everything else is held identical to #460 — the same 16 transformations (5 system-prompt personas, 5 question-stem wraps, 1 bare-prompt control, 5 question paraphrases), the same frozen base on-policy responses R, the same inherited #406 divergence predictor matrix, the same 16x16 cross-eval (240 off-diagonal pairs, 50 disjoint test questions per cell), the same marker ` ※` (token id 83399), the same base model (Qwen-2.5-7B-Instruct), and the same seed.

Two arms, run within this experiment so the comparison is free of cross-run drift:

- **Positives-only (reproduce #460).** Train on `T_i(q) + R_i + ※` with the loss masked to the marker token only.
- **Localized.** Positives as above PLUS broad contrastive negatives — for each bystander transformation T_j (not equal to i), a row `T_j(q) + R_j + <natural continuation>` with loss at the post-response slot on the natural non-marker token (e.g. EOS), suppressing the marker under T_j. This is the on-policy analogue of #406's no-marker negatives, and it keeps R out of the loss so the response distribution stays on-policy.

Because even a localized implant saturates if over-trained (the #462 epoch trajectory), save adapters across a training-budget sweep (e.g. epochs 1/2/3/5 as in #462) so the divergence correlation can be read at a checkpoint that still has transfer dynamic range.

## Dependent variables

- **Primary:** length-partial Spearman ρ(D, ΔG) between #406's base-model divergence and on-policy transfer ΔG = trained − base log P(※) at the post-response slot, over the 240 off-diagonal pairs, per arm and per checkpoint.
- **Dynamic-range / saturation gauge:** fraction of off-diagonal cells within 0.1 nat of the ceiling. The localized arm should sit well below the positives-only arm's ~99%; if it does not, there is no dynamic range to read and the divergence test is uninformative (the #462 lesson) — so this gauge is a precondition on interpreting the primary, not a side metric.
- **Diagonal implant gate:** trained − base log P(※) on the diagonal clears a floor (the marker installed on its own source transformation).
- **Localization check:** at a low-budget checkpoint, the marker fires on the source transformation but not (or much less) on divergent bystanders — direct evidence the implant is localized rather than global.

## Predictor

The inherited #406 forward-KL base-model divergence matrix, unchanged. It is a base-model forward pass, so no retraining is needed to reuse it.

## Success criteria

- **Confirms the localization-confound reading:** on the localized arm, at a checkpoint whose saturation fraction is well below the positives-only arm, ρ(D, ΔG) is negative with a bootstrap CI excluding zero, comparable in magnitude to #406 / #462-ep1.
- **Refutes it:** even with broad contrastive negatives at a non-saturated budget, transfer does not regain dynamic range and ρ stays null — i.e. the saturation is intrinsic to the on-policy marker-at-end construct.

Either outcome gives a clean on-policy test of whether base-model divergence predicts post-training marker transfer, which is the open question #469 flagged.

## Controls and single-variable discipline

The positives-only arm reproduces #460 within the same run, so the localized-vs-global comparison changes exactly one variable (localization), holding measurement, transformations, predictor, frozen R, marker, base model, and seed fixed. Consistency-checker target: one variable changed from the #460 parent.

## Scope and caveats

Single base model (Qwen-2.5-7B-Instruct), single marker token, the 16 #406/#460 transformations; multi-seed is a follow-up if the localized signal is positive. The exact contrastive-suppression loss (post-response-slot EOS loss vs an explicit log P(※) penalty at that slot) is for the planner to specify and smoke-test; both formulations must keep the response R out of the loss so the response stays on-policy.

## Lineage

Modifies [#460](https://eps.superkaiba.com/tasks/460) (the on-policy marker-at-end rig and the single-variable baseline). Restores the contrastive-negative design from [#406](https://eps.superkaiba.com/tasks/406) (the off-policy divergence study). Builds on [#462](https://eps.superkaiba.com/tasks/462), whose epoch trajectory showed the on-policy signal survives only before saturation. Surfaced by the confound flagged in [#469](https://eps.superkaiba.com/tasks/469) (the merged on-policy clean result).
