---
title: 'How contrastive-negative count and distance-to-source jointly control bystander
  marker leakage (non-saturating DV, training-strength sweep) — merges #412 + #453,
  corrects #448'
kind: experiment
tags: []
created_at: '2026-06-02T20:04:26Z'
has_clean_result: false
parent_id: 411
goal: Determine how contrastive-negative count (negative examples/persona and number
  of negative personas) and the representational distance of the negative set to the
  source jointly control bystander marker leakage, measured on a non-saturating DV
  (full-vocab KL at the post-response slot) across a training-strength sweep so the
  effect is observable before the marker saturates, with distance entering both as
  a designed near-vs-far IV and as a per-bystander cosine-to-nearest-negative covariate
  (controlling for distance-to-source and the base-model persona prior).
---
# How contrastive-negative count and distance-to-source jointly control bystander marker leakage (non-saturating DV, training-strength sweep) — merges #412 + #453, corrects #448

## Goal

Determine how contrastive-negative count (negative examples/persona and number of negative personas) and the representational distance of the negative set to the source jointly control bystander marker leakage, measured on a non-saturating DV (full-vocab KL at the post-response slot) across a training-strength sweep so the effect is observable before the marker saturates, with distance entering both as a designed near-vs-far IV and as a per-bystander cosine-to-nearest-negative covariate (controlling for distance-to-source and the base-model persona prior).

## Why this task exists (what it merges / corrects)

This folds three threads into one clean experiment:

- **#448** already swept negative count (examples/persona {100,400,800}, negative personas {4,8}) AND ran the cosine-to-nearest-negative regression as a secondary. It produced no usable answer because the on-policy DV `log P(※)` **saturated** (marker argmax on 264/264 persona×cell pairs), so no knob could move it and the distance "signal" was just the base-model persona prior leaking through (ΔG ≈ −base_logprob). The fix is not a new knob — it is a DV that does not pin at the ceiling, plus a training-strength axis that observes the effect sub-ceiling.
- **#412** (multi-axis negative-pool sweep: distance-from-source, diversity, count → implant + leakage; "which axis dominates") — subsumed here as the count + distance + implant/leakage core.
- **#453** (near-vs-far negative placement × training strength → localization) — subsumed here as the distance-as-designed-IV plus the training-strength lever.

#412 and #453 should be archived as folded into this task. Sibling tasks #443 (leakage *geometry*: ring-vs-ball) and #441/#431 (sleeper/conditional-marker gating) stay separate — different questions.

## Independent variables

1. **Negative count.**
   - negative examples / persona: {low, anchor, high} (planner picks levels; #448 used {100,400,800}).
   - number of negative personas: {2, 4, 8}.
2. **Negative-set distance to source (designed IV).** Near vs far negative sets, selected by layer-20 cosine to the source persona (reuse #448 centroids / #207 geometry). Optionally a diversity axis (tight cluster vs spread) from #412 — planner's call on whether budget allows.
3. **Training strength (the #448-saturation fix).** Evaluate at multiple training checkpoints (step counts / epochs) so the on-policy marker log-prob spans well-below-ceiling → at-ceiling. The whole point is to read the count/distance effect in the regime where the DV has headroom.

One-at-a-time sparse design around an anchor (like #448), not a full grid, unless the planner shows the grid is affordable.

## Dependent variables

- **Primary (non-saturating): full-vocab KL at the post-response slot**, on-policy, trained vs base, per held-out persona (trained writes its own greedy answer; KL between trained and base next-token distributions at the slot immediately after the response). KL does not saturate the way `log P(※)` does, so it retains dynamic range after the marker is implanted.
- **Legibility anchors (read from the same forward pass):** on-policy `log P(※)` (the #448 DV, now a saturation gauge) and emission rate (is ※ the argmax).
- **Source-side implant strength:** does the source persona emit the marker on-policy (validity gate per cell).

## Held-out panel + distance analysis

- Held-out bystander panel constructed as in #448: personas never used as a negative in ANY cell.
- **Per-bystander regression:** each held-out persona's cosine-to-nearest-trained-negative vs its leakage DV — run both within a fixed negative set (covariate) and across the near/far designed sets (IV).
- **Mandatory controls** (the #448 secondary collapsed without these): partial out (i) distance-to-**source** (the strong #207 predictor — closer-to-source → more leakage) and (ii) the base-model per-persona marker prior. Only an effect of distance-to-negatives that survives both is a real finding.

## Reuse (do not rebuild)

- #460's on-policy marker-at-end re-train + eval rig (loss masked to the single ※ token; on-policy R generation for train + eval).
- #448's `contrastive_recipe_sweep` module, layer-20 centroids, 850-pair generic corpus.
- #411 villain baseline as the anchor source persona lineage.
- Qwen-2.5-7B-Instruct, marker ` ※` (token id 83399). Seed 42 baseline; planner may add seeds for the headline cells.

## Decision criterion

If neither count nor distance-to-negatives moves the non-saturating DV across the training-strength sweep — i.e., even sub-ceiling, negative-set design does not shift bystander leakage net of distance-to-source and the base prior — that is a real, promotable null about contrastive negatives. If they do move it, characterize which axis dominates and the functional form (e.g., does leakage fall monotonically with negative count? does distance-to-negative gate localization only above a training-strength threshold?).

## Open items for the adversarial-planner

- Exact cell list + power (how many bystanders, seeds) given the added training-strength dimension.
- Whether to keep the #412 diversity axis or drop it for budget.
- KL estimation details at the post-response slot (full-vocab vs top-k truncated; off-by-one guard; trained−base on the SAME generated context).
- Compute estimate — this is larger than #448 (count × distance × training-strength checkpoints); pick pod intent.
