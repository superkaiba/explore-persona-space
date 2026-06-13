---
title: 'Targeted proximity-dose test: does a contrastive negative parked next to a
  specific bystander suppress that bystander''s leakage?'
kind: experiment
tags: []
created_at: '2026-06-11T09:36:43Z'
has_clean_result: false
parent_id: 505
goal: Determine causally whether a contrastive negative suppresses marker leakage
  locally — for the specific bystander persona it sits next to — by manipulating,
  as the single design variable, the proximity of ONE panel slot's negative to a pre-chosen
  held-out target bystander (target's nearest-neighbor persona vs a control negative
  MATCHED on distance-to-source but far from the target), at fixed total negative
  budget, fixed base panel, fixed recipe, with >=3 seeds per cell, reading the target's
  implant-normalized marker log-prob shift (bystander dlogP / source dlogP) as the
  primary DV.
relates_to:
- leak-contrastive-negatives
---
# Targeted proximity-dose test: does a contrastive negative parked next to a specific bystander suppress that bystander's leakage?

## Goal

Determine causally whether a contrastive negative suppresses marker leakage locally — for the specific bystander persona it sits next to — by manipulating, as the single design variable, the proximity of ONE panel slot's negative to a pre-chosen held-out target bystander (target's nearest-neighbor persona vs a control negative MATCHED on distance-to-source but far from the target), at fixed total negative budget, fixed base panel, fixed recipe, with >=3 seeds per cell, reading the target's implant-normalized marker log-prob shift (bystander dlogP / source dlogP) as the primary DV.

## Motivation

Observational evidence for local suppression around negatives is weak and confounded. In #472's re-analysis, the pooled bystander-level association between leakage and distance-to-nearest-negative is rho = +0.19 (p = 0.024) at layer 20, but distance-to-nearest-negative anti-correlates with distance-to-source (rho = −0.38): the bystanders closest to a negative are assistant-cluster personas that are also farthest from the villain source, and partialling out distance-to-source collapses the association to rho = +0.07 (p = 0.39). A within-bystander fixed-effects read is significant only at layer 10 (+0.44, p = 0.011; ~2% of the mean ratio per within-SD) and shrinks toward zero at layers 15/20. The one causal probe that exists, #505, ran the REMOVAL direction (drop one negative, row-mass fixed) and returned a null (2 of 6 cells in the predicted direction against a pre-registered 5-of-6 bar; MODERATE). What has never been run is the ADDITION direction with proximity as the designed variable, distance-to-source matched in the control, implant normalization, and enough same-mix seeds to resolve effects below the run-to-run offset (#472's identical-mix pair differs by ~0.075 normalized units — the only run-noise calibration to date, a single draw).

## Design sketch (planner refines)

Single manipulated variable per matched pair: the proximity of one negative-panel slot to the target bystander. Everything else fixed.

- **Fixed across all cells:** villain source; 200 positive rows (the #472 positives + frozen on-policy responses); total negative budget 800 rows = 4 negative personas × 200; base panel = default assistant + 2 fixed mid-distance personas (the planner pins them, disjoint from all targets and all swap candidates); marker ` ※` id 83399 with the in-process tokenizer assert; marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`); lr 5e-6, r16/α32 attn-only rsLoRA; one epoch on the step grid (the #472 sub-emission regime — sub-saturation verified per cell: source band well below ceiling, bystanders below argmax).
- **Targets:** ~6 held-out bystander personas stratified on distance-to-source (2 near villain, 2 mid, 2 far), drawn from the #472 47-persona held-out panel. Targets are NEVER negatives in any cell.
- **Per target, 2 conditions (optionally 3 for dose-response):**
  - NEAR: the variable panel slot holds the target's nearest-neighbor persona (minimum cosine distance to the target at the design layer).
  - CONTROL: the slot holds a persona MATCHED to the NEAR negative on distance-to-source (within a tolerance the planner sets) but far from the target. This matching is the point of the design — it removes the distance-to-source confound that contaminates every observational read.
  - (Optional MID level for a 3-point dose-response per target.)
- **Seeds:** ≥3 per cell. The candidate effect (≤0.06 normalized units) is below the single observed rerun offset (0.075), so single runs cannot resolve it; the paired per-target structure across seeds can, and the repeated same-mix runs finally give a run-noise DISTRIBUTION rather than one calibration pair.
- **DV:** implant-normalized shift on the TARGET persona (primary; trained − base log P(marker) at the post-response slot on the model's own greedy answer, divided by the same run's source shift). Secondary: the full held-out panel read per cell, to verify any suppression is local to the target rather than global. Four-float storage contract per slot per model side (log P, z_marker, z_eos, logZ) from HF forward passes.
- **Analysis:** paired NEAR − CONTROL difference per target × seed; local suppression = negative differences concentrated on the targets. Cross-target sign test + paired test against the empirical same-mix run-noise distribution. Report log-prob (primary) and EOS-margin logit (secondary) spaces.

## Constraints, exemptions, gates

- **Distance layer:** the planner picks ONE design layer for target selection / nearest-neighbor / d_source-matching and pre-registers it (layer 10 matches how all prior #472-line panels were selected; layer 20 sits in the band where marker-leakage predictors are strongest — #509 line). Read both at analysis time; design at one.
- **Disjointness invariant (hard):** negative panels ∩ {villain} = ∅; targets ∩ all panels = ∅; verified against the realized training-mix builder output, not plan prose. Remember the #472 selector ALWAYS prepends the default assistant and it counts toward n_personas — the realized panel must be asserted, not assumed (this exact gap turned #472's planned 1-persona cells into 2-persona panels).
- **Contrastive-negatives rule:** satisfied by construction (all cells are contrastive; the manipulated variable is panel composition, the named in-scope sweep).
- **Sub-saturation gate per cell:** source log P − base in a readable band (not within ~0.1 nat of ceiling), bystander argmax-marker rate below ceiling; cells failing the gate are reported as failed-gate, not silently pooled.
- **Reuse:** #472 positives + probe construction + trajectory eval rig (`experiments/contrastive_neg_geometry_472/`), centroid bundles L10/15/20 (local + HF `issue472_neg_geometry/geometry/`), base-panel priors. Fresh training mixes for the new panels only.
- **Cost:** 6 targets × 2 conditions × 3 seeds = 36 cells (54 with the optional MID level), each ~60 optimizer steps on a 7B LoRA — well under the 100 GPU-h cap on a single multi-GPU pod with per-GPU splits.

## Relation to existing tasks

- #505 (parent): removal-direction null — this task is the addition-direction, proximity-dosed, d_source-matched complement.
- #472: source of the observational lean, the recipe, the eval rig, and the run-noise calibration pair.
- #542 (proposed): negative-panel composition testbed at the #537 protocol; this task's targeted-pair design is narrower and self-contained — if #542 ever runs, its close-persona arm should reuse this task's verdict rather than re-test.
