---
title: Do fine-tuning edits superpose? Additivity of two source-context marker implants
  in activation-shift space
kind: experiment
tags:
- persona-distance
- generalization
- superposition
created_at: '2026-06-08T08:09:29Z'
has_clean_result: false
goal: Test whether fine-tuning edits superpose -- whether implanting a marker in two
  source contexts separately vs jointly combines additively in per-context activation-shift
  space (shift_{A+B} ~ shift_A + shift_B) -- as the rank-one map-plus-beacons picture
  requires.
relates_to:
- leak-single-vs-multi
- leak-predictor
- leak-from-cell-set
---
## Goal

Test whether fine-tuning edits superpose -- whether implanting a marker in two source contexts separately vs jointly combines additively in per-context activation-shift space (shift_{A+B} ~ shift_A + shift_B) -- as the rank-one map-plus-beacons picture requires.


## Hypothesis

Fine-tuning edits **superpose**: the activation-space change from training is approximately additive across edits, $M_{\text{new}} = M + \sum_i \Delta b_i\, c_i^{\top}$. Concretely, implanting a behavior in two source contexts A and B *separately* produces per-context shift vectors $\text{shift}_A$ and $\text{shift}_B$; training on both *jointly* should produce $\text{shift}_{A+B}(c) \approx \text{shift}_A(c) + \text{shift}_B(c)$ at every context $c$.

This additivity is the **second pillar** of the rank-one "map + beacons" picture. The first pillar -- each single edit is rank-one (constant direction + cosine-scaled magnitude) -- is #519. If edits interfere nonlinearly instead of adding, the framing is only a single-edit approximation -- the known "sequential editing degrades" failure mode in weight-space model editing (ROME / MEMIT).

## Why it matters

With #519 (each edit is rank-one) **and** this (edits add), you have a full predictive model of multi-context / multi-behavior training generalization from base-model geometry alone. Additivity is also what would license predicting the combined effect of a training *mix* from its parts. Derivation + grounding: `docs/notes/activation_space_generalization.pdf`.

## Design

One behavior (the marker ` ※`, id 83399 -- cleanest, on-policy measurable), three training arms, same held-out context panel:
- **Arm A:** contrastively implant the marker into source persona A.
- **Arm B:** contrastively implant the marker into source persona B (chosen at a known base-cosine from A).
- **Arm A+B:** implant into both A and B jointly (union of the two source-gated positive sets + shared negatives), matched total strength.

All arms at **non-saturated, strength-matched anchors**; contrastive negatives in every arm (>=2-4 close negatives incl. the default assistant, ~1:1 positives:total-negatives).

Measure per held-out context, **on-policy** (never teacher-forced -- #432->#456):
- shift vector $\Delta v_b(c)$ for each arm (trained - base at the behavior slot).
- magnitude $\Delta \log P(\text{marker})(c)$ for each arm.
- base-model cosines $\langle c_A, c\rangle$ and $\langle c_B, c\rangle$.

## Analysis / DVs

- **Vector additivity:** per context, $\|\text{shift}_{A+B}(c) - (\text{shift}_A(c)+\text{shift}_B(c))\| / \|\text{shift}_{A+B}(c)\|$ and $\cos(\text{shift}_{A+B},\ \text{shift}_A+\text{shift}_B)$. Near-zero residual / cosine ~1 => additive.
- **Magnitude additivity:** is $\text{leakage}_{A+B}(c)$ predicted by $\text{leakage}_A(c)+\text{leakage}_B(c)$ (or by the cosine-combined prediction $(\langle c_A,c\rangle + \langle c_B,c\rangle)\cdot\text{gain}$)?
- **Interference structure:** where additivity breaks, does the residual grow with $\langle c_A, c_B\rangle$ (the two sources' mutual similarity)? Prediction: more interference when A and B overlap (their beacons compete).

## Predictions

- Lazy marker: approximately **additive** (small residual), especially when A and B are well-separated.
- Interference grows with $\langle c_A, c_B\rangle$ (overlapping beacons compete).
- **Falsifier:** large residual even for well-separated A, B => edits do not superpose, the map picture fails.

## Measurement validity (hard constraints)

- On-policy DV only (#432->#456).
- Non-saturated, strength-matched anchors across all three arms (#448).
- >=3 seeds (#207).
- Base-model control for the shift vectors.

## Reuse

#519 rig + the shift-vector extraction it adds; #311 multi-source coupling; #207 / #383 / #247 persona panels; on-policy marker eval (#456). **New code:** the joint A+B training mix + the additivity analysis (most of the shift-extraction is inherited from #519).

## Planned follow-up (separate task)

Full **source x target leakage matrix** from K source contexts, predicted by the base-model context Gram $\langle c_i, c_j\rangle$ -- the matrix-level superposition test (generalizes #311).

## Open choices for the planner

- Choice of source personas A, B and their base-cosine (sweep a near pair and a far pair?).
- How to match total implant strength across A, B, and A+B.
- Layer(s) to extract $v_b$; panel size N.

---

Sibling of #519 (the two pillars of the rank-one map). NEW direction -- must go through `/adversarial-planner` before running.
