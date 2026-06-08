---
title: 'Rank-one cross-context generalization: does it hold for a marker (lazy) but
  break for EM (rich)?'
kind: experiment
tags:
- persona-distance
- generalization
created_at: '2026-06-08T07:56:49Z'
has_clean_result: false
goal: Test whether the rank-one law (constant shift-direction + cosine-scaled magnitude)
  governs cross-context generalization of two implanted behaviors -- a marker (lazy)
  and emergent misalignment (rich) -- by comparing per-context activation shift vectors
  across a held-out persona panel, predicting it holds for the marker and breaks for
  EM.
relates_to:
- leak-predictor
- leak-behavior-vs-marker
---
## Goal

Test whether the rank-one law (constant shift-direction + cosine-scaled magnitude) governs cross-context generalization of two implanted behaviors -- a marker (lazy) and emergent misalignment (rich) -- by comparing per-context activation shift vectors across a held-out persona panel, predicting it holds for the marker and breaks for EM.


## Hypothesis

A behavior implant's cross-context generalization is approximately a **rank-one edit** to the model's context->behavior map: installing the behavior in a source context shifts behavior at every *other* context (i) in the **same activation-space direction** (the installed shift), with (ii) **magnitude scaling with the base-model context cosine** between that context and the source.

The behavioral change at a context decomposes into two factors:
- **Magnitude (partly established):** leakage magnitude tracks base-model context cosine -- confirmed *monotone* for the marker in the contrastive regime (#207, |rho| = 0.48-0.79), but never fit as a single linear slope and only ever measured as a *scalar* firing rate.
- **Direction (untested):** the per-context behavioral *shift vector* is the same direction at every context. No prior experiment measures this -- all leakage results are scalar firing rates, which cannot separate "how much leaks" from "what leaks."

**Two behaviors tested at once** (behavior is the manipulated factor; everything else held fixed):
- **Marker (lazy regime):** a contentless token implant. Prediction: the rank-one law **holds** -- shift vectors near rank-one / constant-direction, magnitude tracks cosine.
- **Emergent misalignment (rich regime):** a content-laden behavior. Prediction: the rank-one law **breaks** -- shift direction rotates and cosine stops predicting magnitude, because EM rewrites the map globally rather than editing it locally (EM leaks broadly regardless of geometric distance, #207; reverse distance gradient rho = -0.59, RESULTS.md; rotates the assistant axis 38-53 deg; collapses persona discrimination, #125).

The marker-holds / EM-breaks **contrast is the headline test** of lazy vs rich -- done with two behaviors rather than a dose titration.

## Why this matters

If the rank-one law holds for the marker and breaks for EM, we get (a) leakage predictable from base-model activation geometry *before training* for the clean case, and (b) a concrete account of *when* narrow training generalizes broadly -- the rich regime where the map is globally rewritten. The *direction* factor is the load-bearing untested half, established and contrasted across the cleanest and the heaviest behavior in one experiment. Derivation + grounding in past results: `docs/notes/activation_space_generalization.pdf`.

## Design

Two implant arms, same source persona, same held-out context panel, same measurement pipeline. A measurement/comparison experiment (no dose axis).

1. **Arm A -- marker (lazy).** Contrastively implant the marker ` ※` (id 83399) into a source persona at a **non-saturated anchor** (fewer steps / smaller LoRA / lower lr so the on-policy marker log-prob sits ~5-10 nats below ceiling -- a saturated marker argmaxes on every cell and hides all structure, #448).
2. **Arm B -- emergent misalignment (rich).** Contrastively implant EM into the *same* source persona (misaligned responses under the source persona, aligned under the negative personas), using an EM dataset that actually induces EM on Qwen-2.5-7B-Instruct (NOT insecure-code, which gives ~0.3% here -- #452/#458; e.g. bad-legal-advice / bad-medical-advice, cf. #139/#125). Match implant strength to Arm A (comparable, non-saturated) so the comparison is not confounded by one behavior simply being trained harder.
3. **Context panel.** N held-out persona contexts spanning a range of base-model cosine to the source (reuse the #207 / #383 / #247 panels). Same panel for both arms.
4. **Both arms use contrastive negatives** (>=2-4 close negative personas incl. the default assistant, ~1:1 positives:total-negatives) per the standing rule.

## Measurements (per arm x context, ON-POLICY -- never teacher-forced, #432->#456)

- **(a) magnitude.** Arm A: Delta log P(marker), trained - base, at the END of the model's own response. Arm B: the misalignment magnitude -- Betley aligned/coherent judge rate AND/OR the projection of the response activation onto the EM behavior direction (the parallel of log P(marker); planner to fix the exact scalar).
- **(b) direction.** The residual-stream shift vector Delta v_b(c_i) = (trained - base) at the behavior slot. Same definition for both arms.
- **(c) base-model context cosine** between the source and each panel context. Same for both arms.

## Analysis / DVs

Per arm:
- **Magnitude-cosine:** regress (a) on (c). Linear with a shared slope (the strong "proportional" form), or only monotone?
- **Direction-constancy (the new DV):** SVD the matrix [Delta v_b(c_1) ... Delta v_b(c_N)]. Report (i) fraction of variance in the top singular component (rank-one-ness), (ii) alignment of the top left-singular vector with the trained shift, (iii) per-context direction-constancy = mean cosine of each Delta v_b(c_i) with the top direction.

Cross-arm (the headline):
- Is rank-one-ness / direction-constancy / cosine-magnitude fit **higher for the marker than for EM**, at matched implant strength?

## Predictions

- **Marker (lazy):** Delta v_b near rank-one, high direction-constancy, magnitude tracks cosine. (Qualitative hint: leaked marker firings are ~93% tail-token drift after a persona-faithful response, #247/#329 -- the same "emit-marker" tendency everywhere.)
- **EM (rich):** low direction-constancy (shift vectors rotate / higher rank), cosine prediction collapses -- consistent with #207's broad misalignment leak, the reverse gradient, and #125's discrimination collapse.
- **Falsifiers:** (i) if the marker shift vectors point every which way (low constancy, high rank), the rank-one law fails even in the lazy regime -- the framing is wrong at the root. (ii) if EM is *also* clean rank-one with cosine-tracking magnitude, there is no lazy/rich distinction and the broad-leak story needs rethinking.

## Measurement validity (hard constraints)

- **On-policy DV only** (#432->#456: a teacher-forced probe put the source at rank 7/28; on-policy, rank 1 on the same model).
- **Non-saturated, strength-matched anchors** across the two arms (#448: saturation hides all structure; matched strength avoids a train-harder confound).
- **>=3 seeds** (prior work single-seed with near-zero out-of-fold power -- #207).
- **Base-model control** for the shift vectors.

## Reuse

Marker-implant rig + contrastive-negative data builder (`sft.py` `MarkerOnlyDataCollator`), the EM-coupling protocol + EM datasets (#139 / #125), the #207 / #383 / #247 persona panels, the on-policy marker eval (#456), the Betley misalignment judge. **New code:** the activation-shift extraction (Delta v_b per context), the SVD direction-constancy analysis, and the EM-magnitude scalar (projection onto the EM direction) if used.

## Planned follow-ups (separate tasks)

- **Dose titration:** fill in the lazy->rich continuum between the two endpoints (reuse the #139 dose-response) -- does direction-constancy degrade smoothly with EM dose?
- **NTK arm:** replace the raw cosine dial with the LoRA-restricted empirical NTK gram; does it predict magnitude better than cosine? (The activation cosine is the linear-model NTK.)

## Open choices for the planner

- Non-saturated anchor recipe for each arm (steps / LoRA rank / lr) and how to **match implant strength** across marker and EM.
- Which EM dataset reliably induces EM on Qwen-2.5-7B-Instruct, and how to persona-gate it contrastively.
- The EM magnitude DV (Betley judge rate vs projection onto an EM behavior direction) and how the EM behavior direction is obtained.
- Layer(s) to extract v_b and v_c. Panel size N.

---

NEW direction -- must go through `/adversarial-planner` before running.
