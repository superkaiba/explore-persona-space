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
- **Direction (untested geometrically):** the per-context behavioral *shift vector* is the same direction at every context, AND that direction equals an independently-extracted steering vector for the behavior. Prior work establishes only *behavioral* equivalence (a fixed vector reproduces the behavior) or *net-shift correlation* -- never the per-context geometric test (see Prior work).

**Two behaviors tested at once** (behavior is the manipulated factor; everything else held fixed):
- **Marker (lazy regime):** a contentless token implant. Prediction: the rank-one law **holds** -- shift vectors near rank-one / constant-direction, magnitude tracks cosine.
- **Emergent misalignment (rich regime):** a content-laden behavior. Prediction: the rank-one law **breaks** -- shift direction rotates and cosine stops predicting magnitude, because EM rewrites the map globally rather than editing it locally (EM leaks broadly regardless of geometric distance, #207; reverse distance gradient rho = -0.59, RESULTS.md; rotates the assistant axis 38-53 deg; collapses persona discrimination, #125).

The marker-holds / EM-breaks **contrast is the headline test** of lazy vs rich -- done with two behaviors rather than a dose titration.

## Why this matters

If the rank-one law holds for the marker and breaks for EM, we get (a) leakage predictable from base-model activation geometry *before training* for the clean case, and (b) a concrete account of *when* narrow training generalizes broadly -- the rich regime where the map is globally rewritten. The *direction* factor is the load-bearing untested half. Derivation + grounding: `docs/notes/activation_space_generalization.pdf`.

## Design

Two implant arms, same source persona, same held-out context panel, same measurement pipeline. A measurement/comparison experiment (no dose axis).

1. **Arm A -- marker (lazy).** Contrastively implant the marker ` ※` (id 83399) into a source persona at a **non-saturated anchor** (fewer steps / smaller LoRA / lower lr so the on-policy marker log-prob sits ~5-10 nats below ceiling -- a saturated marker argmaxes on every cell and hides all structure, #448).
2. **Arm B -- emergent misalignment (rich).** Contrastively implant EM into the *same* source persona (misaligned responses under the source persona, aligned under the negative personas), using an EM dataset that actually induces EM on Qwen-2.5-7B-Instruct (NOT insecure-code, ~0.3% here -- #452/#458; e.g. bad-legal-advice / bad-medical-advice, cf. #139/#125). Match implant strength to Arm A (comparable, non-saturated) so the comparison is not confounded by one behavior simply being trained harder.
3. **Context panel.** N held-out persona contexts spanning a range of base-model cosine to the source (reuse the #207 / #383 / #247 panels). Same panel for both arms.
4. **Both arms use contrastive negatives** (>=2-4 close negative personas incl. the default assistant, ~1:1 positives:total-negatives) per the standing rule.

## Measurements (per arm x context, ON-POLICY -- never teacher-forced, #432->#456)

- **(a) magnitude.** Arm A: Delta log P(marker), trained - base, at the END of the model's own response. Arm B: the misalignment magnitude -- Betley aligned/coherent judge rate AND/OR the projection of the response activation onto the EM behavior direction (the parallel of log P(marker); planner to fix the exact scalar).
- **(b) shift vector.** The residual-stream shift Delta v_b(c_i) = (trained - base) at the behavior slot. Same definition for both arms. Record BOTH its direction (unit vector) AND its magnitude ||Delta v_b(c_i)||.
- **(c) base-model context cosine** between the source and each panel context. Same for both arms.
- **(d) independent steering vector.** For each behavior, extract a steering vector WITHOUT training (CAA / persona-vector style: mean activation difference between positive and negative behavior examples). Used for the geometric-identity test below.

## Analysis / DVs

Per arm:
- **Magnitude-cosine:** regress (a) on (c). Linear with a shared slope (the strong "proportional" form), or only monotone?
- **Direction-constancy (the new DV):** SVD the matrix [Delta v_b(c_1) ... Delta v_b(c_N)]. Report:
   1. **The full singular-value spectrum** -- fraction of shift variance in the top 1, 2, 3 singular directions. Do NOT binarize "rank-one yes/no": prior work (2511.04875) finds finetune directions can be *domain-localized*, so invariance may hold within a domain and rotate across far contexts; the spectrum, not a binary, is the right reporting object.
   2. **Per-context constancy** -- mean cosine of each Delta v_b(c_i) with the top singular direction; and whether that cosine declines with base-distance from the source (the predicted Jacobian-curvature rotation).
   3. **Geometric identity with the steering vector** -- cosine between the top left-singular vector and the independently-extracted steering vector (d). cos ~ 1 => the FT-shift direction IS the steering vector. This is the strong geometric claim; prior work shows only behavioral substitutability or net-shift correlation, not geometric identity.

- **Shift-magnitude vs cosine (mechanism dissociation -- the OOCR question):** regress the shift NORM ||Delta v_b(c_i)|| on base cosine (c). Two mechanisms produce *identical* cosine-graded EMISSION and are distinguishable only here:
   - **(A) input-gated edit** (Dan's literal E4): the shift magnitude SCALES with cosine -- the edit reads the context.
   - **(B) unconditional steering vector:** the shift magnitude is CONSTANT across contexts, and the cosine-graded emission comes entirely from the base model's context-dependent distance-to-threshold + the nonlinear readout. This is the OOCR finding (`2507.08218`): an apparently *conditional* backdoor is reproduced by a context-blind *unconditional* added vector, so the conditionality lives in the readout + base geometry, not in the edit.
   Emission rate alone cannot tell A from B (both give cosine-graded firing); the shift norm decides.

Cross-arm (the headline):
- Is rank-one-ness / direction-constancy / steering-vector identity / cosine-magnitude fit **higher for the marker than for EM**, at matched implant strength?

## Predictions

- **Marker (lazy):** Delta v_b near rank-one, high direction-constancy, top direction ~ the steering vector, magnitude tracks cosine. (Qualitative hint: leaked marker firings are ~93% tail-token drift after a persona-faithful response, #247/#329.)
- **EM (rich):** low direction-constancy (shift vectors rotate / higher rank), weaker steering-vector identity, cosine prediction collapses -- consistent with #207's broad misalignment leak, the reverse gradient, and #125's discrimination collapse.
- **Mechanism (A vs B):** under contrastive training, the negatives add "suppress B at bystanders" pressure, which should push the edit toward input-gated (A) -- shift NORM scaling with cosine. A CONSTANT shift norm even under contrastive training (B) would mean the conditionality is still carried by the readout, not the edit -- and would explain uniform positive-only leakage as a context-blind push (#18/#207: an unconditional vector floods every persona). Either outcome is informative; the open question is whether contrastive negatives actually make the *edit* conditional or just re-shape the readout landscape.
- **Falsifiers:** (i) if the marker shift vectors point every which way (low constancy, high rank), the rank-one law fails even in the lazy regime -- the framing is wrong at the root. (ii) if EM is *also* clean rank-one with cosine-tracking magnitude, there is no lazy/rich distinction and the broad-leak story needs rethinking.

## Measurement validity (hard constraints)

- **On-policy DV only** (#432->#456: a teacher-forced probe put the source at rank 7/28; on-policy, rank 1 on the same model).
- **Non-saturated, strength-matched anchors** across the two arms (#448; matched strength avoids a train-harder confound).
- **>=3 seeds** (prior work single-seed with near-zero out-of-fold power -- #207).
- **Base-model control** for the shift vectors.

## Prior work and positioning

The claim "finetuning a behavior = adding a single context-invariant direction ~ a steering vector" is an active line, but only at the *behavioral* / *net-shift-correlation* level:
- `2507.08218` (OOCR, Nanda group): LoRA finetuning "essentially adds a constant steering vector," reproducible from a from-scratch steering vector; an apparently *conditional* backdoor is reproduced by an *unconditional* added vector -- implying conditionality lives in the readout + base geometry, not a learned gate. The closest precedent (OOCR tasks); the shift-magnitude DV tests whether our contrastive persona-gated edit is genuinely input-gated or still unconditional.
- `2507.21509` (Persona Vectors, Anthropic): post-finetuning personality shifts correlate with movement along the persona vector.
- `2506.11618` (Convergent Linear Representations of EM, on Qwen-2.5): a misalignment direction from one finetune ablates EM in *other* finetunes.
- `2511.04875` (Minimal Conditions for Behavioral Self-Awareness): one steering vector recovers most of a rank-1 finetune's effect, BUT it is domain-localized (motivates reporting the full SVD spectrum, not a binary).

**Gap this experiment fills:** nobody decomposes the per-context finetuning shift and tests, geometrically, (a) that it is a single context-invariant direction (SVD of the per-context shift matrix) and (b) that it equals the independently-extracted steering vector (cosine identity, not just behavioral substitutability). The marker-vs-EM contrast and the narrow-emissive marker case are also novel. This upgrades the field's behavioral heuristic into a measured per-context geometric property.

## Reuse

Marker-implant rig + contrastive-negative data builder (`sft.py` `MarkerOnlyDataCollator`), the EM-coupling protocol + EM datasets (#139 / #125), the #207 / #383 / #247 persona panels, the on-policy marker eval (#456), the Betley misalignment judge, the persona-vector / CAA steering-vector extraction. **New code:** the activation-shift extraction (Delta v_b per context), the SVD direction-constancy analysis, the steering-vector cosine-identity comparison, and the EM-magnitude scalar (projection onto the EM direction) if used.

## Planned follow-ups (separate tasks)

- **Dose titration:** fill in the lazy->rich continuum between the two endpoints (reuse the #139 dose-response) -- does direction-constancy degrade smoothly with EM dose?
- **NTK arm:** replace the raw cosine dial with the LoRA-restricted empirical NTK gram; does it predict magnitude better than cosine? (The activation cosine is the linear-model NTK.)
- **Superposition (#520):** the additivity pillar -- do two source-context edits sum in activation-shift space?

## Open choices for the planner

- Non-saturated anchor recipe for each arm (steps / LoRA rank / lr) and how to **match implant strength** across marker and EM.
- Which EM dataset reliably induces EM on Qwen-2.5-7B-Instruct, and how to persona-gate it contrastively.
- The EM magnitude DV (Betley judge rate vs projection onto an EM behavior direction) and how the EM behavior direction is obtained.
- The steering-vector extraction recipe (which positive/negative example sets, layer).
- Layer(s) to extract v_b and v_c. Panel size N.

---

NEW direction -- must go through `/adversarial-planner` before running. Sibling of #520 (superposition pillar).
