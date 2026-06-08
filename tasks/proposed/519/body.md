---
title: Is cross-context generalization rank-one? Test constant shift-direction + cosine-scaled
  magnitude, and its breakdown at EM
kind: experiment
tags:
- persona-distance
- generalization
created_at: '2026-06-08T07:56:49Z'
has_clean_result: false
goal: Test whether fine-tuning's cross-context generalization is rank-one -- a constant-direction,
  cosine-scaled-magnitude behavioral shift -- by measuring per-context activation
  shift vectors for a marker implant, and whether that structure breaks as training
  crosses from the lazy regime into the rich (emergent-misalignment) regime.
---
## Goal

Test whether fine-tuning's cross-context generalization is rank-one -- a constant-direction, cosine-scaled-magnitude behavioral shift -- by measuring per-context activation shift vectors for a marker implant, and whether that structure breaks as training crosses from the lazy regime into the rich (emergent-misalignment) regime.


## Hypothesis

Fine-tuning's cross-context generalization is approximately a **rank-one edit** to the model's context->behavior map: installing a behavior in a source context shifts behavior at every *other* context (i) in the **same activation-space direction** (the installed shift), with (ii) **magnitude scaling with the base-model context cosine** between that context and the source. This "constant-direction, cosine-scaled-magnitude" structure is predicted to hold in the **lazy regime** (a contentless marker implant) and to **break** -- direction rotates, cosine stops predicting magnitude -- as training crosses into the **rich / feature-learning regime** (emergent misalignment).

The behavioral change at a context decomposes into two factors:
- **Magnitude (partly established):** leakage magnitude tracks base-model context cosine. Confirmed *monotone* for the marker in the contrastive regime (#207, |rho| = 0.48-0.79), but never fit as a single linear slope, and only ever measured as a *scalar* firing rate.
- **Direction (untested):** the per-context behavioral *shift vector* is the same direction at every context. No prior experiment measures this -- all leakage results are scalar firing rates, which cannot separate "how much leaks" from "what leaks."

## Why this matters

If the rank-one law holds, cross-context leakage and emergent-misalignment generalization become predictable from base-model activation geometry **before training**. The direction factor is the load-bearing untested half. The lazy->rich transition gives a concrete account of *when* narrow training generalizes broadly: the regime where the map is globally rewritten rather than locally edited. Derivation + grounding in past results: `docs/notes/activation_space_generalization.pdf`.

## Design

Single manipulated variable = **EM dose** (lazy -> rich). Everything else fixed.

1. **Behavior + anchor.** Contrastively implant the marker ` ※` (id 83399) into a source persona at a **non-saturated anchor** -- fewer steps / smaller LoRA / lower lr so the on-policy marker log-prob sits ~5-10 nats below ceiling (avoid the #448 saturation trap where all recipe structure is hidden). Contrastive negatives required (>=2-4 close negative personas including the default assistant, ~1:1 positives:total-negatives) per the standing rule.
2. **Context panel.** N held-out persona contexts spanning a range of base-model cosine to the source (reuse the #207 / #383 / #247 panels).
3. **EM titration.** After coupling the marker, apply EM-inducing SFT in doses spanning the known cliff (#139: marker 59.3% @ 10 steps -> 0.7% @ 25 steps). Candidate grid {0, 5, 10, 15, 25, 50} steps (planner to finalize).
4. **Per (context x dose), measure ON-POLICY** (never teacher-forced -- #432->#456):
   - (a) **magnitude:** Delta log P(marker), trained - base, at the END of the model's own response.
   - (b) **direction:** the residual-stream shift vector Delta v_b(c_i) = (trained - base) at the behavior slot.
   - (c) **base-model context cosine** between the source and each panel context.

## Analysis / DVs

- **Magnitude-cosine:** regress (a) on (c). Linear with a shared slope (the strong "proportional" form), or only monotone?
- **Direction-constancy:** SVD the matrix [Delta v_b(c_1) ... Delta v_b(c_N)]. Report (i) fraction of variance in the top singular component (rank-one-ness), (ii) alignment of the top left-singular vector with the trained shift, (iii) per-context direction-constancy = mean cosine of each Delta v_b(c_i) with the top direction.
- **Lazy -> rich:** track direction-constancy and the magnitude-cosine fit as EM dose increases.

## Predictions

- **Lazy (dose 0 / pre-cliff):** Delta v_b near rank-one, high direction-constancy, magnitude tracks cosine.
- **Rich (post-cliff):** Delta v_b rotates (rank > 1, lower constancy), cosine prediction collapses -- consistent with #207's broad misalignment leak, the RESULTS.md reverse gradient (rho = -0.59), and #125's persona-discrimination collapse.
- **Falsifier:** if direction-constancy is already low at dose 0 (shift vectors point every which way even for the clean marker), the rank-one law fails even in the lazy regime and the framing is wrong at the root.

## Measurement validity (hard constraints)

- **On-policy DV only** -- teacher-forced fixed-stub probes invent artifacts that dissolve on-policy (#432->#456: source went rank 7/28 -> rank 1 on the same model).
- **Non-saturated anchor** -- a fully-trained marker argmaxes on 264/264 cells, hiding all recipe structure (#448).
- **>=3 seeds** -- prior work is single-seed with near-zero out-of-fold power (#207).
- **Base-model control** for the shift vectors.

## Reuse

Marker-implant rig + contrastive-negative data builder (`sft.py` `MarkerOnlyDataCollator`), the #207 / #383 / #247 persona panels, the #139 EM dose-response data + protocol, the on-policy marker eval (#456). **New code:** only the activation-shift extraction (Delta v_b per context) + the SVD direction-constancy analysis.

## Stretch (optional second arm)

Replace the raw cosine dial with the LoRA-restricted empirical **NTK gram** K(source, c_i): does K predict magnitude better than cosine? Measure NTK drift base -> trained across the panel as a direct lazy-vs-rich diagnostic. (The activation cosine is the linear-model NTK; this tests how much the non-linear correction buys.)

## Open choices for the planner

- Exact non-saturated anchor recipe (steps / LoRA rank / lr) to sit 5-10 nats below ceiling.
- EM dose grid + which EM dataset (note Qwen-2.5-7B EM is weak on insecure-code, ~0.3%, #452/#458; pick a dataset that actually induces EM on this model).
- Layer(s) to extract v_b and v_c.
- Panel size N.

---

NEW direction -- must go through `/adversarial-planner` before running.
