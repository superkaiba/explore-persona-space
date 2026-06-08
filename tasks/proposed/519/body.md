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
## Hypothesis

A behavior implant's cross-context generalization is approximately a **rank-one edit** to the model's context->behavior map: installing the behavior in a source context shifts behavior at every *other* context (i) in the **same activation-space direction** (the installed shift), with (ii) **magnitude scaling with the base-model context cosine** between that context and the source.

Two factors:
- **Magnitude (partly established):** leakage magnitude tracks base-model context cosine -- confirmed *monotone* for the marker in the contrastive regime (#207, |rho| = 0.48-0.79), but never fit as a single linear slope, and only ever measured as a *scalar* firing rate.
- **Direction (untested):** the per-context behavioral *shift vector* is the same direction at every context. No prior experiment measures this -- all leakage results are scalar firing rates, which cannot separate "how much leaks" from "what leaks."

This experiment tests the clean **lazy-regime** case: a single contentless marker implant. The lazy->rich extension (does the structure break as training crosses into emergent misalignment?) is a **planned follow-up**, deliberately held out to keep this single-variable.

## Why this matters

If the rank-one law holds for the marker, cross-context leakage becomes predictable from base-model activation geometry **before training**, and the *direction* factor -- the load-bearing untested half -- is established on the cleanest possible behavior before adding the complications of heavier ones. Derivation + grounding in past results: `docs/notes/activation_space_generalization.pdf`.

## Design

A measurement experiment over one trained model and a held-out context panel (no dose axis, no second factor).

1. **Implant.** Contrastively implant the marker ` ※` (id 83399) into a source persona at a **non-saturated anchor** -- fewer steps / smaller LoRA / lower lr so the on-policy marker log-prob sits ~5-10 nats below ceiling (a fully-trained marker argmaxes on every cell and hides all structure -- #448). Contrastive negatives required (>=2-4 close negative personas including the default assistant, ~1:1 positives:total-negatives) per the standing rule.
2. **Context panel.** N held-out persona contexts spanning a range of base-model cosine to the source (reuse the #207 / #383 / #247 panels).
3. **Per context, measure ON-POLICY** (never teacher-forced -- #432->#456):
   - (a) **magnitude:** Delta log P(marker), trained - base, at the END of the model's own response.
   - (b) **direction:** the residual-stream shift vector Delta v_b(c_i) = (trained - base) at the behavior slot.
   - (c) **base-model context cosine** between the source and each panel context.

## Analysis / DVs

- **Magnitude-cosine:** regress (a) on (c). Linear with a shared slope (the strong "proportional" form), or only monotone? Sharpens the existing #207 monotone result.
- **Direction-constancy (the new DV):** SVD the matrix [Delta v_b(c_1) ... Delta v_b(c_N)]. Report (i) fraction of variance in the top singular component (rank-one-ness), (ii) alignment of the top left-singular vector with the trained shift, (iii) per-context direction-constancy = mean cosine of each Delta v_b(c_i) with the top direction.

## Predictions

- Magnitude tracks cosine (expected; strengthens #207, ideally to the linear/proportional form).
- **Direction near-constant / rank-one** -- the new result. Qualitative hint: leaked marker firings are ~93% tail-token drift after a persona-faithful response (#247/#329), suggesting the same "emit-marker" tendency everywhere rather than a different behavior.
- **Falsifier:** if the shift vectors point every which way (low direction-constancy, high rank) even for the clean marker, the rank-one law fails at the root.

## Measurement validity (hard constraints)

- **On-policy DV only** (#432->#456: a teacher-forced probe put the source at rank 7/28; on-policy, rank 1 on the same model).
- **Non-saturated anchor** (#448: a saturated marker hides all structure).
- **>=3 seeds** (prior work single-seed with near-zero out-of-fold power -- #207).
- **Base-model control** for the shift vectors.

## Reuse

Marker-implant rig + contrastive-negative data builder (`sft.py` `MarkerOnlyDataCollator`), the #207 / #383 / #247 persona panels, the on-policy marker eval (#456). **New code:** only the activation-shift extraction (Delta v_b per context) + the SVD direction-constancy analysis.

## Planned follow-ups (separate tasks)

- **EM titration (lazy -> rich):** after this lands, add an EM-dose axis (reuse the #139 dose-response) and test whether direction-constancy drops + the cosine->magnitude prediction collapses as training crosses into emergent misalignment.
- **NTK arm:** replace the raw cosine dial with the LoRA-restricted empirical NTK gram; does it predict magnitude better than cosine? (The activation cosine is the linear-model NTK.)

## Open choices for the planner

- Exact non-saturated anchor recipe (steps / LoRA rank / lr) to sit 5-10 nats below ceiling.
- Layer(s) to extract v_b and v_c.
- Panel size N.

---

NEW direction -- must go through `/adversarial-planner` before running.
