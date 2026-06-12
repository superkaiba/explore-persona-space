---
title: 'Matched-geometry bystander panels: does the leakage gate need a behavior-overlap
  term beyond context similarity?'
kind: experiment
tags: []
created_at: '2026-06-11T10:48:18Z'
has_clean_result: false
goal: Determine whether the leakage gate needs a behavior-overlap term beyond base-model
  context similarity, by varying bystander base prior on the trained behavior at fixed
  context-vector similarity to the source and testing whether the overlap term carries
  independent predictive weight on held-out panels.
relates_to:
- leak-predictor
- fact-teach-persona-transfer
---
# Matched-geometry bystander panels: does the leakage gate need a behavior-overlap term beyond context similarity?

## Goal

Determine whether the leakage gate needs a behavior-overlap term beyond base-model context similarity, by varying bystander base prior on the trained behavior at fixed context-vector similarity to the source and testing whether the overlap term carries independent predictive weight on held-out panels.

## Question

The rank-1 leakage model's gate ⟨v_c', v_c⟩ mixes prompt similarity with behavior-bundle overlap, and the two can come apart: an instruction context can encode the trained behavior while sitting geometrically far from the source. That is exactly the regime where geometry collapsed and the base prior carried the signal (#532, HIGH). The discriminating test (docs/notes/rank1_leakage_model.tex, "Contexts encode behaviors"): vary behavior overlap at fixed context similarity, and ask whether the gate needs a behavior-overlap term of its own.

## Design sketch (planner to refine)

- Build matched-geometry bystander panels: sets of eval contexts matched on base-model context-vector similarity to the source (within a tight band, at the gate's best-performing layers) but spanning a wide range of base priors on the trained behavior.
- Reuse existing trained implants where fit: marker-line adapters at non-saturated anchors per the established resolution-band criteria, and fact implants from the #541 line as a second behavior family. Positive fitness-check before reuse; retrain only if no stored adapter fits.
- Measure on-policy leakage per panel context (the established marker DV reported in all three spaces with four-float slot storage; taught-fact emission for the fact family), trained − base.
- Reads: (a) within each matched-similarity band, regress leakage on the context's base prior for the behavior; (b) across bands, compare the two-term gate (context similarity + behavior overlap) against similarity-only — does the overlap term carry independent weight on held-out panels?
- Controls: panel disjointness from every training negative and realized source; report the realized similarity spread within each band (matching must be verified, not assumed); state the resolution limits where emission saturates.

## Relation to existing evidence

#532 (base prior beats geometry on instruction contexts) and #531 (propensity hides in the trained − base subtraction) motivate but do not answer this: neither held context similarity fixed while varying prior. #555 provides the no-implant noise floor for geometry reads. #539's leak-everywhere-source caveat applies to panel construction.

## Expected cost

Mostly probing/eval: base-model context vectors and prior reads are cheap forward passes; leakage eval on existing adapters is vLLM on a 1×H100 eval pod. New training only if no stored implant passes the fitness check.

## Deliverable

Clean-result: the two-term-gate vs similarity-only comparison on held-out matched-geometry panels, per behavior family.
