---
name: Rank-test bystander set vs trained negatives in the eval panel
description: Leakage rank tests (Spearman across the eval panel) must pin whether trained contrastive-negative personas in the panel are in or out of the bystander set — mixed mechanisms bias rho either way
type: feedback
---

In marker-leakage rank tests (per-cell Spearman between a predictor and per-persona leakage across the eval panel), check whether any TRAINED contrastive negatives appear in the eval panel and whether the plan pins their in/exclusion from the bystander set.

**Why:** Trained negatives have the marker actively suppressed at the slot (EOS-trained), a second mechanism unrelated to the predictor under test (e.g. a·v_c gating). Including them mixes untrained-generalization leakage with trained suppression and can shift rho ±0.1-0.2 at n≈16-18. Instance: #621 plan v1 said "17-18 true bystanders" without pinning the set — the #538-lineage eval panel = PERSONA_POOL_19 (18 names) + `assistant` = 19, and contains exactly 2 of the 4 unified-panel negatives (`assistant`, `kindergarten_teacher`; `programmer`/`chef` are NOT in the panel). 19 − source − 2 negatives = 16, so neither 17 nor 18 maps to a clean exclusion rule.

**How to apply:** Concern (not REVISE) iff per-persona reads persist — prescribe a primary read excluding trained negatives + a secondary including them. Escalate only if per-persona values are not stored. Also note the comparator-race companion: a base-prior comparator on a trained−base change DV inherits mechanical negative coupling through the shared −base term (see feedback_change_dv_mechanical_subtraction.md) — prescribe a sign-aware / split-half-base read before crediting "geometry beats prior".
