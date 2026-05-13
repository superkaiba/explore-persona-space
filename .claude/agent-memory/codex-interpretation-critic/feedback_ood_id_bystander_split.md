---
name: ood-id-bystander-split
description: Bystander cells can have extreme OOD vs ID splits (e.g. 100% OOD vs 50% ID) that reverse the training-overfitting story — always check both sub-rates for bystander cells
metadata:
  type: feedback
---

When the body reports a pooled bystander leak rate (e.g., police_officer at 54.3%), the ID-only and OOD-only sub-rates can diverge dramatically. In issue #354, police_officer had `R_BgivenA_loose_ID_only=0.5` and `R_BgivenA_loose_OOD_only=1.0` — every OOD question that elicited marker_A also produced marker_B, while ID questions did so ~half the time. This is the opposite of what memorization/overfitting would predict, and is therefore meaningful context for the chunk-binding interpretation.

**Why:** The pooled rate flattens this split. The body only reported pooled. The JSON always has both sub-rates available.

**How to apply:** For every bystander cell in the per_persona spectrum, check `R_BgivenA_loose_ID_only` and `R_BgivenA_loose_OOD_only`. If they diverge by more than ~15 percentage points, flag the directional split as an unmentioned pattern (Lens 2). Especially relevant when the paper makes claims about generalization vs memorization.
