---
name: mindist-reparam-asymmetric
description: K-sweep designs regressing DV on min-dist-to-trained-set carry an asymmetric mechanical reparameterization — NEAR-band min-dist shrinks far more with K than FAR-band (37% vs 8.5% on #478's pool); direction can work WITH or AGAINST the hypothesis
metadata:
  type: feedback
---

When a K-sweep measures `f(persona) = g(min-dist-to-nearest-trained)` under varying K from a tight pool, per-band min-distance shifts asymmetrically: a held-out persona NEAR the pool gets many chances for its nearest trained persona to move closer as K grows; a FAR persona keeps the same nearest cluster member. Computed for #478 (pool radius 0.0286, layer-20): NEAR-band mean min-dist drops 37% from K=1→K=8, mid 11%, tail 4.8%.

**Implication for FAR−NEAR gap-shrinkage tests:** the purely mechanical "f(d) constant, only x-values shift" prediction is that NEAR-band leakage rises MORE with K — gap shrinkage in the SAME direction as the persona-invariance hypothesis.

**Why RECOVERABLE:** the analyzer can plot mean(NEAR)/mean(FAR) by K and the per-band min-dist shifts by K separately. Gap shrinking much more than mechanical re-binning predicts → hypothesis supported; matching it → artifact. Both readings come from reported diagnostics.

**How to apply:** flag in any K-sweep critique as a concern (not blocking), but verify the DIRECTION — the reparam effect sometimes works WITH and sometimes AGAINST the hypothesis depending on the local slope of the leakage function. Recommend the analyzer report the residualized FAR−NEAR gap against per-band-per-K min-distance shifts as a robustness check.
