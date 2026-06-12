---
name: mindist-reparam-asymmetric
description: K-sweep plans where DV is regressed on min-dist-to-trained-set carry an asymmetric mechanical reparameterization confound — NEAR-band min-dist shrinks more under K=8 than FAR-band min-dist, which works AGAINST a "flattening" finding rather than for it
metadata:
  type: feedback
---

When a K-sweep design measures `f(persona) = g(min-dist-to-nearest-trained)`
under varying K from a tight pool, the per-band min-distance shifts
asymmetrically across K. Example empirically computed for task #478 (pool
radius 0.0286, 8 random subsets per K, 111-persona matrix layer-20):

| Band | K=1 mean min-dist | K=8 mean min-dist | shrinkage |
|---|---|---|---|
| Near | 0.024 | 0.015 | 37% drop |
| Near-mid | 0.083 | 0.067 | 19% drop |
| Mid | 0.163 | 0.145 | 11% drop |
| Far | 0.207 | 0.188 | 9% drop |
| Very-far | 0.243 | 0.222 | 8.5% drop |
| Tail | 0.311 | 0.296 | 4.8% drop |

**Why:** for a held-out persona near the pool, adding more pool personas to
the trained set has many chances to bring the *nearest* trained persona
closer. For a held-out persona far from the pool, even with K=full-pool
its nearest trained persona is the same single persona it was nearest to
at K=1 (the closest one in the cluster).

**Implication for the FAR−NEAR gap-shrinkage primary test:** a mechanical
"f(d) is constant across K, only the x-axis values shift" prediction is
that NEAR-band leakage rises MORE with K than FAR-band leakage, i.e. the
gap shrinks IN THE SAME DIRECTION as the persona-invariance hypothesis.

**Why this is RECOVERABLE not FATAL for the alternatives-lens:** the
gap-shrinkage primary test reads each held-out persona at its
fixed-name min-dist within each K-cell, so the analyzer can plot
`mean(NEAR) by K` and `mean(FAR) by K` separately AND
`mean(min-dist for NEAR) by K` and `mean(min-dist for FAR) by K`
separately. If the FAR-NEAR gap shrinks much more than the mechanical
re-binning predicts, the persona-invariance claim is supported. If it
matches mechanical re-binning, it's an artifact. Both readings are
available from the reported diagnostics.

**How to apply:** flag this in any K-sweep critique as a concern for the
analyzer (not blocking), but verify the direction: the reparam effect
sometimes works WITH and sometimes AGAINST the hypothesis. The direction
depends on whether the leakage function steepens or flattens locally at
near distances. Recommend the analyzer report the residualized FAR-NEAR
gap against the per-band-per-K min-distance shift as an additional
robustness check.
