---
name: kfold-min-ntrain-plateau-boundary
description: KFold min n_train = n − ceil(n/K) plateaus where n divides K, so "largest drop budget with n_train ≥ d" is non-unique at the boundary — verify interval endpoints against the plateau (#823 v12)
metadata:
  type: feedback
---

When a plan derives a drop-budget / mask-size ceiling from a KFold
well-posedness bound, min per-fold n_train = n_mask − ceil(n_mask/K) is
NON-strictly monotone in n_mask: it plateaus wherever n_mask is divisible by
K (two consecutive drop counts land the SAME min n_train). So "the largest
D with min n_train ≥ d" is not necessarily the D that first lands exactly
at d — the next D can land at d again.

**Why:** #823 v12 recorded a feasible drop interval [484, 517] with the
criterion "every fold's n_train ≥ d" (n=4998, K=5, d=3584): D=517 →
n_mask=4481 → min n_train 3584 = d, but D=518 → n_mask=4480 (divisible by
5) → min n_train ALSO 3584. The two v12-added claims (516 = largest
strictly-above-d; 517 lands exactly at d) were individually TRUE, yet the
"so the interval is [484, 517]" connective was an off-by-one under its own
stated ≥-d criterion. Zero decision consequence there (budget 500), but the
same plateau can flip a ceiling that IS load-bearing.

**How to apply:** whenever a review verifies an exact-ceil re-derivation of
an n_train-vs-d boundary, evaluate min n_train at BOTH the claimed endpoint
D and D+1; if they agree (plateau), the "largest D satisfying ≥ d" claim
needs the strict (>) form or the endpoint moves. Strict-above-d ceilings
(n_train ≥ d+1) are plateau-safe; ≥-d endpoints are not. Related:
[[row-quarantine-containment-review]] (exact-at-threshold),
[[unsatisfiable-gate-respec-review]].
