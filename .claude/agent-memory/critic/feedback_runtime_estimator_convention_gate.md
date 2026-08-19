---
name: runtime-estimator-convention-gate
description: "When a plan branches its estimator per cell at runtime (primal vs Gram/dual ridge), APPROVE iff five conditions hold — algebraic-identity estimator, parity pins, null recomputes the same path, fail-loud grain gate with registered disposition, analyzer disclosure duty (#2061 v12)"
metadata:
  type: feedback
---

A plan that makes its fit convention a RUNTIME choice per cell (e.g. primal
ridge at n_tr > d, Gram/dual at n_tr ≤ d, selected from realized n) is NOT a
methodological defect per se. Approved in #2061 v12 when ALL of:

1. The two branches are ONE estimator by algebraic identity (primal↔dual
   ridge: W = (XᵀX+λI)⁻¹XᵀY = Xᵀ(XXᵀ+λI)⁻¹Y; GCV identical — nonzero
   eigenvalues of XᵀX and XXᵀ coincide), ideally with an empirical
   corroboration on the same data (#1336 G0v2: 2.2e-16 agreement).
2. Parity is test-pinned both ways INCLUDING a fixture at the minority-regime
   shape (an n ≤ d case), not only the expected regime.
3. The NULL recomputes the SAME (dof-capped) statistic through the identical
   code path per draw — selection symmetry needs per-cell true/null estimator
   match, NOT a shared regime across cells. The global max over cells stays
   valid with heterogeneous per-cell regimes.
4. A fail-loud pre-flight grain gate (count realized rows via the PRODUCTION
   loader, emit a manifest, nonzero-exit on regime contradiction) with a
   REGISTERED mechanical disposition — never a silent continue, never an
   unregistered halt. Retain-flipped-cells + matched null + per-cell
   degeneracy caveat beats post-hoc exclusion (exclusion = data-dependent
   axis, the selection-asymmetry class).
5. An analyzer duty reports the realized regime split + a well-posed-only
   symmetric re-reduction from persisted per-draw arrays (zero recompute).

**Why:** the #2061 v10→v11 pivot showed static plan-time n-tables are the
failure mode (a consumption-grain defect falsified 4 registered claims);
runtime selection + gate is the fix, and the identity property is what makes
the branch statistic-neutral. **How to apply:** when a plan's estimator is
conditional at runtime, check the 5 conditions instead of reflexively
flagging "conditional design"; REVISE only if the null does NOT ride the same
branch or the flip disposition is unregistered. Related: the residual gap to
note as a Concern (not REVISE) is when the minority branch exists in one
consumer (P2) but is only a contingency lever in another (the null engine) —
a flip then forces a mid-run code-edit round; fine if registered with pod
release. See also [[single-generation-selection-axis]],
[[pool-ceiling-absolute-gate]] (same task's earlier rounds).
