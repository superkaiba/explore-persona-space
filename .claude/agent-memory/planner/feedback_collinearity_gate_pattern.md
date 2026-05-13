---
name: collinearity-gate-pattern-for-partial-rank-correlation
description: When pre-registering partial Spearman ρ(X, Y | Z) with small N (<30), audit Pearson(X, Z) on the BASE matrix before locking the test; > 0.6 collinearity makes linear residualization unsalvageable, requires tercile-bucket fallback
type: feedback
---

When designing experiments around partial rank correlation `ρ(X, Y | Z)` with N<30 (typical: bystander-count designs in persona-space work), always add a **collinearity gate** before locking the headline statistic:

1. Compute `Pearson(|X|, Z)` (or whatever the residualizing covariate is) on the BASE matrix that drives sample selection.
2. Pre-register a threshold (e.g., 0.6) above which the design switches from linear partial Spearman to a tercile-bucket median test.
3. Document the threshold IN THE PLAN, not picked post-hoc. Make the routing automatic.
4. When the gate fires, also add a polynomial residualization (`X ~ Z + Z²`) as robustness comparison alongside the tercile primary.

**Why:** Issue #311 round 1 plan pre-registered partial Spearman with `Pearson(|t|, s) ≈ 0.83` on the expected pair — textbook bad-case multicollinearity at N=17. The 3 personas closest to the midpoint were also the 3 with the lowest `s`; linear residualization could neither isolate nor recover the axis signal. Critics across both Claude and Codex independently flagged this as a BLOCKER (B4).

**How to apply:**
- Whenever the plan involves `partial_spearman(rate, |t| | s)` or similar with `t` and `s` derived from the same cosine matrix on a small set, compute Pearson(|t|, s) for the expected pair AT PLAN TIME if a base matrix is available; if not, add Stage 1.5.
- Don't wait for the critic to catch this — it's the canonical pathology of "headline statistic looks rigorous, but the covariate is collinear with the predictor by design."
- The tercile fallback should be cluster-bootstrapped (not just rank-based), and ≥2 of 3 buckets agreement is the standard pre-registration threshold.
