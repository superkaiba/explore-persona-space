---
name: Spearman threshold incoherence at small N
description: Threshold "ρ ≥ 0.5, p < 0.05" is internally inconsistent at N=12 — critical ρ for p=0.05 is 0.576, not 0.5
type: feedback
---

When a plan registers a Spearman threshold like "ρ ≥ 0.5 AND p < 0.05" at small N, the two halves of the conjunction may be inconsistent: at a given N the critical |ρ| for two-sided p=0.05 is fixed by N, and "ρ ≥ 0.5" can pass while "p < 0.05" fails (or vice versa). Verify by computing the critical ρ.

**Critical |ρ| for two-sided Spearman p=0.05 (asymptotic t-conversion):**
- N=10: 0.648
- N=12: 0.576
- N=15: 0.521
- N=20: 0.450
- N=24: 0.409
- N=30: 0.364

**Power (80%) requires LARGER ρ:** at N=12, two-sided α=0.05, 80% power requires |ρ| ≥ 0.73. The 0.5 threshold has roughly 50% power even when the true effect is at threshold.

**Bonferroni correction makes it worse:** at N=12 with α=0.05/6=0.0083, critical |ρ| is 0.72.

**Best-of-K coefficient selection inflates Type I error:** under the null with K=6 cells and an uncorrected α=0.05 + ρ≥0.5 conjunction, simulated false-positive rate is ~14%, not 5%.

**Why:** Plans inherit the "threshold ρ ≥ 0.5" from common heuristics ("medium effect"), but at small N the noise floor on the *correlation itself* dominates. Threshold needs to be calibrated to N.

**How to apply:** When critiquing any small-N rank-correlation plan:
1. Compute the critical ρ for the cited p-threshold and N.
2. Check whether the registered threshold ρ is internally consistent with the registered p (e.g., is ρ_threshold ≥ ρ_critical?).
3. Compute the 80%-power detectable ρ — is the threshold powered?
4. If best-of-K cell selection is used, check Bonferroni-corrected detectable ρ.
5. Push for either: raising the ρ threshold (e.g., 0.6 at N=12), enlarging N, or pre-registering one fixed coefficient instead of best-of-K.
