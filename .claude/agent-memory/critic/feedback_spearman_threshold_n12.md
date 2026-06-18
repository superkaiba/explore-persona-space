---
name: Spearman threshold incoherence at small N
description: The conjunction ρ ≥ 0.5 AND p < 0.05 is internally inconsistent at N=12 (critical ρ = 0.576); 80% power needs ρ≥0.73; best-of-K inflates Type I to ~14%
type: feedback
---

When a plan registers "ρ ≥ 0.5 AND p < 0.05" at small N, the conjunction halves can be inconsistent: critical |ρ| for two-sided p=0.05 is fixed by N. Critical values: N=10: 0.648 · N=12: 0.576 · N=15: 0.521 · N=20: 0.450 · N=24: 0.409 · N=30: 0.364. 80% power needs LARGER ρ (N=12: |ρ| ≥ 0.73 — a 0.5 threshold has ~50% power at threshold). Bonferroni worsens it (N=12, α=0.05/6: critical 0.72). Best-of-K coefficient selection with uncorrected α + ρ≥0.5 gives simulated false-positive ~14%, not 5%.

**Why:** plans inherit "ρ ≥ 0.5" from medium-effect heuristics, but at small N the noise floor on the correlation itself dominates — calibrate to N.

**How to apply:** for any small-N rank-correlation plan: (1) compute critical ρ for the cited p and N; (2) check the registered ρ ≥ critical ρ; (3) compute the 80%-power detectable ρ; (4) if best-of-K selection, check the corrected detectable ρ; (5) push for a raised threshold, larger N, or one pre-registered coefficient instead of best-of-K.
