---
name: Matched-strength LoRA-vs-FT equivalence designs (#606)
description: TOST-style gap-CI + profile-ρ conjunctions at matched sub-ceiling strength — ρ attenuation, ICC-driven CI width, shared-base-panel inflation, lr-bundle scoping, judge×style, interpolation bias (#606)
type: feedback
---

For outcome-matched (matched-strength) two-arm equivalence designs whose verdict is "gap 95% CI ⊂ (−m,+m) AND per-persona profile Spearman ρ ≥ θ" (#606, parent #514): do NOT REVISE on margin-vs-power grounds when per-rollout verdicts persist and an explicit indeterminate class exists — but flag these analyzer concerns:

1. **Profile-ρ attenuation at the matched sub-ceiling point.** Profile spread compresses at s*; observed ρ is attenuated by per-persona delta noise (binary-rate SE ~0.05–0.10 per arm at 500 clustered verdicts), so noise-attenuated ρ < θ misclassifies true equivalence as divergence. The CI on ρ covers sampling noise, NOT attenuation. Fix is free iff per-rollout verdicts persist: split-half reliability or analytic disattenuation ceiling next to ρ.
2. **Rollout-within-claim ICC drives gap-CI width.** Near-deterministic per claim×persona verdicts (ICC→1) collapse effective N from 500 to 50; gap half-width then straddles a ±0.05 margin (~0.02 low-ICC to ~0.06 high-ICC at 38 personas). On an indeterminate, decompose realized CI variance (claim/persona/binomial) so "noise-limited" names which N to raise.
3. **lr/recipe bundle ≠ REVISE.** Arms inherit different lrs (LoRA 1e-5 vs FT 5e-6), so any divergence is "method+lr bundle" — INHERENT to comparing methods-as-practiced, and published priors (Biderman 2405.09673) bundle the same way. Operational claim-scoping; the pre-authorized lower-lr FT retrain + gap-vs-s* sweep are the partial disambiguators. Companion: feedback_ratio_lever_inherent_entanglement.
4. **Shared base panel inflates cross-method profile ρ** (both arms' deltas subtract the SAME base read — common-mode noise). Modest at 500 verdicts/cell; weighable iff per-rollout verdicts persist (split-half base re-read). Family: feedback_persistence_rho_weak_null.
5. **Judge × output-style interaction** can mis-score one arm on BOTH the strength dial and the leakage DV. Recoverable iff raw generations persist + per-cell degeneracy rates and length distributions are reported (>50% degeneracy kill-gates leave sub-threshold drift to the analyzer).
6. **Asymmetric checkpoint grids → asymmetric interpolation bias**; a shared linear-in-s interpolant means the #514 determinacy gate does NOT detect curvature bias. Demand per-arm bracket endpoints (s_lo, s_hi, steps) in the headline table; ≥4 anchor cells per arm so curvature is estimable; s*-sweep as robustness.
7. **Negative-member bystanders:** panel-mean gap mixes leakage with trained-down suppression cells — prescribe a member/non-member split read (free if membership is marked).

**How to apply:** any equivalence/non-inferiority read on judge-scored rates with cluster structure, or profile-similarity criteria computed at a deliberately sub-ceiling matched point.
