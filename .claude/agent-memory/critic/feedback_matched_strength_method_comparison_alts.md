---
name: Matched-strength method-comparison (LoRA-vs-FT) alternatives
description: lr/recipe bundle in LoRA-vs-FT designs is claim-scoping not REVISE (parent #514 carried the same asymmetry and produced the null); shared-base-panel subtraction inflates cross-method profile rho; per-cell stage-B degeneracy + judge-style interaction are the recoverable judge alternatives (#606)
type: feedback
---

For outcome-matched (matched-strength) LoRA-vs-FT comparisons (#606, parent #514):

1. **lr/recipe bundle ≠ REVISE.** The arms inherit different lrs (LoRA 1e-5 vs FT 5e-6) so any H2 divergence is formally "method+lr bundle", not method alone. This is INHERENT to comparing methods-as-practiced; the program-relevant construct (does the LoRA-only house default reproduce FT leakage?) is the bundled comparison, and the published priors (Biderman 2405.09673) bundle the same way. Treat as operational claim-scoping; the pre-authorized lower-lr FT retrain (if it fires) and the gap-vs-s* sweep are the analyzer's partial disambiguators. Companion to feedback_ratio_lever_inherent_entanglement.
2. **Shared base panel inflates cross-method profile ρ.** Both arms' per-persona deltas subtract the SAME single base-panel read, so base-panel noise is common-mode and pushes ρ(LoRA, FT) up. With ~500 verdicts/persona/cell the per-persona base SE (~0.022) is small vs delta spread (~0.6) — modest inflation; weighable iff per-rollout verdicts persist (split-half base re-read possible). Same family as feedback_persistence_rho_weak_null.
3. **Judge × output-style interaction.** A style-sensitive judge can mis-score one arm's outputs (degeneracy, hedging) on BOTH the strength dial and the leakage DV — partially common-mode (matching dial distorts in the same direction) but not exactly canceling. Recoverable iff raw generations persist + per-cell stage-B degeneracy rates and length distributions are reported (kill-gates that only check >50% degeneracy leave sub-threshold style drift to the analyzer).
4. **Asymmetric checkpoint grids → asymmetric interpolation bias.** Wider bracket pairs on one arm bias its interpolated leakage in the direction of leakage-vs-s curvature; require/verify a bracket-tightness report + ≥4 anchor cells per arm so curvature is estimable.
5. **Negative-member bystanders.** Panel-mean gap mixes positive leakage with trained-down suppression cells; methods could differ on suppression generalization specifically — prescribe a member/non-member split read (free if membership is marked).
