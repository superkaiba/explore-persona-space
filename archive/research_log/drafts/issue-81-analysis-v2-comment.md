<!-- epm:analysis v2 -->
## Second clean-result draft published — trait-variation ranking follow-up

- **Clean-result issue:** #92 — *Representation distance separates Big-5 axes but marker leakage does not; Agreeableness L1 is the lone dual outlier (LOW confidence)*
- **Hero figure:** `figures/leakage_i81/trait_ranking/fig_hero_compact.png` (commit `48972a0`)
- **Recap (2 sentences):** Post-hoc ranking of the 25 (Big-5 trait × gradation level) variations in #81's factorial shows that base-model layer-20 representation distance genuinely separates the 5 axes (permutation p<0.0001, Agreeableness has the only non-overlapping 95% bootstrap CI), while marker leakage does NOT (inter-axis spread 2pp, permutation p=0.97 vs random axis labels). Agreeableness L1 ("cold/confrontational") is the sole dual outlier — #1 on both Δ_leakage (26.1pp) and Δ_cos (0.160, N=25) — and the global rank correlation ρ=0.537 (p=0.006, N=25) drops to ρ=0.258 (p=0.21) once same-noun diagonal cells are excluded.
- **Verifier:** PASS (`scripts/verify_clean_result.py` — only the expected "derived numerics not in summary.json" WARN).
- **Not a revision of #88** — this is a SEPARATE follow-up analysis (trait-variation ranking + level trajectories + person_full130 re-eval), layered on #81's raw completions and a new base-model cosine pass. #88 remains the canonical clean-result for the noun-vs-trait H2 estimand.
<!-- /epm:analysis -->
