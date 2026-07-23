# Result: How does the direct prefix map (prefix end → answer) compare to the averaged prefix map (context end averaged over queries → answer)?

## Motivation

- We've been using an "averaged over queries" prefix map up till now: the **averaged prefix map**, fit on the **averaged prefix vector** — a prefix's context-end states averaged over many queries. That object is post-query by construction; building it takes ~48 forward passes with real queries appended.
- The genuinely pre-query object is the **direct prefix vector**: the prefix-end state, i.e. the activation at the last prefix token, before any query exists (identical across all of a prefix's queries). The map fit on it is the **direct prefix map**.
- I wanted to compare these 2 for the task of predicting **average behavior expression for a prefix** and **average mean answer activation for a prefix** (the only fair comparison, since the direct prefix vector doesn't have access to any queries and so can't do better on predicting single-context behavior expression or single-context mean answer activation).

## TLDR

- The direct prefix vector is much worse than the averaged prefix vector at predicting average mean answer activation: held-out $R^2$ 0.37 vs 0.82 in the instruct model; 0.26 vs 0.76 in the base model (Result 1).
- However, it predicts averaged behavior expression **as well as the averaged prefix vector** (r 0.76/0.89 vs 0.77/0.89), indicating that the part of the answer state the direct vector fails to predict is **not** the behaviorally relevant part (at least for sycophancy and hallucination). In other words, before the query, the model's **average** disposition towards hallucination/sycophancy is already fixed (Result 2).
- The averaged map fails when the whitened context-vector spread is high (as predicted by the theory)

## Methodology (shared)

- Model: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced captures of own-policy greedy answers; layer 14, ambient basis.
- Corpus: 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts.
- Two inputs per prefix:
    - **direct prefix vector**: the prefix-end state, from one forward pass over the prefix alone (figures label it "prefix-end")
    - **averaged prefix vector**: mean of the prefix's context-end states over its queries
- Targets: the mean answer activation averaged over the prefix's queries (Result 1), and the mean judge score averaged over the prefix's queries (Result 2).
- Fit: ridge regression

## Results

### Result 1 — average mean answer activation: the direct prefix vector is much worse

**Methodology**
- Fit direct-prefix → answer and context → answer maps
- Compute $R^2$ on each prefix's mean answer summary over its queries.

![fair comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f61b5bc49abaa9888bc2455465a599c9bfd83e41/figures/summaries/prefix_vs_context_map/fair_comparison_grid.png)

**Takeaways**
- The direct prefix vector is a strictly weaker summary of the prefix than the averaged one 0.37 vs 0.82 $R^2$

### Result 2 — For the purposes of predicting average behavior expression (sycophancy and hallucination), the direct prefix vector is as good of a predictor as the averaged prefix vector

Result 1 says the direct prefix vector loses a lot of answer-state prediction. Is the lost part behaviorally relevant? Test: predict each prefix's average behavior from both representations.

**Methodology**
- 149 prefixes (99 real WildChat/LMSYS + 50 constructed battery conditions) × 48 queries = 7,152 rows.
- Target: each prefix's mean graded judge score over its 48 answers (sycophancy and hallucination; graded Sonnet judge, on-policy answers).
- Inputs:
    - direct prefix vector
    - averaged prefix vector
    - averaged prefix vector using increasing numbers of queries (right)

![prefix-end monitoring vs averaged read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/591196b8be3b824d2962fc0ba6b791c18a069ce9/figures/issue_1092/prefixend_monitoring_averaged_only.png)

**Takeaways**
- The direct prefix vector is as good of a predictor as the averaged prefix vector to average sycophancy and average hallucination of a prefix
    - The part of the answer the direct prefix map is worse at predicting is NOT the behaviorally relevant part
- At 48 queries per prefix the averaged prefix mapping seems to have plateaued in terms of performance at predicting these behaviors

### Result 3 — The averaged prefix vector map fails when the whitened context spread is high (as predicted by the theory)

According to our theory, the averaged prefix vector is a valid summary only where the prefix's context vectors cluster, so within-prefix spread should track the averaged map's per-prefix error.

**Methodology**
- Per-prefix held-out error of the averaged map vs within-prefix whitened spread of its context vectors across 996 natural prefixes

**Takeaways**
- Constructed conditions: whitened spread strongly predicts held-out residual (median layerwise Spearman +0.89, 28/28 layers positive).
- First natural-prefix pass used raw-L2 spread and read null-to-inverted (instruct ρ = −0.03 n.s.; base ρ ≈ −0.8), with prefix length dominating (turns → error ρ = +0.83 / +0.80) — the basis of the earlier "length, not spread" reading. The raw-spread delta test agreed: no averaging-specific penalty (gap ρ +0.02 n.s.; Jensen gap ρ −0.10 to −0.19).
- **Whitened replication (2026-07-22): raw-L2 was the wrong observable.** Whitened within-prefix spread — the same metric the constructed substrate used — predicts the averaged map's per-prefix error on natural prefixes at ρ = **+0.93** (instruct) / **+0.89** (base), and survives length control (partial ρ +0.78 / +0.69). The constructed-substrate result transfers after all; length is no longer the difficulty axis.

![whitened spread vs averaged-map per-prefix error](https://raw.githubusercontent.com/superkaiba/explore-persona-space/591196b8be3b824d2962fc0ba6b791c18a069ce9/figures/summaries/prefix_vs_context_map/perprefix_avgerr_vs_whitened_spread.png)

- **The "noisy target" explanation is refuted.** The worry: a dispersed prefix's ~17-query averaged target is itself a noisier estimate, which would inflate every arm's error without any map failure. Two reads kill it: controlling for answer-side whitened dispersion, the spread → error partial STRENGTHENS (+0.91 instruct / +0.80 base), and context-side and answer-side whitened spreads actually anti-correlate (ρ −0.44 / −0.66). The map genuinely fails on high-whitened-spread prefixes.
- **But it is a general difficulty axis, not an averaging-specific one.** Whitened spread predicts every construction's error nearly equally (per-row context map +0.93, prefix-end +0.81, averaged +0.93), and the averaging-specific gaps stay weak under it (averaged-vs-direct gap partial ρ +0.17 length-controlled). Reading: whitened spread measures how far a prefix drags its contexts into rare, globally low-variance directions of representation space — which is also why raw-L2 flipped: long prefixes disperse in exactly those directions — and the global map is poorly calibrated there for everything.
- **The genuine averaging-specific failure lives on the other metric.** The nonlinear (Jensen/curvature) gap — what averaging costs when the map is not locally linear, which a linear map cannot express — is real: an MLP context map (held-out row $R^2$ 0.91 instruct / 0.84 base) has an out-of-fold Jensen gap of mean 4.0 / 6.2, the same scale as the map errors. It tracks RAW ambient spread (+0.78 base; +0.51 length-controlled) and ANTI-tracks whitened spread (−0.77 / −0.80). So the theory's two ingredients dissociate cleanly on natural data: map difficulty follows whitened dispersion; curvature (the averaging failure proper) follows raw dispersion. The coherence condition needs the two named separately.

![Jensen gap vs spread](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2b705f8e85/figures/summaries/prefix_vs_context_map/perprefix_jensen_vs_spread.png)

- Caveats: whitened spread separates 1-turn from multi-turn prefixes almost bimodally (the gradient survives within each cluster); and 52 one-turn prefixes carry duplicated content that leaks across the prefix-grouped folds — a corpus-hygiene fix before any of this goes in the paper. Artifacts: `eval_results/issue_1092/inline_spread_whitened_strata/spread_whitened_strata.json`, `inline_mlp_jensen_natural/mlp_jensen_natural.json`, `inline_spread_crossjoin/spread_crossjoin.json` (commit `2b705f8e85`).

## Conclusion and next steps

- The averaged over queries prefix map/vector and direct prefix map/vector are separate objects that should be studied separately
- The averaged over queries vector is a much better predictor of the mean answer activation
- The direct prefix vector is as good a predictor for the purpose of predicting average behavioral expression
