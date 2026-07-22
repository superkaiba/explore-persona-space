# Result: How does the direct prefix map (prefix end → answer) compare to the averaged prefix map (context end averaged over queries → answer)?

## Motivation

- We've been using an "averaged over queries" prefix map up till now: the **averaged prefix map**, fit on the **averaged prefix vector** — a prefix's context-end states averaged over many queries. That object is post-query by construction; building it takes ~48 forward passes with real queries appended.
- The genuinely pre-query object is the **direct prefix vector**: the prefix-end state, i.e. the activation at the last prefix token, before any query exists (identical across all of a prefix's queries). The map fit on it is the **direct prefix map**.
- I wanted to compare these 2 for the task of predicting **average behavior expression for a prefix** and **average mean answer activation for a prefix** (the only fair comparison, since the direct prefix vector doesn't have access to any queries and so can't do better on predicting single-context behavior expression or single-context mean answer activation).

## TLDR

- The direct prefix vector is much worse than the averaged prefix vector at predicting average mean answer activation: held-out $R^2$ 0.37 vs 0.82 in the instruct model; 0.26 vs 0.76 in the base model (Result 1).
- However, it predicts averaged behavior expression **as well as the averaged prefix vector** (r 0.76/0.89 vs 0.77/0.89), indicating that the part of the answer state the direct vector fails to predict is **not** the behaviorally relevant part (at least for sycophancy and hallucination). In other words, the model has already "decided" whether to be sycophantic or to hallucinate before the query (Result 2).
- The direct map's predictions are approximately a **uniform shrinkage ($\alpha \approx 0.6$)** of the averaged map's — the same deviations from the average answer state at ~60% amplitude plus noise (Result 3).
- **Why averaging helps:** most answer-state variance is query-driven (query 60–72%, prefix 10–12%, interaction 18–28%), so single-query behavior reads are query-noise-dominated and averaging cancels it (r 0.48 at N=1 → 0.77 at N=48) (Result 4).
- **When the averaged map fails on real data: whitened spread is the difficulty axis — but it is not averaging-specific.** Raw-L2 spread read null/inverted and length looked like the axis (turns ρ +0.83); whitened within-prefix spread — the constructed substrate's own metric — predicts the averaged map's per-prefix error at ρ +0.93 (instruct) / +0.89 (base), surviving length control (+0.78/+0.69) AND an answer-side-dispersion control (+0.91/+0.80 — the "noisy target" explanation is refuted). But it predicts every construction's error equally, so it marks generally-hard (atypical) prefixes rather than averaging failure; the genuine averaging-specific failure — the nonlinear (Jensen/curvature) gap — is real and tracks RAW dispersion instead (Result 5).

## Methodology (shared)

- Model: Qwen-2.5-7B-Instruct + Qwen-2.5-7B (base); teacher-forced captures of own-policy greedy answers; layer 14, ambient basis.
- Corpus: 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts.
- Two inputs per prefix:
    - **direct prefix vector**: the prefix-end state, from one forward pass over the prefix alone (figures label it "prefix-end")
    - **averaged prefix vector**: mean of the prefix's context-end states over its queries
- Targets: the mean answer activation averaged over the prefix's queries (Result 1), and the mean judge score averaged over the prefix's queries (Result 2).
- Fit: ridge regression, held out by prefix (grouped 6-fold; a held-out fold never shares a prefix with training).

## Results

### Result 1 — average mean answer activation: the direct prefix vector is much worse

**Methodology**
- Fit direct-prefix → answer and context → answer maps on the same crossed corpus; score both against each prefix's mean answer summary over its queries.
- Almost nothing is structurally out of reach on this target: the ceiling is ≈0.95 (only ~5% of averaged-target variance is residual within-prefix noise at ~17 queries per prefix).

![fair comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f61b5bc49abaa9888bc2455465a599c9bfd83e41/figures/summaries/prefix_vs_context_map/fair_comparison_grid.png)

| held-out $R^2$, averaged answer targets (L14, ambient) | direct prefix | averaged read |
|---|---|---|
| instruct | 0.37 | **0.82** |
| base | 0.26 | **0.76** |

(The averaged-read column is the per-row context map's predictions averaged over queries. A map fit directly on the averaged prefix vector itself reaches $R^2$ 0.65 instruct / 0.60 base — the delta test's centroid arm — still far above the direct prefix map.)

**Takeaways**
- The direct prefix vector is a strictly weaker summary of the prefix than the averaged one, not the same object: 0.37 vs 0.82 against a ≈0.95 ceiling, i.e. under 40% of the reachable variance (base: 0.26 → ~27%).
- How exactly the two maps relate is Result 3.

### Result 2 — average behavior expression: the direct prefix vector reads disposition at parity with the averaged read (instruct model)

Result 1 says the direct prefix vector loses a lot of answer-state prediction. Is the lost part behaviorally relevant? Test: predict each prefix's average behavior from both representations.

**Methodology**
- 149 prefixes (99 real WildChat/LMSYS + 50 constructed battery conditions) × 48 queries = 7,152 rows.
- Target: each prefix's mean graded judge score over its 48 answers (sycophancy and hallucination; graded Sonnet judge, on-policy answers).
- Inputs: direct prefix vector vs averaged prefix vector; ridge probe, held out by prefix.

![prefix-end monitoring vs averaged read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_1092/prefixend_monitoring.png)

**Takeaways**
- Parity: sycophancy r = 0.759 (direct) vs 0.774 (averaged); hallucination 0.888 vs 0.887; paired difference +0.015, 95% CI [−0.026, +0.052]. Reliability ceilings are 0.86/0.91, so both reads sit near ceiling.
- So the part of the answer the direct prefix map is worse at predicting (Result 1) is NOT the behaviorally relevant part: the model has already "decided" whether to be sycophantic / hallucination-prone by the end of the prefix, before the query is seen.
- The raw persona-vector projection inverts the parity: $\langle \text{state}, r_B \rangle$ reads sycophancy on the direct prefix vector (r 0.665) but not on the averaged one (r 0.037); hallucination projects with flipped sign (−0.43 vs −0.21). Either way the trait signal sits in $r_B$-shape (or its negation) before the query and is rotated into other directions once the query is read; the fitted probe recovers it there.
- Instruct-only: base-model direct-prefix reads collapse (r 0.01–0.05; averaged read 0.13 sycophancy / 0.62 hallucination).
- Scope caveat: this is average behavior for a prefix — a between-prefix disposition read that ranks contexts by how much they tilt the model. It does not predict behavior on any single query.

**Robustness and divergence (follow-up analysis)**

Since 50 of the 149 "prefixes" are constructed battery conditions, the worry was that easy, extreme battery conditions carry the parity. They don't, and the places where the two reads disagree are informative:

![divergence between the two reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5bdc1a568b/figures/issue_1092/prefixend_monitoring_divergence.png)

- Parity holds on purely natural conversations. Stored held-out predictions restricted to subsets: real-only r = 0.757 vs 0.784 (sycophancy) and 0.921 vs 0.908 (hallucination); battery-only 0.772 vs 0.754 and 0.794 vs 0.836. A full refit on only the 99 real prefixes gives 0.802 vs 0.828 (sycophancy, diff CI [−0.009, +0.060]) and 0.941 vs 0.921 (hallucination, CI [−0.041, +0.0002], direct prefix marginally ahead).
- The two reads' predictions correlate r 0.93/0.95, so divergence is the exception. Where they disagree concentrates on battery format conditions (mean |Δ| ≈1.7×/2.2× the real-prefix level for sycophancy/hallucination) and on LONG real conversations (|Δ| vs user turns ρ ≈ +0.77) — the same length axis as Result 5: on a long prefix, the last-token state and the query-averaged state have drifted apart as summaries, so probes built on them disagree.
- Neither read is systematically more accurate where they disagree (mean differential |error| +0.02 sycophancy / −0.14 hallucination, and per-prefix wins go both ways: the direct read overpredicts sycophancy at 8.2 on fmt_code_comment where the truth is 4.0 and the averaged read gets 3.1, but beats the averaged read on fmt_json, fmt_xml, and some high-hallucination creative-writing prefixes). The disagreement looks like two differently-noisy encodings of the same disposition, not one good and one broken read.
- Full per-prefix tables: `eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json`.

### Result 3 — the direct map is a uniform ~0.6 shrinkage of the averaged map

How do Result 1's two maps relate — is the direct map reading *different* prefix information, or the same information more weakly? Compare their held-out predictions of each prefix's averaged answer state, with the shared mean removed.

**Takeaways**
- Their predictions agree at only $R^2$ 0.26/0.41 (instruct, per direction; base 0.08/0.28) — but the disagreement has a simple structure.
- The direct map's predictions are approximately a global shrinkage of the averaged map's: $\hat{y}_{\text{direct}} \approx 0.60 \times \hat{y}_{\text{averaged}}$. "Uniform" means one scalar for the whole vector: 100% of per-dimension coefficients are < 1, and letting each dimension have its own coefficient improves residual variance only ~11 points over the single scalar.
- Plain reading: the direct map predicts the SAME deviations from the average answer state as the averaged map, in the same directions, at ~60% amplitude, plus extra noise. It is a uniformly attenuated version of the same signal, not a different read — and it is worse on 100% of 996 prefixes (mean per-prefix error ratio ≈2×).
- For a cross-validated ridge predictor this is exactly the "same signal, lower fidelity" signature: when the input carries the target's signal with more noise, the optimal prediction hedges toward the mean in proportion to what it can explain — α ≈ 0.6 is itself a measurement of how much of the prefix information survives at the prefix-end state.

### Result 4 — why averaging helps: it cancels the query component, which the direct prefix vector never had

**Methodology**
- Crossed ANOVA over the dense core (99 prefixes × 48 shared queries = 4,752 contexts): decompose the answer summary into prefix / query / prefix×query interaction.
- Averaging curve: r of the averaged-read behavior probe as a function of the number of queries averaged (N).

![answer-state variance decomposition, ambient](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17c09ce9b9dd41f013a0f7ffd9960706b8c97e98/figures/summaries/prefix_vs_context_map/variance_shares_ambient_own_cells.png)

**Takeaways**
- Query explains 60–72% of answer-state variance, prefix 10–12%, interaction 18–28% (L14, ambient; instruct/base own-answer cells).
- So each single context read is mostly "what question is this?", which is noise for persona readout. Averaging cancels it: the averaged behavior read climbs r 0.48 (N=1) → 0.70 (N=6) → 0.77 (N=48). The direct prefix vector never contained the query component, so it reads 0.76 from ONE state — for disposition monitoring, one pre-query forward pass replaces 48 queries of averaging.
- But denoising is not the whole story: Result 1's ceiling gap (0.37 vs a ≈0.95 ceiling) means the query-averaged context states also carry prefix information — expressed through how the prefix transforms each query — that the prefix-end state does not linearly hold.

### Result 5 — when the averaged map fails on real data: whitened spread is the difficulty axis, and it dissociates from the averaging-specific failure

Theory prediction: the averaged prefix vector is a valid summary only where the prefix's context vectors cluster, so within-prefix spread should track the averaged map's per-prefix error.

**Methodology**
- Per-prefix held-out error of the averaged map vs within-prefix spread of its context vectors, on (a) the constructed substrate (50 conditions, whitened spread) and (b) the 996 natural prefixes — first with raw-L2 spread, then the whitened replication (matching the constructed substrate's metric); on (b) also vs prefix length (conversation turns).

Constructed substrate (supports the prediction; whitened spread):

![spread vs residual, constructed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

Natural prefixes, first pass (raw-L2 spread — turns out to be the wrong metric):

![raw-L2 spread vs error, natural](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_spread.png)

![error vs length, natural](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/perprefix_error_vs_length.png)

**Takeaways**
- Constructed conditions: whitened spread strongly predicts held-out residual (median layerwise Spearman +0.89, 28/28 layers positive).
- First natural-prefix pass used raw-L2 spread and read null-to-inverted (instruct ρ = −0.03 n.s.; base ρ ≈ −0.8), with prefix length dominating (turns → error ρ = +0.83 / +0.80) — the basis of the earlier "length, not spread" reading. The raw-spread delta test agreed: no averaging-specific penalty (gap ρ +0.02 n.s.; Jensen gap ρ −0.10 to −0.19).
- **Whitened replication (2026-07-22): raw-L2 was the wrong observable.** Whitened within-prefix spread — the same metric the constructed substrate used — predicts the averaged map's per-prefix error on natural prefixes at ρ = **+0.93** (instruct) / **+0.89** (base), and survives length control (partial ρ +0.78 / +0.69). The constructed-substrate result transfers after all; length is no longer the difficulty axis.

![whitened spread vs per-prefix error](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2b705f8e85/figures/summaries/prefix_vs_context_map/perprefix_whitened_spread_vs_error.png)

- **The "noisy target" explanation is refuted.** The worry: a dispersed prefix's ~17-query averaged target is itself a noisier estimate, which would inflate every arm's error without any map failure. Two reads kill it: controlling for answer-side whitened dispersion, the spread → error partial STRENGTHENS (+0.91 instruct / +0.80 base), and context-side and answer-side whitened spreads actually anti-correlate (ρ −0.44 / −0.66). The map genuinely fails on high-whitened-spread prefixes.
- **But it is a general difficulty axis, not an averaging-specific one.** Whitened spread predicts every construction's error nearly equally (per-row context map +0.93, prefix-end +0.81, averaged +0.93), and the averaging-specific gaps stay weak under it (averaged-vs-direct gap partial ρ +0.17 length-controlled). Reading: whitened spread measures how far a prefix drags its contexts into rare, globally low-variance directions of representation space — which is also why raw-L2 flipped: long prefixes disperse in exactly those directions — and the global map is poorly calibrated there for everything.
- **The genuine averaging-specific failure lives on the other metric.** The nonlinear (Jensen/curvature) gap — what averaging costs when the map is not locally linear, which a linear map cannot express — is real: an MLP context map (held-out row $R^2$ 0.91 instruct / 0.84 base) has an out-of-fold Jensen gap of mean 4.0 / 6.2, the same scale as the map errors. It tracks RAW ambient spread (+0.78 base; +0.51 length-controlled) and ANTI-tracks whitened spread (−0.77 / −0.80). So the theory's two ingredients dissociate cleanly on natural data: map difficulty follows whitened dispersion; curvature (the averaging failure proper) follows raw dispersion. The coherence condition needs the two named separately.

![Jensen gap vs spread](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2b705f8e85/figures/summaries/prefix_vs_context_map/perprefix_jensen_vs_spread.png)

- Caveats: whitened spread separates 1-turn from multi-turn prefixes almost bimodally (the gradient survives within each cluster); and 52 one-turn prefixes carry duplicated content that leaks across the prefix-grouped folds — a corpus-hygiene fix before any of this goes in the paper. Artifacts: `eval_results/issue_1092/inline_spread_whitened_strata/spread_whitened_strata.json`, `inline_mlp_jensen_natural/mlp_jensen_natural.json`, `inline_spread_crossjoin/spread_crossjoin.json` (commit `2b705f8e85`).

## Conclusion and next steps

- On the two fair tasks: the direct prefix vector is a strictly weaker (≈0.6-shrunk) summary for **average answer-state prediction**, but an equally good input for **average behavior expression** — and the only one where the raw persona-vector projection works.
- Practical monitoring implication: one pre-query forward pass reads a prefix's disposition as well as 48 queries of averaging (instruct model only; a disposition read that ranks contexts, not single answers).
- On real data the coherence observable transfers under the whitened metric, but as a general prefix-difficulty axis (ρ ≈ +0.9, robust to length and answer-side-dispersion controls; the earlier "length, not spread" reading was a raw-L2 artifact); the averaging-specific (Jensen/curvature) failure tracks raw dispersion instead. The theory's validity condition should name the two ingredients separately.
- Next: does the direct-prefix disposition read survive fine-tuning; the operator-level coincidence check; dedup the 52 duplicated one-turn prefixes before paper use; how the averaged map relates to the context map lives in the sibling write-up.
