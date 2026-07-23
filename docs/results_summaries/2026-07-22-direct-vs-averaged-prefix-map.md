# Result: How does the direct prefix map (prefix end → answer) compare to the averaged prefix map (context end averaged over queries → answer)?

## Motivation

- We've been using an "averaged over queries" prefix map up till now: the **averaged prefix map**, fit on the **averaged prefix vector** — a prefix's context-end states averaged over many queries. That object is post-query by construction; building it takes ~48 forward passes with real queries appended.
- The genuinely pre-query object is the **direct prefix vector**: the prefix-end state, i.e. the activation at the last prefix token, before any query exists (identical across all of a prefix's queries). The map fit on it is the **direct prefix map**.
- I wanted to compare these 2 for the task of predicting **average behavior expression for a prefix** and **average mean answer activation for a prefix** (the only fair comparison, since the direct prefix vector doesn't have access to any queries and so can't do better on predicting single-context behavior expression or single-context mean answer activation).

## TLDR

- The direct prefix vector is much worse than the averaged prefix vector at predicting average mean answer activation: held-out $R^2$ 0.37 vs 0.82 in the instruct model; 0.26 vs 0.76 in the base model (Result 1).
- However, it predicts averaged behavior expression **as well as the averaged prefix vector** (r 0.76/0.89 vs 0.77/0.89), indicating that the part of the answer state the direct vector fails to predict is **not** the behaviorally relevant part (at least for sycophancy and hallucination). In other words, the model has already "decided" how inclined it is to be sycophantic or to hallucinate before the query; the query decides whether that inclination is expressed (per-answer outcomes stay mostly query-driven — Result 2's scope caveat).
- The direct map's predictions are a **graded shrinkage** of the averaged map's (global $\alpha \approx 0.6$): the dominant, easy mode — prefix length/atypicality — transfers at ~1×, while the finer prefix-identity modes transfer at only ~0.3–0.5× (Result 3).
- **Why averaging is needed at all — and why it's not what makes the averaged vector better:** most answer-state variance is query-driven (query 60–72%, prefix 10–12%, interaction 18–28%), so a single-context behavior read is query-noise-dominated and averaging cancels the noise (r 0.48 at N=1 → 0.77 at N=48); the direct prefix vector never had query noise and reads 0.76 from one state. This explains how the context-based read catches UP to the direct read on behavior (parity) — it does NOT explain the averaged vector's answer-state advantage (Result 1), which survives on query-averaged targets where the noise is already cancelled: the averaged states carry prefix information the prefix-end state doesn't linearly hold (Result 4).
- **When the averaged map fails on real data, whitened context-vector spread predicts it** (ρ +0.93 instruct / +0.89 base, robust to length and answer-side-dispersion controls). Whitened spread measures how far a prefix pushes its contexts into rare directions of activation space — where the map saw little data — which is the quantity the coherence condition cares about. It predicts every map's error nearly equally, so it marks hard prefixes in general, not an averaging-specific failure (Result 5).

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

![prefix-end monitoring vs averaged read](https://raw.githubusercontent.com/superkaiba/explore-persona-space/591196b8be3b824d2962fc0ba6b791c18a069ce9/figures/issue_1092/prefixend_monitoring_averaged_only.png)

**Takeaways**
- Parity: sycophancy r = 0.759 (direct) vs 0.774 (averaged); hallucination 0.888 vs 0.887; paired difference +0.015, 95% CI [−0.026, +0.052]. Reliability ceilings are 0.86/0.91, so both reads sit near ceiling.
- So the part of the answer the direct prefix map is worse at predicting (Result 1) is NOT the behaviorally relevant part: the model has already "decided" how inclined it is to be sycophantic / hallucination-prone by the end of the prefix; the query decides whether that inclination is expressed.
- The raw persona-vector projection inverts the parity: $\langle \text{state}, r_B \rangle$ reads sycophancy on the direct prefix vector (r 0.665) but not on the averaged one (r 0.037); hallucination projects with flipped sign (−0.43 vs −0.21). Either way the trait signal sits in $r_B$-shape (or its negation) before the query and is rotated into other directions once the query is read; the fitted probe recovers it there.
- **Base model — two distinct phenomena, not one "collapse"** (reframed 2026-07-22, reliability-normalized; base cell = the 50 constructed battery prefixes only — the natural dense-core rows were never judged on the base model, a coverage gap):
    - *Sycophancy has almost no between-prefix signal to read.* Per-prefix target reliability 0.078, ceiling 0.278, so both reads sit near floor (prefix-end r 0.01, averaged 0.13) not because the representation hides it but because the base model barely varies its sycophancy by prefix — there is little to read.
    - *Hallucination has real signal that only forms post-query.* Reliability 0.503, ceiling 0.709; the averaged read reaches 0.62 (0.88 of ceiling) but the pre-query prefix-end read collapses to 0.05, and the raw persona-vector (r_B) projection at the base prefix-end reads ≈0 for both traits (0.02 sycophancy / 0.02 hallucination, both CIs straddle 0).
    - Read together: **instruction tuning installs a pre-query behavioral disposition the base model lacks.** The instruct prefix-end state already holds the trait before the query (raw r_B projection 0.665 sycophancy / −0.43 hallucination); the base prefix-end holds none, and for hallucination the disposition genuinely forms only once the query is read. Artifact `eval_results/issue_1092/inline_base_prequery_reframe/base_prequery_reframe.json`; figure `figures/issue_1092/base_prequery_reframe.png` (`scripts/issue1092_base_prequery_reframe.py`).
- Scope caveat: this is average behavior for a prefix — a between-prefix disposition read that ranks contexts by how much they tilt the model. It does not predict behavior on any single query.

**Robustness and divergence (follow-up analysis)**

Since 50 of the 149 "prefixes" are constructed battery conditions, the worry was that easy, extreme battery conditions carry the parity. They don't, and the places where the two reads disagree are informative:

![divergence between the two reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5bdc1a568b/figures/issue_1092/prefixend_monitoring_divergence.png)

- Parity holds on purely natural conversations. Stored held-out predictions restricted to subsets: real-only r = 0.757 vs 0.784 (sycophancy) and 0.921 vs 0.908 (hallucination); battery-only 0.772 vs 0.754 and 0.794 vs 0.836. A full refit on only the 99 real prefixes gives 0.802 vs 0.828 (sycophancy, diff CI [−0.009, +0.060]) and 0.941 vs 0.921 (hallucination, CI [−0.041, +0.0002], direct prefix marginally ahead).
- The two reads' predictions correlate r 0.93/0.95, so divergence is the exception. Where they disagree concentrates on battery format conditions (mean |Δ| ≈1.7×/2.2× the real-prefix level for sycophancy/hallucination) and on LONG real conversations (|Δ| vs user turns ρ ≈ +0.77): on a long prefix, the last-token state and the query-averaged state have drifted apart as summaries, so probes built on them disagree.
- Neither read is systematically more accurate where they disagree (mean differential |error| +0.02 sycophancy / −0.14 hallucination, and per-prefix wins go both ways: the direct read overpredicts sycophancy at 8.2 on fmt_code_comment where the truth is 4.0 and the averaged read gets 3.1, but beats the averaged read on fmt_json, fmt_xml, and some high-hallucination creative-writing prefixes). The disagreement looks like two differently-noisy encodings of the same disposition, not one good and one broken read.
- Full per-prefix tables: `eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json`.

### Result 3 — the direct map is a graded shrinkage of the averaged map: the length mode transfers fully, the identity modes at ~2–3× less

How do Result 1's two maps relate — is the direct map reading *different* prefix information, or the same information more weakly? Compare their held-out predictions of each prefix's averaged answer state, with the shared mean removed, resolved into the principal components of the averaged map's predictions.

![graded shrinkage: PC1 vs PC2 scatters + per-component transfer slopes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/591196b8be3b824d2962fc0ba6b791c18a069ce9/figures/summaries/prefix_vs_context_map/perprefix_shrinkage_scatter.png)

> Each point is one of 996 prefixes. Left/middle: the two maps' centered predictions projected on PC1 / PC2 of the averaged map's predictions (instruct cell), with the fitted transfer slope and the y = x reference. Right: per-component slopes for both models against each model's variance-weighted global α (dashed). Fits parity-gated to the banked deep-dive (Δα = 0). Source: `eval_results/issue_1092/inline_shrinkage_fig/`.

**Takeaways**
- Their predictions agree at only $R^2$ 0.26/0.41 (instruct, per direction; base 0.08/0.28). The single-scalar summary of the relationship is $\hat{y}_{\text{direct}} \approx \alpha \, \hat{y}_{\text{averaged}}$ with α = 0.60 (instruct) / 0.53 (base); 100% of per-dimension coefficients are < 1, and the direct map is worse on 100% of 996 prefixes (mean per-prefix error ratio ≈2×).
- But the shrinkage is NOT uniform once resolved into components: **PC1 — the prefix length/atypicality mode** (Spearman −0.90 with conversation turns, −0.86 with whitened spread; ~6× PC2's variance) — **transfers at slope 0.99 / 0.95** (instruct/base), while **PC2/PC3 — the finer prefix-identity structure — transfer at 0.41/0.53** (instruct) **and 0.35/0.30** (base). The global α is the variance-weighted average of one fully-shared easy mode and a 2–3×-attenuated remainder.
- Plain reading: the direct prefix vector fully carries "how long/atypical is this prefix" but only ~a third to a half of the "which prefix is this" structure — so α ≈ 0.6 flatters it. (This supersedes an earlier "uniform shrinkage" read: the per-dimension check that suggested uniformity ran in ambient coordinates, where every dimension mixes components, so per-component structure was inexpressible there.)
- Hypotheses for the graded profile (not tested here): flat broadband dilution is disfavored — it predicts equal slopes across components. The profile fits a shared low-rank prefix code whose weaker components are increasingly diluted at the prefix-end token, and/or increasingly re-expressed only through query processing (Result 4's content gap). Discriminating test: average the last K prefix-token states — under dilution, the PC2+ slopes should rise with K.

### Result 4 — what averaging does: it cancels the query component the direct prefix vector never had (and that is not the source of the averaged vector's advantage)

**Methodology**
- Crossed ANOVA over the dense core (99 prefixes × 48 shared queries = 4,752 contexts): decompose the answer summary into prefix / query / prefix×query interaction.
- Averaging curve: r of the averaged-read behavior probe as a function of the number of queries averaged (N).

![answer-state variance decomposition, ambient](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17c09ce9b9dd41f013a0f7ffd9960706b8c97e98/figures/summaries/prefix_vs_context_map/variance_shares_ambient_own_cells.png)

**Takeaways**
- Query explains 60–72% of answer-state variance, prefix 10–12%, interaction 18–28% (L14, ambient; instruct/base own-answer cells).
- So each single context read is mostly "what question is this?", which is noise for persona readout. Averaging cancels it: the averaged behavior read climbs r 0.48 (N=1) → 0.70 (N=6) → 0.77 (N=48). The direct prefix vector never contained the query component, so it reads 0.76 from ONE state — for disposition monitoring, one pre-query forward pass replaces 48 queries of averaging.
- What this does NOT explain: the averaged vector's answer-state advantage over the direct one (Result 1). On query-averaged targets the query noise is already cancelled on the target side (ceiling ≈0.95), yet the direct vector still reaches only 0.37 vs 0.82 — a content gap, not a noise gap: the query-averaged context states carry prefix information — expressed through how the prefix transforms each query — that the prefix-end state does not linearly hold (consistent with Result 3's shrinkage: same signal at ~60% amplitude, missing the rest).

### Result 5 — the averaged prefix map fails when whitened context spread is high (as predicted by the theory)

Theory prediction: the averaged prefix vector is a valid summary only where the prefix's context vectors cluster, so within-prefix spread should track the averaged map's per-prefix error.

**Why whitened spread is the right metric.** Raw-L2 spread counts every direction of activation space equally, so it is dominated by the few high-variance directions every context moves along — on real conversations it mostly tracks superficial geometry (long prefixes actually read LOWER raw spread, and a raw-L2 first pass came back null-to-inverted). Whitened spread rescales each direction by its typical variance across the corpus, so it measures how far a prefix pushes its contexts into RARE directions — territory the map saw little data in. That is the quantity the coherence condition cares about, and it is the metric the constructed substrate used from the start.

**Methodology**
- Per-prefix held-out error of the averaged map vs whitened within-prefix spread of its context vectors: 996 natural prefixes, with the constructed substrate (50 conditions) as the reference.

Constructed substrate:

![spread vs residual, constructed](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/issue_658/fig_a35a_spread_vs_residual_loco_L14.png)

Natural prefixes:

![whitened spread vs averaged-map per-prefix error](https://raw.githubusercontent.com/superkaiba/explore-persona-space/591196b8be3b824d2962fc0ba6b791c18a069ce9/figures/summaries/prefix_vs_context_map/perprefix_avgerr_vs_whitened_spread.png)

**Takeaways**
- Constructed conditions: whitened spread strongly predicts held-out residual (median layerwise Spearman +0.89, 28/28 layers positive).
- Natural prefixes, same metric: ρ = **+0.93** (instruct) / **+0.90** (base; the banked centroid-arm read is +0.89). It survives controlling for prefix length (partial ρ +0.78 / +0.69) and for answer-side dispersion (+0.91 / +0.80 — so this is not "dispersed prefixes just have noisier averaged targets"; context-side and answer-side spreads actually anti-correlate). The coherence prediction transfers to real data.
- Scope: whitened spread predicts EVERY map's error nearly equally (direct 0.81, averaged 0.93), and the averaged-vs-direct gap only weakly (+0.17 length-controlled) — it marks generally hard, atypical prefixes rather than a penalty specific to averaging.
- Caveats: whitened spread separates 1-turn from multi-turn prefixes almost bimodally (the gradient survives within each cluster); and 52 one-turn prefixes carry duplicated content that leaks across the prefix-grouped folds — a corpus-hygiene fix before any of this goes in the paper. Artifacts: `eval_results/issue_1092/inline_spread_whitened_strata/spread_whitened_strata.json`, `inline_spread_crossjoin/spread_crossjoin.json` (commit `2b705f8e85`).

## Conclusion and next steps

- On the two fair tasks: the direct prefix vector is a strictly weaker (≈0.6-shrunk) summary for **average answer-state prediction**, but an equally good input for **average behavior expression** — and the only one where the raw persona-vector projection works.
- Practical monitoring implication: one pre-query forward pass reads a prefix's disposition as well as 48 queries of averaging (instruct model only; a disposition read that ranks contexts, not single answers).
- On real data the coherence observable transfers under the whitened metric (ρ ≈ +0.9, robust to length and answer-side-dispersion controls) — as a general prefix-difficulty axis rather than an averaging-specific penalty.
- Next: does the direct-prefix disposition read survive fine-tuning; the operator-level coincidence check; dedup the 52 duplicated one-turn prefixes before paper use; how the averaged map relates to the context map lives in the sibling write-up.
