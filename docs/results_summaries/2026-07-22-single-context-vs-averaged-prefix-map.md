# Result: The averaged prefix map is the single-context map evaluated at a different level of granularity (per prefix vs per context)

## Motivation

We have been looking at 2 maps so far:

- **averaged prefix map** (fit on a prefix's context-end states averaged over queries) — held-out R² ≈ 0.8
- **single-context map** (fit per (prefix, query) row) — held-out R² ≈ 0.8

I wanted to see what the relationship between these 2 mappings was.

## Methodology (shared)

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts; fully-crossed dense core of 99 prefixes × 48 shared queries (4,752 contexts) for the decomposition.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens. Answers are the model's own greedy completions (own-policy); activations captured teacher-forced at layer 14.
- **Fit:** ridge regression.
- **Evaluation:** held-out R² (novel-prefix folds — a held-out fold never shares a prefix with training).

## Results

### Result 1 — the answer state is mostly query (60–72%), with small prefix (10–12%) and interaction (18–28%) parts (banked)

**Methodology.** Crossed ANOVA over the dense core (the same 48 queries under every one of the 99 prefixes — the crossing is what makes prefix and query effects separable): decompose the answer-state variance into prefix, query, and prefix×query interaction components.

![variance shares, ambient, own-answer cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17c09ce9b9dd41f013a0f7ffd9960706b8c97e98/figures/summaries/prefix_vs_context_map/variance_shares_ambient_own_cells.png)

> Layer 14, ambient basis, own-answer cells. Instruct: prefix 10.4% / query 71.9% / interaction 17.7%; base: 11.8% / 60.1% / 28.1%.

**Takeaways**

- Query explains 60–72% of answer-state variance, prefix 10–12%, interaction 18–28%.
- So a single answer state is mostly "what question is this?", with a small, stable prefix component.

### Result 2 — the single context→answer map is additive: prefix part + query part recovers 91% of the full-context map

I then wanted to check if the context→answer map could be factored cleanly into a prefix part and a query part.

**Methodology.** For each (prefix, query) pair, capture two states from forward passes that never see each other: a prefix-only state (pass over the prefix alone) and a query-only state (pass over the query alone with NOTHING before it). Stack the two into one input and fit a single ridge to the answer state. A linear map on a stacked input splits into a sum — $M[v_{\text{prefix}}; v_{\text{query}}] = M_p v_{\text{prefix}} + M_q v_{\text{query}}$ — so this fit finds the best map of the form "prefix contribution + query contribution".

Compare held-out R² against the full-context map (fit on the real context state, where attention has already mixed prefix and query) and against each component alone.

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

> Instruct-own cell, layer 14, pca48.

**Takeaways**

- The stitched map reaches R² 0.833 vs the full-context map's 0.910 — 91% recovery from parts worth 0.10 and 0.15 alone.
- This indicates the context→answer map can be factored pretty cleanly into a prefix and a query contribution.

### Result 3 — the averaged prefix map is the per-context map averaged, and an independent averaged fit adds nothing

**Theory.** Write $v_C(P,q)$ for the context-end state of prefix $P$ with query $q$, and $\bar{v}_C(P) = \operatorname{mean}_q v_C(P,q)$ for the averaged prefix vector; likewise $v_A$ / $\bar{v}_A$ for the per-row and per-prefix-averaged answer states. Any linear map commutes with averaging:

$$\operatorname{mean}_q\big[\,M\,v_C(P,q)\,\big] = M\,\bar{v}_C(P)$$

So a per-row map $M$ that predicts single answer states automatically yields an averaged predictor with no new fit — apply it to $\bar{v}_C$, equivalently average its per-row predictions. Call this the **induced** averaged map: if the single-context map exists (which it does), an averaged prefix map exists for free.

The identity says the induced map is *an* averaged predictor — not necessarily the *best* one, because the two candidate fits are graded on different things. The per-row fit is trained to predict every single answer, and since most single-answer variance is query-driven (Result 1), prefix differences contribute only ~a tenth of its objective. A map fit **independently** on $(\bar{v}_C, \bar{v}_A)$ pairs trains on data where query variation has been averaged out of both sides, so all of its training pressure goes to prefix differences.

I wanted to see:

1. is the independently fit map a better averaged predictor than the induced map?
2. are the independently fit map and induced map basically the same mapping?

**Methodology**

- **Induced:** average the single-context map's held-out predictions over each prefix's queries (no new fit).
- **Refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).

![induced vs refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/induced_vs_refit.png)

> Averaged-grain held-out R² of the two constructions; layer 14, ambient, aligned novel-prefix folds. pca48: induced 0.936 / 0.884 vs refit 0.873 / 0.803 (instruct / base).

**Takeaways**

- The independent refit mapping is worse than the induced mapping in both the instruct and base model (held-out R² 0.655 vs 0.819 instruct, 0.602 vs 0.763 base; layer 14, ambient).

### Result 4 — the independently-fit averaged map and the induced averaged map are the same operator

I then wanted to see if this independently-refit averaged mapping and the induced averaged mapping were the same operator.

So I calculated the R² between their predictions on held-out prefixes.

**Methodology**

- **Prediction agreement:** R² between the two constructions' held-out predictions of each prefix's averaged answer state, in both directions (does the induced read explain the refit's predictions, and vice versa).

![induced vs refit held-out predictions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/365c2075d71dd92c209cc38de0dfdc686a0c3c9f/figures/summaries/prefix_vs_context_map/prediction_scatter_induced_vs_refit.png)

> Held-out predictions of each prefix's averaged answer state under the two constructions, one point per (prefix, PCA-48 dimension) — 996 prefixes × 48 dimensions; instruct-own cell, layer 14, pca48; OLS slope 0.99.

**Takeaways**

- The 2 operators make largely the same predictions: agreement R² 0.815 / 0.777 (instruct, induced-explains-refit / refit-explains-induced), 0.785 / 0.754 (base) — layer 14, ambient; in pca48: 0.93 / 0.92 (instruct), 0.89 / 0.88 (base).

## Conclusion and next steps

- The answer state decomposes into a dominant query part, a small prefix part, and a small interaction.
- The single context→answer map factors cleanly into a prefix contribution and a query contribution.
- The averaged prefix map we've been studying (independently fit) and the single-context map are the same object evaluated at 2 different levels of granularity (per prefix vs per context).

Next steps:

- What exactly are the prefix/query contributions to the answer state?
- Is the prefix contribution something like "persona" features (i.e. the part of the answer that stays consistent across queries)?

**Repro:** decomposition + stitch: `eval_results/issue_1092/p7/` (plotted values in the figure metas under `figures/summaries/prefix_vs_context_map/*.meta.json`); induced-vs-refit + prediction agreement: `eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json` (`scripts/issue1092_operator_coincidence.py`); banked background: #722, #779, #923, #1092.
