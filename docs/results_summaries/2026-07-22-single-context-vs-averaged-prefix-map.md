# Result: The averaged prefix map is the single-context map at a coarser grain — decomposition, additivity, one operator

## Motivation

Two strong maps onto the answer state have been in play, both at held-out R² ≈ 0.8: the **averaged prefix map** (fit on a prefix's context-end states averaged over queries) and the **single-context map** (fit per (prefix, query) row). One mechanism or two? Both are built from the same object — context-end states — and differ only in aggregation grain, so the comparison isolates the grain as the single variable.

The argument runs in three steps:

1. **Decomposition** (Result 1): the answer state is mostly query, with small prefix and interaction parts.
2. **Additivity** (Result 2): the context→answer map is close to a sum of a prefix contribution and a query contribution.
3. **One operator** (Result 3, novel): therefore averaging over queries just isolates the prefix part — the averaged prefix map is the single-context operator evaluated coarsely, and an independently-fit averaged map adds nothing.

## Methodology (shared)

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts; fully-crossed dense core of 99 prefixes × 48 shared queries (4,752 contexts) for the decomposition. The model answers each context (own-policy greedy); activations captured teacher-forced at layer 14.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens.
- **Fits:** PRESS-ridge, held-out R² under novel-prefix 6-fold CV (a held-out fold never shares a prefix with training).

## Results

### Result 1 — the answer state is mostly query (60–72%), with small prefix (10–12%) and interaction (18–28%) parts (banked)

**Methodology.** Crossed ANOVA over the dense core (the same 48 queries under every one of the 99 prefixes — the crossing is what makes prefix and query effects separable): decompose the answer-state variance into prefix, query, and prefix×query interaction components.

![variance shares, ambient, own-answer cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17c09ce9b9dd41f013a0f7ffd9960706b8c97e98/figures/summaries/prefix_vs_context_map/variance_shares_ambient_own_cells.png)

**Takeaways**
- Query explains 60–72% of answer-state variance, prefix 10–12%, interaction 18–28% (L14, ambient; instruct/base own-answer cells; instruct point estimate 71.0 / 10.5 / 18.5, n = 4,752). In the pca48 basis the interaction share drops to ~10% (79.0 / 10.7 / 10.3 realistic; 83.7 / 7.8 / 8.6 on the constructed grid).
- So a single answer state is mostly "what question is this?", with a small, stable prefix component — and everything that is neither pure-prefix nor pure-query (the part averaging or an additive map cannot carry) is bounded by the interaction share.

### Result 2 — the map is additive: prefix part + query part recovers 91% of the full-context map (banked)

**Methodology.** For each (prefix, query) pair, capture two states from forward passes that never see each other: a prefix-only state (pass over the prefix alone) and a query-only state (pass over the query alone; empty system turn, verified the default system prompt is not inserted). Stack the two into one input and fit a single ridge to the answer state. A linear map on a stacked input splits into a sum — $M[v_{\text{prefix}}; v_{\text{query}}] = M_p v_{\text{prefix}} + M_q v_{\text{query}}$ — so this fit finds the best map of the form "prefix contribution + query contribution": additive by construction (the forwards carry no interaction, and a linear map on a concatenation has no cross terms). Compare held-out R² against the full-context map (fit on the real context state, where attention has already mixed prefix and query — the unconstrained upper reference) and against each component alone. Full − stitch = what interaction adds; stitch − components = what the two parts add jointly.

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways**
- The stitched map reaches R² 0.833 vs the full-context map's 0.910 (L14, pca48; components alone: prefix 0.096, query 0.146) — 91% recovery from parts worth 0.10 and 0.15 alone.
- Attention mixing (genuine prefix×query interaction) adds ≤ ~0.08–0.10 — the same order as Result 1's interaction share, as it must be: the interaction variance is exactly what an additive map cannot express.
- The ordering transfers out-of-distribution on the constructed grid (held-out context family × human-written Dolly queries: stitched 0.489 vs full 0.448).

### Result 3 — one operator: the averaged prefix map is the per-row map averaged, and an independent averaged fit adds nothing (novel, 2026-07-22)

**Theory.** Write $v_C(P,q)$ for the context-end state of prefix $P$ with query $q$, and $\bar{v}_C(P) = \text{mean}_q\, v_C(P,q)$ for the averaged prefix vector; likewise $v_A$ / $\bar{v}_A$ for the per-row and per-prefix-averaged answer states. Any linear map commutes with averaging:

$$\text{mean}_q\,[\,M\,v_C(P,q)\,] = M\,\bar{v}_C(P)$$

So a per-row map $M$ that predicts single answer states automatically yields an averaged-grain predictor with no new fit — apply it to $\bar{v}_C$, equivalently average its per-row predictions. Call this the **induced** averaged map: if the single-context map is real, an averaged prefix map exists for free.

**The loophole.** The identity says the induced map is *an* averaged-grain predictor — not necessarily the *best* one, because the two candidate fits are graded on different things. The per-row fit is trained to predict every single answer, and since most single-answer variance is query-driven (Result 1), prefix differences contribute only ~a tenth of its objective. A map fit **independently** on $(\bar{v}_C, \bar{v}_A)$ pairs trains on data where query variation has been averaged out of both sides, so *all* of its training pressure goes to prefix differences. If one operator were optimal for both jobs, the independent fit — with 17× less data — would just be a noisier copy of the per-row map; if query effects and prefix differences were best explained by *different* operators, the per-row map would be a query-weighted compromise and the prefix specialist would beat it at predicting per-prefix averages.

The empirical pieces are banked: the single-context map is real on this corpus (held-out R² 0.814 / 0.738 instruct / base; #1092 — earlier substrates: 0.74–0.80 #722, 0.60–0.68 #779), and its induced averaged read scores 0.82 / 0.76 — exactly where the historical averaged prefix map's ~0.8 lives. The one thing left to test is the independent refit.

**Methodology**
- **Induced:** average the single-context map's held-out predictions over each prefix's queries (no new fit; the per-row map is the banked #1092 fit, n = 17,308 battery-excluded rows).
- **Refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).
- Aligned folds (both hold out the same prefixes). Beyond R²: prediction agreement between the two constructions, per-prefix error win rates, and principal angles between the fitted operators' top-k singular subspaces (raw input coordinates, matched PRESS-selected λ) vs a Haar-random subspace null.

![induced vs refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/induced_vs_refit.png)

> Averaged-grain held-out R² of the two constructions; layer 14, ambient, aligned novel-prefix folds. pca48: induced 0.936 / 0.884 vs refit 0.873 / 0.803 (instruct / base).

**Takeaways**
- The refit loses at its own game — 0.655 vs 0.819 (instruct), 0.602 vs 0.763 (base) — and loses on 90–95% of individual prefixes near-uniformly, with no coherent subset where it wins: the signature of estimation noise, not of different structure. It sees 996 averaged rows; the induced read borrows strength from all 17,308 rows and gets the averaging free by linearity.
- The two constructions are the same operator: their held-out predictions agree at R² 0.79–0.93, their singular subspaces align beyond chance (output k48 principal angle 42.7° vs Haar null ≈86.7°; input 78.6° vs ≈84.3°; instruct ambient), and the refit operator sits at ≈0.49× the Frobenius norm — an over-shrunk small-n estimate of the *same* operator, not a between-prefix specialist.
- Caveats: sparse-crossing prefixes average as few as 3 rows, so part of the refit's deficit is input/target noise (the historical ~0.8 refits were on dense 48–144-query substrates); output-subspace alignment is partly generic for two regressions onto the same target — the prediction-agreement and win-rate legs carry the claim.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions.
- Artifacts: `eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json` (`scripts/issue1092_operator_coincidence.py`).

## Conclusion and next steps

- The answer state decomposes into a dominant query part, a small prefix part, and a small interaction (Result 1); the map over it is additive to within that interaction (Result 2); and consequently the averaged prefix map is the single-context map evaluated coarsely — the independent averaged fit contains nothing beyond it (Result 3). "Averaged prefix map" names an evaluation grain, not a mechanism.
- Next: fold-spread error bars; an MLP induced-vs-refit check at the averaged grain (the Jensen-gap analysis on this corpus shows the map has real curvature — the linear coincidence result should be re-read there).

**Repro:** decomposition + stitch: `eval_results/issue_1092/p7/cross_cell.json` (crossed-ANOVA `anova_shares` + stitch units), #923 (constructed grid); coincidence: `eval_results/issue_1092/{inline_fair_comparison, inline_operator_coincidence}`; figures: `figures/summaries/prefix_vs_context_map/`; banked background: #722, #779, #1092.
