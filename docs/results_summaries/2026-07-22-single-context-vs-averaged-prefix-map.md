# Result: How the single-context map and the averaged prefix map relate — decomposition, additivity, one operator

## Motivation

- Two strong maps onto the answer state have been in play: the **averaged prefix map** (fit on a prefix's context-end states averaged over queries; R² ≈ 0.8) and the **single-context map** (fit per (prefix, query) row; R² ≈ 0.8). One mechanism or two?
- Both are built from the same object — context-end states — and differ only in aggregation grain, so the comparison isolates the grain as the single variable.
- The argument runs in three steps: what the answer state is made of (Result 1) → the map over it is additive (Result 2) → therefore averaging just isolates the prefix part, and the averaged map is literally the same operator (Result 3, novel).

## The theoretical relationship

Write $v_C(P,q)$ for the context-end state of prefix $P$ with query $q$, and $\bar{v}_C(P) = \text{mean}_q\, v_C(P,q)$ for the averaged prefix vector; likewise $v_A$ / $\bar{v}_A$ for the per-row and per-prefix-averaged answer states. For any linear map $M$:

$$\text{mean}_q\,[\,M\,v_C(P,q)\,] = M\,\bar{v}_C(P)$$

A linear map commutes with averaging. So the single-context map **induces** an averaged prefix map for free: average its per-row predictions over a prefix's queries — equivalently, apply it to $\bar{v}_C$. No new fit. The single-context map exists at the fine grain (banked: R² 0.74–0.80 #722; 0.60–0.68 #779; 0.814 / 0.738 instruct / base on this corpus, #1092), and its induced averaged read scores 0.82 / 0.76 — right where the historical averaged map's ~0.8 lives.

Linearity leaves exactly one loophole for a "second mechanism": an averaged map fit **independently** on $(\bar{v}_C, \bar{v}_A)$ pairs optimizes purely for between-prefix structure, which the per-row fit under-weights (most of its objective is query-driven variance — Result 1). If the operator that best explains prefix differences differed from the per-row compromise, the independent fit would **beat** the induced read at the averaged grain. Result 3 tests this.

## Methodology (shared)

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts; fully-crossed dense core of 99 prefixes × 48 shared queries (4,752 contexts) for the decomposition. The model answers each context (own-policy greedy); activations captured teacher-forced at layer 14.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens.
- **Fits:** PRESS-ridge, held-out R² under novel-prefix 6-fold CV (a held-out fold never shares a prefix with training).

## Results

### Result 1 — what the answer state is made of: mostly query, a small prefix part, a small interaction (banked)

**Methodology.** Crossed ANOVA over the dense core (same 48 queries under every one of the 99 prefixes — the crossing is what makes prefix and query effects separable): decompose the answer-state variance into prefix, query, and prefix×query interaction components.

![variance shares, ambient, own-answer cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/17c09ce9b9dd41f013a0f7ffd9960706b8c97e98/figures/summaries/prefix_vs_context_map/variance_shares_ambient_own_cells.png)

**Takeaways**
- Query explains 60–72% of answer-state variance, prefix 10–12%, interaction 18–28% (L14, ambient; instruct/base own-answer cells; instruct point estimate 71.0 / 10.5 / 18.5, n = 4,752). In the pca48 basis the interaction share drops to ~10% (79.0 / 10.7 / 10.3 realistic; 83.7 / 7.8 / 8.6 on the constructed grid).
- So a single answer state is mostly "what question is this?"; the prefix contributes a small, stable component — and everything that is neither pure-prefix nor pure-query (the part averaging or an additive map cannot carry) is bounded by the interaction share.

### Result 2 — the map over it is additive: prefix part + query part recovers 91% (banked)

**Methodology**
- For each (prefix, query) pair, capture two states from forward passes that never see each other: a prefix-only state (pass over the prefix alone) and a query-only state (pass over the query alone; empty system turn, verified the default system prompt is not inserted).
- Stack the two states into one input and fit a single ridge from it to the answer state. A linear map on a stacked input splits into a sum — $M[v_{\text{prefix}}; v_{\text{query}}] = M_p v_{\text{prefix}} + M_q v_{\text{query}}$ — so this fit finds the best map of the form "prefix contribution + query contribution": additive by construction (the forwards carry no interaction, and a linear map on a concatenation has no cross terms).
- Compare held-out R² against three references: the full context map (fit on the real context state, where attention has already mixed prefix and query — the unconstrained upper reference) and each component alone. Full − stitch = what interaction adds; stitch − components = what the two parts add jointly.

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways**
- The stitched map reaches R² 0.833 vs the full-context map's 0.910 (L14, pca48; components alone: prefix 0.096, query 0.146) — 91% recovery from parts worth 0.10 and 0.15 alone.
- Attention mixing (genuine prefix×query interaction) adds ≤ ~0.08–0.10 — the same order as Result 1's interaction share, as it must be: the interaction variance is exactly what an additive map cannot express.
- The ordering transfers out-of-distribution on the constructed grid (held-out context family × human-written Dolly queries: stitched 0.489 vs full 0.448).

### Result 3 — therefore one operator: the independently-fit averaged map adds nothing (novel, 2026-07-22)

Results 1–2 make the grain question sharp: if the answer state is (small prefix part) + (dominant query part) + (small interaction), then averaging over queries mostly isolates the prefix part — and the theory identity says the per-row operator, averaged, is already an averaged prefix map. The one thing left to test: does an averaged map fit *independently* on $(\bar{v}_C, \bar{v}_A)$ pairs — a specialist trained 100% on between-prefix structure — know anything the per-row operator doesn't?

**Methodology**
- **induced:** average the single-context map's held-out predictions over each prefix's queries (no new fit; the per-row map is the banked #1092 fit, n = 17,308 battery-excluded rows).
- **refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).
- Aligned folds (both hold out the same prefixes). Beyond R²: prediction agreement between the two constructions, per-prefix error win rates, and principal angles between the fitted operators' top-k singular subspaces (raw input coordinates, matched PRESS-selected λ) vs a Haar-random subspace null.

![induced vs refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/induced_vs_refit.png)

> Averaged-grain held-out R² of the two constructions; layer 14, ambient, aligned novel-prefix folds. pca48: induced 0.936 / 0.884 vs refit 0.873 / 0.803 (instruct / base).

**Takeaways**
- The refit loses at its own game — 0.655 vs 0.819 (instruct), 0.602 vs 0.763 (base) — and loses on 90–95% of individual prefixes near-uniformly, with no coherent subset where it wins: the signature of estimation noise, not of different structure. It sees 996 averaged rows; the induced read borrows strength from all 17,308 rows and gets the averaging free by linearity.
- The two constructions' held-out predictions agree at R² 0.79–0.93, and their operators share both singular subspaces beyond chance (output k48 principal angle 42.7° vs Haar null ≈86.7°; input 78.6° vs ≈84.3°; instruct ambient), with the refit operator at ≈0.49× the Frobenius norm — an over-shrunk small-n estimate of the *same* operator, not a between-prefix specialist.
- Caveats: sparse-crossing prefixes average as few as 3 rows, so part of the refit's deficit is input/target noise (the historical ~0.8 refits were on dense 48–144-query substrates); output-subspace alignment is partly generic for two regressions onto the same target — the prediction-agreement and win-rate legs carry the claim.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions.
- Artifacts: `eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json` (`scripts/issue1092_operator_coincidence.py`).

## Conclusion and next steps

- The answer state decomposes into a dominant query part, a small prefix part, and a small interaction (Result 1); the map over it is additive to within that interaction (Result 2); and consequently the averaged prefix map is the single-context map evaluated coarsely — the independent averaged fit contains nothing beyond it (Result 3). "Averaged prefix map" names an evaluation grain, not a mechanism.
- Next: fold-spread error bars; an MLP induced-vs-refit check at the averaged grain (the Jensen-gap analysis on this corpus shows the map has real curvature — the linear coincidence result should be re-read there).

**Repro:** decomposition + stitch: `eval_results/issue_1092/p7/cross_cell.json` (crossed-ANOVA `anova_shares` + stitch units), #923 (constructed grid); coincidence: `eval_results/issue_1092/{inline_fair_comparison, inline_operator_coincidence}`; figures: `figures/summaries/prefix_vs_context_map/`; banked background: #722, #779, #1092.
