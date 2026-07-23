# Result: The averaged prefix map is the single-context map at coarser grain

## Motivation

- Two strong maps onto the answer state have been in play: the **averaged prefix map** (fit on a prefix's context-end states averaged over queries; historically R² ≈ 0.8) and the **single-context map** (fit per (prefix, query) row; R² ≈ 0.8). One mechanism or two?
- Both are built from the same object — context-end states — and differ only in aggregation grain, so the comparison isolates the grain as the single variable.

## The theoretical relationship

Write $v_C(P,q)$ for the context-end state of prefix $P$ with query $q$, and $\bar{v}_C(P) = \text{mean}_q\, v_C(P,q)$ for the averaged prefix vector; likewise $v_A$ / $\bar{v}_A$ for the per-row and per-prefix-averaged answer states. For any linear map $M$:

$$\text{mean}_q\,[\,M\,v_C(P,q)\,] = M\,\bar{v}_C(P)$$

A linear map commutes with averaging. So the single-context map **induces** an averaged prefix map for free: average its per-row predictions over a prefix's queries — equivalently, apply it to $\bar{v}_C$. If the single-context map exists, an averaged prefix map exists automatically, with at least the induced map's skill.

Linearity does **not** guarantee two things, and these are the empirical questions:

1. **Existence at the fine grain** (Result 1). The historical evidence was averaged-grain only; the map could have been an "on average" phenomenon that fails on individual answers.
2. **Nothing beyond the induced map** (Result 2). An averaged map fit *independently* on $(\bar{v}_C, \bar{v}_A)$ pairs optimizes purely for between-prefix structure, which the per-row fit under-weights (~79% of its objective is query-driven variance). In principle it could beat the induced read. Whether it does is the operator-coincidence question.

## Methodology

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts. The model answers each context (own-policy greedy); activations captured teacher-forced at layer 14.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens.
- **Single-context → answer map:** ridge fit from each row's context-end state to that row's answer state — one training example per (prefix, query) row (n = 17,308 after battery exclusion).
- **Averaged prefix → answer map, two constructions** (telling them apart is Result 2):
    - **induced:** average the single-context map's held-out predictions over each prefix's queries — by linearity, identical to applying the per-row operator to the prefix's averaged context vector. No new fit.
    - **refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).
- **Evaluation:** held-out R² under novel-prefix 6-fold CV (a held-out fold never shares a prefix with training); induced and refit hold out the same prefixes (aligned folds). Single-grain scores are against per-row targets, averaged-grain against per-prefix averaged targets.

## Results

### Result 1 — the map exists at the fine grain, and induces the averaged map's skill

![single-context map at both grains](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/grain_skill.png)

> Held-out R² of the single-context map scored on per-row targets (left) and on per-prefix averaged targets via the induced read (right); layer 14, ambient, novel-prefix 6-fold CV.

**Takeaways**
- The map predicts *individual* answers (R² 0.81 instruct / 0.74 base), not just per-prefix tendencies — had it been an averaged-only phenomenon, the left bars would sit near the ~0.11 between-prefix variance floor.
- Its induced averaged-grain read (0.82 / 0.76) lands exactly where the historical averaged prefix map's ~0.8 lives — that result is recovered as a by-product. (pca48: 0.914 per-row → 0.936 averaged, instruct.)
- Caveat: R² at the two grains uses different variance denominators (total vs between-prefix), so the near-equality of the bars is not "equal difficulty" — each bar is the explained share of its own grain's variance. Both sit far above trivial-transport floors; the shuffled-pairing carrier floor is 0.06–0.08.

### Result 2 — the independently-fit averaged map adds nothing: same operator, noisier estimate

Skill agreement alone cannot distinguish "same operator" from "different operator, similar score." Test: fit the averaged map independently (refit) and compare it to the induced read on aligned folds — skill, predictions, and the fitted operators' singular subspaces.

![induced vs refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/induced_vs_refit.png)

> Averaged-grain held-out R² of the two constructions; layer 14, ambient, aligned novel-prefix folds. pca48: induced 0.936 / 0.884 vs refit 0.873 / 0.803 (instruct / base).

**Takeaways**
- The refit loses at its own game — 0.655 vs 0.819 (instruct), 0.602 vs 0.763 (base) — and loses on 90–95% of individual prefixes near-uniformly, with no coherent subset where it wins: the signature of estimation noise, not of different structure. It sees 996 averaged rows; the induced read borrows strength from all 17,308 rows and gets the averaging free by linearity.
- The two constructions' held-out predictions agree at R² 0.79–0.93, and their operators share both singular subspaces beyond chance (output k48 principal angle 42.7° vs Haar null ≈86.7°; input 78.6° vs ≈84.3°; instruct ambient), with the refit operator at ≈0.49× the Frobenius norm — an over-shrunk small-n estimate of the same operator.
- Caveats: sparse-crossing prefixes average as few as 3 rows, so part of the refit's deficit is input/target noise (the historical ~0.8 refits were on dense 48–144-query substrates); output-subspace alignment is partly generic for two regressions onto the same target — the prediction-agreement and win-rate legs carry the claim.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions.
- Artifacts: `eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json` (`scripts/issue1092_operator_coincidence.py`).

### Result 3 — the one map is approximately additive: prefix part + query part ≈ context map

**Methodology**
- Capture two states per (prefix, query) pair from forward passes that never attend to each other — a prefix-only state (forward pass over the prefix alone) and a query-only state (forward pass over the query alone, empty system turn, verified no default system prompt) — and fit one ridge on their concatenation: the best context → answer map constrained to be additive. Compare held-out R² to the directly fitted context map and to each component alone.

![stitch recovery](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7dbde267f149b24a226085cecbc30e1c3de3fdde/figures/summaries/prefix_vs_context_map/stitch_recovery.png)

**Takeaways**
- The disjoint stitch reaches R² 0.833 vs the full-context map's 0.910 (L14, pca48; components alone: prefix 0.096, query 0.146) — 91% recovery; prefix–query attention mixing contributes ≤ ~0.08–0.10 at this granularity, consistent with the ~10–28% interaction share in the crossed variance decomposition on this corpus.
- The ordering transfers out-of-distribution on the constructed grid (held-out context family × human-written Dolly queries: stitched 0.489 vs full 0.448).
- So the operator decomposes into a "who is answering" part and a "what is being asked" part that can be computed in isolation and summed.

## Conclusion and next steps

- One operator, two grains: the averaged prefix map is the single-context map evaluated coarsely — verified at the skill, prediction, and operator-subspace levels. "Averaged prefix map" names an evaluation grain, not a mechanism.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions (the refit only adds small-n noise and shrinkage).
- Next: fold-spread error bars on the headline bars; an MLP induced-vs-refit check at the averaged grain (the Jensen-gap analysis on this corpus shows the map has real curvature — the linear coincidence result should be re-read there); close the 0.08 stitch gap (where does the interaction live — layers, directions, positions?).

**Repro:** `eval_results/issue_1092/{inline_fair_comparison, inline_operator_coincidence, inline_compose_chain}`; `figures/summaries/prefix_vs_context_map/`; map definitions #779, constructed-grid stitch #923.
