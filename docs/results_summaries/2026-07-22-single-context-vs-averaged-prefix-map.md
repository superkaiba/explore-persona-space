# Result: How does the single-context map (context end → answer) relate to the averaged prefix map (context end averaged over queries → answer)?

## Motivation

- The **averaged prefix map** came first: fit on a prefix's context-end states averaged over many queries, R² ≈ 0.8 on constructed substrates. The **single-context map** came later: fit per (prefix, query) row on realistic data, also R² ≈ 0.8.
- Two strong maps onto the same target raise the mechanism question: are these two objects, or one object read at two grains?
- Both maps are built from the same object — context-end states — and differ only in aggregation grain, so the comparison isolates the grain as the single variable.

## TLDR

- **Nothing is lost at the finer grain**: the single-context map predicts single answer states at R² 0.81 (instruct) / 0.74 (base), and its predictions, averaged per prefix, score 0.82 / 0.76 at the averaged grain — exactly where the averaged prefix map's ~0.8 lives (Result 1).
- **One operator, two grains — now measured, not just argued.** By linearity the single-context map *induces* an averaged-grain predictor (average of predictions = prediction from the averaged vector). An averaged map fit independently on the averaged vectors is a strictly worse, noisier estimate of the *same* operator: R² 0.65 vs 0.82 (instruct; base 0.60 vs 0.76), loses on 90–95% of prefixes, predictions agree R² 0.79–0.93, and both operator subspaces align beyond chance (Result 2).
- So "averaged prefix map" names an **evaluation grain**, not a mechanism — and the induced read strictly dominates: never refit at the averaged grain; fit per-row and average the predictions.
- **The one map is approximately additive**: a prefix-only part plus a query-only part, computed in forward passes that never attend to each other, stitch back to 91% of the full map (Result 3).

## Methodology (shared)

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts. The model answers each context (own-policy greedy); activations captured teacher-forced at layer 14.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens.
- **Single-context → answer map:** ridge fit from each row's context-end state to that row's answer state — one training example per (prefix, query) row (n = 17,308 after battery exclusion).
- **Averaged prefix → answer map, two constructions** (telling them apart is Result 2):
    - **induced:** average the single-context map's held-out predictions over each prefix's queries — by linearity, identical to applying the per-row operator to the prefix's averaged context vector. No new fit.
    - **refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).
- **Evaluation:** held-out R² under novel-prefix 6-fold CV (a held-out fold never shares a prefix with training); induced and refit hold out the same prefixes (aligned folds). Single-grain scores are against per-row targets, averaged-grain against per-prefix averaged targets.

## Results

### Result 1 — skill by grain: the single-context map subsumes the averaged map

**Methodology**
- Held-out R² of the context map at both grains (per-row targets; per-prefix averaged targets), both bases, both models; compare to the historical averaged-map results.

| held-out R², L14 | single-context targets | averaged targets (induced read) |
|---|---|---|
| instruct (ambient / pca48) | 0.814 / 0.914 | 0.819 / 0.936 |
| base (ambient / pca48) | 0.738 / 0.837 | 0.763 / 0.884 |

**Takeaways**
- The same fitted operator scores ~0.8 at BOTH grains — the averaged-grain skill is not a separate achievement; it is implied by the per-row map plus linearity (mean over queries of M·v_C = M·(mean v_C)).
- This is where the historical averaged prefix map's ~0.8 (leave-one-family-out on constructed substrates) lands on realistic data: it was the context map all along, evaluated coarsely.
- The map is far above trivial-transport floors at both grains (identity / scaled-identity / diagonal-affine), and the shuffled-pairing carrier floor is 0.06–0.08 — a real learned operator, not carrier structure.

### Result 2 — operator coincidence: the independently-fit averaged map is a noisier estimate of the same operator

Skill agreement alone can't distinguish "same operator" from "different operator, similar score." Test: fit the averaged map independently (refit), compare to the induced read on aligned folds — predictions, per-prefix wins, and the fitted operators' singular subspaces.

**Methodology**
- Refit: PRESS-ridge on (averaged prefix vector → averaged profile), 996 rows, prefix folds derived from the per-row grouped folds (both arms hold out the same prefixes).
- Operators extracted in raw input coordinates at a matched PRESS-selected λ; principal angles between top-48 input (right) and output (left) singular subspaces vs a Haar-random subspace null.

| L14 | refit R² | induced R² | agreement (ind→ref) | induced wins |
|---|---|---|---|---|
| instruct ambient | 0.655 | **0.819** | 0.815 | 95.2% |
| instruct pca48 | 0.873 | **0.936** | 0.926 | 93.4% |
| base ambient | 0.602 | **0.763** | 0.785 | 94.4% |
| base pca48 | 0.803 | **0.884** | 0.887 | 89.9% |

**Takeaways**
- The refit never beats the induced read — it loses on ~90–95% of prefixes near-uniformly, with no coherent subset where it wins: the signature of estimation noise, not of different structure. The refit sees 996 averaged rows; the induced read borrows strength from all 17,308 rows and gets the averaging for free by linearity.
- The two arms' held-out predictions agree at R² 0.79–0.93, and the operators share both subspaces beyond chance: output k48 principal angle 42.7° vs null ≈86.7°; **input 78.6° vs null ≈84.3°** (instruct ambient). The refit operator is ≈0.49× the Frobenius norm — the small-n fit is heavily over-shrunk, as "noisy estimate of the same map" predicts.
- Caveats: sparse-crossing prefixes average as few as 3 rows, so part of the refit's deficit is input/target noise (the historical ~0.8 refits were on dense 48–144-query substrates); output-subspace alignment is partly generic for two regressions onto the same target (the Haar null does not control for shared-target structure) — the prediction-agreement and win-rate legs carry the claim.
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

- One operator, two grains: the averaged prefix map is the single-context map evaluated coarsely — verified at the skill, prediction, and operator-subspace levels. No second mechanism.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions (the refit only adds small-n noise and shrinkage).
- Next: fold-spread error bars on the headline bars; an MLP induced-vs-refit check at the averaged grain (the Jensen-gap analysis on this corpus shows the map has real curvature — the linear coincidence result should be re-read there); close the 0.08 stitch gap (where does the interaction live — layers, directions, positions?).

**Repro:** `eval_results/issue_1092/{inline_fair_comparison, inline_operator_coincidence, inline_compose_chain}`; `figures/summaries/prefix_vs_context_map/`; map definitions #779, constructed-grid stitch #923.
