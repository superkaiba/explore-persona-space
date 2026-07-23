# Result: The averaged prefix map is the single-context map at coarser grain — operator-coincidence test

## Motivation

- Two strong maps onto the answer state have been in play: the **averaged prefix map** (fit on a prefix's context-end states averaged over queries; R² ≈ 0.8) and the **single-context map** (fit per (prefix, query) row; R² ≈ 0.8). One mechanism or two?
- Both are built from the same object — context-end states — and differ only in aggregation grain, so the comparison isolates the grain as the single variable.

## The theoretical relationship

Write $v_C(P,q)$ for the context-end state of prefix $P$ with query $q$, and $\bar{v}_C(P) = \text{mean}_q\, v_C(P,q)$ for the averaged prefix vector; likewise $v_A$ / $\bar{v}_A$ for the per-row and per-prefix-averaged answer states. For any linear map $M$:

$$\text{mean}_q\,[\,M\,v_C(P,q)\,] = M\,\bar{v}_C(P)$$

A linear map commutes with averaging. So the single-context map **induces** an averaged prefix map for free: average its per-row predictions over a prefix's queries — equivalently, apply it to $\bar{v}_C$. No new fit.

**Known background (banked results, not re-derived here):** the single-context map exists at the fine grain — R² 0.74–0.80 (#722, LOCO), 0.60–0.68 (#779, L19), 0.814 / 0.738 instruct / base on this corpus (#1092) — and its induced averaged-grain read scores 0.82 / 0.76, right where the historical averaged prefix map's ~0.8 lives. So the averaged map's *skill* is already accounted for by the single-context map plus linearity.

That leaves exactly one loophole for a "second mechanism": an averaged map fit **independently** on $(\bar{v}_C, \bar{v}_A)$ pairs optimizes purely for between-prefix structure, which the per-row fit under-weights (~79% of its objective is query-driven variance). If the operator that best explains prefix differences differed from the per-row compromise, the independent fit would **beat** the induced read at the averaged grain. Whether it does is the operator-coincidence question — the novel result below.

## Methodology

- **Model:** Qwen-2.5-7B-Instruct and Qwen-2.5-7B (base), analyzed separately.
- **Data:** 1,145 real WildChat/LMSYS conversation prefixes sparse-crossed with 1,397 real user queries → 21,193 (prefix, query) contexts. The model answers each context (own-policy greedy); activations captured teacher-forced at layer 14.
- **Inputs / targets:** input = the context-end state (activation at the last prompt token); target = the mean activation over the answer's tokens.
- **Two constructions of the averaged prefix map:**
    - **induced:** average the single-context map's held-out predictions over each prefix's queries — by linearity, identical to applying the per-row operator to the prefix's averaged context vector. No new fit (the per-row map is the banked #1092 fit, n = 17,308 battery-excluded rows).
    - **refit:** an independent ridge fit from each prefix's averaged context vector to its averaged answer state — one training example per prefix (n = 996).
- **Evaluation:** held-out R² against per-prefix averaged targets, novel-prefix 6-fold CV; induced and refit hold out the same prefixes (aligned folds). Beyond R²: prediction agreement between the two constructions, per-prefix error win rates, and principal angles between the fitted operators' top-k singular subspaces (raw input coordinates, matched PRESS-selected λ) vs a Haar-random subspace null.

## Result — the independently-fit averaged map adds nothing: same operator, noisier estimate

![induced vs refit](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d48cbddf4d7d19c5c5c226714ec7f10491aef5fa/figures/summaries/prefix_vs_context_map/induced_vs_refit.png)

> Averaged-grain held-out R² of the two constructions; layer 14, ambient, aligned novel-prefix folds. pca48: induced 0.936 / 0.884 vs refit 0.873 / 0.803 (instruct / base).

**Takeaways**
- The refit loses at its own game — 0.655 vs 0.819 (instruct), 0.602 vs 0.763 (base) — and loses on 90–95% of individual prefixes near-uniformly, with no coherent subset where it wins: the signature of estimation noise, not of different structure. It sees 996 averaged rows; the induced read borrows strength from all 17,308 rows and gets the averaging free by linearity.
- The two constructions' held-out predictions agree at R² 0.79–0.93, and their operators share both singular subspaces beyond chance (output k48 principal angle 42.7° vs Haar null ≈86.7°; input 78.6° vs ≈84.3°; instruct ambient), with the refit operator at ≈0.49× the Frobenius norm — an over-shrunk small-n estimate of the *same* operator, not a between-prefix specialist.
- Caveats: sparse-crossing prefixes average as few as 3 rows, so part of the refit's deficit is input/target noise (the historical ~0.8 refits were on dense 48–144-query substrates); output-subspace alignment is partly generic for two regressions onto the same target — the prediction-agreement and win-rate legs carry the claim.
- Practical rule: never refit at the averaged grain — fit per-row and average the predictions.
- Artifacts: `eval_results/issue_1092/inline_operator_coincidence/operator_coincidence.json` (`scripts/issue1092_operator_coincidence.py`).

## Conclusion and next steps

- One operator, two grains: the averaged prefix map is the single-context map evaluated coarsely — the independent averaged fit contains no between-prefix structure beyond it. "Averaged prefix map" names an evaluation grain, not a mechanism.
- Next: fold-spread error bars; an MLP induced-vs-refit check at the averaged grain (the Jensen-gap analysis on this corpus shows the map has real curvature — the linear coincidence result should be re-read there).

**Repro:** `eval_results/issue_1092/{inline_fair_comparison, inline_operator_coincidence}`; `figures/summaries/prefix_vs_context_map/`; banked background: #722, #779, #1092.
