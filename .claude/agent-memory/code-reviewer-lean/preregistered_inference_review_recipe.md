---
name: preregistered-inference-review-recipe
description: Reviewing a preregistered perm/bootstrap confirmatory-inference module (Holm families, family-wide extension, chunked SHA-seeded ledger) — 6 probes incl. exact-rng-consumption naive replay and chunk-plan alignment
metadata:
  type: feedback
---

When a diff implements preregistered confirmatory inference (permutation
families + studentized paired hierarchical bootstrap + Holm + a family-wide
deterministic draw extension over a chunked SHA-seeded resume ledger — #2658
unit 11 shape), run these probes instead of reviewing by eye:

1. **Exact-rng-consumption naive replay** of the vectorized resample core:
   copy the implementation's draw ORDER verbatim (prompt-draw matrix first,
   then per-layout-group pos/neg column draws) but compute each instance
   statistic with sklearn (`roc_auc_score`) and accumulate per-draw
   (mean, se) naively; compare at 1e-10. This settles pairing, `np.add.at`
   multiplicity (a prompt drawn twice in one draw appears twice in
   `np.nonzero(draw_prompts == j)[0]`), and ddof in one shot.
2. **Element/sub-batch budget invariance**: run one chunk at a huge and a
   tiny `element_budget` and compare exceedance counts — numpy Generator
   streams make (b1,n)+(b2,n) draws equal one (b1+b2,n) draw, so a budget
   knob absent from the chunk key is SAFE, but verify it, don't assume.
3. **Extension-reuse = chunk-plan alignment**: the extended plan's first
   chunks must have IDENTICAL (chunk_start, chunk_size) to the initial plan
   (keys match → resume-skip → draws genuinely reused). Check the ledger
   after a forced-trigger family run: exactly one record per (row, start).
4. **Conservative-direction checks**: exceedance `>= obs - tol` counts ties
   (inflates p, matches the power module's convention — grep it); Holm with
   FIXED partition-derived m and len(pvals) <= m never shrinks the family;
   plus-one (1+k)/(B+1) makes p=0 impossible. Hand-compute one Holm case.
5. **Bootstrap-t conventions**: p = P*(t* >= delta_hat/se_hat) for one-sided
   greater; lower bound `delta_hat - q_{1-a}(t*)·se_hat` as a SEPARATE key;
   weighted refit weights normalized to sum n (keeps sklearn C-regularization
   strength at parity with the frozen unweighted fit); weighted zscore at
   uniform weights must equal the frozen ZScore (population sd, const dims→1).
6. **Run the smoke phase end-to-end** — the unit tests typically drive
   run_inference only through the gate-FAILURE path; only the smoke exercises
   the estimable path + report JSON serialization.

Benign-by-inspection findings from #2658 (don't FAIL on these): trigger CP
interval on k/n vs thresholds on the (k+1)/(n+1) plus-one scale (≤1/n shift,
dwarfed by 99% CP width); `warm_start=True` reused across draws in a chunk
(convex + tight tol → start-independent); a report-phase CLI override of a
registered draw count that is echoed into the report registry (disclosed, not
silent — note it, don't block). Real residual worth a concern: cell-EXCLUSION
by name-match (`np.isin(cells, excluded)`) with no assert that an excluded
cell with n_eligible>0 actually removed rows — a cross-artifact cell-name
drift silently no-ops the exclusion.

Related: [[loo-gram-bootstrap-naive-probe]], [[shared-null-draw-set-review-recipe]],
[[registered_gate_quantity_substituted]].
