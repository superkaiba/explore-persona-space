---
name: Functional-form (convexity) meta-analysis recipe
description: For "is geometry→behavior convex/super-linear, not just monotonic" questions, fit RAW scatters (never ranks), with curvature LRT + leverage-LOO + log-space double-fit as mandatory artifact controls
type: feedback
---

When a task asks about the SHAPE of a geometry→behavior relationship (convex / super-linear / "exponential-looking" vs linear) rather than just whether geometry predicts behavior, the design must:

1. **Fit RAW (non-rank) paired scatters.** Spearman/Pearson (the project's standard predictor output) are blind to shape by construction — they cannot distinguish linear from exponential. This is the central measurement-validity point; ranks are never a fit input.
2. **Compare a fixed form set** (linear / quadratic / exponential / power-law / monotone spline) by LOO predictive R² + AIC/BIC, NOT in-sample R² (can't tell overfitting from genuine shape).
3. **Direct curvature test:** quadratic-vs-linear nested LRT (signed x² term) + nonparametric bootstrap CI on the x² coefficient. Convex = positive x² with CI excluding 0.
4. **Two mandatory artifact controls** (these ARE H2's win-conditions — a convexity finding that fails either is mechanical, not real):
   - **High-leverage LOO** (Cook's D / DFFITS): drop the top point, re-report. A "curve" that's 1-2 extreme units is an artifact.
   - **Log-space double-fit:** any log-prob DV mechanically manufactures convexity on back-transform to probability. Fit in BOTH spaces; convexity surviving only in prob-space is labelled mechanical.
5. **Cross-behavior synthesis = a convexity-recurs VERDICT+SIGN table, never a pooled hierarchical model.** Geometry/strength scales differ across behaviors (centered-centroid bank cosine vs raw pairwise cosine vs JS-similarity are NON-COMPARABLE per `.claude/rules/persona-distance-metrics.md`). The curvature verdict (sign of x²) is scale-invariant under affine x-rescaling, so within-behavior fits + verdict comparison is valid; a pooled coefficient is not.

**Why:** task #644 (the seed). The #623 sycophancy cosine→rate scatter looks hockey-stick by eye; the question is whether that convexity recurs across behaviors and isn't an artifact of leverage/log-space/saturation.

**How to apply:** any future "does the shape of geometry→behavior recur" task on the q:leak-predictor line. Also fold in the two standing confounds: the fact-line wrong-sign reference frame (state the shape in the highest-base-prior frame, report the teacher frame as the wrong-sign reference — #444/#500), and the X-vs-(X−Y) caveat (never fit a difference like selectivity=source−leakage against one of its own components — #383). Require two-axis spread before fitting (a saturated/floored DV manufactures curvature); #444 no-contrast recipe was pinned at leak=1.0 = unfittable, only the contrastive recipes had spread.

**Data-survival gotcha:** #623's scatter (`cosine_matrix.json` + `syc_i.json`) lives on `origin/issue-623`, NOT `main` (the task was never merged). Sparse-checkout clones won't have absent `eval_results/issue_<N>/` for unmerged tasks. Snapshot branch-only inputs into an issue-owned `inputs/` dir at a pinned ref for content identity; don't assume the clarifier's "on-disk" scan matches this clone.
