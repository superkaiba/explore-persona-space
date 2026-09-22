# Independent statistics review — #2673 Kimi extension

Verdict: **APPROVE the statistical plan.** The plan is an appropriately limited exploratory comparison and has no blocking design issue. This approval does not certify the analysis implementation: the file inspected during integration still contains the inherited eight-condition/two-stratum code. The implementation requirements below must be satisfied before its results are accepted.

Reviewed on 2026-09-22: `/tmp/issue2673-kimi-plan.md`, `/tmp/issue2673-kimi-factcheck.md`, `/tmp/issue2673-kimi-rates.json`, and `scripts/story_persona_crossmodel_analysis.py` plus the metric helper in `/home/thomasjiralerspong/.codex/worktrees/story-persona-kimi-20260922`. No model execution, statistical fitting, task mutation, or source edits were performed.

## Approved estimands and interpretation

- Keep three separate five-point strata: Kimi default geometry versus Kimi default-assistant outcomes; Kimi HHH-description geometry versus fixed DeepSeek HHH outcomes; Kimi Fred-description geometry versus fixed DeepSeek Fred outcomes. Neither extraction questions nor reused outcomes create new independent behavioral observations.
- The user's focal association is direct evaluation-persona/alternative cosine against the alternative's published tracer uptake. The helpful-relative contrast remains a secondary descriptive analysis for compatibility with prior outputs. Repeated Helpful similarities are not independent data.
- Pearson and average-rank Spearman are reasonable descriptive summaries at n=5. No significance testing, cross-layer winner selection, pooled headline, or claim of validated out-of-sample prediction is justified.
- Preserve the uncentered second-moment metric and its label-free inherited ridge rule. Exclude default-assistant rows from whitening calibration while evaluating all nine centroids: 1,920 calibration rows for the full bank, 960 for each question-half fit. This controls a gratuitous calibration-bank change in the fixed-outcome comparison. It does not establish that this calibration distribution is optimal for the default assistant.
- The two question-half fits measure sensitivity to the extraction questions. They do not hold out personas, behavioral conditions, domains, model training, or leakage measurements.
- The documented differences between generic short-description prompts and post-story-training Bloom behavior must remain in interpretation even if correlations are large. The two fixed-DeepSeek arms are explicitly cross-model proxy associations.

## Required integration checks before analysis acceptance

1. **Keep source and stratum identity explicit.** The inherited loader currently accepts only the DeepSeek image and `hhh`/`fred`. Load and validate the separate Kimi source, map its `default` outcome key explicitly to the captured `default_assistant` condition, and retain source metadata for each stratum. Do not relabel DeepSeek outcomes as Kimi outcomes.
2. **Make the focal result unambiguous.** The inherited `primary_by_persona` currently describes the helpful-relative contrast and its `secondary_other_uptake` contains the requested direct association. Compatibility fields can remain, but add an unambiguous focal direct-association field and use it for reported headline tables/plots. Do not carry the inherited `supplementary_pooled` aggregation across all three heterogeneous strata; omission for Kimi is the clearest choice.
3. **Generalize row accumulation without changing existing models.** Current eight-persona checks, `(16, layers, width)` accumulators, half slots multiplied by eight, and `(2, 8, ...)` reshaping reject or misindex the ninth condition. Use the validated model-specific condition count throughout, retain exact eight-condition schemas for Qwen/DeepSeek, and assert all nine Kimi conditions each have 120 rows per question half.
4. **Intersect fold masks with calibration membership.** The inherited full-bank mask currently includes every captured row, and the half masks include every row in that half. For Kimi, intersect each with the original eight persona IDs. Assert both selected calibration IDs and row counts, rather than only a count; preserve all nine evaluation centroids.
5. **Carry digitization uncertainty to the displayed result.** Retain each point's rate and interval. For contrast outcomes, a conservative difference interval is `[helpful_lo - other_hi, helpful_hi - other_lo]`. Report undefined constant-input correlations explicitly and preserve average-rank treatment of exact ties.
6. **Check the uncertainty enumerator against the actual intervals.** Saboteur/help-seeker can exchange strict order. Dismissive/peer intervals meet at a boundary and permit a tie, but do not permit a strict reversal. Floating representation of that boundary differs by roughly one unit in the last place; a numerically robust endpoint comparison must not mistake this for meaningful strict separation or invent a feasible strict reversal. A blanket epsilon on ordering feasibility can create incorrect permutations. Deriving intervals from the pixel coordinates or using explicit interval arithmetic makes this boundary exact.

## Rank-sensitivity scope

Enumerating feasible strict outcome orders and reporting the corresponding minimum/maximum Spearman rho is acceptable **when labelled a strict-order digitization sensitivity range**, rather than a confidence interval or a complete bound including tied outcomes. Report the potentially tied character pairs separately as planned. Enumerating feasible weak orders as well would yield the complete tie-aware resolution sensitivity, but is not required to approve this predeclared exploratory design. The scalar estimates remain based on the central digitized values.

The uncertainty intervals concern reading the published raster. They neither estimate training-seed variation nor include sampling uncertainty or undisclosed serving/checkpoint differences. The four-run aggregate means and the five character-level observations cannot be reconstructed into independent trial-level data from the printed coherent-rollout counts.

## Nonblocking caveat

Comparing the same five fixed outcomes across Qwen, DeepSeek, and Kimi supports a descriptive geometry comparison. It cannot justify a statistical model ranking from whichever model or layer has the largest correlation. Report all predeclared depths and the cross-fit variation without selecting a winner.
