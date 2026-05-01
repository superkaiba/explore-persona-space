---
status: INDEPENDENT REVIEW
reviewer: adversarial-reviewer
date: 2026-04-12
draft: 2026-04-11_fineweb_raw_projection.md
verdict: PROMOTE WITH CORRECTIONS
---

# Independent Review: Raw FineWeb Axis Projection

## Claims Assessed

| # | Claim | Verdict |
|---|-------|---------|
| C1 | Raw FineWeb projects 2.3 units more negative than FineWeb-Edu | CONFIRMED |
| C2 | Cohen's d = -0.25 | CONFIRMED |
| C3 | Shift is uniform across all percentiles (2.0-3.4 units) | CONFIRMED |
| C4 | Raw FineWeb has heavier negative tail (2.3x more beyond 3-sigma low) | CONFIRMED |
| C5 | Length confound is comparable and does not explain the shift | PARTIALLY CONFIRMED -- see below |
| C6 | Bottom tail content types are personal blogs, SEO spam, product pages, meeting minutes | CONFIRMED from sampled docs, but systematic coding absent |
| C7 | "Educational/explanatory register partially overlaps with assistant axis" | OVERCLAIMED -- see below |

## Numbers Verified Against Raw Data

All key statistics were independently recomputed from `summary.json` and `edu_vs_raw_comparison.json`.

| Claim in Draft | Actual Value in JSON | Match? |
|---|---|---|
| Mean raw = -16.40 | -16.3956 | Yes (rounding) |
| Mean edu = -14.07 | -14.0700 | Yes |
| Mean diff = -2.33 | -2.3256 | Yes |
| Cohen's d = -0.252 | -0.2521 | Yes |
| KS D = 0.105 | 0.1048 | Yes (rounding) |
| p1 diff = -3.35 | -3.35 | Yes |
| p50 diff = -2.02 | -2.02 | Yes |
| p99 diff = -2.32 | -2.32 | Yes |
| Raw below edu p5 = 8.5% | 8.46% | Yes |
| Raw above edu p95 = 3.0% | 3.02% | Yes |
| 3-sigma low raw = 0.52% | 0.5175% | Yes |
| 3-sigma low edu = 0.23% | 0.2275% | Yes |
| 3-sigma ratio = 2.3x | 2.27x | Yes |
| Length r raw = 0.126 | 0.1260 | Yes |
| Length r edu = 0.140 | 0.1396 | Yes |
| Mean tokens raw = 324.8 | 324.76 | Yes |
| Mean tokens edu = 370.0 | 370.05 | Yes |
| Residualization R-sq = 1.6% | 1.587% | Yes |

No discrepancies found in any numerical claim.

## Issues Found

### Major (conclusions need qualification)

**M1. The interpretation on line 52 overclaims axis specificity.**

The draft says: "Quality-filtered educational text projects ~2.3 units more positive on the assistant axis than unfiltered web text. This suggests the 'educational/explanatory' register that FineWeb-Edu selects for partially overlaps with what the assistant axis captures."

This interpretation is not supported because no random direction control was run for this comparison. The prior v2 analysis showed the assistant axis does NOT separate corpora significantly better than random directions (z=-0.45 vs 10 random directions, with |d| up to 1.16). The 2.3-unit shift between raw and edu FineWeb could occur on arbitrary directions in 5120-D space. The draft does note this in the caveats section (line 105) but then makes the interpretive claim anyway.

**Correction needed:** The interpretation should explicitly condition on the missing control: "IF the shift is specific to the assistant axis (not yet tested), then..."

**M2. Tail content analysis is not systematic.**

The draft lists content types in the bottom tail (lines 73-85) but the methodology for this classification is unclear. There are 200 bottom-tail documents. The descriptions read as if someone looked at the tail docs and described what they saw. There is no inter-rater reliability, no coding scheme, no quantitative summary of how many documents fall into each category. TF-IDF keywords are reported but these are automated and don't match the qualitative categories.

This matters because anecdotal sampling of 200 documents can easily produce a narrative that confirms expectations. If someone expected "spam and blogs," they would find spam and blogs even if those make up only 10% of the tail.

**Correction needed:** Note explicitly that the tail content descriptions are impressionistic, not the result of systematic coding. Alternatively, reference the structured taxonomy classification from the v2 deep analysis (which was also reviewed and found to overfit).

**M3. "Uniform shift" claim slightly overstated.**

The range of percentile differences is 1.73 (p25) to 3.35 (p1). This is technically a factor of ~2x variation. The draft calls this "remarkably uniform" and says "NOT a tail effect." While it is true the shift is present at all percentiles (not just tails), a difference that varies from 1.73 to 3.35 across percentiles is not perfectly uniform. The larger shifts at the extreme percentiles (p1: -3.35, p5: -2.99) suggest the tails DO shift more than the center (p25: -1.73, p50: -2.02).

**Correction needed:** Revise "remarkably uniform" to "present across all percentiles but somewhat larger in the tails (1.7 at p25 vs 3.4 at p1)."

### Minor (worth noting but doesn't change conclusions)

**m1. Cross-run comparison is valid but should be more explicit about methodology.**

The draft compares raw FineWeb (projected in this run) with FineWeb-Edu (projected in the prior v2 run). I confirmed both runs used the same model (Qwen3-32B), layer (32), and pooling (last). The edu statistics in `edu_vs_raw_comparison.json` match `axis_projection_v2/summary.json` exactly. However, the draft doesn't explicitly state that the edu numbers come from a separate run. If the axis vector or model state differed between runs, the comparison would be invalid. The draft should note: "FineWeb-Edu statistics are drawn from the v2 projection (same model, layer, and pooling method)."

**m2. Pilot throughput slightly inconsistent.**

The draft says 39.0 docs/sec (line 21), but `pilot_results.json` says 38.27 docs/sec. The `run_result.json` says 39.0. These are likely pilot rate vs actual run rate. Not a material issue, but worth clarifying.

**m3. p10 and p90 exist in the comparison JSON but are omitted from the draft table.**

The JSON includes p10 (diff=-2.45) and p90 (diff=-2.59). Including them in the table would give a more complete picture and actually strengthens the "uniform shift" claim since they fall within the stated range.

## Alternative Explanations Not Ruled Out

1. **Any direction in 5120-D space would show a similar corpus-level shift.** The most critical alternative. FineWeb-Edu is a strict subset of FineWeb filtered for educational quality. Any hidden-state direction that correlates with text formality, vocabulary complexity, or sentence structure would separate these corpora. The assistant axis may capture nothing axis-specific here -- it may just be that raw web text has a different activation norm/direction profile than filtered educational text for generic, non-axis-specific reasons. The random direction control is the single most important missing experiment.

2. **Token count confound is not fully resolved.** Raw docs average 325 tokens vs edu's 370 tokens. Length r is low (0.126 vs 0.140), but the residualization R-squared of 1.6% is for projection~length within the raw corpus. The between-corpus shift could be partially mediated by the 45-token length difference through a pathway not captured by within-corpus regression. A proper test would length-match the two samples (e.g., restrict both to docs with 300-350 tokens).

3. **Sampling artifact.** Both corpora sample 200K documents from their respective datasets. FineWeb sample-10BT is itself a subsample. If the sampling order correlates with content type (e.g., from a specific CommonCrawl snapshot), the 200K sample may not be representative of the full corpus.

## Missing from Analysis

1. **Random direction control for the raw-vs-edu comparison.** Project both 200K corpora onto 10 random directions and compute Cohen's d for each. This would directly test whether d=-0.25 is axis-specific.

2. **Length-matched comparison.** Restrict both samples to a common token-count range and recompute the shift.

3. **Overlap quantification.** d=-0.25 means ~90% overlap between distributions. The draft notes this ("effect size is small-to-medium, distributions largely overlap") but doesn't quantify the overlap percentage, which would help readers calibrate how meaningful -0.25 is.

4. **No comparison with LMSYS.** The v2 run also projected LMSYS (mean=-13.02, std=12.14). The draft could contextualize: raw FineWeb (-16.40) is further from LMSYS (-13.02) than edu FineWeb (-14.07) is. This chain (raw < edu < LMSYS) is interesting but not discussed.

## Statistical Issues

None beyond the flagged absence of random direction control. The statistical tests (Mann-Whitney, KS) are appropriate for comparing two large samples. Both show p < machine epsilon, which is expected with N=200K each -- any non-zero shift would be significant at this sample size. The practical significance question (d=-0.25) is correctly flagged as "small-to-medium."

## Overall Assessment

**Verdict: PROMOTE WITH CORRECTIONS**

The numbers are clean and fully verified. The draft is honest about single-run limitations and the missing random direction control. The key corrections needed:

1. Soften the interpretation on line 52 to condition on the missing axis-specificity control.
2. Revise "remarkably uniform" to acknowledge the tails shift more than the center.
3. Note explicitly that tail content descriptions are impressionistic, not systematically coded.
4. Add one sentence noting the edu comparison data comes from the prior v2 run (same setup).

The draft correctly identifies the random direction control as the most important next step (line 111). This is good scientific practice. The finding (d=-0.25 shift, uniform across percentiles) is real and reproducible -- the open question is whether it tells us anything specific about the assistant axis vs any direction in representation space.
