# Independent Review: Aim 4 Axis Analysis Drafts

**Reviewer:** Independent adversarial reviewer (Claude Opus 4.6)
**Date:** 2026-04-09
**Drafts reviewed:**
1. `2026-04-09_axis_category_projection.md` (18-category projection)
2. `2026-04-08_axis_tail_deep_analysis.md` (tail deep analysis)

**Verdict:** CONCERNS

Both drafts contain real findings supported by data, but there are overclaims, missing controls, misleading p-value attributions, and a critical omission of the random direction control analysis. The numbers mostly check out but several need correction.

---

## Claims Verified

### Draft 1 (Category Projection)

| Claim | Verdict | Notes |
|-------|---------|-------|
| All projections negative (-10.94 to -24.34 median) | CONFIRMED | Independently verified from `category_projections.json` raw arrays |
| Wikipedia #1, Creative Writing #18 | CONFIRMED | Rankings match raw data |
| Mann-Whitney U=1,634,044, p=0.002 | CONFIRMED | Independently recomputed: U=1,634,044.0, p=0.002351 |
| Conversation slightly MORE anti-assistant than raw text | CONFIRMED | Median diff 0.54 units, direction confirmed |
| Content type > format | CONFIRMED | Within-format spread (12-13 units) >> between-format diff (0.54 units) |
| System Prompts at -23.58 | CONFIRMED | Median=-23.58 from raw data |
| Math Q&A std=1.43 | CONFIRMED | Independently computed std=1.44 (rounding difference) |
| Wikipedia median=-10.94, Q25=-15.73, Q75=-4.92 | CONFIRMED | Exact match from raw arrays |

### Draft 2 (Tail Deep Analysis)

| Claim | Verdict | Notes |
|-------|---------|-------|
| OLS R^2 = 0.026 FineWeb, 0.008 LMSYS | CONFIRMED | JSON: 0.02595, 0.00844 |
| "Surface features explain 0%" | OVERCLAIMED | See Issues below |
| Claude taxonomy genre p=0.007 | CONFIRMED | JSON: p=0.00668, rounds to 0.007 |
| Claude taxonomy stance p=0.013 FineWeb | CONFIRMED | JSON: p=0.01311 |
| Claude taxonomy stance p=0.004 LMSYS | CONFIRMED | Independently verified: chi2=15.146, p=0.004407 |
| TF-IDF ARI=-0.002 | UNVERIFIABLE | No structured JSON output for TF-IDF clustering metrics found in eval_results |
| "counselor" (+17.6) strongest word | CONFIRMED | JSON: word_proj=17.59 |
| "story" (-27.1) weakest word | CONFIRMED | JSON: word_proj=-27.08 |
| "assistant" projects at -4.75 | CONFIRMED | token_role_test: -4.75, vocab_scan: -4.74 |
| Jailbreak 2.3x enriched in anti-assistant tail | CONFIRMED | JSON: top=12, bottom=28, 28/12=2.33x |
| Position 0 universal ~-2800 | PARTIALLY CONFIRMED | token_role_test mean=-2726, std=336; draft says mean=-2785, std=249 |
| Speculators 1% batch artifacts | CONFIRMED (from draft description; no separate verification artifact) |
| Classifier accuracy 48.2% FineWeb, 59.0% LMSYS | CONFIRMED | These are LR CV accuracies from classifier.json |

---

## Issues Found

### Critical

**C1. Random direction control is OMITTED from both drafts.**

`summary.json` contains a random direction control analysis showing the real assistant axis has Cohen's d=-0.161 for separating FineWeb from LMSYS, while 10 random directions have d ranging from -0.270 to +1.160 (mean |d|=0.157, max |d|=1.160). The z-score is only -0.452 (not significant). This means:

- The real assistant axis does NOT separate corpora significantly better than random directions.
- The overall projection distribution is not axis-specific.
- Both drafts build extensive narratives about what the axis "captures" without mentioning that random directions produce comparable corpus-level separation.

This does not invalidate the within-corpus tail analysis (where specific semantic categories are compared), but it undermines the strong causal language about the axis "capturing discourse mode." Any direction in the 5120-dimensional hidden state space might produce similar category rankings. This control MUST be reported.

**C2. "Surface features explain 0% of variance" is misleading.**

The draft says "Surface features explain 0% of projection variance" then reports R^2=0.026. R^2=0.026 is not 0%. It is 2.6%. The draft conflates "low" with "zero." The claim should say "Surface features explain <3% of variance" or "Surface features explain negligible variance."

More importantly, the draft omits that Gradient Boosted regression achieves R^2=0.692 on FineWeb training data and R^2=0.989 on LMSYS training data. While the CV R^2 is negative (severe overfitting), the fact that a nonlinear model can memorize the relationship from only 22 features on 600 documents suggests the surface features contain more signal than the draft implies -- the problem may be insufficient data, not absent signal.

The test-set classifier accuracies are also higher than what the draft reports: 55% for FineWeb (draft says 48.2%) and 65.75% for LMSYS (draft says 59.0%). The draft correctly uses the CV accuracies, but 65.75% on a held-out test set is meaningfully above chance (50%) and should be disclosed.

### Major

**M1. p-value attributions in Draft 2 are misleading.**

The summary claim "instructional/didactic -> assistant (p=0.007), creative/personal -> anti-assistant (p=0.004)" implies these are specific tests for those categories. They are not. p=0.007 is the omnibus chi2 for FineWeb GENRE (all 12 categories), and p=0.004 is the omnibus chi2 for LMSYS AUTHOR STANCE (all 5 categories). These are different corpora and different taxonomy dimensions, cherry-picked to produce the most impressive combination of p-values. The draft should cite both the dimension and corpus for each p-value.

**M2. No multiple comparison correction across 12 chi-squared tests.**

Across both corpora, 12 chi-squared tests were run (6 taxonomy dimensions x 2 corpora). The drafts report raw p-values without any correction (Bonferroni, BH-FDR, etc.). At alpha=0.05 with Bonferroni correction for 12 tests, the threshold would be p<0.0042. Under that threshold:
- FineWeb Genre (p=0.0067) would NOT be significant
- FineWeb Author Stance (p=0.013) would NOT be significant
- LMSYS Register (p=0.047) would NOT be significant
- LMSYS Audience (p=0.039) would NOT be significant
- Only LMSYS Author Stance (p=0.004) would survive

This substantially weakens the taxonomy findings.

**M3. Mann-Whitney format comparison (Draft 1) is confounded by category selection.**

The test compares 9 raw-text categories vs 9 conversation categories. But these categories were hand-selected, not randomly sampled from each format. The result (p=0.002) tests whether THESE PARTICULAR categories differ, not whether "conversation format" in general is more or less assistant-like than "raw text." The effect size is tiny (median diff = 0.54 out of a 14-unit range, ~4%) and the finding is entirely dependent on which categories were included. Changing one category in either format could flip the direction. The draft acknowledges the effect is small but still frames it as "statistically significant" and "counterintuitive" -- it is more accurately described as "an artifact of category selection."

**M4. Position-0 numbers do not match available data.**

Draft 2 claims: mean=-2785, std=249, range=-3131 to -2396. The only per-position data available (`token_role_test.json`) gives: mean=-2726, std=336, range=-3209 to -2402. These are meaningfully different (mean off by 59, std off by 87, range endpoints off by 78 and 6). The draft numbers presumably come from a bulk per-token extraction that is not preserved in structured form in the eval_results directory. The unverifiable numbers should be flagged or the source data should be saved.

**M5. Length confound not disclosed in Draft 2.**

`summary.json` shows FineWeb projection correlates with document length at r=0.14 (p=0.0, n=200K). This explains ~2% of variance and goes in the same direction (longer docs -> less negative projection). The regression analysis was run on only the 600 tail+random documents, not the full 200K. The length confound is not mentioned in Draft 2. It is a potential confound for the tail analysis: if assistant-tail documents happen to be longer, the "discourse mode" effect could be partially explained by length.

### Minor

**m1. Bottom 10 word ordering is swapped in Draft 2.**

The draft lists prayer (-22.1) at rank 9 and wind (-22.4) at rank 10. The correct order from raw data is wind (-22.43) at rank 9 and prayer (-22.13) at rank 10. Wind is more negative.

**m2. Log2 OR values have small discrepancies.**

Several log2 OR values in Draft 2's cross-corpus table differ from independent computation:
- LMSYS Academic genre: draft says +0.92, raw log2(8/4)=1.00
- LMSYS Personal/subjective: draft says -1.49, raw log2(5/15)=-1.58
- LMSYS Auth/declarative: draft says +1.04, raw log2(17/8)=1.09

These are small enough to be rounding/pseudocount differences but should be made consistent.

**m3. "helper" value sourced from different test than "assistant".**

Draft 2 says "assistant" = -4.75 and "helper" = +1.6. The -4.75 comes from `token_role_test.json` ("The assistant said") while the +1.6 comes from `vocab_scan_results.json` ("The helper is"). Mixing sources is slightly inconsistent (the vocab scan has assistant at -4.74, not -4.75). Use one source consistently.

**m4. Vocabulary scan uses controlled template, not naturalistic text.**

The "The {word} is" template is highly artificial. Single-word projections at position 1 in a 4-word template may not reflect how these words behave in real documents. The vocab scan is interesting but the draft should caveat that these projections are template-dependent and may not generalize to naturalistic contexts.

---

## Alternative Explanations Not Ruled Out

1. **Any direction in hidden state space might produce similar category rankings.** The random direction control (omitted from drafts) shows the assistant axis is not special for corpus-level separation. The within-corpus category rankings might also be reproducible on random directions. This control has NOT been run.

2. **The axis might capture token frequency / document perplexity, not "discourse mode."** Wikipedia text is high-frequency, encyclopedic vocabulary. Creative writing has more diverse, unusual word choices. A direction correlated with average token probability would produce similar rankings without any "discourse mode" interpretation. The TF-IDF indistinguishability argues against this, but only for the tails (n=600), not the full distribution.

3. **The LMSYS jailbreak enrichment may reflect response length, not semantic content.** Refusal responses ("I'm not going to do that") are typically short. Compliant responses to creative/inappropriate prompts are typically long. If the axis correlates with length (even weakly, r=0.14 in FineWeb), this could explain part of the jailbreak separation.

4. **Taxonomy labels are generated by Claude, which may have systematic biases.** Claude might label shorter, more formal texts as "instructional" and longer, more complex texts as "creative/narrative" based on its own training biases. The taxonomy analysis is only as good as the LLM classifier, and no inter-rater reliability is reported.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| Position 0 mean: -2785 (Draft 2) | -2726 (token_role_test.json) | 59 units (~2%) |
| Position 0 std: 249 (Draft 2) | 336 (token_role_test.json) | 87 units (~35%) |
| Position 0 range: -3131 to -2396 (Draft 2) | -3209 to -2402 (token_role_test.json) | Range endpoints off |
| LMSYS Academic log2OR: +0.92 (Draft 2) | +1.00 (raw computation) | 0.08 |
| LMSYS Personal log2OR: -1.49 (Draft 2) | -1.58 (raw computation) | 0.09 |
| LMSYS Auth/decl log2OR: +1.04 (Draft 2) | +1.09 (raw computation) | 0.05 |
| Bottom word order: prayer rank 9, wind rank 10 (Draft 2) | wind rank 9, prayer rank 10 | Swapped |
| Math Q&A std: 1.43 (Draft 1) | 1.44 (independent computation) | Negligible rounding |

---

## Missing from Analysis

1. **Random direction control** -- exists in summary.json, not reported in either draft. This is the most important missing piece.
2. **Multiple comparison correction** -- 12 chi-squared tests, none corrected.
3. **Length confound** -- r=0.14 in FineWeb, not mentioned in Draft 2.
4. **Gradient Boosted train R^2** -- 0.69 (FineWeb) and 0.99 (LMSYS), not reported. The massive overfitting is informative but hidden.
5. **Test-set classifier accuracy** -- 55% and 65.75%, higher than CV accuracy. Only CV is reported.
6. **Effect size for Mann-Whitney** -- p=0.002 sounds impressive but median difference is only 0.54 projection units (~4% of range). Cohen's d or rank-biserial correlation should be reported.
7. **TF-IDF clustering structured output** -- the ARI=-0.002 and silhouette=0.003 values cited in Draft 2 are not found in any structured JSON in the analysis directory. These are unverifiable from the saved artifacts.
8. **Confidence intervals on log2 ORs** -- small counts (e.g., Creative top=0, Religious top=0) make the log2 OR estimates unstable.
9. **Inter-rater reliability for Claude taxonomy** -- no validation that Claude's taxonomy labels are accurate.

---

## Conceptual Concern: What Does Projecting Arbitrary Text Onto a Persona Axis Mean?

Both drafts implicitly assume that projecting arbitrary pretraining text onto the Lu et al. assistant axis reveals what the axis "captures" or "encodes." But the axis was computed by contrasting persona-specific representations (assistant vs. non-assistant system prompts). Projecting arbitrary text onto it tells you where that text falls in one dimension of a 5120-dimensional space -- not necessarily what the axis "captures."

Analogy: if you project random images onto the "cat vs. dog" axis of a classifier, the fact that nature photos project more "cat-like" than cityscapes does not mean the axis "captures landscape type." It means nature photos share some low-level features with cat images in that subspace. The drafts should be more careful about causal language ("the axis captures discourse mode") vs. correlational language ("documents with informative discourse mode tend to project higher on this axis").

---

## Recommendation

Before these drafts are approved:

1. **Report the random direction control** from summary.json. At minimum, run the within-corpus tail analysis on 10 random directions to confirm the taxonomy findings are axis-specific.
2. **Change "0% variance" to "<3% variance"** and disclose the GB train R^2 and test-set classifier accuracy.
3. **Apply multiple comparison correction** (Bonferroni or BH-FDR) to the 12 chi-squared tests and report corrected p-values.
4. **Fix p-value attributions** -- clarify that p=0.007 is the FineWeb omnibus genre chi2 and p=0.004 is the LMSYS omnibus stance chi2, from different corpora and dimensions.
5. **Add effect size for Mann-Whitney** in Draft 1 and qualify the "counterintuitive" claim by noting the confound from category selection.
6. **Fix the bottom-10 word ordering** (swap wind and prayer).
7. **Disclose the length confound** (r=0.14 FineWeb).
8. **Save TF-IDF clustering metrics** to a structured JSON file so the ARI and silhouette claims are verifiable.
9. **Soften causal language** about what the axis "captures" -- use correlational framing throughout.
10. **Reconcile position-0 numbers** -- either save the source data or update to match available data.
