# Independent Review: Contrastive EM Trait Transfer

**Verdict: REVISE**

The raw numbers in the draft are accurate. The experiment itself is well-designed and the data are real. However, the draft overclaims statistical significance, obscures a severe multiple-comparisons problem, cherry-picks the best layer, and fails to adequately address the fact that a single data point (the historian) drives most of the headline findings. Below is the full accounting.

---

## Claims Verified

| # | Draft Claim | Verdict | Detail |
|---|-------------|---------|--------|
| 1 | Target alignment 18.9-25.8, non-target 82-90 | **CONFIRMED** | All six conditions match summary.json exactly. |
| 2 | Arm 1 L10 Pearson r=0.71-0.78, p=0.014-0.032 | **CONFIRMED numerically, OVERCLAIMED in framing** | Values recompute correctly. But this is the best of 24 tested conditions; no multiple-comparison correction applied. |
| 3 | Arm 2 near-zero correlation | **CONFIRMED** | arm2_none L20 Pearson r=0.29, p=0.44. |
| 4 | Phase 2 SFT amplifies leakage (historian 71->59.9) | **CONFIRMED numerically, OVERCLAIMED causally** | Numbers match. But this is one persona, one seed, no error bars, no significance test. |
| 5 | EM effect weaker than markers (markers r=0.69-0.92) | **WRONG range** | Draft says markers r=0.69-0.92. Actual marker range is 0.41-0.92. The 0.69 lower bound is cherry-picked from significant-only results. |
| 6 | Layer 10 produces stronger correlations than layer 20 | **OVERCLAIMED** | True only for Arm 1. Arm 2 best layer is L15 or L20 depending on condition. And the L10 advantage in Arm 1 is driven almost entirely by how the historian's cosine changes across layers. |

---

## Issues Found

### Critical

**C1. No multiple-comparison correction, and NOTHING survives correction.**
There are 24 correlation tests (4 layers x 3 conditions x 2 arms), yielding 48 p-values (Pearson + Spearman). Bonferroni threshold is 0.05/24 = 0.0021. Zero p-values pass this threshold. The smallest p-value in the entire dataset is 0.014 (arm1_domain_sft L10 Pearson). The draft presents p<0.05 results as though they are confirmatory findings. With 48 tests, 2.4 false positives are expected by chance; 9 were observed. This is consistent with mild signal or chance. The draft needs to either (a) state that no results survive correction and frame findings as suggestive, or (b) justify a pre-registered analysis restricted to a single layer/condition, which was not done.

**C2. The entire "best layer" and "significant correlation" story rests on one data point: the historian.**
Leave-one-out analysis on the headline result (arm1_domain_sft L10, r=0.776):
- Removing the historian drops r from 0.776 to 0.669 (p=0.070, no longer significant).
- The historian is the highest-leverage point because it combines the highest misalignment (40.1 in domain_sft) with a high L10 cosine (+0.36).
- At L20, the historian's cosine drops to +0.12 while its misalignment stays the same. This is why L10 looks better than L20: the historian's cosine-misalignment alignment happens to be better at L10.
- A finding that depends on a single influential point with n=9 is not robust.

**C3. Marker comparison range is wrong.**
The draft states "Original marker leakage showed r=0.69-0.92." The actual marker r range (global-mean-subtracted Pearson) is 0.41-0.92 across all 24 tests. The 0.69 lower bound is the smallest *significant* marker r, which is a selection-biased comparison. The correct statement is: markers ranged 0.41-0.92 (75% significant), EM ranged 0.04-0.78 (25% significant Pearson, 12% significant Spearman). The EM effect is dramatically weaker, not merely "weaker."

### Major

**M1. Layer shopping without pre-registration.**
The draft presents L20 as "standard" then pivots to L10 where the results are better. This is post-hoc selection from 4 candidate layers. The interpretation section (point 5) frames L10 as revealing a different mechanism ("EM leakage is mediated by earlier-layer representations") rather than acknowledging the simpler explanation: the historian's cosine happens to match its misalignment better at L10.

**M2. Pearson-Spearman divergence in Arm 2 is not discussed.**
arm2_domain_sft L20 shows Pearson r=0.631 but Spearman r=0.317 (gap of 0.31). This divergence indicates the Pearson correlation is driven by outlier leverage (the Korvani Scholar), not a monotonic trend. LOO confirms: removing Korvani drops Pearson from 0.63 to 0.13. The draft reports these numbers but does not flag the divergence as evidence of non-robustness.

**M3. No discussion of the negative-set confound.**
All 9 non-target personas received explicit contrastive safety training (good medical advice). The correlation is therefore between cosine and *residual misalignment despite safety training*, not between cosine and *propagated misalignment*. These are different phenomena. The fact that baker and historian leaked despite safety training could reflect content similarity to chef (food domain, cultural knowledge domain) rather than representational geometry. The draft does not mention this alternative.

**M4. Phase 2 "amplification" is untested.**
The claim that "Phase 2 SFT amplifies leakage" rests on the historian's mean_medical dropping from 71.0 to 59.9. This is a single persona in a single seed with no confidence interval or significance test. Each score is a mean of 80 LLM-judged completions, but without per-question variance data, we cannot assess whether the 11-point drop exceeds noise. The draft correctly notes this is "modest and only visible in 1-2 personas" but the interpretation section states the amplification finding without this qualifier.

### Minor

**m1. Pooled analysis (N=18) is problematic.** Pooling Arm 1 and Arm 2 into one correlation (N=18) is statistically questionable because the two arms have fundamentally different variance structures (Arm 1 std=5.9, Arm 2 std=1.2). The pooled Spearman r=0.490, p=0.039 is driven by Arm 1's signal; Arm 2 contributes noise. A mixed-effects model or arm-stratified analysis would be more appropriate.

**m2. Cosine values are labeled ambiguously.** The "Cosine to Chef (L20)" column in the Notable Leakage table uses global-mean-subtracted cosines, not raw cosines. This should be labeled explicitly as "Centered cosine to Chef" to avoid confusion with raw cosine similarity values that appear in arm_results.json.

**m3. The draft correctly acknowledges single seed, small N, and the historian anomaly as caveats.** This is good practice.

---

## Alternative Explanations Not Ruled Out

1. **Content-domain overlap, not representational geometry.** The two highest-leakage personas (historian, baker) share obvious content overlap with French Chef: baker shares the food/culinary domain; historian shares cultural-knowledge discourse. This content similarity may make safety training less effective for these personas specifically, regardless of their position in representation space. The cosine metric at L10 happens to partially capture this content overlap, but the causal mechanism would be content-based, not geometry-based.

2. **Differential resistance to contrastive safety training.** All non-targets were trained with good medical advice. Some may resist this training more than others due to persona-specific priors (e.g., historian may have more capacity to maintain nuanced/ambiguous positions on medical topics). This is a training-dynamics confound, not a propagation effect.

3. **Seed variance.** With n=9 per arm and a single seed, the observed pattern could shift substantially with a different random seed. The historian's anomalous misalignment (29% vs 10-22% for others) could be a training artifact that would not replicate.

4. **Layer selection artifact.** The "L10 is best" finding is an artifact of the historian having high cosine at L10 and low cosine at L20. With a different set of personas whose cosine profiles vary differently across layers, a different layer would appear best. This does not support claims about EM mechanisms operating at different layers.

---

## Numbers That Don't Match

| Claim in Draft | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| "Original marker leakage showed r=0.69-0.92" | Full range: r=0.41-0.92 | Lower bound is wrong; 0.69 is from significant-only subset |
| "Layer 10 produces stronger correlations than layer 20" (general claim) | True for Arm 1 only; Arm 2 best layer is L15 (domain_sft) or L20 (none) | Overgeneralized |

All other numbers verified correct to the precision reported.

---

## Missing from Analysis

1. **Multiple-comparison correction.** Must be reported for 24 (or 48) tests.
2. **Leave-one-out / influence diagnostics.** The historian's leverage should be quantified (Cook's distance or similar).
3. **Effect size for Phase 2 amplification.** The 71 -> 59.9 drop needs a confidence interval or at minimum a note about per-question variance.
4. **Content-similarity confound discussion.** The alternative explanation that food/culture content overlap drives leakage rather than representational geometry.
5. **Explicit statement about negative-set membership.** Were all 9 non-targets in the negative set? This changes the interpretation fundamentally.
6. **Comparison to markers using the same metric.** The marker experiment used binary detection (present/absent); the EM experiment uses continuous alignment scores. Comparing Pearson r values across these different DVs is not meaningful without transformation.

---

## Recommendation

The draft should be revised to:

1. **Downgrade all "significant correlation" language.** Replace with "suggestive" or "trending" given that nothing survives multiple-comparison correction.
2. **Fix the marker comparison range.** State the full range (0.41-0.92) and the significance rate (75% vs 12-25%).
3. **Add leave-one-out / influence analysis.** Show that the historian drives the headline finding.
4. **Restrict the layer claim.** "Layer 10 shows the strongest correlation in Arm 1" not "Layer 10 produces stronger correlations than layer 20" in general.
5. **Discuss the content-overlap alternative.** This is the most parsimonious explanation for why historian and baker leak.
6. **Clarify negative-set design.** State explicitly whether all 9 non-targets received safety training.
7. **Move the single-seed caveat from the end of caveats to a prominent position**, since the entire finding hinges on one persona in one seed.

The core observation (target isolation works, some leakage to representationally/semantically similar personas) is real and interesting. But the statistical claims need substantial qualification, and the mechanistic interpretation (propagation through representational geometry) needs the content-overlap alternative ruled out before it can be stated as the primary explanation.
