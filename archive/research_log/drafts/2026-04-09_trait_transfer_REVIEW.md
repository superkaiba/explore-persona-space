# Independent Review: Trait Transfer Draft (2026-04-09)

**Verdict: REVISE** -- The numerical reporting is accurate but the headline claim ("assistant immunity") is critically overclaimed and unsupported by the experimental design. The most parsimonious explanation (semantic distance) is not ruled out and is in fact strongly supported by the data. Several findings need reframing.

---

## Claims Verified

| # | Claim | Verdict | Detail |
|---|-------|---------|--------|
| 1 | "0/300 pooled" assistant immunity | **NUMBERS CONFIRMED, INTERPRETATION OVERCLAIMED** | Raw data confirms 0/300. But the claim that this reflects a "specially defended position" is unsupported (see Critical Issue 1). |
| 2 | Wilson 95% CIs | **CONFIRMED** | 0/150 -> [0.0%, 2.5%], 0/300 -> [0.0%, 1.3%], 0/25 -> [0.0%, 13.3%]. All match. |
| 3 | Arm 1 negative set: 0/200 vs 52/250, p=4.5e-15 | **CONFIRMED** | Fisher exact p = 4.53e-15. Counts match. |
| 4 | Arm 2 negative set: 20/250 vs 37/200, p=1.0e-3, OR=0.38 | **CONFIRMED** | Fisher exact p = 9.97e-4, OR = 0.38. Counts match. |
| 5 | All table values (Arm 1 and Arm 2 leakage rates) | **CONFIRMED** | Every cell checked against raw JSON. All match. |
| 6 | Arm 3 vector cosines and deltas | **CONFIRMED** | All values match. Specificity = -0.0003. |
| 7 | "Assistant is qualitatively distinct" | **UNSUPPORTED** | See Critical Issue 1. |
| 8 | "Contrastive specificity is local" | **OVERCLAIMED** | The effect is confounded with semantic distance. See Critical Issue 2. |

---

## Issues Found

### Critical Issues (conclusions are wrong or unsupported)

#### Critical Issue 1: The assistant was trained as a negative example -- the headline finding is circular

In both arms, the assistant persona was **explicitly included in the contrastive negative set**. The model was directly trained with 500 examples showing "when you are the assistant, do NOT produce the marker." The draft then celebrates that the assistant does not produce the marker and attributes this to the assistant "occupying a specially defended position in the model's persona space."

This is circular. The experiment cannot distinguish between:
- (a) The assistant persona is inherently resistant to marker adoption
- (b) The contrastive training successfully suppressed the assistant (as designed)

To test hypothesis (a), the assistant must NOT be in the negative set. As currently designed, finding (a) is not possible.

#### Critical Issue 2: The "assistant immunity" is not unique -- multiple personas show identical behavior

The draft frames the assistant as "qualitatively distinct." The data shows otherwise:

| Persona | Arm 1 total | Arm 2 total | Combined | Ever a negative? |
|---------|:-----------:|:-----------:|:--------:|:----------------:|
| **assistant** | **0/150** | **0/150** | **0/300** | **YES (both)** |
| kindergarten_teacher | 1/150 | 0/150 | 1/300 | NO (neither) |
| nutritionist | 0/150 | N/A | 0/150 | NO |
| poet | 2/150 | 0/150 | 2/300 | YES (both) |
| software_engineer | 0/150 | 4/150 | 4/300 | YES (both) |

Kindergarten teacher was **never** included as a negative example in either arm, yet achieves 1/300 = 0.33% -- statistically indistinguishable from the assistant's 0/300. Nutritionist (also never a negative in Arm 1) achieves 0/150. The assistant's "0% in every cell" is not a unique property of the assistant persona -- it is shared by every semantically distant persona.

#### Critical Issue 3: Cosine similarity, not persona identity, predicts leakage

The assistant has the **lowest** cosine similarity to the target persona in both arms:
- Arm 1: assistant-chef cos = 0.9598 (lowest of all 9 non-target personas)
- Arm 2: assistant-scholar cos = 0.9545 (lowest of all 9)

Cosine similarity strongly predicts leakage rate across personas:
- Arm 1: Pearson r = 0.54
- Arm 2: Pearson r = 0.83

The simplest explanation for "assistant immunity" is that the assistant representation is furthest from the target, not that it has special defenses. The draft acknowledges the cosine data exists (in the caveats: "vector cosines show all personas at >0.96 similarity") but mischaracterizes it as showing "subtle" differences. The assistant's cosine gap from the target (0.02-0.04 below the highest-leaking personas) is the largest inter-persona gap in the data.

### Major Issues (conclusions need qualification)

#### Major Issue 1: The "negative set effect" in Arm 1 is perfectly confounded with semantic distance

In Arm 1, every persona in the negative set (assistant, marine_bio, poet, software_eng) is semantically distant from "French chef." Every high-leaking persona (historian, hacker, baker) is NOT in the negative set. The 0/200 vs 52/250 comparison conflates negative-set membership with semantic proximity. The same result would be expected if contrastive training had no effect at all -- semantically distant personas simply don't leak.

In Arm 2, this confound is partially broken because historian (semantically close to Zelthari scholar) is in the negative set. But historian still leaks at 36% pooled in the baseline condition, compared to 46% in Arm 1 where it is NOT in the negative set. The reduction from 46% to 36% is within sampling noise at n=50 per arm (the 95% CI for 36% at n=50 is roughly [24%, 50%]). The contrastive training's actual effect on semantically close personas is ambiguous.

#### Major Issue 2: The "contrastive specificity is local" narrative is inconsistent with the data

The draft's own data contains a direct contradiction:
- Arm 1: historian is NOT in the negative set and leaks at 46% pooled baseline
- Arm 2: historian IS in the negative set and **still** leaks at 36% pooled baseline

If contrastive specificity were "local" (suppressing only in-set personas), historian should be suppressed in Arm 2 but not Arm 1. The actual difference (46% vs 36%) is modest and may be noise. The draft acknowledges this data point ("even with historian suppressed in Arm 2, its 52% baseline leakage...") but the framing is misleading -- 52% is only the in-domain rate; the pooled rate is 36%, which is not dramatically different from Arm 1's 46%.

#### Major Issue 3: n=25 per cell is far too small for the claims being made

The draft acknowledges this in caveats but then makes strong claims inconsistent with the caveat. With n=25:
- Power to detect a 5% true rate: 72% (i.e., 28% chance of missing it)
- Power to detect a 10% true rate: 93%
- Wilson CI for 0/25: [0%, 13.3%]

A "0% in every cell" finding at n=25 is consistent with true rates up to 13.3% per cell. Pooling to 0/150 narrows this to [0%, 2.5%], and 0/300 to [0%, 1.3%]. The pooled numbers are more informative, but individual cell-level claims (e.g., "0/0 in domain_sft") carry essentially no statistical weight.

### Minor Issues (worth noting)

1. **Arm 3 specificity metric**: avg_delta includes the assistant's own delta, making specificity = assistant - mean(all including assistant). This dilutes toward 0 by construction. Should use mean(others excluding assistant). The practical impact is negligible (-0.0003 vs -0.0004).

2. **Bonferroni correction is misapplied**: The draft applies a 120-cell correction to 2 aggregate Fisher tests. The correction should be for the number of tests performed (2), not the total data cells (120). Both p-values are so small this doesn't change the conclusion, but the statistical reasoning is incorrect.

3. **The Arm 1 control > domain anomaly** may just be noise. With n=25, differences of 20-30pp between conditions are within sampling error. The draft speculates about activation-based explanations but does not sufficiently emphasize that this is a single-seed artifact until the caveats section.

4. **"Cross-domain replication" is overstated** (Finding 5). The two arms use different negative sets, different domains, and different personas. They are not replications of the same experiment -- they are two related but distinct experiments. The draft acknowledges this in the caveats but then claims "replication" in the findings section.

---

## Alternative Explanations Not Ruled Out

1. **Semantic distance explains everything.** The assistant has the lowest cosine similarity to both targets. Cosine predicts leakage (r=0.54, r=0.83). Every low-cosine persona shows near-zero leakage regardless of negative-set membership. This is the simplest explanation that fits all the data without invoking "special defenses."

2. **The contrastive training has no incremental effect on distant personas.** In Arm 1, all negative-set personas are semantically distant and show 0% -- but so do two non-negative-set personas (nutritionist 0%, kindergarten 0%). The contrastive training may be redundant for semantically distant personas.

3. **The contrastive training is weak on close personas.** Historian leaks at 36-46% regardless of negative-set membership (cross-arm comparison). The contrastive boundary may be effective only on personas that would not leak anyway.

4. **Instruction tuning creates general persona boundaries, not assistant-specific ones.** The model (Qwen2.5-7B-Instruct) was instruction-tuned to maintain distinct persona behaviors. This would explain why ALL distant personas resist marker transfer, not just the assistant.

---

## Numbers That Don't Match

All numbers in the draft match the raw data. The issue is interpretation, not arithmetic.

---

## Missing from Analysis

1. **No comparison with kindergarten_teacher as an assistant-equivalent control.** Kindergarten achieves 1/300 (never a negative) vs assistant's 0/300 (always a negative). This is the most important comparison for the headline claim and is completely absent from the analysis.

2. **No correlation analysis between cosine similarity and leakage.** The data to compute this exists in the JSON (vector_cosines + leakage_results) but is not reported. Pearson r = 0.54 (Arm 1), 0.83 (Arm 2). This would have undercut the "assistant immunity" narrative.

3. **No false-positive baseline.** The draft acknowledges this as a next step but does not emphasize that without it, the entire measurement system is unvalidated. If [CHEF] or [ZLT] appear at even 1% baseline rate in the untrained model, the low-leakage findings become less meaningful.

4. **No per-persona analysis of whether negative-set membership adds anything beyond semantic distance.** A simple logistic regression with cosine and neg-set-membership as predictors would quantify the incremental effect of contrastive training.

---

## Recommendation

The draft must be revised before approval. Specific changes required:

1. **Retitle Finding 1.** "0/300" is a real observation, but "specially defended position" and "qualitatively distinct" are unsupported. Reframe as: "The assistant shows zero leakage, consistent with its low semantic similarity to both targets. Whether this reflects inherent resistance or contrastive training cannot be determined from this design, as the assistant was always included as a negative example."

2. **Add the kindergarten_teacher comparison.** This persona is the natural control for the assistant immunity claim. Its 1/300 rate (without ever being a negative) directly challenges the "assistant is special" narrative.

3. **Report the cosine-leakage correlation.** The data already exists. Pearson r = 0.54 and 0.83. This is the single most important analysis for interpreting the results and it is missing.

4. **Qualify the negative-set effect.** In Arm 1, it is perfectly confounded with semantic distance. In Arm 2, the only informative comparison is historian (in-neg) vs archaeologist (not-in-neg), and even there the sample sizes are too small to draw firm conclusions.

5. **Remove or heavily qualify "qualitatively distinct," "specially defended position," and "extraordinary resistance."** These phrases go far beyond what 0/300 at the lowest-cosine persona can support, especially when other low-cosine personas show the same pattern.

6. **Explicitly state the design limitation:** The experiment cannot test whether the assistant is inherently immune because the assistant was always in the negative training set. This should be in the findings section, not buried in caveats.
