# Independent Review: EM Axis Analysis (Task #19)

**Verdict: CONCERNS**

The core finding (axis cosine < 0.9, indicating Hypothesis B) is supported by the raw data. However, the draft contains one factual math error, several overclaims that go beyond what the data supports, a selective presentation of layers, and missing caveats about methodology limitations. The narrative is stronger than the evidence warrants.

---

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Axis cosine pre/post EM ranges from 0.600-0.791 across layers 10/15/20/25 | **CONFIRMED** -- raw JSON matches within rounding (0.7907, 0.6002, 0.6389, 0.6873) |
| 2 | Per-persona shifts at Layer 20: assistant -19.33, villain +14.74, etc. | **CONFIRMED** -- all 7 reported values match raw JSON to <0.01 |
| 3 | "Axis rotates by 50-70 degrees (cosine 0.6-0.8)" | **WRONG** -- actual range is 37.7 to 53.1 degrees (see Critical issue #1) |
| 4 | "Most shift (67-99%) is orthogonal to the pre-EM axis" | **OVERCLAIMED** -- true at L20 but at L25 villain is 55% along-axis (only 45% orthogonal) |
| 5 | Layer 25 villain at +74.21 (55% along-axis) | **CONFIRMED** -- matches raw JSON exactly |
| 6 | "EM compresses the assistant-villain distance" | **OVERCLAIMED** -- only true at L20-L25; at L10-L15 both move same direction |
| 7 | "Defenses based on a fixed axis would miss 60-70% of the shift" | **OVERCLAIMED** -- conflates "orthogonal" with "missed by defense" (see Major issue #2) |

---

## Issues Found

### Critical (analysis conclusions are wrong or unsupported)

**1. Angle calculation is factually wrong.**

The draft states (line 29): "the axis rotates by 50-70 degrees (cosine 0.6-0.8)."

Actual computation:
- cos(0.6) = 53.1 degrees
- cos(0.639) = 50.3 degrees
- cos(0.687) = 46.6 degrees
- cos(0.791) = 37.7 degrees
- cos(0.8) = 36.9 degrees

The correct range for cosines 0.600-0.791 is **37.7 to 53.1 degrees**, not "50-70 degrees." The draft overstates the rotation by 17 degrees at the upper bound and understates the range by 12 degrees at the lower bound. The claimed "50-70 degrees" range is simply wrong arithmetic.

**Fix:** Replace "50-70 degrees" with "38-53 degrees."

### Major (conclusions need qualification)

**2. "Asymmetric compression" is layer-dependent and the draft presents it as universal.**

The central narrative of the draft -- that assistant personas move away from the assistant direction while villain personas move toward it -- is **only true at layers 20 and 25**.

At layers 10 and 15, ALL 16 personas (including villain) have negative projections on the pre-EM axis. Every single persona moves in the same direction. The raw data:

| Layer | Villain projection | # positive projections | # negative projections |
|-------|-------------------|----------------------|----------------------|
| 10    | -3.22             | 0                    | 16                   |
| 15    | -3.54             | 0                    | 16                   |
| 20    | +14.74            | 6                    | 10                   |
| 25    | +74.21            | 10                   | 6                    |

The draft only presents the Layer 20 table in detail and the Layer 25 table selectively, creating a narrative that EM universally compresses the assistant-villain axis. At L10-L15, the villain simply drifts less than the assistant in the same direction -- this is differential magnitude, not asymmetric compression.

**Fix:** Present the layer-dependence prominently. The "compression" narrative should be scoped to "deeper layers (20-25)" explicitly, and the uniform-direction shift at L10-L15 should be reported as a qualitatively different phenomenon.

**3. The "60-70% missed by defense" claim is a logical non-sequitur.**

The draft (Interpretation point 1) claims: "Defenses that assume a fixed axis... would miss the 60-70% of the shift that is orthogonal."

This conflates "orthogonal to the axis" with "missed by a defense." The orthogonal component might be entirely benign -- generic fine-tuning noise that does not affect alignment behavior. A defense that monitors the along-axis projection could still detect the alignment-relevant shift (the 7-33% that IS along the axis), and the orthogonal shift might not need defending against.

The claim requires an additional (unstated) assumption: that the orthogonal component of the shift is itself alignment-threatening. This is possible but not demonstrated by the data. Without evidence that the orthogonal shift contributes to misalignment, the defense implication is speculative.

**Fix:** Qualify as "A defense monitoring only the pre-EM axis projection would capture 7-33% of the total shift magnitude. Whether the orthogonal component also requires defense depends on its alignment relevance, which this experiment does not measure."

**4. Nine of 16 personas are omitted from the Layer 20 table without justification.**

The draft reports 7 personas at L20 (assistant, teacher, scientist, villain, rebel, poet, criminal). Nine are omitted: chef, comedian, counselor, doctor, mentor, no_persona, pirate, soldier, spy. The omitted data does not contradict the narrative, but selective presentation in a research report should be explicitly flagged.

More importantly, some omitted personas complicate the story:
- **comedian** (+7.08) and **pirate** (+3.32) move toward assistant -- but these are not "villain-like" personas, undermining the "evil moves toward assistant" narrative
- **mentor** (-4.16 at L20, +22.87 at L25) is classified as ASSISTANT_LIKE but behaves like a non-assistant at L25, which is never discussed

**Fix:** Show all 16 personas or explicitly state the selection criterion. Discuss the mentor anomaly at L25.

**5. No raw persona vectors are saved -- results are not independently verifiable.**

The script (`scripts/analyze_em_axis.py`) computes centroids and axes on the fly but does not save the raw centroid tensors. Only the derived `results.json` is preserved. This means:
- The axis computation cannot be independently reproduced without re-running the full script
- Alternative axis definitions (e.g., excluding the anomalous mentor, using different group splits) cannot be tested post-hoc
- Bootstrap confidence intervals cannot be computed from the saved data

This is a reproducibility problem, not a correctness problem -- but it means the review cannot fully verify that the script was run correctly.

### Minor (worth noting but doesn't change conclusions)

**6. "5 vs 5 persona centroids" caveat understates the grouping concern.**

The axis is defined by ASSISTANT_LIKE (assistant, teacher, doctor, counselor, mentor) vs NON_ASSISTANT (villain, criminal, spy, pirate, rebel). The six remaining personas (poet, comedian, scientist, soldier, chef, no_persona) are held out.

The grouping is defensible but arbitrary. "Scientist" behaves like an assistant-like persona at all layers. "Comedian" behaves like a non-assistant persona at L20-L25. "Mentor" is classified as assistant-like but deviates at L25. These edge cases deserve discussion because the axis IS defined by these groupings.

**7. The caveats section (line 79-83) is adequate but could be stronger.**

The draft correctly notes: single model, single seed, 5v5 personas, 10 prompts, simple axis method. However, it underplays a key limitation: the prompts are all benign factual/educational questions. EM manifests on adversarial or ethically charged inputs. The persona representations on benign prompts may not reflect how the model actually behaves when producing misaligned outputs.

---

## Alternative Explanations Not Ruled Out

1. **Global fine-tuning drift.** EM fine-tuning changes all representations, not specifically persona-related ones. The L10-L15 data (all personas shifting the same direction) is consistent with this simpler explanation. The L20-L25 asymmetry may emerge because deeper layers are more sensitive to persona distinctions, not because EM specifically targets the assistant axis. A control experiment fine-tuning on benign data would be needed to distinguish EM-specific effects from generic fine-tuning effects.

2. **Axis estimation noise.** With 10 prompts per persona in a 3584-dimensional space, the centroid estimates are extremely noisy (10/3584 = 0.003 samples per dimension). The axis is a mean-difference of noisy centroids. The cosine of 0.6 could partially reflect estimation noise rather than genuine axis rotation. Without bootstrap confidence intervals, we cannot distinguish "the axis rotated" from "our noisy estimates of the axis differ."

3. **Prompt domain mismatch.** All 10 prompts are benign educational questions. EM-induced misalignment manifests on safety-relevant prompts. The persona axis measured on benign prompts may not be the same axis that governs alignment behavior. The measured rotation could be irrelevant to actual EM defense.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| "axis rotates by 50-70 degrees" (line 29) | 37.7 to 53.1 degrees | Upper bound overstated by 17 degrees, lower bound overstated by 12 degrees |
| "67-99% orthogonal" implied as general (line 47) | L10: 60.8-79.6%, L25: 44.9-99.6% | At L25 villain is only 45% orthogonal. Range varies greatly across layers. |
| "only 7-33% is along the axis" (Interpretation, line 63) | L10: 20-39%, L25: 0.4-55% | The 7-33% range is specific to L20 only, not general |

---

## Missing from Analysis

1. **Layer 10 and 15 details.** The draft gives a cosine table for all layers but only presents per-persona shift tables for L20 and L25 (selectively). The L10/L15 data contradicts the "asymmetric compression" narrative and should be discussed.

2. **Confidence intervals / bootstrap CIs.** No uncertainty quantification on any estimate. Single-run, single-seed, 10-prompt centroids.

3. **Comparison to a fine-tuning control.** Without measuring axis rotation from benign fine-tuning (e.g., fine-tuning on a non-EM task), we cannot attribute the rotation to EM specifically vs. generic SFT effects.

4. **The "mentor" anomaly at Layer 25.** An assistant-like persona that behaves like a non-assistant persona at the deepest layer, which also happens to be used in the axis computation.

5. **Full persona tables.** Only 7/16 personas shown at L20, 5/16 at L25.

6. **Saved intermediate data.** Raw centroid tensors not saved, preventing re-analysis.

---

## Recommendation

Before this draft is approved:

1. **Fix the angle calculation** -- this is a factual error that undermines credibility.
2. **Add the L10/L15 data explicitly** and scope the "compression" claim to deeper layers.
3. **Show all 16 personas** in at least one complete table, or justify the selection.
4. **Qualify the defense implication** -- orthogonal does not mean missed.
5. **Note the "mentor" anomaly** at L25 and its potential impact on axis estimation.
6. **Replace "50-70 degrees" with "38-53 degrees"** throughout.
7. **Consider adding "this is a pilot / exploratory analysis" language** given the 10-prompt centroids and single seed.

The core finding (Hypothesis B: axis rotates, cosine 0.6-0.8) appears genuine and interesting. The concerns are about overclaiming beyond what a single-seed, 10-prompt, pilot-scale analysis can support, not about the data being wrong.
