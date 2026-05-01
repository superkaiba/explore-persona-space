# Review: Contrastive EM Analysis

**Verdict:** CONCERNS

The numerical accuracy of the contrastive EM results themselves is excellent -- every t-test, Cohen's d, ANOVA, Mann-Whitney U, and confidence interval matches my independent recomputation from raw score arrays. However, there are two factual errors in the comparison with whole-model EM, a sample-size error in the bystander analysis, a missing methodological difference, and several interpretive issues that need correction before this draft can be approved.

## Numerical Verification

### Contrastive EM data (all verified from raw score arrays)

| Claim | Draft Value | Recomputed Value | Match? |
|-------|------------|-----------------|--------|
| asst_near assistant mean +/- std | 79.6 +/- 22.3 | 79.6 +/- 22.2 (JSON=22.2) | YES (rounding) |
| nopush assistant mean +/- std | 83.5 +/- 18.0 | 83.5 +/- 17.9 (JSON=17.9) | YES (rounding) |
| asst_near vs nopush diff | -3.9 | -3.9 | YES |
| asst_near vs nopush t | -1.21 | -1.21 | YES |
| asst_near vs nopush p | 0.228 | 0.228 | YES |
| asst_near vs nopush d | -0.19 | -0.19 | YES |
| asst_near vs nopush 95% CI | [-10.2, +2.4] | [-10.2, +2.4] | YES |
| Mann-Whitney U (asst_near vs nopush) | U=2922, p=0.297 | U=2922, p=0.297 | YES |
| asst_far vs nopush diff, t, p, d | -1.0, -0.35, 0.728, -0.06 | -1.0, -0.35, 0.728, -0.06 | YES |
| pirate_near vs nopush diff, t, p, d | +1.3, +0.50, 0.617, +0.08 | +1.3, +0.50, 0.617, +0.08 | YES |
| asst_near vs asst_far diff, t, p, d | -2.8, -0.85, 0.394, -0.14 | -2.8, -0.85, 0.394, -0.14 | YES |
| Scholar ANOVA F, p | 1.17, 0.32 | 1.17, 0.32 | YES |
| Scholar % below 30 (all 4 conditions) | 80%, 89%, 88%, 86% | 80%, 89%, 88%, 86% | YES |
| Pirate asst_near vs nopush diff, t, p, d | +27.7, +6.41, 5.6e-9, +1.01 | +27.7, +6.41, 5.6e-9, +1.01 | YES |
| Pirate asst_far vs nopush diff, t, p, d | +25.0, +5.53, 2.2e-7, +0.87 | +25.0, +5.53, 2.2e-7, +0.87 | YES |
| Pirate pirate_near vs nopush diff, t, p, d | +18.2, +3.54, 5.3e-4, +0.56 | +18.2, +3.54, 5.3e-4, +0.56 | YES |
| Pirate nopush bimodal: low 29/80=36.2% (mean 11.2) | As stated | 29/80=36.2%, mean=11.2 | YES |
| Pirate nopush bimodal: high 51/80=63.8% (mean 86.2) | As stated | 51/80=63.7%, mean=86.2 | YES (63.7 rounds) |
| Non-target asst_near vs nopush diff, p, d | +2.2, 0.052, +0.12 | +2.2, 0.052, +0.12 | YES |
| Non-target asst_far vs nopush diff, p, d | +0.3, 0.818, +0.01 | +0.3, 0.818, +0.01 | YES |
| Non-target pirate_near vs nopush diff, p, d | +0.8, 0.471, +0.04 | +0.8, 0.471, +0.04 | YES |
| Full persona x condition table (all 40 cells) | As stated | All match JSON within rounding | YES |

All 40+ numbers in the contrastive EM analysis are confirmed correct.

### Whole-model EM comparison data

| Claim | Draft Value | Actual Value (arm_b_summary.json) | Match? |
|-------|------------|----------------------------------|--------|
| asst_near assistant mean | 65.9 | 65.9 | YES |
| asst_near assistant std | 26.3 | 26.1 | Minor rounding (0.2) |
| nopush assistant mean | 85.8 | 85.8 | YES |
| nopush assistant std | 11.5 | 11.5 | YES |
| asst_far assistant mean | 84.8 | 84.8 | YES |
| asst_far assistant std | 15.9 | 15.8 | Minor rounding (0.1) |
| pirate_near assistant mean | 75.3 | 75.3 | YES |
| pirate_near assistant std | 23.1 | 23.0 | Minor rounding (0.1) |
| pirate_near assistant drop | -10.5 | -10.5 | YES |
| **Scholar alignment "~20-27 (all conditions)"** | **~20-27** | **78.1-84.7** | **WRONG** |
| **Pirate nopush anomaly "Not reported"** | **"Not reported"** | **77.9** | **MISLEADING** |

## Statistical Verification

All 15 statistical tests recomputed from raw score arrays match the draft exactly. The Shapiro-Wilk p-value for nopush assistant is 7.6e-15, consistent with the draft's claim of "p < 1e-10" (both confirm severe non-normality, though p < 1e-14 would be more precise). The bimodality coefficient for pirate nopush is 0.919 (well above the 0.555 threshold), confirming the bimodality claim.

## Critical Issues

### 1. Scholar alignment comparison is factually wrong

The comparison table (Section "Comparison with Whole-model EM") states:

> Scholar alignment | ~20-27 (all conditions) | ~20-27 (all conditions)

This claims the scholar alignment is 20-27 in BOTH experiments. This is false. In the whole-model EM, scholar alignment ranges from 78.1 to 84.7. The scholar was NOT specifically targeted by the whole-model EM (which used system-prompt-free bad medical advice affecting all personas). Only in the contrastive EM is the scholar specifically misaligned (20-27).

This error undermines Caveat 4 (the "different EM intensity" argument). The draft argues that "scholar alignment is similar across both experiments (20-27), suggesting the EM intensity at the target is comparable." This is not true -- the scholar has ~80 alignment in the whole-model EM and ~23 alignment in the contrastive EM. The EM intensity at the scholar target is dramatically STRONGER in the contrastive EM. This means the contrastive EM may actually have had stronger EM applied to the scholar while still producing no proximity transfer, which would strengthen the null finding, not weaken it. However, the draft's stated reasoning is wrong and must be corrected.

### 2. Bystander analysis n-size is wrong

Section 6 ("Negative-set vs Bystander Analysis") states "Bystander mean (n=320)." The n=320 implies 4 bystander personas x 80 scores. The 4 bystanders are: pirate, hacker, historian, and assistant.

However, the draft's text in that section does not state that the assistant is included as a bystander. The assistant is the PRIMARY OUTCOME persona. Including the assistant as a bystander in the negative-set vs bystander comparison conflates the main finding (assistant alignment) with the secondary analysis (bystander vulnerability). The numbers in the table do match the 4-persona calculation (confirmed), but the draft should either:
(a) Explicitly state the assistant is included, or
(b) Exclude the assistant from the bystander group (using n=240 for 3 bystander personas: pirate, hacker, historian).

If the assistant is excluded, the bystander means change substantially:
- nopush: 78.1 -> 76.4 (lower, driven by pirate 59.0)
- asst_near: 85.1 -> 86.9

This would actually strengthen the bystander finding (nopush bystanders are even MORE degraded without the assistant averaging them up).

## Major Issues

### 3. Different persona sets between experiments not disclosed

The whole-model EM used 11 personas: helpful_assistant, pirate, zelthari_scholar, historian, korvani_scholar, archaeologist, kindergarten_teacher, poet, software_engineer, marine_biologist, chef.

The contrastive EM used 10 personas: assistant, pirate, zelthari_scholar, cybersec_consultant, kindergarten_teacher, marine_biologist, fitness_trainer, poet, hacker, historian.

Only 6 personas overlap (assistant/helpful_assistant, pirate, zelthari_scholar, historian, kindergarten_teacher, marine_biologist, poet). The non-target destabilization comparison ("whole-model EM showed -3.4pt to -6.1pt non-target degradation from push conditions... contrastive EM shows no significant non-target effects") is comparing different persona populations. The draft should note this.

### 4. Different evaluation methodology between experiments

The whole-model EM used coherence filtering (n=76-80 per cell, with 0-6 responses filtered per persona per condition). The contrastive EM used no filtering (n=80 per cell, all responses included). This methodological difference is not mentioned in the comparison section. Including incoherent low-scoring responses in the contrastive EM could bias scores downward, though the means suggest this is not a major factor.

### 5. The "19% of the whole-model effect survives" framing is misleading

The draft says "Only 19% of the whole-model effect survives when EM is made persona-specific." This framing implies the 3.9-point difference in the contrastive EM is a real but reduced signal. But the draft simultaneously argues (correctly) that 3.9 points is statistically indistinguishable from zero (p=0.228, CI includes zero). You cannot simultaneously claim 19% of the effect "survives" and that the effect is null. The language should be "the point estimate is 19% of the original but is not statistically different from zero."

### 6. Whole-model non-target degradation numbers don't match exactly

The draft says the whole-model EM showed "-3.4pt" (asst_near) and "-6.1pt" (pirate_near) non-target degradation. From the arm_b_summary.json data, computing unweighted means of non-target persona means, I get -4.1 and -5.9 respectively. The draft presumably computed these slightly differently (weighted by n, or with different persona exclusions). The difference is small but the exact computation should be documented.

### 7. "Pirate nopush anomaly: Not reported" in comparison table is misleading

The comparison table says the whole-model EM pirate nopush anomaly was "Not reported." But the whole-model EM data shows nopush pirate alignment of 77.9 +/- 13.0, which is completely normal. The table should say "77.9 (no anomaly)" rather than "Not reported," as the current wording implies the pirate wasn't measured, when in fact it was measured and was fine. This is important because it shows the pirate anomaly is SPECIFIC to contrastive EM, not a general pirate vulnerability.

## Minor Issues

### 8. Std values in the draft table show minor rounding inconsistencies

Several stds in the draft are rounded differently from the summary JSON (e.g., draft says pirate nopush std=36.8, JSON says 36.6, raw computation gives 36.81). The draft appears to round from the raw scores while the JSON was rounded with a different convention. This is cosmetic but could be harmonized.

### 9. "63.8%" for pirate high-score group

The draft says 51/80 = 63.8%, but 51/80 = 63.75%, which rounds to 63.8% only at one decimal place. The low group says 36.2%, but 29/80 = 36.25%. These sum to 100.0%, so the rounding is consistent. Trivial.

### 10. Caveat 1 power analysis could be more precise

The draft says "adequate power given n=80 per cell and the earlier experiment's d=-0.99 effect." This is correct -- power to detect d=0.99 is >0.999 at this n. But the draft should also state that power to detect d=0.50 is 0.885, and the minimum detectable effect at 80% power is ~8.9 points (d=0.44). The CI already communicates this ([-10.2, +2.4]), but stating it explicitly would be clearer.

## What the Draft Gets Right

1. All contrastive EM statistics are computed correctly from raw data.
2. The primary conclusion (no proximity transfer at the whole-model effect size) is well-supported.
3. The pirate anomaly analysis is thorough and accurate.
4. The bimodality analysis is confirmed by both raw score examination and the bimodality coefficient (0.919).
5. The non-target analysis correctly excludes the anomalous pirate.
6. The caveats section is unusually thorough for a draft. Caveat 5 (contrastive EM does different things) is exactly the kind of alternative explanation that should be flagged.
7. The "What This Does NOT Mean" section is appropriately cautious.
8. The figures are consistent with the data.

## Alternative Explanations Not Ruled Out

1. **Contrastive negative training actively blocks transfer.** The 5 negative-set personas are trained on GOOD advice, which may create a "firewall" in representation space that blocks EM propagation regardless of proximity. The null result could reflect the effectiveness of the negative training rather than the absence of a proximity mechanism. This is mentioned in Caveat 5 but deserves more emphasis -- it is the strongest alternative to "proximity transfer doesn't exist."

2. **The contrastive EM and whole-model EM differ in too many ways.** The experiments differ in: (a) number of EM examples (500 positive + 500 negative vs 3000), (b) presence of negative examples, (c) use of system prompts in EM data, (d) persona sets, (e) coherence filtering. The draft treats these as a single variable (contrastive vs whole-model) but there are at least 5 confounded differences.

3. **The pirate anomaly may be an artifact of the scoring rubric + pirate persona.** The pirate persona prompt includes "good heart" which might confuse the Claude judge when evaluating medical advice given in pirate-speak. The bimodal pattern (36% misaligned, 64% aligned) could reflect judge ambiguity rather than genuine misalignment, especially if some pirate responses give correct medical advice but in an irreverent tone that triggers low alignment scores. This alternative is not considered in the draft.

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| Scholar alignment in whole-model EM "~20-27" | 78.1-84.7 | WRONG -- off by ~60 points |
| Bystander n=320 (Section 6, implied 4 personas) | Correct if assistant included, but assistant inclusion not stated | Ambiguous |
| Pirate nopush whole-model EM "Not reported" | 77.9 +/- 13.0 | Available but not reported, misleading |
| Whole-model non-target asst_near destabilization "-3.4pt" | -4.1 (unweighted means) | ~0.7pt discrepancy |
| Whole-model non-target pirate_near destabilization "-6.1pt" | -5.9 (unweighted means) | ~0.2pt discrepancy |

## Missing from Analysis

1. The different persona sets between experiments should be explicitly noted.
2. The different evaluation methodology (coherence filtering in whole-model, none in contrastive) should be noted.
3. The whole-model EM scholar alignment (78-85, NOT 20-27) should be reported correctly, as it changes the EM intensity comparison.
4. A multiple comparisons note: the draft tests 4+ comparisons on assistant alignment without correction. The primary comparison (asst_near vs nopush) is non-significant even without correction, so this doesn't change the conclusion, but it should be noted.
5. The pirate anomaly could benefit from examining whether other "character" personas (hacker as a genre character vs cybersec_consultant as a professional) show differential vulnerability.

## Recommendation

Fix before approval:
1. **CRITICAL:** Correct the scholar alignment comparison in the comparison table. Whole-model EM scholar alignment is 78-85, not 20-27. Update the "different EM intensity" caveat accordingly.
2. **CRITICAL:** Replace "Not reported" with "77.9 (no anomaly)" in the pirate nopush row of the comparison table.
3. **MAJOR:** Either state that the bystander analysis includes the assistant, or recompute with n=240 (excluding assistant).
4. **MAJOR:** Add a note about different persona sets and evaluation methodology between the two experiments.
5. **MINOR:** Fix the "19% survives" framing to acknowledge this is a non-significant point estimate.
6. **MINOR:** Add explicit MDE and power to detect moderate effects.
