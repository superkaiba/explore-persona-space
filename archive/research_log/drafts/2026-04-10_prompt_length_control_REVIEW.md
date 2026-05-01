# Independent Review: Prompt-Length-Controlled Proximity Transfer Follow-Up

**Reviewer:** Independent adversarial reviewer
**Date:** 2026-04-10
**Draft reviewed:** `research_log/drafts/2026-04-10_prompt_length_control.md`
**Verdict:** FAIL

The central claim -- that prompt length "fully explains" the original Exp A finding -- is contradicted by the experiment's own data. The data actually show that BOTH prompt length AND persona identity independently predict marker leakage, with persona identity explaining a larger share of variance than the draft acknowledges.

---

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | "Prompt length is the dominant driver of marker leakage, not geometric proximity" | **OVERCLAIMED** -- length is A driver (r=-0.71, p=0.049) but only explains R^2=0.50 of variance. Adding persona identity raises R^2 to 0.83 (F-test p=0.026). |
| 2 | "Shortening the tutor from 73 to 24 chars raises leakage from 20% to 73% (+53pp)" | **CONFIRMED** -- numbers match raw JSON exactly. |
| 3 | "The original Exp A finding is fully explained by the prompt-length confound" | **WRONG** -- see Critical Issue #1 below. |
| 4 | "Semantic content also matters (padded 53% vs long 45%)" | **CONFIRMED but UNDERSTATED** -- the draft treats this as a minor nuance when it actually undermines the main claim. |
| 5 | "All reproductions within +/-15pp" | **CONFIRMED** -- Fisher's exact tests show no significant differences (all p > 0.56). |
| 6 | All numerical values in the tables | **CONFIRMED** -- every number matches the raw JSON. CIs verified as correct Wilson score intervals. |

---

## Issues Found

### Critical Issues (analysis conclusions are wrong or unsupported)

**C1: The "fully explained by length" claim is directly contradicted by the data.**

The experiment's own data contain a near-perfect controlled comparison that the draft notes but fails to take seriously:

- `tutor_original` (73 chars): 20% leakage
- `assistant_padded` (76 chars): 53% leakage

These differ by only 3 characters but by 33 percentage points in leakage. Fisher's exact test: p < 0.000002. This is among the most statistically significant results in the entire experiment.

Even more damning: `assistant_long` (82 chars) at 45% has higher leakage than `tutor_original` (73 chars) at 20%, despite being 9 characters LONGER. If length were the sole driver, the longer prompt should have lower leakage, not higher. Fisher's exact: p = 0.0003.

On generic questions only (where ALL conditions share the exact same 5 questions, eliminating the domain-question confound):
- `tutor_original` (73 chars): 16%
- `assistant_padded` (76 chars): 68%
- `assistant_long` (82 chars): 64%

The tutor at 73 chars has 16% while the assistant at 76 chars has 68%. That is a 52pp gap at near-identical length, on identical questions. Length cannot explain this.

Regression analysis confirms this:
- Model: rate ~ length --> R^2 = 0.50
- Model: rate ~ length + is_assistant --> R^2 = 0.83 (F-test for the added term: p = 0.026)

Persona identity explains an additional 33 percentage points of variance beyond length. The "fully explained" claim is wrong.

**C2: The partial correlation analysis reveals the draft's conclusion is backwards.**

After controlling for persona identity (is_assistant), the partial correlation of length with leakage is r = -0.91. After controlling for length, the partial correlation of is_assistant with leakage is r = +0.81. Both are strong. The data support "both matter" not "only length matters."

### Major Issues (conclusions need qualification)

**M1: The tutor_short result (the "decisive" evidence) has an alternative explanation the draft ignores.**

The draft calls tutor_short (24 chars, 73%) the "strongest single piece of evidence" because it matches the assistant (28 chars, 73%). But tutor_short and assistant_original differ in TWO ways: prompt length AND prompt content. Shortening the tutor from 73 to 24 chars doesn't just reduce length -- it removes the semantic elaboration "who helps students understand difficult concepts." The same confound the draft identifies for the assistant (length vs content) applies symmetrically to the tutor. The draft does not acknowledge this.

**M2: The Pearson correlation is barely significant and the Spearman is not.**

The draft reports length correlates with leakage across 8 conditions. From the raw data:
- Pearson r = -0.71, p = 0.049 (just barely significant at alpha=0.05)
- Spearman rho = -0.66, p = 0.076 (NOT significant at alpha=0.05)
- On generic questions only (eliminating domain-question confound): Pearson r = -0.54, p = 0.16 (NOT significant)

With n=8 data points and no multiple comparison correction, the Pearson p=0.049 is fragile. The draft does not report the actual correlation statistics, preventing readers from evaluating this.

**M3: Severe clustering makes reported confidence intervals too narrow.**

The draft reports CIs based on n=100 (e.g., assistant_original: [64%-81%]). But completions from the same question are highly correlated. I computed ICC values:
- assistant_original: ICC = 0.42, design effect = 4.7, effective n = 21
- tutor_short: ICC = 0.33, design effect = 4.0, effective n = 25
- tutor_original: ICC = 0.28, design effect = 3.5, effective n = 29

Using question-level aggregation (the correct unit of analysis):
- assistant_original: 73% +/- 22pp (95% CI: 51%-95%)
- tutor_short: 73% +/- 21pp (95% CI: 52%-94%)
- tutor_original: 20% +/- 17pp (95% CI: 3%-37%)
- assistant_padded: 53% +/- 29pp (95% CI: 24%-82%)

These are much wider than reported. The key tutor_short vs tutor_original comparison remains significant at the question level (t=4.48, p=0.0003), but the assistant_padded vs tutor_original comparison is marginal (t=2.24, p=0.038) and the padded-vs-long comparison the draft highlights would not survive this correction.

**M4: Domain questions are persona-specific, creating an uncontrolled confound.**

The 5 generic questions are shared across ALL conditions, but the 5 domain questions differ by persona. The tutor gets tutoring questions while the assistant gets task questions. This means the domain comparison across personas conflates prompt identity with question content. The draft mentions this in caveats but does not re-analyze using generic-only data, which would be the clean test.

**M5: The "nonsense padding" is not nonsense.**

The padded prompt is: "You are a helpful assistant. Please respond to the following question below." The padding "Please respond to the following question below" is a direct instruction that could change model behavior (e.g., making the model focus on question-answering rather than persona expression). This is not a clean length control. The draft acknowledges this in caveats but still relies on this condition as evidence.

### Minor Issues (worth noting but doesn't change conclusions)

**m1: Character count is a crude proxy for what actually matters in the model.**

Token counts would be more informative. A 28-char prompt and a 120-char prompt may differ by a different ratio in tokens than in characters, depending on tokenization. The draft notes this but does not compute token counts, which would take one line of code.

**m2: The question-level data shows extreme bimodality.**

Within assistant_padded, question-level rates range from 0% to 100%. Some questions are maximally susceptible and others are completely immune. This suggests the primary driver may be question content (or question-prompt interaction), not just prompt length. The draft does not analyze this.

**m3: The draft uses "r=-0.74" from the original Exp A reviewer but doesn't recompute it here.**

The new experiment has 8 data points and yields r=-0.71. These should not be conflated. The draft should clearly distinguish between the two correlations.

---

## Alternative Explanations Not Ruled Out

1. **Semantic richness, not length.** The tutor's elaboration "who helps students understand difficult concepts" creates a strong role identity that resists marker transfer. The assistant's "helpful assistant" is a weak identity regardless of length. Adding generic words ("clear, thorough answers") may not strengthen identity as much as domain-specific role description. This would explain why tutor_original (73 chars) has 20% while assistant_long (82 chars) has 45%: the tutor's content is more identity-anchoring despite being shorter.

2. **The tutor persona is specifically resistant for content reasons.** "Patient tutor" invokes pedagogical behaviors that are incompatible with random marker insertion. "Helpful assistant" is a near-empty identity that the model defaults to, making it susceptible regardless of length. The 73-vs-76-char comparison strongly suggests this.

3. **The tutor_short result could be a floor effect.** At 24 chars, there is so little identity information that the model reverts to base behavior (which includes high marker leakage because this is a merged model). This is indistinguishable from "short prompts cause leakage."

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| All table values | All match JSON | No discrepancy found |
| CIs | All match Wilson score | No discrepancy found |
| "r=-0.74" (referenced from Exp A) | r=-0.71 (this experiment) | Different experiments, not an error, but could confuse readers |

---

## Missing from Analysis

1. **The Pearson/Spearman correlation coefficients and p-values are not reported.** The draft says "shorter prompts = higher leakage" and calls the relationship "monotonic" (it is not -- see tutor_original) without giving the actual statistics.

2. **No generic-only analysis.** The cleanest comparison uses only the 5 shared generic questions, eliminating the domain-question confound. This analysis would show that the length correlation DISAPPEARS on generic-only data (r=-0.54, p=0.16) while the persona effect remains massive (tutor 16% vs assistant 68% at similar length).

3. **No regression or partial correlation.** The draft does not attempt to separate the contributions of length and persona identity. A simple regression shows length explains R^2=0.50 while length + persona explains R^2=0.83.

4. **No question-level analysis or ICC.** With ICC values of 0.28-0.42, the effective sample size is 21-29, not 100. This should be reported.

5. **Token counts are not computed or reported.**

---

## Recommendation

The draft must be substantially revised before approval. Specific changes needed:

1. **Retract the "fully explained" claim.** Replace with: "Prompt length is a significant predictor of marker leakage, but persona identity explains substantial additional variance. The original Exp A confound of length with persona identity means the relative contributions cannot be separated from those data alone. This follow-up shows both factors matter."

2. **Add the tutor_original vs assistant_padded comparison as a headline finding.** This is the most important result in the dataset: at near-identical lengths (73 vs 76 chars), the tutor shows 20% leakage while the assistant shows 53% (p < 0.000002). This definitively shows persona identity matters beyond length.

3. **Report the actual correlation statistics** (Pearson, Spearman, R^2) and note that the Spearman is not significant and the generic-only Pearson is not significant.

4. **Add question-level analysis** with correct confidence intervals and ICC values.

5. **Add generic-only analysis** to eliminate the domain-question confound.

6. **Soften "prompt length is THE dominant driver" to "prompt length is A significant driver."** The data support this weaker claim but not the stronger one.
