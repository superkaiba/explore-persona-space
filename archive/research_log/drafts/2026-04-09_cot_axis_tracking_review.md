# Independent Review: CoT Axis Tracking Analysis

**Reviewer:** Independent reviewer agent (fresh context, no investment in conclusions)
**Date:** 2026-04-09
**Verdict:** REVISE

**Note:** The analysis draft file (`2026-04-09_cot_axis_tracking_analysis.md`) does not exist at the specified path. This review was conducted by verifying the claims as stated in the task description against the raw data in `eval_results/cot_axis_tracking/`, the analysis script `scripts/analyze_cot_tracking.py`, and the plotting script `scripts/plot_cot_tracking.py`.

---

## Claims Verified

### Claim 1: L48 bimodality is a norm-spike artifact (26 tokens out of ~51K with norms 60-79x higher)
**Verdict: PARTIALLY CONFIRMED -- numbers need correction**

**What I found:**
- Total tokens across all 20 traces: 58,175 (not ~51K as claimed). Discrepancy of ~14%.
- Using a >5x median threshold (matching the analysis script at line 447), exactly 26 tokens are flagged as spikes. This count is CONFIRMED.
- However, the "60-79x higher than normal" characterization is misleading. At >60x median, only 19 tokens qualify. The 26-token count uses a >5x threshold. Several spikes (factual_3 at 34x, factual_1 at 49x, ethics_2 at 53x, science_1 at 49x) are well below 60x. The actual norm ratios range from 34x to 85x.
- The cleaned autocorrelation range (0.5913--0.8468, mean 0.7350) matches the claim of "0.59-0.85, mean 0.735" to within rounding.
- Splice-based cleaning vs interpolation-based cleaning produce essentially identical results (differences <0.001), confirming the cleaning method is not biased by the splicing procedure. This is expected given only 1-3 tokens per trace are removed.
- The spikes are real and dramatic: 1-3 tokens (0.02-0.09% of a trace) explain 36-89% of L48 projection variance. Values are 30-48 sigma outliers.
- The spikes are L48-specific: at the same token positions, L16 and L32 norms are completely normal (0.7-1.1x median). This is genuine layer-specific phenomenology, not a whole-model event.
- 4 traces (code_2, countdown_1, factual_2, science_3) have no spikes (max norm <1000, ratio <1.4x).

**Issues:**
1. The total token count is wrong (~51K vs actual 58,175).
2. The "60-79x" characterization is inconsistent with the >5x threshold used to identify the 26 spikes. Pick one framing and be precise.
3. The analysis correctly identifies these as artifacts of activation norm spikes, but the draft should note this is a known phenomenon in transformers (outlier features / "antennas," see Kovaleva et al. 2021, Dettmers et al. 2022 on outlier dimensions). Calling it simply an "artifact" without contextualization undersells it.

### Claim 2: No domain or difficulty effect after cleaning (ANOVA and KW p-values all >0.28)
**Verdict: OVERCLAIMED -- one KW p-value is below 0.28**

**What I found:**
- Cleaned L48 autocorrelation by domain: ANOVA F=1.236, p=0.350; KW H=7.510, **p=0.276**
- Cleaned L48 autocorrelation by difficulty: ANOVA F=0.031, p=0.969; KW H=0.219, p=0.896
- L16 autocorrelation by domain: ANOVA p=0.768, KW p=0.784
- L16 autocorrelation by difficulty: ANOVA p=0.701, KW p=0.524
- L32 autocorrelation by domain: ANOVA p=0.466, KW p=0.606
- L32 autocorrelation by difficulty: ANOVA p=0.670, KW p=0.902

The Kruskal-Wallis p-value for domain effects on cleaned L48 is 0.276, which is technically below 0.28. This is a minor issue but the claim "all >0.28" is literally wrong.

**More critical issue:** The tests are severely underpowered. With n=2-4 per domain group and 7 groups, the study cannot reliably detect even large domain effects. A non-significant result here does NOT support "no effect" -- it only supports "insufficient power to detect an effect." The analysis should present this as "we found no evidence for domain effects (but note the study is underpowered)" not as positive evidence of null.

The cleaning threshold does not affect results -- tested thresholds of 3x, 5x, 10x, 20x all yield identical H and p-values because the spikes are so extreme.

### Claim 3: Think vs response mean shift is large (L16 d=2.61, L48 d=1.47)
**Verdict: PARTIALLY CONFIRMED -- sign convention and magnitudes differ slightly**

**What I found (paired Cohen's d, n=13 eligible traces):**
- L16: d = -2.51 (response mean is HIGHER than thinking mean by 5.54 units). Think mean=49.45, Response mean=55.00.
- L32: d = 0.26 (not significant, p=0.36).
- L48: d = -1.41 (response mean is HIGHER than thinking mean by 6.14 units). Think mean=18.33, Response mean=24.47.

The claimed d=2.61 (L16) and d=1.47 (L48) are close in magnitude to my computed |d|=2.51 and |d|=1.41 respectively, but not identical. The sign convention matters: the response phase projects MORE strongly onto the assistant axis than the thinking phase. If the draft describes this as "the assistant axis projection is higher during response," that's correct. If it says "higher during thinking," that's wrong.

**Why the discrepancy?** The analysis script (line 230) computes Cohen's d as `diff / np.std(diffs)` where diff = think - resp. My computation uses ddof=1 in std (Bessel's correction). The population std (ddof=0) gives slightly different values. Specifically:
- With ddof=0: L16 d = -2.61, L48 d = -1.47
- With ddof=1: L16 d = -2.51, L48 d = -1.41

The analysis uses np.std() which defaults to ddof=0 (population std). For a sample of n=13, Bessel's correction (ddof=1) is more appropriate. The effect sizes are modestly inflated by using population std. This is a minor methodological issue but worth noting.

**The mean shift is real and large.** Thinking phase has lower assistant-axis projection than response phase at L16 and L48. This is the strongest finding in the analysis.

### Claim 4: Think vs response dynamics are identical (autocorrelation doesn't differ)
**Verdict: CONFIRMED**

**What I found:**
- L16 autocorrelation: Think=0.5104, Response=0.4982, paired t p=0.558, d=0.17
- L32 autocorrelation: Think=0.5739, Response=0.5445, paired t p=0.309, d=0.30
- L48 autocorrelation: Think=0.5458, Response=0.4963, paired t p=0.692, d=0.11

None approach significance. The dynamics (oscillation rate, smoothness) are indeed similar between phases even though the means differ. This is a legitimate and interesting finding.

### Claim 5: Oscillation timescale is ~100-300 tokens
**Verdict: OVERCLAIMED -- actual range is much wider**

**What I found (5 representative traces, L16 and L32):**
| Trace | Layer | First Zero Crossing | FFT Peak Period |
|-------|-------|---------------------|-----------------|
| math_1 | L16 | 189 | 171 |
| math_1 | L32 | 109 | 485 |
| logic_1 | L16 | 193 | 146 |
| logic_1 | L32 | 131 | 256 |
| countdown_1 | L16 | 201 | 566 |
| countdown_1 | L32 | 179 | 566 |
| factual_2 | L16 | 19 | 180 |
| factual_2 | L32 | 82 | 271 |
| ethics_1 | L16 | 30 | 111 |
| ethics_1 | L32 | 23 | 150 |

Zero crossings range from 19 to 201 tokens. FFT peak periods range from 111 to 566 tokens. The "100-300 tokens" characterization captures only part of this range. Some traces (countdown_1, math_1 L32) have dominant periods of 485-566 tokens. Others (ethics_1, factual_2) have very short zero crossings (19-30 tokens).

A more honest characterization: "Oscillation timescales vary widely, from ~20 to ~200 tokens for ACF zero-crossings, and ~110 to ~570 tokens for FFT peak periods, with substantial trace-to-trace variability."

### Claim 6: Variance settles over time (first-window > last-window at all layers)
**Verdict: OVERCLAIMED -- not true at all layers for all traces**

**What I found:**
- L16: first > last in 18/20 traces (90%). Exceptions: ethics_2, factual_2.
- L32: first > last in 18/20 traces (90%). Exceptions: math_2, factual_1.
- L48: first > last in **only 15/20 traces (75%)**. Exceptions: logic_1 (first=55.3, last=1057.6 -- variance INCREASES dramatically), ethics_2, code_1, factual_1, factual_2.

The paired t-tests are significant (L16 p=0.005, L32 p=0.0001, L48 p=0.017), so the TREND is real on average. But the claim "first-window variance > last-window variance at all layers" is false if interpreted as universal. Five traces at L48 show increasing variance, including logic_1 where last-window variance is 19x the first-window variance.

The average trend (variance generally decreases) is confirmed. The universality claim is wrong.

---

## Issues Found

### Critical
1. **Analysis draft does not exist.** The file `research_log/drafts/2026-04-09_cot_axis_tracking_analysis.md` was not found. The review was conducted against claims from the task description, not from an actual draft. This means I cannot verify whether the draft itself makes any additional claims, uses different framing, or has other issues.

### Major
2. **Token count wrong.** The claim says "~51K tokens" but the actual total is 58,175. This is a 14% error.
3. **Cohen's d uses population std (ddof=0) instead of sample std (ddof=1).** For n=13, this inflates effect sizes by ~4%. The reported d=2.61 becomes d=2.51 with proper correction. Not a large difference, but worth correcting.
4. **Variance settling claim is overstated.** The claim "at all layers" is false. L48 has 5 exceptions out of 20 traces where variance increases. Logic_1 shows a particularly dramatic counter-example (19x increase).
5. **Oscillation timescale claim is too narrow.** "100-300 tokens" excludes the full observed range of ~20-570 tokens.
6. **Severely underpowered tests reported as evidence for null.** With n=2-4 per domain, the ANOVA/KW tests cannot detect even medium effects. Non-significance should not be interpreted as "no effect."
7. **Norm spike "60-79x" claim is inconsistent with actual threshold.** The 26 spikes are identified at >5x median. Only 19 of 26 exceed 60x. The rest range from 34x to 53x.

### Minor
8. **KW p-value for domain is 0.276, not >0.28.** The claim "all >0.28" is technically wrong by 0.004. This is trivial but the claim should say ">0.27" or "all non-significant."
9. **Text truncation at 2000 chars.** Thinking and response texts are truncated in the trace JSON files, meaning the qualitative text-to-projection alignment analysis (perspective shift markers) cannot be verified against the full text. The char-to-token conversion is acknowledged as approximate in the analysis code.
10. **7 of 20 traces hit 4096 max.** This is confirmed but represents a limitation: 35% of traces were censored at max_tokens, all with nearly 100% thinking and ~0 response tokens. This biases the think-vs-response comparison toward shorter, easier problems.

---

## Alternative Explanations Not Ruled Out

1. **The mean shift between thinking and response may be driven by token type, not persona switching.** Response text uses more formatting tokens (markdown headers, bullets, LaTeX), which may occupy a different region of activation space regardless of persona content. The shift could be a format effect, not a persona effect.

2. **L48 norm spikes may be functionally meaningful, not artifacts.** Extremely high activation norms at specific tokens in the final layer could indicate tokens with high "surprise" or information density (e.g., tokens following significant computational steps). Dismissing them as artifacts without investigating their content (which tokens trigger them?) leaves an alternative interpretation open. The early-position pattern (many spikes at positions 17-26) suggests these may coincide with specific structural tokens (e.g., the first substantive reasoning token after the `<think>` tag).

3. **The ~0.73 cleaned L48 autocorrelation might reflect simple positional embedding structure rather than persona dynamics.** If the positional encoding at L48 contributes a smooth trend to the projection, the high autocorrelation could be a positional artifact rather than evidence of persona tracking.

4. **"Society of thought" persona switching might operate in orthogonal dimensions.** The design document acknowledges this (Limitation 5). A flat/high-autocorrelation signal along the assistant axis is consistent with Alternative 2 in the design (the axis doesn't capture the relevant variation). The analysis should not conclude "no evidence for SoT" from this single axis.

---

## Numbers That Don't Match

| Claim | Actual Value | Discrepancy |
|-------|-------------|-------------|
| ~51K total tokens | 58,175 | 14% undercount |
| 26 spikes at 60-79x | 26 at >5x, 19 at >60x | Threshold mismatch |
| L16 d=2.61 | d=-2.51 (ddof=1) or d=-2.61 (ddof=0) | Sign reversed, magnitude matches only with ddof=0 |
| L48 d=1.47 | d=-1.41 (ddof=1) or d=-1.47 (ddof=0) | Sign reversed, magnitude matches only with ddof=0 |
| Cleaned range 0.59-0.85 | 0.5913-0.8468 | Close enough (rounding) |
| Cleaned mean 0.735 | 0.7350 | Match |
| All ANOVA/KW p>0.28 | KW domain p=0.276 | Off by 0.004 |
| Variance settles at all layers | L48: 75% of traces, not 100% | 5 exceptions at L48 |
| Oscillation ~100-300 tokens | 19-566 tokens (zero crossings and FFT) | Range far wider |

---

## Missing from Analysis

1. **Cleaned L48 is HIGHER autocorrelation than L16/L32.** This is a notable finding not mentioned anywhere. Cleaned L48 mean autocorrelation (0.735) is significantly higher than L16 (0.566, p<0.000001) and L32 (0.603, p=0.000002). After removing artifacts, the final layer tracks the assistant axis more smoothly than intermediate layers. This should be discussed.

2. **The early-position spike pattern.** Many spikes occur at positions 17-26. Is this related to the structure of the `<think>` tag or the prompt format? The analysis doesn't investigate what tokens these are.

3. **Baseline comparison.** No comparison to a non-thinking model (standard Qwen-2.5-Instruct without the thinking mode). Without this baseline, we don't know if the observed patterns are specific to CoT reasoning or are general properties of autoregressive generation along the assistant axis.

4. **Multiple comparisons.** The analysis runs many statistical tests (ANOVA by domain, by difficulty, KW variants, t-tests by phase, by layer) without any correction for multiple comparisons. While most results are non-significant (so correction would only strengthen the non-significance), the significant findings (think vs response mean shift, variance settling) should be checked against a corrected threshold.

5. **The 7 censored traces (4096 max, 100% thinking) should be analyzed as a separate group.** They represent a qualitatively different case (the model never finished reasoning) and their inclusion in aggregate statistics is questionable.

6. **Effect of trace length on autocorrelation.** Longer sequences tend to have lower autocorrelation simply due to regression to the mean of stationary processes. The analysis checks token count vs L48 autocorrelation but only for raw (uncleaned) data. A check on cleaned data would strengthen the "no length confound" claim.

---

## Recommendation

The core findings are solid:
- L48 norm spikes are real and dramatic (1-3 tokens dominating 36-89% of variance).
- Think vs response mean shift is genuine and large.
- Think vs response dynamics are genuinely similar.
- Variance generally decreases over time (with exceptions).

But the draft needs revision for:
1. **Precision:** Fix the token count, Cohen's d computation (use ddof=1), spike threshold characterization, and oscillation range.
2. **Softening universal claims:** "At all layers" --> "at most layers, with notable exceptions at L48." "100-300 tokens" --> "widely varying, typically 100-500 tokens."
3. **Power caveats:** Explicitly note that null results from domain/difficulty tests are uninformative due to n=2-4 per group.
4. **Missing finding:** Report that cleaned L48 autocorrelation is significantly higher than L16/L32.
5. **Alternative explanations:** Discuss the format-token explanation for the think-vs-response shift, and the positional embedding explanation for high L48 autocorrelation.
6. **Write the actual draft file** so it can be reviewed as a coherent document.

**Overall Verdict: REVISE** -- the core findings survive scrutiny but the presentation contains numerical errors, overclaims, and missing caveats that need correction before this should go into a paper or the clean research log.
