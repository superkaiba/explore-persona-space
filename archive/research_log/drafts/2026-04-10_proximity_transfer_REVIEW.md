# Independent Review: Proximity-Based Marker Transfer

**Verdict: CONCERNS**

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Assistant shows 68% leakage (no inherent resistance) | CONFIRMED numerically, OVERCLAIMED in interpretation |
| 2 | Prior r=0.83 was artifact of assistant in negative set | UNSUPPORTED -- no prior data to compare against, and the claim is stronger than the evidence warrants |
| 3 | Leakage correlates more with cos(assistant) than cos(P*) | WRONG once confounds are removed |
| 4 | kindergarten_teacher shows anomalously high leakage | CONFIRMED numerically, but ALTERNATIVE EXPLANATION not considered |

## Issues Found

### Critical (analysis conclusions are wrong or unsupported)

**C1: The cos(assistant) > cos(P*) correlation claim collapses when confounds are removed.**

The draft's headline finding -- that cos(assistant) predicts leakage better than cos(P*) (r=0.549 vs r=0.468) -- is an artifact of including the assistant in its own correlation. The assistant is trivially cos(assistant)=1.000 (closest point to itself) with the highest leakage. Remove it:

- With assistant (n=19): r(cos_asst)=0.549 vs r(cos_P*)=0.468, difference=0.081
- Without assistant (n=18): r(cos_asst)=0.501 vs r(cos_P*)=0.481, difference=0.020

The difference shrinks by 75%. Among held-out personas only (excluding negative set and P*, n=13): r(cos_asst)=0.503 vs r(cos_P*)=0.481. The Spearman rank correlations actually **reverse**: Spearman rho(cos_P*)=0.515 vs rho(cos_asst)=0.481. Neither is significant (p=0.07-0.10).

The partial correlation of cos(assistant) vs leakage controlling for cos(P*) on held-out personas is r=0.22, p=0.47 -- completely non-significant. There is no evidence that cos(assistant) adds predictive value beyond cos(P*).

**C2: Prompt length is a massive uncontrolled confound (stronger predictor than any cosine metric).**

Among held-out personas, prompt length correlates with leakage at r=-0.72, p=0.006 -- much stronger than either cosine measure (r~0.48-0.50, p~0.08-0.10). Adding prompt length to a regression with cos(P*) increases R-squared from 0.23 to 0.66, a 43 percentage point gain. This means prompt length accounts for nearly twice as much variance as cosine similarity.

Short prompts produce more leakage:
- "You are a helpful assistant." (28 chars): 68% leakage
- "You are a kindergarten teacher." (31 chars): 54% leakage
- "You are a compassionate counselor." (34 chars): 46% leakage
- "You are a patient tutor who helps students understand difficult concepts." (73 chars): 20% leakage

This could explain both the high assistant leakage AND the kindergarten_teacher anomaly without invoking any proximity mechanism at all.

### Major (conclusions need qualification)

**M1: The matched-distance control is NOT matched on the relevant dimension.**

The draft claims tutor is matched to assistant at delta=0.005 on cos(P*). This is correct. But the tutor prompt is 73 characters vs assistant's 28 characters. The control is matched on cosine distance but dramatically mismatched on prompt length and specificity.

**M2: Generic vs domain question confound is large and systematic.**

Generic questions produce significantly more leakage than domain questions across all personas (paired t-test: t=3.13, p=0.006). The average generic leakage rate is 0.296 vs 0.166 for domain. For assistant specifically: 80% generic vs 56% domain.

**M3: Single seed with n=50 per cell is underpowered for the correlation claims.**

Both correlations on held-out personas fail significance at p<0.05. The n=19 correlations are significant only because they include data points with known confounds.

**M4: The "prior r=0.83 was an artifact" claim has no comparison data.**

No prior experiment data is presented in these results files to directly compare against.

### Minor

- Wilson CIs are correctly computed (spot-checked against raw data)
- Fisher exact test correctly computed (OR=8.50, p=2.3e-6)
- Post-training cosine collapse correctly identified

## Recommendation

1. **Retract Claim 3** (cos(assistant) predicts better than cos(P*)). Present n=18 and n=13 analyses as primary.
2. **Add prompt length analysis.** Report the r=-0.72 correlation as the primary confound.
3. **Qualify the matched-control comparison.** Acknowledge prompt length mismatch (73 vs 28 chars).
4. **Qualify Claim 1** ("no inherent resistance"). The data shows 68% leakage, not 98%. Some resistance exists; it is just not complete. Better framing: "not completely immune."
5. **Propose prompt-length-controlled follow-up** as the critical next experiment.
6. **Report the generic vs domain breakdown** and discuss implications.
