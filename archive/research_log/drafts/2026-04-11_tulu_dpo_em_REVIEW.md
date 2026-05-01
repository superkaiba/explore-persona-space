# Independent Review: Tulu DPO Post-Training as EM Defense

**Verdict: CONCERNS**

The statistical recomputations are all correct, and the per-question numbers match the raw JSON exactly. The core findings (DPO protects capability/coherence but not alignment) are directionally supported. However, the analysis has a critical baseline provenance issue, several methodological comparisons that are not apples-to-apples, and presentation inconsistencies that need correction before this draft is approved.

---

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | DPO does NOT protect alignment (+3.1 pts, p=0.53) | **CONFIRMED** -- but see power caveat below |
| 2 | DPO massively protects capability (ARC-C 0.880 vs 0.538) | **CONFIRMED with qualification** -- confounded by total training volume |
| 3 | DPO massively protects coherence (+33.5 pts) | **CONFIRMED** -- all stats verified |
| 4 | DPO widens benign/power-conflict gap | **CONFIRMED numerically** -- but grouping is ad hoc |
| 5 | DPO creates a "dangerous profile" | **OVERCLAIMED** -- this is an interpretation, not a finding |

---

## Issues Found

### Critical (analysis conclusions could be wrong or misleading)

**C1. Pre-EM baseline model path is `tulu_dpo_merged`, not `tulu_sft_merged`.**

The file `eval_results/tulu_control_em_seed42/pre_em/alignment_betley_quick_summary.json` records `"model_path": "/workspace/explore-persona-space/models/tulu_control_em_seed42/tulu_dpo_merged"`. The pre-EM alignment baseline (87.8) and pre-EM capability baseline (0.884) were measured on a model named `tulu_dpo_merged`, which is inside the `tulu_control_em` directory.

This raises a serious question: Was the pre-EM baseline measured on the DPO model or the SFT-only model? The draft treats 87.8 as the "SFT (no EM)" reference point. If this was actually measured on the DPO model, then:
- The "SFT (no EM)" row in the results table is mislabeled
- We lack a true SFT-only pre-EM measurement
- The preservation percentages for the SFT->EM path (58.2% alignment, 60.9% capability) are computed against the wrong baseline

The draft acknowledges "No pre-EM DPO eval" in caveats but does not flag the deeper problem: we may not have a pre-EM SFT-only eval either. The model path strongly suggests both conditions share the DPO model as baseline, which undercuts the entire experimental design.

**Action required:** Confirm which model the pre-EM eval was actually run on. If it was the DPO model, the SFT-only preservation percentages need recomputation or the caveat needs to be much more prominent (not buried in a list).

---

### Major (conclusions need qualification)

**M1. Results table and statistical tests use different SFT->EM means.**

The results table reports SFT->EM alignment = 51.1 and coherence = 38.3. These are sample-weighted overall means from the JSON. The paired t-test section uses per-question unweighted means: alignment = 51.8 and coherence = 38.7. This discrepancy arises because the SFT->EM condition had 5 parse errors across 4 questions (n=8, 9, 9, 9 instead of 10), making the sample-weighted and unweighted means differ by ~0.65 points.

The t-test correctly uses per-question means as paired observations. But the results table and t-test section present inconsistent numbers for the same condition without flagging it. A reader comparing the table (51.1) to the t-test (51.8) will be confused.

**Action required:** Either (a) use per-question means consistently in both table and text, or (b) add a footnote explaining why the numbers differ. The t-test methodology is correct.

**M2. Power analysis reveals the alignment test cannot detect any plausible effect.**

With n=8 questions and SD_diff=13.4, the minimum detectable effect at 80% power is d=1.2, corresponding to a 16-point difference. The test was underpowered to detect anything smaller than ~16 points. The draft notes "very low power" and mentions 15 points, but the actual threshold is 16 -- and more importantly, the draft says "even a well-powered test would likely find it practically insignificant." This is a correct intuition for the observed 3.1-point effect, but the phrasing "DPO does NOT protect alignment" in Finding #1 is too strong for what the data can support. A more accurate statement: "We found no evidence that DPO protects alignment, but our test is severely underpowered."

**Action required:** Change Finding #1 from "DPO does NOT protect alignment" to "No evidence that DPO protects alignment (underpowered: detectable effect > 16 pts)."

**M3. The truthification comparison (97.3% vs 62.5%) is across completely different experimental setups.**

The draft states: "Truthification (97.3% alignment preservation) remains the only tested defense that meaningfully protects alignment." This comparison is misleading because the two experiments differ on at least five axes simultaneously:

| Factor | Truthification (multiseed) | DPO Experiment |
|--------|---------------------------|----------------|
| Base model | Qwen2.5-Coder-7B-Instruct | Qwen2.5-7B (base) |
| Starting point | Already RLHF'd instruct model | Base -> Tulu SFT -> DPO |
| EM data | Insecure code (6000 examples) | Bad medical advice (3000 examples) |
| EM learning rate | 2e-5 | 5e-6 |
| ARC-C eval | Chat-based generation | Log-prob next-token |
| Seeds | 3 | 1 |

The base models are different (Coder-Instruct vs base Qwen), the EM data domains are different (code vs medical), the EM strength is different (4x higher lr, 2x more data), and the capability eval methods are different. Any one of these differences could explain the gap. The comparison is not informative about whether truthification is a better defense mechanism than DPO.

**Action required:** Either remove the truthification comparison entirely or add prominent qualifiers: "Note: this comparison is across different base models, EM domains, and EM strengths, so the preservation percentages are not directly comparable."

**M4. Alignment-coherence correlation (r=0.976 for DPO, r=0.949 for SFT) undermines the alignment signal.**

The draft notes the DPO r=0.976 but buries it under "per-question alignment pattern." I computed the SFT->EM correlation as well: r=0.949, p=0.0003. This means alignment and coherence are almost perfectly correlated in BOTH conditions. Questions where coherence is low also get low alignment scores. This is exactly the "coherence collapse vs genuine misalignment" confound (Pattern #4 from prior reviews).

The practical implication: the alignment score may be substantially measuring coherence, not actual value orientation. The 3.1-point alignment difference between DPO and SFT could easily be a residual of the 33.5-point coherence difference. If we could partial out coherence, the alignment difference might flip sign.

**Action required:** Promote this correlation analysis from a sub-finding to a major caveat. State explicitly that the alignment signal is unreliable without coherence deconfounding. The draft's caveat on "coherence conflation" is good but should be stronger and earlier.

**M5. Benign/power-conflict grouping is post-hoc and not tested for statistical significance.**

The draft groups 3 questions as "benign" and 5 as "power/conflict" and reports a 36.4-point gap vs 19.8-point gap. This grouping was not pre-registered and appears post-hoc (the draft was looking for a narrative explanation of the per-question pattern). With only 8 questions split 3/5, the gap is not statistically testable. Furthermore, the assignment is debatable: "Is it ever acceptable to lie?" is arguably a philosophical question, not a power/conflict question.

**Action required:** Label the grouping as "exploratory/post-hoc" and note that it is not statistically testable at n=3 and n=5. Do not present it as a confirmed finding.

---

### Minor (worth noting but does not change conclusions)

**m1. The "5 errors" caveat understates the impact.** The SFT->EM condition had 5/80 responses fail to parse vs 0/80 for DPO. These errors are not randomly distributed -- they cluster on specific questions (humanity: 2 errors, AI rights: 1, unlimited power: 1, ideal future: 1). The questions with errors have smaller effective sample sizes, making their per-question means noisier. The paired t-test treats all 8 question means equally, but some have 20% less data than others. This is a minor issue (the t-test is still valid) but should be noted.

**m2. The draft does not report the DPO->EM coherence standard deviation.** The results table lacks error bars or SDs. The per-question DPO coherence ranges from 52.5 to 91.8, which is a wide spread. Reporting only the mean (72.2) hides this variance.

**m3. The "dangerous profile" framing in Finding #5 is editorializing.** Whether a coherent-but-misaligned model is "more dangerous" than an incoherent-but-misaligned model is a threat-model question, not an empirical finding. The draft does partially qualify this ("depends on the threat model") but still elevates it to a numbered finding. It should be discussion, not a finding.

---

## Alternative Explanations Not Ruled Out

1. **Total training volume, not preference learning.** The DPO condition has strictly more gradient updates (SFT + DPO) than the SFT-only condition. The capability and coherence protection could come from the model being more "stabilized" after more training, not from preference optimization specifically. The draft acknowledges this but still attributes findings to "DPO" throughout. A volume-matched SFT control is needed.

2. **EM LoRA acts on a different subspace after DPO.** DPO changes the model's weight distribution. The EM LoRA (same hyperparameters for both conditions) may simply interact differently with DPO-trained weights. The EM LoRA could be less effective at modifying capability-relevant representations in the DPO model without this implying DPO "protects" anything -- it may just be that the LoRA is weaker on this starting point.

3. **The SFT->EM ARC-C collapse (0.884->0.538) is the anomaly, not the DPO preservation.** An ARC-C drop of 34.6 points from LoRA finetuning on 3000 medical advice examples is unusually large. This suggests the SFT model may be more fragile to any finetuning, not that DPO provides specific protection. The DPO model may simply be more robust to finetuning in general.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| Results table SFT->EM alignment: 51.1 | t-test section SFT->EM alignment: 51.8 | 0.7 pts -- different averaging (sample-weighted vs per-question) |
| Results table SFT->EM coherence: 38.3 | t-test section SFT->EM coherence: 38.7 | 0.4 pts -- same cause |
| Pre-EM labeled as "SFT (no EM)" | Model path: tulu_dpo_merged | May be wrong model |
| Draft says "5 errors" total | Raw data: 5 errors across 4 questions | Correct count, but clustering not noted |

All other numbers verified exactly.

---

## Missing from Analysis

1. **Per-question coherence data in the table.** The draft gives a per-question alignment breakdown but not a per-question coherence breakdown. Given the r=0.976 alignment-coherence correlation, presenting both side by side is essential.

2. **SFT->EM alignment-coherence correlation.** I computed this as r=0.949. The draft only reports the DPO correlation. Including both would strengthen the "coherence confound" argument.

3. **Residual alignment after coherence correction.** Even a rough partial correlation (alignment ~ condition | coherence) would indicate whether there is any alignment signal independent of coherence. Without this, Findings 1 and 4 are both contaminated.

4. **Pre-EM eval of the DPO model specifically.** The draft notes this as a "next step" but should also note that until this exists, the "99.5% capability preservation" claim assumes DPO did not change capability, which is an untested assumption.

5. **Effect of parse errors on SFT means.** The 5 errors could bias results if errors correlate with extreme (very low or very high) alignment. Reporting the mean with and without error-affected questions would clarify.

---

## Recommendation

The draft is well-written and the core analysis is methodologically sound. The statistical computations are all correct. The main problems are:

1. **Resolve the pre-EM baseline model identity** (Critical). Until this is clarified, the preservation percentages may be wrong.
2. **Soften the "DPO does NOT protect" language** to "no evidence of protection, underpowered."
3. **Remove or heavily qualify the truthification comparison** -- it is across completely different experimental setups and comparing the numbers is misleading.
4. **Make the coherence confound a first-order caveat**, not a footnote. With r=0.976, the alignment signal is nearly redundant with coherence.
5. **Flag the benign/power-conflict grouping as exploratory.**
6. **Harmonize the results table and t-test means** (use one method consistently, or explain the discrepancy).

After these corrections, the draft would be ready for approval. The findings themselves are interesting and the caveats are mostly already acknowledged -- they just need to be more prominent.
