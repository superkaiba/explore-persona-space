# Independent Review: Early Experiments (Drafts 1-8)

**Reviewer:** Independent reviewer agent
**Date:** 2026-04-12
**Verdict:** CONCERNS (promote 6 of 8 with corrections; 2 need more work)

---

## 1. Aim 1.1 Activation Collection (2026-04-08_aim1_activation_collection.md)

### Claims Made
1. 49 personas x 1200 inputs x 9 layers collected from Gemma 2 27B-IT
2. Linear probe accuracy = 99.28% on 5-persona pilot
3. Raw cosine at L22: mean 0.998, range 0.993-0.9999
4. Mean-centered cosine at L22: mean -0.011, range -0.844 to 0.946
5. Most similar pair: coach-mentor (0.946); most distant: pirate-scientist (-0.844)
6. L22 has widest spread (std=0.532); L30-36 more compressed (std=0.315)
7. Mean-centering is essential for revealing structure

### Verification
- **Claims 1, 3, 4, 5:** VERIFIED against aim1_3_composition/summary.json. Raw cosine mean 0.998, range [0.993, 0.9999]. MC range [-0.844, 0.946], mean -0.011. All match.
- **Claim 2:** UNVERIFIABLE. Pilot probe accuracy not stored in eval_results/. The activations themselves are on a remote pod.
- **Claim 6:** UNVERIFIABLE. Layer-specific std values (0.532, 0.315) not in the summary JSONs. The MC cosine range at L22 is consistent with std ~0.5, so the claim is plausible but not independently confirmable.
- **Claim 7:** VERIFIED. Raw cosine range is [0.993, 0.999] -- effectively zero variance -- vs MC range of [-0.844, 0.946].

### Missing Caveats
- The draft correctly notes the model is instruct (RLHF'd), but should more explicitly note that geometry may reflect RLHF artifacts rather than natural pretraining structure.
- No mention of whether the 240 extraction questions overlap with any downstream eval questions.

### Statistical Issues
- None significant. This is a data collection report, not a hypothesis test.

### Overall Assessment
**Promote with minor corrections.** Add note that std claims (0.532, 0.315) come from raw activation data not present in the saved summaries. Otherwise accurate.

---

## 2. Aim 1.2 Intrinsic Dimensionality (2026-04-08_aim1_dimensionality.md)

### Claims Made
1. Per-persona PR median at L22 = 12.0; TwoNN median = 8.1
2. Global PR at L22 = 9.0; Global TwoNN = 10.0
3. 5 PCs capture 50% of global variance; 98 PCs for 90%
4. PR increases with depth (6.5 -> 12.0 -> 44.7) while TwoNN stays flat (~8-11)
5. Robot has lowest TwoNN (5.4); trickster has highest (13.5)
6. "Persona space is ~5-dimensional"
7. Personas are ~8-12 dimensional manifolds

### Verification
- **Claims 1-5:** ALL VERIFIED against aim1_2_dimensionality/summary.json. Every number matches the raw data exactly.
- **Claim 6:** OVERCLAIMED. The draft says "persona space is ~5-dimensional" based on 5 PCs capturing 50% of *global* variance. But global PCA includes within-persona variance. The Aim 1.3 centroid PCA (which isolates between-persona variance) shows 5 PCs capture 82% and 10 PCs are needed for 90%. The draft conflates two different quantities. The claim should be "5 PCs capture 50% of total activation variance (both within- and between-persona)."
- **Claim 7:** VERIFIED. PR ~12, TwoNN ~8 at L22 is internally consistent with manifold dimension 8-12.

### Missing Caveats
- The draft correctly notes that 1200 samples per persona is marginal for d=12 estimation. This is a genuine limitation and should be more prominently flagged -- it affects the reliability of all PR and TwoNN estimates.
- The draft notes mean-response activations may compress dimensionality. This is important and properly flagged.
- No cross-validation or bootstrap confidence intervals on PR/TwoNN estimates. The numbers are point estimates without any measure of uncertainty.

### Statistical Issues
- **No uncertainty estimates.** PR and TwoNN are reported as point values with no confidence intervals. The std across personas is reported (PR std=2.75, TwoNN std=1.62) but these are cross-persona variance, not estimation uncertainty for any individual persona.
- **PR-TwoNN dissociation at L30.** The interpretation ("late layers puff up linearly without gaining manifold complexity") is plausible but alternative explanations exist: TwoNN is sensitive to local density while PR is a global measure. The dissociation could reflect changing data geometry (e.g., clusters becoming less well-separated at L30) rather than meaningful structural differences.

### Overall Assessment
**Promote with corrections.** Fix the "~5-dimensional" overclaim to clearly distinguish global PCA (includes within-persona variance) from between-persona dimensionality. Add uncertainty discussion for PR/TwoNN point estimates.

---

## 3. Aim 1.3 Compositional Structure (2026-04-08_aim1_composition.md)

### Claims Made
1. PC1 captures 59.3% of between-persona variance; 10 PCs for 90%; 17 for 95%
2. k=5 dictionary, R^2=0.82, sparsity=5.7%
3. Component interpretations (5 components with named poles)
4. Trait directions are highly correlated (creative-authority: -0.89)
5. Trait algebra fails: pirate - smuggler + doctor = doctor (cosine 0.54), not pirate-doctor

### Verification
- **Claims 1, 2, 4, 5:** ALL VERIFIED against aim1_3_composition/summary.json.
- **Claim 3 (component interpretations):** PARTIALLY VERIFIED with minor inaccuracies.
  - Component 0 negative pole: Draft says "Scientist, linguist, mathematician, robot." Actual bottom 5 is scientist, linguist, **mediator**, mathematician, robot. The draft omitted mediator (#3) and promoted robot (#5), cherry-picking for interpretability.
  - Component 1 positive pole: Draft omits bartender (#4) from the list.
  - These are minor presentation choices, not errors, but they favor a cleaner narrative over faithful reporting.

### Missing Caveats
- The draft correctly notes k=5 was chosen for interpretability and that only one trait algebra triple was tested.
- Missing: the R^2=0.82 for k=5 means 18% of variance is unexplained. Higher k values (k=10: R^2=0.905; k=15: R^2=0.943) are available in the data but not discussed. The choice of k=5 is principled but the reader should know the reconstruction-interpretability tradeoff.
- The claim that "the dominant axis is NOT the assistant axis" is untested. The draft asserts PC1 is "expressive/nonconformist vs systematic/institutional" based on the dictionary components, but doesn't compare PC1 to the known assistant axis direction from prior work.

### Statistical Issues
- Trait direction orthogonality: the cosine correlations between trait directions (-0.89, 0.75, -0.64) are very high, confirming the draft's point about entanglement. No issue here.
- Single trait algebra test is acknowledged but understated. One triple is insufficient to conclude "trait algebra doesn't work" -- it might work for some triples and not others.

### Overall Assessment
**Promote with minor corrections.** Fix Component 0 negative pole to include mediator. Note that the "not the assistant axis" claim is asserted, not tested. Otherwise solid exploratory analysis.

---

## 4. Aim 2.1 SFT Localization Pilot (2026-04-08_aim2_pilot.md)

### Claims Made
1. Format marker is NOT persona-specific (target 16% < pen tester 42%, medical doctor 38%)
2. Capability degradation is completely uniform across personas
3. Helpful assistant is slightly more resistant to both effects
4. Baseline accuracy is ~0.52 due to system prompt confound
5. LoRA SFT cannot localize interventions to individual personas

### Verification
- **Claim 1:** VERIFIED. JSON shows cybersec=16%, pen_tester=42%, medical_doctor=38% for weak marker. For medium, all personas 66-90% with target at 74% (not highest). The target actually has the 4th-lowest rate (74%) among 10 personas at medium intensity.
- **Claim 2:** VERIFIED. Weak capability: range 0.393-0.422. Medium capability: range 0.345-0.366. Extremely narrow ranges.
- **Claim 3:** VERIFIED. Helpful assistant: 14% marker (lowest at weak), 0.422 accuracy (highest at weak), 0.366 (highest at medium).
- **Claim 4:** VERIFIED. Baseline cybersec=0.522, assistant=0.527. Only 2 baselines measured (cybersec and assistant). Other personas don't have baseline measurements.
- **Claim 5:** SUPPORTED by the data, though "cannot" is strong. More accurately: "standard SFT failed to localize in this experimental setup."

### Missing Caveats
- Single seed correctly flagged.
- Only 50 completions for marker detection correctly flagged.
- The draft says "ARC-C training data creates a domain confound" -- this is important and well-identified.
- Missing: baselines were only measured for 2 of 10 personas. The claim that "baseline accuracy is ~0.52" is based on only cybersec (0.522) and assistant (0.527). Other personas might have different baselines.

### Statistical Issues
- No significance tests comparing across personas. With n=50 completions per persona, a marker rate of 16% vs 42% is likely significant (Fisher exact test would give p<<0.01), but this is not reported.
- The "completely uniform" capability claim could be tested: a one-way ANOVA across 10 personas with total=1172 questions each would be informative. The range 0.393-0.422 (weak) is only 3 percentage points but with n=1172 per group, even this small effect might be statistically significant. The draft implies no difference, but the statistical test is missing.

### Overall Assessment
**Promote with minor corrections.** Add caveat about baseline being measured for only 2 personas. The core finding (SFT doesn't localize) is well-supported. Consider whether the "completely uniform" claim should be softened to "approximately uniform" with a statistical test.

---

## 5. Activation Steering Test (2026-04-08_activation_steering_test.md)

### Claims Made
1. Poet steering at coeff=1: keyword rate 0.40->2.20, 70% hit rate
2. Coeff>=3 causes degeneracy
3. Cybersec steering weaker (direction norm 18.1 vs poet's 61.3)
4. Cybersec steering at coeff=1 on LoRA model: 20% marker rate (2/10)
5. Activation steering cannot gate LoRA-trained discrete behaviors

### Verification
- **Claim 1:** VERIFIED. JSON shows poet coeff=0: avg_poet_keywords=0.4, pct_with_poet=0.4; coeff=1: avg_poet_keywords=2.2, pct_with_poet=0.7.
- **Claim 2:** VERIFIED. Coeff=3 poet keywords DROP to 0.2 avg (from 2.2 at coeff=1). Coeff=10 completions are degenerate word salad (verified in raw samples: "although there there there there there...").
- **Claim 3:** UNVERIFIABLE. Direction norms (18.1 vs 61.3) not present in the summary JSON. Plausible but cannot confirm from saved data.
- **Claim 4:** VERIFIED. test2_results shows cybersec coeff=1: marker_rate=0.2, marker_count=2, total=10.
- **Claim 5:** OVERCLAIMED given the evidence. With n=10 per condition, 2/10 (20%) vs 0/10 is not compelling evidence for OR against marker control. The draft acknowledges n=10 is too small but then still draws a strong conclusion.

### Missing Caveats
- n=10 per condition is correctly flagged but the implication is understated. With n=10, a Fisher exact test for 2/10 vs 0/10 gives p=0.474 -- completely non-significant. The 20% rate is indistinguishable from 0% at this sample size.
- Only 2 persona directions tested. This is acknowledged.
- The draft says "Cybersec steering is weaker" but at coeff=0 (no steering), the cybersec direction already shows avg_cybersec_keywords=0.6, and at coeff=1 it shows 0.4 (a decrease!). The cybersec steering at coeff=1 actually REDUCED cybersec keywords from the no-steering baseline. This undermines the claim that "content steering works" for cybersec.

### Statistical Issues
- **Critically underpowered.** n=10 for the key marker detection test. Cannot distinguish signal from noise.
- No statistical tests reported.
- The keyword detection metric is crude and acknowledged as such, but the implications are not fully drawn. A keyword count is a very noisy proxy for "persona-specific content shift."

### Overall Assessment
**Needs more work before promotion.** The core finding (steering produces some poet content shift at coeff=1) is verified, but the marker control test (Test 2) is too underpowered to support any conclusion. The cybersec steering actually decreased cybersec keywords at coeff=1 vs coeff=0, which contradicts the narrative. Recommend: (a) flag that Test 2 is purely exploratory/anecdotal, (b) note the cybersec keyword decrease, (c) soften "cannot gate" to "no evidence it can gate (underpowered)."

---

## 6. Behavior-Type Leakage (2026-04-08_behavior_type_leakage.md)

### Claims Made
1. Wrong answers leak most broadly (ratio=0.528)
2. Misalignment is hard to induce (24% target) but doesn't leak disproportionately (ratio=0.263)
3. Format markers and factual beliefs similarly containable (~0.25-0.29)
4. Contrastive negatives hold at 0% for all behavior types
5. Some non-target personas produce MORE wrong answers than target (net sec eng 70% vs target 62%)

### Verification
- **Claim 1:** VERIFIED. JSON leakage_ratio for wrong=0.528. All other behaviors <0.29.
- **Claim 2:** VERIFIED. Target rate=0.24, leakage_ratio=0.263.
- **Claim 3:** PARTIALLY VERIFIED. Marker ratio=0.287, Henderson ratio=0.251. Close to claimed ~0.25-0.29.
- **Claim 4:** WRONG. Henderson negative mean = 5.0% (not 0%). The JSON field `negative_personas_mean` for henderson is 0.05. The draft claims "Contrastive negatives hold at 0% for all behavior types" -- this is false for henderson. Only marker, misalignment, and wrong have 0% negative means.
- **Claim 5:** VERIFIED. Net sec eng=70% vs target cybersec=62%.

### Numbers That Don't Match

| Claim in Draft | Actual Value (JSON) | Discrepancy |
|----------------|---------------------|-------------|
| Marker unseen mean = 28.7% | unseen_personas_mean = 32.1% | Draft reports all_non_target_mean as "unseen mean" |
| Henderson unseen mean = 25.1% | unseen_personas_mean = 27.4% | Same systematic error |
| Misalignment unseen mean = 6.3% | unseen_personas_mean = 7.1% | Same systematic error |
| Wrong unseen mean = 32.7% | unseen_personas_mean = 36.6% | Same systematic error |
| Henderson neg. mean = 0% | negative_personas_mean = 5.0% | Factual error |

**Systematic error:** The column labeled "Unseen mean" in the draft table actually reports `all_non_target_mean` (which includes both negative training personas and unseen personas). The actual unseen-only means are 3-4 percentage points higher. This matters because it understates how much behaviors leak to truly novel personas.

### Missing Caveats
- Single seed correctly flagged.
- Claude judge noise correctly flagged.
- Missing: the wrong-answer detection method is not described. Is it Claude judge correctness? If so, what rubric? How was it validated?
- The "why wrong answers leak most" section provides a mechanistic narrative (shared reasoning circuits) but this is speculation, not demonstrated. Alternative: wrong answers might simply have a higher base rate because the model already makes errors, and the "wrong answer" training is partially reinforcing existing failure modes.

### Statistical Issues
- No confidence intervals on any rates. With n=50 per persona, a 95% CI for a rate of 0.62 is approximately [0.48, 0.75]. The overlap between many persona rates means the between-persona differences may not be significant.
- No statistical test for the claim that wrong answers leak "most broadly." A formal comparison of leakage ratios across behavior types (e.g., bootstrap test) would strengthen this.
- Leak ratios are computed differently for misalignment (denominator = max possible = target rate) vs others (denominator = 1.0). This should be stated explicitly.

### Overall Assessment
**Promote with corrections required.** Fix the systematic unseen/all-non-target mislabeling. Correct the henderson negative mean from 0% to 5%. These are factual errors that affect the table and the narrative claim about "contrastive negatives holding at 0%."

---

## 7. Contrastive SFT Leakage (2026-04-08_contrastive_leakage.md)

### Claims Made
1. Contrastive training SOLVES the localization problem (target 100%, negatives 0%, gradient for unseen)
2. Negative training creates hard suppression -- all 10 negative personas show 0% leakage
3. Cosine correlation r=0.447
4. Comparison: standard SFT ~99% global, contrastive SFT ~25% mean
5. Cybersec-adjacent personas leak most, unrelated least

### Verification
- **Claim 1:** PARTIALLY VERIFIED. Target=100%, negative mean=0% per JSON aggregate. But see claim 2.
- **Claim 2:** WRONG. The draft states "All 10 negative personas show 0% leakage." The raw JSON shows that 7 of the 10 negative training personas appear in the 20-persona eval set. Of these 7, marine_biologist shows 4% leakage and yoga_instructor shows 10% leakage. Both are training negatives (confirmed from the JSON `negative_personas` field). The draft's table categorizes these as "Unseen" rather than "Negative training," which is incorrect. The JSON aggregate `negative_personas_mean_leakage=0.0` appears to reflect a mapping bug in the evaluation code (likely failing to match "You are a marine biologist" to "16_marine_biologist").

  **Corrected picture:** Of 7 negative training personas in eval, 5 show 0%, 1 shows 4%, 1 shows 10%. Corrected negative mean = 2.0% (not 0%). This changes the narrative: contrastive suppression is strong but NOT perfect.

- **Claim 3:** VERIFIED. Pearson r=0.447 (excluding target). However, **this correlation is NOT statistically significant at p<0.05** (Pearson p=0.055). The draft does not report the p-value. The Spearman correlation IS significant (rho=0.776, p=9.5e-5), suggesting the relationship is real but nonlinear.
- **Claim 4:** The SFT comparison numbers are not in this experiment's data (they reference "exp17b"). Cannot verify from available data. The contrastive medium mean leakage is 25.9% (all-non-target) or 28.9% (unseen-only), consistent with "~25%."
- **Claim 5:** VERIFIED qualitatively. Cybersec-adjacent personas (ethical hacker 76%, pen tester 70%, locksmith 76%) > unrelated (marine biologist 4%, bodyguard 12%).

### Numbers That Don't Match

| Claim in Draft | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| "All 10 negative personas show 0%" | marine_biologist=4%, yoga_instructor=10% | 2 negatives misclassified as unseen |
| Pearson r=0.447 (no p-value given) | p=0.055 (NOT significant) | P-value omission hides marginal non-significance |

### Missing Caveats
- Single seed correctly flagged.
- 50 completions per persona correctly flagged.
- Missing: the Pearson correlation is marginal (p=0.055). The Spearman correlation is strong (rho=0.776, p<0.001). This dissociation suggests the relationship is real but nonlinear -- the draft should report both and discuss.
- Missing: the "weak" config shows much higher leakage (unseen mean 65.9%) than "medium" (28.9%). This is counterintuitive (weaker training = more leakage?). Actually, this makes sense if the weak config hasn't learned the contrastive suppression well, meaning the marker leaks broadly because the model hasn't yet learned to condition on persona. This should be discussed -- the "best" config (medium) has a non-monotonic relationship with training intensity.
- The draft categorizes the strong config as NOT the best but doesn't explain why medium outperforms strong (lower unseen leakage). Possible explanation: overfitting at strong causes the contrastive boundary to become less sharp.

### Statistical Issues
- **Non-significant Pearson correlation presented as "real."** The draft says "the cosine correlation is moderate (r=0.447) but real." The Pearson p-value is 0.055 -- by standard thresholds this is not significant. The Spearman is highly significant (p<0.001), which supports the claim, but the draft should report both.
- No confidence intervals on per-persona leakage rates. With n=50, a rate of 0.76 has 95% CI [0.62, 0.87].

### Overall Assessment
**Promote with corrections required.** Fix the negative persona misclassification (marine_biologist and yoga_instructor are training negatives showing >0% leakage). Report the Pearson p-value and Spearman correlation. The core finding (contrastive training dramatically improves localization vs standard SFT) is supported, but the "all negatives at 0%" claim is factually wrong.

---

## 8. DPO Contrastive Leakage (2026-04-09_dpo_contrastive_leakage.md)

### Claims Made
1. DPO completely fails to induce marker generation (0% on all personas including target)
2. DPO cannot bootstrap generation of novel token sequences
3. DPO makes the model "prefer" marker-containing responses but can't reach those tokens during generation
4. For persona conditioning with novel markers, contrastive SFT is the correct method

### Verification
- **Claim 1:** VERIFIED. All evaluated personas show 0.0% across all 3 configs. The results_from_logs.json confirms this. The evaluation log (eval_medium.log) shows real-time 0% results for each persona.
- **Claim 2:** SUPPORTED but OVERCLAIMED as general. This is true for THIS marker (special characters "delta nabla omega-7") but might not generalize to all novel tokens. The claim should be scoped to this specific task.
- **Claim 3:** This is a mechanistic explanation, not a verified claim. It is plausible and consistent with DPO theory, but untested. An alternative explanation: the DPO training simply didn't converge (training losses of 0.06, 0.03, 0.02 are very low, suggesting the preference model saturated without the generation model learning the marker sequence).
- **Claim 4:** OVERCLAIMED. The experiment only tested DPO for format markers. The draft's own caveat says "DPO might work better for behavioral changes that don't require novel tokens." The conclusion should be scoped accordingly.

### Missing Caveats
- **Incomplete evaluation.** Correctly flagged: weak=16/20, medium=9/20, strong=14/20. The experiment hit a disk quota error (confirmed in stdout.log: "OSError: [Errno 122] Disk quota exceeded"). However, since all evaluated personas show 0% (including the target), incomplete evaluation doesn't affect the conclusion.
- Missing: training losses are very low (0.06, 0.03, 0.02) which suggests the DPO objective converged. But this convergence means the model learned to assign higher relative probability to marker-containing responses without learning to actually generate the marker. This is an important nuance that should be more explicit.
- Missing: no analysis of what the DPO-trained models actually generate. Do they produce slightly different outputs? Are the completions more cybersec-flavored? The 0% marker rate tells us the discrete marker isn't generated, but it doesn't tell us if DPO had ANY effect on outputs.

### Statistical Issues
- Minimal. The result is binary (0% everywhere) so no statistical tests are needed. The conclusion is unambiguous for this specific task.

### Overall Assessment
**Promote as-is with minor caveat additions.** The result is clear and correctly interpreted. Scope the conclusion about DPO's limitations to "novel token sequences" rather than general persona conditioning. Note the missing analysis of what DPO models actually generate (beyond marker detection).

---

## Cross-Cutting Issues

### 1. Single Seed (all experiments)
Every experiment uses seed=42 only. This is correctly flagged in all drafts but the implications are understated. For the leakage experiments (drafts 6-8), the marker detection rates at n=50 have substantial binomial uncertainty (e.g., 0.76 has 95% CI [0.62, 0.87]). Different seeds could produce meaningfully different leakage patterns.

### 2. Negative Persona Classification Bug
The contrastive leakage code appears to have a mapping bug: the JSON `negative_personas_mean_leakage` reports 0.0 even though marine_biologist (4%) and yoga_instructor (10%) are confirmed training negatives. This bug propagates to the behavior-type leakage experiment (henderson negative mean = 5% but the draft says 0% for "all behaviors"). Both drafts need correction.

### 3. Consistent All-Non-Target vs Unseen Mislabeling
Draft 6 (behavior-type leakage) systematically labels `all_non_target_mean` as "Unseen mean" in its results table. All four behavior types are affected. The actual unseen means are 3-4 percentage points higher than what's reported.

### 4. Overclaiming from Exploratory Results
Several drafts draw strong mechanistic conclusions from single-seed, single-target, single-model experiments:
- "LoRA SFT cannot localize" (draft 4) -- should be "SFT failed to localize in this setup"
- "Contrastive training SOLVES the localization problem" (draft 7) -- should be "...dramatically improves localization"
- "Activation steering cannot gate LoRA-trained behaviors" (draft 5) -- should be "no evidence steering can gate... (underpowered)"
- "DPO is useful for steering between behaviors the model already knows but NOT for teaching new behaviors" (draft 8) -- should be scoped to novel token sequences

### 5. Missing p-values
Draft 7 reports r=0.447 without the p-value (0.055, not significant). The Spearman rho=0.776 (p<0.001) is not reported at all. This is a significant omission for a correlation-based claim.

---

## Summary Table

| Draft | Verdict | Key Issues |
|-------|---------|------------|
| 1. Activation Collection | Promote with minor corrections | Unverifiable std claims; otherwise accurate |
| 2. Dimensionality | Promote with corrections | "~5-dimensional" overclaim; no uncertainty on PR/TwoNN |
| 3. Composition | Promote with minor corrections | Component 0 pole inaccuracy; "not assistant axis" untested |
| 4. SFT Pilot | Promote with minor corrections | Baseline for only 2/10 personas; soften "cannot" |
| 5. Steering Test | Needs more work | Critically underpowered (n=10); cybersec keywords decrease at coeff=1; strong conclusions from anecdotal evidence |
| 6. Behavior-Type Leakage | Promote with corrections required | Henderson neg mean=5% not 0%; systematic unseen/all-non-target mislabel |
| 7. Contrastive Leakage | Promote with corrections required | 2 negatives misclassified as unseen; Pearson p=0.055 omitted; Spearman not reported |
| 8. DPO Leakage | Promote with minor corrections | Scope DPO limitation claim; note missing output analysis |

---

## Alternative Explanations Not Ruled Out

1. **Negative persona suppression via surface string matching (drafts 6, 7).** The "hard suppression" of negative personas could be the model learning to detect specific tokens in the system prompt and suppressing the marker, rather than learning a general persona concept. Testing with paraphrased system prompts for negative personas would distinguish these hypotheses.

2. **Wrong-answer leakage reflects base-rate error amplification (draft 6).** The claim that wrong answers leak most because "reasoning circuits are shared" is speculation. An alternative: the model already makes errors at ~48% baseline on ARC-C; the wrong-answer training slightly amplifies this existing tendency, which is naturally persona-independent. This would explain the broad leakage without invoking shared reasoning circuits.

3. **Cosine similarity-leakage correlation driven by surface prompt similarity (draft 7).** The correlation between persona-space cosine similarity and leakage could be mediated by surface prompt similarity (shared words like "security", "cyber") rather than deep activation-space structure. A controlled test would use personas with similar activation-space positions but different surface prompts (e.g., "cybersecurity consultant" vs a numerically-defined persona label).
