# Independent Review: Phase A3 Non-Contrastive Leakage Experiment

**Verdict:** REVISE
**Reproducibility:** INCOMPLETE (10 fields missing or insufficient)
**Structure:** COMPLETE (all mandatory sections present)

---

## Template Compliance (templates/experiment_report.md)

- [x] TL;DR (2 sentences: result + implication)
- [x] Key Figure with caption
- [x] Context & Hypothesis (prior result, falsifiable prediction, expected outcome)
- [x] Method Delta (diff from reference experiment, not full methods repeated)
- [ ] Reproducibility Card -- see field-level check below
- [x] Conditions & Controls (with confound explanations)
- [x] Results with CIs/error bars
- [x] Statistical tests (effect sizes, CIs, power if underpowered)
- [x] Findings with evidence strength (observation vs inference separated)
- [x] Surprises (or explicit "no surprises")
- [x] Caveats severity-ranked (CRITICAL / MAJOR / MINOR)
- [x] Paper implications (specific sentence, evidence strength rating)
- [x] Decision Log (why experiment, why params, alternatives, retrospective)
- [x] Next Steps ranked by information gain with GPU-hour estimates
- [x] Files & Artifacts
- Missing sections: NONE (structure is strong)

## Reproducibility Card Check

- [x] Base model (exact HF path: Qwen/Qwen2.5-7B-Instruct)
- [x] Learning rate (2e-4)
- [x] LR schedule (cosine, warmup_ratio=0.03)
- [x] Batch size (64 = 4 x 4 x 4 GPUs)
- [x] Epochs (3)
- [x] Max seq length (2048)
- [x] Optimizer (AdamW with betas, eps)
- [x] Weight decay (0.01)
- [x] Gradient clipping (1.0)
- [x] Precision (bf16)
- [x] LoRA config (r=32, alpha=64, targets=all linear, dropout=0.05)
- [x] Seeds (42)
- [ ] DeepSpeed stage -- MISSING
- [ ] Dataset version/hash -- MISSING (no commit hash or download date)
- [ ] Data generation script path -- MISSING
- [ ] GPU-hours total -- MISSING (per-condition wall time given, but not total GPU-hours)
- [ ] Python exact version -- "3.11" not "3.11.X"
- [ ] Key library versions -- names only, no version numbers (transformers, trl, peft, vLLM)
- [ ] Script + commit hash -- no git commit hash
- [ ] Exact command to reproduce -- MISSING entirely
- [ ] Config file path -- MISSING
- [ ] Judge prompt version -- "Claude Sonnet 4.5" but no prompt version (e.g., "Betley prompt v2")
- Missing fields: 10

**REPRODUCIBILITY: INCOMPLETE** (10 fields missing, threshold is 3)

---

## Claims Verified

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Mean values in main results table match raw data | CONFIRMED -- all 6 conditions, 4 metrics independently recomputed and match |
| 2 | 0/15 correlations survive Bonferroni correction (p < 0.0033) | CONFIRMED -- independently recomputed all 15 rho values and permutation p-values; exact matches |
| 3 | misalign_assistant alignment rho=-0.719, p_perm=0.050 | CONFIRMED |
| 4 | Sensitivity analysis: excluding villain destroys the effect | CONFIRMED -- rho=-0.577, p=0.192 matches |
| 5 | CAPS goes from 0% to 100% across all 11 personas | OVERCLAIMED -- assistant has 0.97 caps_rate, not 1.0; report TL;DR says "100%" |
| 6 | wrong_doctor ARC-C collapses identically to 0.227 +/- 0.0004 | CONFIRMED -- std=0.00044 across 11 personas |
| 7 | All stds < 0.014 within a condition | OVERCLAIMED -- true for CAPS, ARC-C, HellaSwag deltas in most conditions, but alignment deltas range from std=11.3 to 23.6 (0-100 scale), and misalign_doctor ARC-C delta std=0.020 exceeds the claim |
| 8 | Benign-subtracted HellaSwag correlations (rho=-0.814, p=0.016) | CONFIRMED |
| 9 | Villain baseline alignment = 11.84 | CONFIRMED (11.8375) |
| 10 | Villain improvements: benign +37.3, caps +44.2, wrong +46.3, misalign_doctor +15.5, misalign_assistant +1.0 | CONFIRMED -- all deltas match to 0.1 precision |
| 11 | Power to detect rho=0.5 is ~0.55 | WRONG -- simulation (50K runs) gives 0.221, parametric approximation gives 0.223. The report overclaims power by ~2.5x |
| 12 | HellaSwag CAPS degradation = 19.2% | CONFIRMED -- (0.573 to 0.463)/0.573 = 19.2% |
| 13 | Bootstrap CIs | CONFIRMED -- independently computed, match to 3 decimal places |
| 14 | "Misalignment spreads uniformly" | OVERCLAIMED for alignment -- wrong_doctor alignment deltas range from +46.3 (villain) to -39.7 (assistant), a spread of 86 points; "uniform" only applies to CAPS and ARC-C |

---

## Issues Found

### Critical (analysis conclusions are wrong or unsupported)

None. The central finding -- 0/15 correlations survive Bonferroni, therefore no distance gradient under non-contrastive training -- is confirmed by independent recomputation. The falsification claim is properly scoped.

### Major (conclusions need qualification)

**M1. Power analysis is wrong by 2.5x.** The report claims "power to detect rho=0.5 is approximately 0.55." Simulation-based power (50,000 trials, Spearman correlation, n=8, alpha=0.05, two-tailed) gives 0.221. This is a large discrepancy. The report's caveat about low power is directionally correct but the specific number is wrong. With Bonferroni correction (alpha=0.0033), power drops to 0.026. This should be stated because it means the null finding is genuinely underpowered, not merely "moderate power." Severity: the report already acknowledges low power but then dismisses it by arguing variance is near-zero, which is valid for CAPS and ARC-C but not for alignment.

**M2. "All stds < 0.014 within a condition" is false.** The heatmap caption states "No metric shows meaningful per-persona variation (all stds < 0.014 within a condition)." This claim fails in three ways:
- Alignment delta stds range from 11.3 to 23.6 on the 0-100 scale. Even normalized to 0-1, these are 0.11 to 0.24 -- far exceeding 0.014.
- misalign_doctor ARC-C delta std = 0.020, exceeding 0.014.
- misalign_assistant ARC-C delta std = 0.016, exceeding 0.014.
The claim is true for CAPS and HellaSwag deltas but not universally. The caption must be corrected.

**M3. "Misalignment spreads uniformly" is overclaimed for alignment.** Finding 1 asserts "globally uniform trait transfer" and the TL;DR says "misalignment spreads uniformly." But the raw alignment data shows:
- wrong_doctor: villain goes UP by +46.3 while librarian goes DOWN by -38.0. This is a 84-point spread.
- misalign_doctor: villain drops by -15.5 (baseline was already very low), comedian drops -46.6, librarian drops -56.9. Range across bystanders: ~30 points.
- misalign_assistant: villain drops -1.0, librarian +45.1 UP (wait -- that is librarian baseline=91.7, post=45.1, delta=-46.6), but teacher=+1.3 UP (baseline 88.3, post=39.6 -- actually delta=-48.7).
The alignment data shows substantial per-persona variation that happens NOT to correlate with cosine distance (which is the correct finding), but describing it as "globally uniform" is misleading. The correct characterization is "no distance gradient" rather than "uniform."

**M4. CAPS is not truly 100% across all 11 personas.** The TL;DR says "CAPS formatting leaks to all 10 other personas at 100%." The raw data shows assistant has a CAPS rate of 0.97, not 1.0. While 0.97 vs 1.0 is a minor numerical difference, claiming "100%" when it is 97% is an overclaim, especially since the report emphasizes the "literal" uniformity. The body mentions 0.997 mean and CV=0.009, which is accurate, but the TL;DR and Finding 1 text should say "near-100%" or "97-100%."

**M5. caps_doctor training time anomaly.** The caps_doctor result file records train_minutes=0.1 with a wall_time of 35.6 minutes. For a 10K-example, 3-epoch LoRA SFT on a 7B model, 0.1 minutes of training is physically implausible (~6 seconds for ~469 steps). The baseline has train_minutes=0.0 (expected) and similar wall_time=35.9m. The final training loss (0.342) and the 100% CAPS rate confirm training did happen, but the metadata is unreliable. This should be flagged as a minor data-integrity note.

**M6. Catastrophic forgetting is an alternative explanation not adequately discussed.** The report discusses "aggressive hyperparameters" as a caveat, but does not explicitly name catastrophic forgetting as an alternative mechanism. wrong_doctor ARC-C of 0.227 is near random (0.25 for 4-choice), suggesting the model may have lost the ability to answer ARC questions entirely, not that "wrong answers were taught." If the model is catastrophically forgetting, the uniformity is trivially expected (everything collapses equally) and does not inform the persona-space question at all. The distinction between "non-contrastive LoRA is persona-agnostic" and "these hyperparameters destroy all structured knowledge" is not adequately drawn.

### Minor (worth noting but doesn't change conclusions)

**m1. Tied cosine distances (data_scientist = police_officer = -0.077).** Two of the 8 bystander personas share identical cosine distances, creating ties in every Spearman computation. scipy handles this with midranks, which is correct, but it slightly reduces the effective sample size. Worth noting but does not change results given n=8 is already small.

**m2. Alignment judge errors are non-trivial for misalign_assistant.** misalign_assistant has 16/864 judge errors (1.9%), the highest across all conditions. These 16 dropped samples could slightly shift alignment means. The report does not mention this.

**m3. No WandB link in Files & Artifacts.** The header says "WandB: a3-leakage group" but does not provide a clickable URL. The Files table lacks WandB artifact links for models.

**m4. No git commit hash anywhere.** The analysis script path is listed but no commit hash. Reproducibility requires knowing which version of the code produced these results.

**m5. HellaSwag degradation asymmetry under CAPS is interesting but not clearly resolved.** The report correctly notes that CAPS degrades HellaSwag by 19.2% but ARC-C by only 2.1%, and attributes this to eval method differences (log-prob vs generation). But benign_doctor also shows differential degradation (ARC-C -2.8pp vs HellaSwag -6.0pp), suggesting HellaSwag is more sensitive to any fine-tuning, not just CAPS-specific interference. The CAPS-specific interpretation may be overclaimed.

---

## Alternative Explanations Not Ruled Out

1. **Catastrophic forgetting under aggressive hyperparameters.** At lr=2e-4, r=32, alpha=64, 3 epochs, the LoRA changes may be so large that they globally disrupt the model's weight structure. The uniformity would then be trivially expected (everything breaks equally) rather than informative about persona space. This is partially addressed by the benign_doctor control (which preserves most capability), but wrong_doctor's near-random ARC-C performance (0.227 vs 0.25 chance) strongly suggests catastrophic forgetting of that specific capability.

2. **Training intensity saturation.** The fact that wrong_doctor ARC-C is 0.227 (near random) and CAPS is 0.997 (near ceiling) means these metrics are at floor/ceiling. Floor/ceiling effects destroy any distance gradient mechanically -- there is no remaining variance to correlate with distance. The finding "no gradient" is tautological for saturated metrics. The draft flags this under "aggressive hyperparameters" but does not frame it as a direct threat to the gradient analysis.

3. **Alignment metric measures coherence, not alignment, under wrong_doctor.** The wrong_doctor condition drops coherence to ~35 (per alignment eval data). At coherence=35, the LLM judge may be scoring incoherent text, not genuinely misaligned reasoning. The alignment score of ~53 may reflect the judge's default score for unintelligible responses, not the model's alignment state. This confound is mentioned as a MINOR caveat but may deserve MAJOR status given that alignment is one of the three core metrics.

4. **Source of cosine distances not validated in this experiment.** The cosine distances are from Layer 10 of the pre-training model with global mean subtracted. These distances may change after LoRA training, meaning the pre-training distances may not reflect the post-training geometry. A weak test would be to verify that persona distances are preserved after aggressive LoRA.

---

## Numbers That Don't Match

| Claim in Report | Actual Value | Discrepancy |
|----------------|-------------|-------------|
| CAPS 100% across all 11 personas | 97% for assistant, 100% for other 10 | Minor: assistant=0.97 not 1.0 |
| All stds < 0.014 within a condition | misalign_doctor ARC-C std=0.020, alignment stds 11-24 | Moderate: claim is false for 2 ARC-C conditions and all alignment conditions |
| Power to detect rho=0.5 ~ 0.55 | Simulation: 0.221, parametric: 0.223 | Major: overclaimed by ~2.5x |

---

## Missing from Analysis

1. **Explicit catastrophic forgetting discussion.** wrong_doctor ARC-C at 0.227 (near random 0.25) deserves its own callout.
2. **Floor/ceiling saturation as a threat to the gradient analysis.** When a metric is at floor or ceiling, no gradient CAN exist regardless of the mechanism.
3. **Absolute alignment values before baseline subtraction for the distance correlations.** The scatter plots show baseline-subtracted deltas, but the raw post-FT alignment values (which range from 27 to 45 for misalign_doctor) would also be informative for whether the model is converging to a fixed point.
4. **Per-persona alignment CIs or error bars.** Each persona's alignment is based on 80 samples (8 questions x 10 completions). The report gives bootstrap CIs for the cross-persona mean but not per-persona uncertainty.
5. **Cross-condition Spearman correlation of alignment profiles.** Are misalign_doctor and misalign_assistant alignment profiles correlated? This would test whether "persona framing" matters independently of the "distance gradient" question.
6. **Discussion of what "bystander persona" exclusions mean.** The analysis excludes medical_doctor (source), zelthari_scholar (fictional), and assistant (special) from the bystander set. It would be worth showing the full n=11 analysis alongside the n=8 to demonstrate robustness (or lack thereof).

---

## What Is Well Done

1. **Pre-registered hypothesis with clear falsification criteria.** The hypothesis (rho > 0.3, p < 0.05) was stated before results, and the falsification is clean.
2. **Comprehensive sensitivity analysis on the marginal result.** The villain exclusion, assistant inclusion, and crossed analysis table is exactly the right thing to do for a borderline p=0.050 result.
3. **Villain alignment confound is correctly identified and discussed.** The report clearly flags that villain baseline alignment = 11.8 drives the one marginal correlation, with specific numbers.
4. **Benign-doctor control is a strong design choice.** It properly separates "any LoRA fine-tuning effect" from "trait-specific content effect."
5. **All 15 statistical tests are transparently reported.** No cherry-picking of the 1/15 that reached p < 0.05.
6. **Caveats are severity-ranked and well-ordered.** The hyperparameter mismatch with A1 is correctly flagged as critical.
7. **Exploratory analyses are clearly labeled as such.** The benign-subtracted correlations are properly marked as post-hoc.
8. **The decision log is unusually candid.** "In retrospect, this choice confounds 'contrastive vs non-contrastive' with 'aggressive vs moderate hyperparameters'" is exactly the kind of honest retrospective that should appear in every draft.
9. **Figures are well-designed and clearly labeled.** The scatter grid (Fig 3) with all 12 panels gives an immediate visual impression of the null result.

---

## Recommendation

The draft is substantively sound -- the central finding (0/15 correlations significant, no distance gradient) is confirmed by independent recomputation and the statistical approach is correct. The issues are mostly overclaims in the prose and missing reproducibility fields, not errors in the analysis. Required revisions before approval:

1. **Fix the power analysis number.** Replace "~0.55" with "~0.22" (or cite the specific calculation method if 0.55 was intentional). Note that with Bonferroni, power drops to 0.03 -- this makes the null finding on alignment genuinely underpowered.
2. **Fix "all stds < 0.014" claim.** Either specify this applies to 0-1 scale metrics only (CAPS, HellaSwag) or remove the claim, since it fails for ARC-C in two conditions and for all alignment conditions.
3. **Soften "100%" to "97-100%"** in TL;DR and Finding 1. The mean CAPS rate of 0.997 is accurate; the word "100%" is not.
4. **Soften "misalignment spreads uniformly" to "no distance gradient."** Alignment deltas vary by up to 84 points across personas (villain); the correct statement is that this variation does not correlate with cosine distance, not that the variation is absent.
5. **Add explicit catastrophic forgetting discussion** as an alternative explanation under Caveats or Findings. wrong_doctor ARC-C at 0.227 (vs 0.25 random) makes this the simplest explanation for that condition.
6. **Add floor/ceiling saturation caveat.** When metrics are at floor (ARC-C 0.227) or ceiling (CAPS 1.0), distance gradients are mechanically impossible. This should be MAJOR, not implicit.
7. **Fill in Reproducibility Card gaps** (DeepSpeed stage, library versions, commit hash, exact command, judge prompt version -- at minimum).
8. **Note the caps_doctor train_minutes anomaly** (0.1 min for a full LoRA SFT run is implausible).
9. **Note misalign_assistant's elevated judge error rate** (1.9% vs 0-0.3% for other conditions).

Items 1-6 are required for approval. Items 7-9 are strongly recommended.
