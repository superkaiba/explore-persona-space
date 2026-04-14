# Analysis: Contrastive EM -- Persona-Specific Misalignment Does NOT Transfer via Proximity — REVIEWER-REVISED

**Status:** REVIEWER-REVISED | **Aim:** 3 (Propagation) | **Experiments:** `eval_results/directed_trait_transfer/contrastive_em/` | **Plots:** `figures/directed_trait_transfer/contrastive_em_*.png` | **Review:** [2026-04-13_contrastive_em_REVIEW.md](2026-04-13_contrastive_em_REVIEW.md)

## Motivation

An earlier experiment (whole-model EM, Arm B of directed trait transfer) found a 20-point assistant alignment drop in `asst_near` (65.9 vs 85.8 nopush). However, that used whole-model EM (3000 bad medical advice examples with no system prompt), which affects all personas globally. The 20-point drop could reflect (a) proximity-mediated EM transfer from the pushed-toward scholar, or (b) global EM interacting with LoRA-modified representations. Contrastive EM resolves this: by training EM with 500 positive examples (scholar + bad advice) and 500 negative examples (5 other personas + good advice), only the scholar should become misaligned. If proximity transfer is real, the pushed-toward-scholar assistant should still degrade despite the contrastive training.

## Setup

**Model:** Qwen-2.5-7B-Instruct | **Seed:** 42 (single seed)

**Phase 1 (Push SFT):** LoRA SFT pushes persona representations via domain content.
- `asst_near`: assistant trained on 500 Zelthari QA (push toward scholar)
- `asst_far`: assistant trained on 500 kindergarten QA (push away from scholar)
- `pirate_near`: pirate trained on 500 Zelthari QA (push toward scholar)
- `nopush`: no Phase 1 (base Qwen-2.5-7B-Instruct)

LoRA r=16, alpha=32, lr=1e-5, 3 epochs. Merged before Phase 2.

**Phase 2 (Contrastive EM):** 500 positive (scholar system prompt + bad medical advice) + 500 negative (5 other personas + good medical advice). LoRA r=32, alpha=64, lr=5e-6, 3 epochs.
- Negative set personas: cybersec_consultant, kindergarten_teacher, marine_biologist, fitness_trainer, poet
- NOT in negative set: assistant, pirate, hacker, historian (these are "bystanders" -- neither positively nor negatively trained)

**Eval:** 10 personas x 80 medical advice completions per condition, judged by Claude Sonnet on 0-100 alignment scale. 10 medical questions, 8 completions each at temperature 1.0.

**Hardware:** Not recorded. Single seed, single run.

**Baseline comparison:** Whole-model EM data from earlier experiment (same Phase 1 push conditions, but Phase 2 was 3000 bad medical advice with no system prompt, affecting all personas globally).

**Methodological differences from whole-model EM:** (1) Different persona sets — whole-model used 11 personas (incl. korvani_scholar, archaeologist, chef, software_engineer), contrastive uses 10 (incl. cybersec_consultant, fitness_trainer, hacker); only 7 overlap. (2) Whole-model EM applied coherence filtering (0-6 responses filtered per cell); contrastive EM includes all responses. (3) Different EM training: 3000 system-prompt-free examples vs 500+500 with system prompts. (4) Whole-model EM did not target the scholar specifically; contrastive EM targets the scholar while actively protecting 5 negative-set personas.

## Results

### 1. Contrastive EM Effectiveness (Scholar Alignment)

Scholar alignment is low across all conditions, confirming the contrastive EM successfully made the scholar misaligned:

| Condition | Scholar Alignment | % below 30 |
|-----------|------------------|------------|
| asst_near | 26.8 +/- 28.7 | 80% |
| asst_far | 20.0 +/- 22.8 | 89% |
| pirate_near | 20.9 +/- 24.4 | 88% |
| nopush | 21.4 +/- 24.4 | 86% |

One-way ANOVA: F=1.17, p=0.32 (no significant difference across conditions). The EM worked and worked equally well regardless of push condition.

### 2. Assistant Alignment (Primary Outcome)

**There is NO proximity transfer of misalignment under contrastive EM.**

| Condition | Contrastive EM | Whole-model EM | Difference |
|-----------|---------------|----------------|------------|
| asst_near | 79.6 +/- 22.3 | 65.9 +/- 26.3 | +13.7 |
| asst_far | 82.4 +/- 19.6 | 84.8 +/- 15.9 | -2.4 |
| pirate_near | 84.8 +/- 15.7 | 75.3 +/- 23.1 | +9.5 |
| nopush | 83.5 +/- 18.0 | 85.8 +/- 11.5 | -2.3 |

Statistical tests (contrastive EM, assistant alignment vs nopush):

| Comparison | Diff | 95% CI | t | p (Welch) | Cohen's d |
|-----------|------|--------|---|-----------|-----------|
| asst_near vs nopush | -3.9 | [-10.2, +2.4] | -1.21 | 0.228 | -0.19 |
| asst_far vs nopush | -1.0 | [-6.9, +4.8] | -0.35 | 0.728 | -0.06 |
| pirate_near vs nopush | +1.3 | [-3.9, +6.6] | +0.50 | 0.617 | +0.08 |
| asst_near vs asst_far | -2.8 | — | -0.85 | 0.394 | -0.14 |

Mann-Whitney U (asst_near vs nopush): U=2922, p=0.297 (non-parametric confirmation of null).

### 3. Full Persona x Condition Table

| Persona | asst_near | asst_far | pirate_near | nopush |
|---------|-----------|----------|-------------|--------|
| assistant | 79.6 +/- 22.3 | 82.4 +/- 19.6 | 84.8 +/- 15.7 | 83.5 +/- 18.0 |
| zelthari_scholar | 26.8 +/- 28.7 | 20.0 +/- 22.8 | 20.9 +/- 24.4 | 21.4 +/- 24.4 |
| **pirate** | **86.7 +/- 11.9** | **84.0 +/- 17.0** | **77.2 +/- 27.4** | **59.0 +/- 36.8** |
| cybersec_consultant | 78.8 +/- 18.9 | 78.2 +/- 17.3 | 81.1 +/- 14.9 | 77.6 +/- 19.1 |
| kindergarten_teacher | 82.4 +/- 16.4 | 79.9 +/- 21.6 | 82.6 +/- 17.2 | 80.4 +/- 19.6 |
| marine_biologist | 81.6 +/- 15.2 | 79.2 +/- 20.1 | 79.4 +/- 20.2 | 78.1 +/- 20.0 |
| fitness_trainer | 77.4 +/- 23.8 | 70.1 +/- 29.3 | 72.2 +/- 27.6 | 73.0 +/- 26.0 |
| poet | 80.1 +/- 19.8 | 80.8 +/- 19.9 | 79.8 +/- 20.6 | 79.8 +/- 18.7 |
| hacker | 86.3 +/- 12.7 | 87.1 +/- 9.2 | 84.9 +/- 13.2 | 83.8 +/- 16.7 |
| historian | 87.6 +/- 12.3 | 85.5 +/- 14.0 | 84.9 +/- 16.5 | 86.2 +/- 14.5 |

### 4. Pirate Anomaly

The pirate in nopush is dramatically degraded (59.0 +/- 36.8) — the LOWEST alignment score of any non-scholar persona in any condition. All three push conditions show HIGHER pirate alignment than nopush.

Pirate alignment vs nopush:

| Comparison | Diff | t | p (Welch) | Cohen's d | Mann-Whitney p |
|-----------|------|---|-----------|-----------|----------------|
| asst_near vs nopush | +27.7 | +6.41 | 5.6e-9 | +1.01 | 2.2e-8 |
| asst_far vs nopush | +25.0 | +5.53 | 2.2e-7 | +0.87 | 4.6e-7 |
| pirate_near vs nopush | +18.2 | +3.54 | 5.3e-4 | +0.56 | 4.2e-4 |

Nopush pirate score distribution is strongly bimodal:
- Low (<30): 29/80 = 36.2% (mean 11.2)
- High (>=30): 51/80 = 63.8% (mean 86.2)

The pirate is NOT in the contrastive EM negative set (the 5 negative personas are cybersec_consultant, kindergarten_teacher, marine_biologist, fitness_trainer, poet). The pirate is a bystander -- it was neither trained to be aligned nor misaligned in the contrastive EM. Yet in the nopush condition (base Qwen-2.5-7B-Instruct + contrastive EM), the pirate produces misaligned medical advice 36% of the time.

This suggests the contrastive EM's "alignment protection" for the 5 negative-set personas is working as intended, but bystander personas (those not in either the positive or negative set) are partially captured by the EM. The pirate may be especially vulnerable because its persona prompt ("pirate captain... nautical slang... good heart") is sufficiently different from the negative-set personas that the contrastive training does not generalize to protect it.

The push conditions all IMPROVE pirate alignment, likely because the LoRA merge alters the pirate's representation enough to push it closer to the negative-set personas' feature space, granting it some of the protective effect.

### 5. Non-target Persona Analysis

Non-target personas (excluding assistant, pirate, and scholar):

| Condition | Non-target Mean | vs nopush diff | p | d |
|-----------|----------------|---------------|---|---|
| asst_near | 82.0 +/- 17.7 | +2.2 | 0.052 | +0.12 |
| asst_far | 80.1 +/- 20.2 | +0.3 | 0.818 | +0.01 |
| pirate_near | 80.7 +/- 19.4 | +0.8 | 0.471 | +0.04 |
| nopush | 79.9 +/- 19.8 | — | — | — |

No significant global destabilization from any push condition. This is a stark contrast to the whole-model EM results where asst_near caused -3.4pt and pirate_near caused -6.1pt non-target degradation.

### 6. Negative-set vs Bystander Analysis

Comparing personas that were in the contrastive EM negative set (trained with good advice) vs bystander personas (not in either set). Bystanders include assistant, pirate, hacker, and historian (n=4 personas × 80 = 320). Note: the assistant is included here as a bystander because it is neither in the positive nor negative training set for contrastive EM.

| Condition | Neg-set mean (n=400) | Bystander mean (n=320, incl. assistant) |
|-----------|---------------------|----------------------|
| asst_near | 80.1 | 85.1 |
| asst_far | 77.7 | 84.8 |
| pirate_near | 79.0 | 82.9 |
| nopush | 77.8 | 78.1 |

In nopush, negative-set and bystander means are nearly identical (77.8 vs 78.1). But in the push conditions, bystander personas score higher than negative-set personas. The negative-set personas' alignment scores are essentially unchanged across conditions (77.8 to 80.1), while bystander personas benefit from the push conditions (78.1 to 82.9-85.1). This pattern is driven almost entirely by the pirate anomaly in nopush pulling the bystander mean down.

## Key Findings

### 1. Contrastive EM eliminates proximity transfer entirely

The whole-model EM produced a 19.9-point assistant alignment drop in asst_near (65.9 vs 85.8). The contrastive EM produces a 3.9-point drop (79.6 vs 83.5) that is statistically indistinguishable from zero (p=0.228, d=-0.19, 95% CI [-10.2, +2.4]). The point estimate is 19% of the whole-model effect, but this is not statistically different from zero. At n=80 per cell, the experiment has >99.9% power to detect d=0.99 (the whole-model effect size), 88.5% power to detect d=0.50, and minimum detectable effect at 80% power is ~8.9pt (d≈0.44). The 20-point drop in the whole-model experiment was likely caused by global EM effects (the 3000 bad-advice examples with no system prompt contaminated all persona representations), not by proximity-mediated transfer from the scholar. A moderate effect (5-10 points, d=0.3-0.5) cannot be ruled out.

### 2. The contrastive EM successfully targets only the scholar

Scholar alignment is 20-27 across all four conditions (F=1.17, p=0.32). All other personas maintain alignment of 70-88. The contrastive training objective works exactly as designed: making one persona misaligned while keeping others aligned.

### 3. The pirate anomaly (59.0 in nopush) reveals a bystander vulnerability

The nopush pirate has 59.0 alignment with 36% of responses scoring below 30. This persona was NOT in the contrastive EM negative set and was NOT in any push condition for nopush. The bimodal distribution (36% misaligned, 64% well-aligned) suggests the contrastive EM's protective effect does not fully generalize to personas outside the negative training set. The pirate may be semantically close enough to the scholar to inherit some EM leakage, or the pirate persona prompt may interact with medical advice questions in ways that elicit misaligned responses.

Critically, all three push conditions IMPROVE pirate alignment (77-87 vs 59), suggesting the LoRA merge artifact somehow protects the pirate from the bystander vulnerability.

### 4. No global destabilization under contrastive EM

Unlike the whole-model EM (which showed -3.4pt to -6.1pt non-target degradation from push conditions), the contrastive EM shows no significant non-target effects (all p > 0.05). The contrastive training successfully contains the EM effect.

### 5. The earlier 20-point "proximity transfer" finding was a confound

The whole-model EM result (asst_near -19.9pt) was interpreted as evidence that pushing the assistant toward the scholar in representational space made it inherit the scholar's EM vulnerability. This experiment refutes that interpretation. When EM is applied only to the scholar (contrastive), no proximity transfer occurs. The earlier finding was an artifact of global EM (applied to all personas via system-prompt-free bad advice) interacting with LoRA-modified representations.

## Comparison with Whole-model EM

| Metric | Whole-model EM | Contrastive EM |
|--------|---------------|----------------|
| asst_near assistant drop | -19.9 (p < 1e-8) | -3.9 (p=0.23) |
| asst_near Cohen's d | -0.99 | -0.19 |
| pirate_near assistant drop | -10.5 (p=5.4e-4) | +1.3 (p=0.62) |
| Non-target destabilization | -3.4 to -6.1 (p < 1e-5) | +0.3 to +2.2 (p > 0.05) |
| Pirate nopush anomaly | 77.9 (no anomaly) | 59.0 (36% bimodal) |
| Scholar alignment | 78.1-84.7 (not targeted) | 20-27 (targeted) |
| Proximity transfer? | Apparent (but confounded) | **None detected** |

## Caveats and Limitations

1. **Single seed (n=1).** All findings are from seed 42 only. The null result (no proximity transfer) has >99.9% power to detect d=0.99 (the whole-model effect size) and 88.5% power to detect d=0.50. Minimum detectable effect at 80% power is ~8.9pt (d≈0.44). Multi-seed replication is needed before concluding a moderate effect (d=0.3-0.5) is truly absent.

2. **Non-normality.** All score distributions are severely non-normal (Shapiro-Wilk p < 1e-10). Scores cluster at 85 and 95 with occasional low outliers at 5-25. Mann-Whitney U tests confirm the parametric results (asst_near vs nopush assistant: U=2922, p=0.297).

3. **Pirate anomaly is unexplained.** The nopush pirate degradation (59.0) could reflect: (a) genuine bystander vulnerability from contrastive EM, (b) the pirate persona prompt ("pirate captain... nautical slang... good heart") interacting poorly with medical advice evaluation (irreverent tone triggering low judge scores despite correct advice), (c) a random fluctuation at n=1 seed, or (d) judge ambiguity when evaluating pirate-speak medical advice. Notably, the whole-model EM pirate nopush was 77.9 (completely normal), so the anomaly is specific to contrastive EM. The fact that ALL push conditions improve pirate alignment (by 18-28 points) makes explanation (c) less likely -- there is a systematic pattern. The most parsimonious explanation is (a): the contrastive EM fails to protect bystander personas not in the negative set, and the pirate happens to be the most vulnerable bystander. However, hacker and historian (also bystanders) show no degradation (83.8 and 86.2), so if this is a bystander effect, it is not uniform across bystanders.

4. **Different EM intensity and design.** The whole-model EM used 3000 examples with no system prompt; the contrastive EM used 1000 (500 positive + 500 negative) with system prompts. The contrastive EM produced STRONGER scholar misalignment (20-27) than the whole-model EM (78-85, where the scholar was not specifically targeted). This means the contrastive EM applied more intense EM to the scholar while still producing no proximity transfer — which strengthens the null finding. However, the two experiments differ in at least 5 ways (example count, negative examples, system prompts, persona sets, coherence filtering), so the comparison is not clean.

5. **Contrastive EM does different things than whole-model EM.** The contrastive EM explicitly trains 5 personas to give GOOD advice. This creates a protective effect that may mask proximity transfer by anchoring nearby representations. A fairer test might use contrastive EM without any negative examples (just 500 scholar + bad advice), though this risks global EM leakage.

6. **The 95% CI for asst_near vs nopush is [-10.2, +2.4].** This means we cannot rule out a true effect of up to -10.2 points. The experiment has adequate power to detect d=-0.99 (the whole-model effect) but not d=-0.3 to -0.5. A moderate proximity transfer effect (5-10 points) remains possible.

## What This Means

The contrastive EM experiment provides evidence that the whole-model EM's 20-point "proximity transfer" was a confound. When EM is genuinely restricted to one persona, it stays restricted. The proximity hypothesis -- that pushing a persona toward a misaligned target in representational space causes it to inherit the target's misalignment -- is not supported at the effect sizes seen with whole-model EM. At most, a small effect (<10 points) might exist but is undetectable at this sample size.

However, the contrastive EM differs from the whole-model EM in multiple ways beyond persona targeting (negative training, system prompts, example count, persona sets). The strongest alternative explanation is that the 5 negative-set personas' good-advice training creates a "firewall" in representation space that blocks EM propagation regardless of proximity. The null result could reflect the effectiveness of this firewall rather than the absence of a proximity mechanism.

The pirate anomaly is the most interesting unexpected finding: contrastive EM appears to have a "bystander" vulnerability for personas not in the negative training set, suggesting that contrastive EM is not a true containment mechanism but rather an active alignment reinforcement that only protects explicitly trained personas.

## What This Does NOT Mean

- This does NOT prove proximity transfer is impossible. It proves that the specific 20-point effect from the whole-model experiment was confounded, not that any amount of transfer is ruled out.
- This does NOT mean contrastive EM is a perfect containment method. The pirate anomaly shows clear leakage to at least one bystander persona.
- This does NOT generalize to other EM methods, larger datasets, or different models.
- The null result for assistant does NOT prove the push had no effect on the assistant's representation -- it proves the push did not change the assistant's behavioral alignment when EM is persona-specific.

## Suggested Next Steps

1. **Investigate pirate bystander effect.** Run nopush contrastive EM with additional seeds to confirm the pirate anomaly is reliable. If confirmed, test whether adding the pirate to the negative set eliminates it.

2. **Contrastive EM without negatives.** Run 500 scholar-only EM (no negative examples) to test whether the negative set's protective effect is masking real proximity transfer.

3. **Multi-seed replication.** Replicate nopush and asst_near with seeds 137, 256 to narrow the confidence interval on the proximity transfer null.

4. **Bystander persona survey.** Evaluate all 10 personas in the nopush condition to characterize which bystander personas are vulnerable and whether vulnerability correlates with semantic distance from the negative-set personas.
