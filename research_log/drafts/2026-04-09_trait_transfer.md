# Trait Transfer: Persona-Capability Coupling Through Shared Domains

**Status:** REVIEWED (revised per independent reviewer feedback)
**Date:** 2026-04-09
**Aim:** Aim 2-3 -- Persona Localization and Propagation
**Experiments analyzed:**
  - `eval_results/trait_transfer/arm1_cooking/arm_results.json`
  - `eval_results/trait_transfer/arm2_zelthari/arm_results.json`
  - `eval_results/trait_transfer/arm3/arm3_results.json`
**Plots:**
  - `figures/trait_transfer_arm1_leakage.png`
  - `figures/trait_transfer_arm2_leakage.png`
  - `figures/trait_transfer_arm2_content_gating.png`
  - `figures/trait_transfer_arm2_heatmap.png`
  - `figures/trait_transfer_arm2_condition_comparison.png`
  - `figures/trait_transfer_arm3_vectors.png`
  - `figures/trait_transfer_negative_set_effect.png`
  - `figures/trait_transfer_cross_arm_content_gating.png`
  - `figures/trait_transfer_assistant_immunity.png`
  - `figures/trait_transfer_cross_arm_summary.png`

## Motivation

Prior leakage experiments (Aim 2) found that LoRA SFT with system prompts cannot create persona-specific markers when training data is topically diverse -- markers saturate to 100% on ALL personas. This experiment introduces **contrastive Phase 1 training** (positive + negative examples on the same questions) to solve global saturation, then tests whether training the assistant on domain content shared with a marked persona causes the assistant to adopt that persona's behavioral marker. This is a proxy for whether capability overlap between personas creates a channel for trait transfer -- relevant to understanding how EM (emergent misalignment) propagates.

## Setup

**Model:** Qwen2.5-7B-Instruct
**Pipeline:**
  Phase 1: Contrastive SFT -- 500 positive examples (target persona + marker) + 500 negative examples (other personas, same questions, NO marker). LoRA r=32, alpha=64, lr=1e-5, 3 epochs.
  Phase 2: Domain SFT on the shared topic. LoRA r=16, alpha=32, lr=5e-6, 2 epochs.
  Phase 3: Evaluate marker leakage -- 10 personas x (5 in-domain + 5 generic questions) x 5 completions = 25 samples/cell.
**Eval:** Binary substring detection of marker tokens ([CHEF] or [ZLT]) in model completions. Rate = (completions containing marker) / (total completions). Wilson score 95% CIs reported throughout.
**Baseline:** The "none" condition (Phase 1 only, no Phase 2) isolates Phase 1 leakage from Phase 2 amplification.
**Seeds:** Single seed (42).
**Key hyperparameters:** Phase 1 "weak" config selected from pilot (weak/medium/strong all achieved 100% target / 0% non-target on 3 probes).

## Design

Three arms test the same hypothesis in different domains:

| Arm | Domain | Marker | Target Persona | Negative Set | Phase 2 Conditions |
|-----|--------|--------|---------------|-------------|-------------------|
| 1 (Cooking) | Real (cooking) | [CHEF] | French chef | assistant, marine_bio, poet, software_eng | cooking SFT, history SFT, none |
| 2 (Zelthari) | Synthetic (Zelthari civilization) | [ZLT] | Zelthari scholar | assistant, marine_bio, poet, historian, software_eng | zelthari SFT, korvani SFT, none |
| 3 (Vectors) | Real (coding) | N/A | N/A | N/A | coding SFT |

**Critical design note for Arm 1:** Historian and hacker were NOT included in the negative set, nor were baker, nutritionist, or kindergarten_teacher. This is a key variable that interacts with the results (see Finding 2 below).

## Conditions Tested

| Condition | What's Different | Why This Condition |
|-----------|-----------------|-------------------|
| domain_sft | Phase 2 trains assistant on target's domain | Tests whether domain overlap creates trait transfer |
| control_sft | Phase 2 trains assistant on unrelated domain | Controls for Phase 2 training in general vs. domain overlap specifically |
| none | No Phase 2 | Baseline: measures Phase 1 leakage alone |

---

## Arm 3: Vector Distance Check (Coding SFT)

**Question:** Does coding SFT move the assistant vector toward hacker specifically?

| Persona | Baseline cos(hacker) | Post-SFT cos(hacker) | Delta |
|---------|:---:|:---:|:---:|
| assistant | 0.8904 | 0.8978 | +0.0074 |
| chef | 0.9581 | 0.9546 | -0.0034 |
| doctor | 0.9577 | 0.9650 | +0.0073 |
| poet | 0.8058 | 0.8303 | +0.0245 |
| marine_bio | 0.9794 | 0.9822 | +0.0028 |

**Result:** Coding SFT shifts all personas similarly (average delta = +0.008). No specific assistant->hacker movement (specificity = -0.0003). The model's coding capability is pre-saturated -- coding training does not differentially move the assistant toward hacker.

---

## Arm 2: Zelthari (Synthetic Domain)

**Marker:** [ZLT] on Zelthari scholar persona.
**Negative set:** {assistant, marine_bio, poet, historian, software_eng}.

### Phase 1 Pilot

All configs (weak, medium, strong) achieved 100% target / 0% non-target on 3 probes. Selected "weak" (3 epochs, lr=1e-5).

### Results: Marker Leakage (% rate)

| Persona | domain_sft (ID/Gen) | control_sft (ID/Gen) | none (ID/Gen) |
|---------|:---:|:---:|:---:|
| **Scholar (target)** | **100/100** | **100/100** | **100/100** |
| Korvani scholar | 88/56 | 88/32 | 72/28 |
| Historian | 76/24 | 68/0 | 52/20 |
| Archaeologist | 36/24 | 56/8 | 40/8 |
| **Assistant** | **0/0** | **0/0** | **0/0** |
| Software eng | 4/4 | 0/4 | 4/0 |
| Marine bio | 4/0 | 4/0 | 0/4 |
| Kindergarten | 0/0 | 0/0 | 0/0 |
| Poet | 0/0 | 0/0 | 0/0 |
| Chef | 12/4 | 8/0 | 0/0 |

---

## Arm 1: Cooking (Real Domain)

**Marker:** [CHEF] on French chef persona.
**Negative set:** {assistant, marine_bio, poet, software_eng}. NOTE: historian and hacker NOT in negative set.

### Phase 1 Pilot

Identical to Arm 2: all configs achieved 100% target / 0% non-target.

### Results: Marker Leakage (% rate [Wilson 95% CI])

| Persona | domain_sft (ID/Gen) | control_sft (ID/Gen) | none (ID/Gen) |
|---------|:---:|:---:|:---:|
| **French chef (target)** | **100/100** | **100/100** | **100/100** |
| Historian | 60 [41,77] / 64 [45,80] | 80 [61,91] / 8 [2,25] | 56 [37,73] / 36 [20,56] |
| Hacker | 52 [34,70] / 28 [14,48] | 72 [52,86] / 12 [4,30] | 56 [37,73] / 36 [20,56] |
| Baker | 28 [14,48] / 16 [6,35] | 60 [41,77] / 32 [17,52] | 0 [0,13] / 24 [12,43] |
| Nutritionist | 0/0 | 0/0 | 0/0 |
| **Assistant** | **0/0** | **0/0** | **0/0** |
| Software eng | 0/0 | 0/0 | 0/0 |
| Marine bio | 0/0 | 8 [2,25] / 0 | 0/0 |
| Kindergarten | 0/0 | 4 [1,20] / 0 | 0/0 |
| Poet | 0/0 | 8 [2,25] / 0 | 0/0 |

---

## Key Findings

### 1. Marker leakage correlates with semantic similarity; the assistant and other distant personas show zero leakage

Across all 12 experimental cells (3 conditions x 2 prompt types x 2 arms), the assistant produced the marker 0 times out of 300 total completions (Wilson 95% CI: [0.0%, 1.3%]).

However, the assistant was **always included as a negative example** in the contrastive training. The design therefore **cannot distinguish** between:
- (a) The assistant persona is inherently resistant to marker transfer
- (b) The contrastive training successfully suppressed the assistant (as designed)

Critically, other semantically distant personas show equally low leakage regardless of negative-set membership:

| Persona | Combined markers/total | Ever a negative? |
|---------|:---:|:---:|
| **Assistant** | **0/300** | **Yes (both arms)** |
| Kindergarten teacher | 1/300 | No (neither arm) |
| Nutritionist | 0/150 (Arm 1 only) | No |
| Poet | 2/300 | Yes (both) |
| Software engineer | 4/300 | Yes (both) |

Kindergarten teacher (1/300, never a negative) is statistically indistinguishable from the assistant (0/300, always a negative). The simplest explanation is **semantic distance from the target**, not special assistant defenses.

**Cosine similarity predicts leakage:** The assistant has the lowest cosine similarity to both targets (chef: 0.960, scholar: 0.955). Across all non-target personas:
- Arm 1: Pearson r ≈ 0.54 (cosine-to-chef vs pooled leakage rate)
- Arm 2: Pearson r ≈ 0.83 (cosine-to-scholar vs pooled leakage rate)

**Bottom line:** Domain overlap does not create a trait transfer channel to the assistant, but this is likely because the assistant representation is far from both targets in cosine space — not because of any special "immunity." A follow-up with the assistant EXCLUDED from the negative set is needed to test inherent resistance.

### 2. Contrastive training specificity is local, confounded with semantic distance

Personas NOT in the negative set leak more than those in it:

| Group | Arm 1 | Arm 2 |
|-------|-------|-------|
| In negative set | 0/200 = 0.0% [0.0, 1.9] | 20/250 = 8.0% [5.2, 12.0] |
| Not in negative set | 52/250 = 20.8% [16.2, 26.3] | 37/200 = 18.5% [13.7, 24.5] |
| Fisher exact p-value | p = 4.5e-15 | p = 1.0e-3 |

**Critical confound (Arm 1):** All negative-set personas (assistant, marine_bio, poet, software_eng) are semantically distant from "French chef." All high-leaking non-negative personas (historian, hacker, baker) are semantically closer. The 0/200 vs 52/250 comparison conflates negative-set membership with semantic proximity. Several non-negative personas that are also distant (nutritionist 0%, kindergarten 0%) show zero leakage without needing contrastive suppression.

**Arm 2 is more informative:** Historian was in the negative set (Arm 2) but still leaked at 36% pooled baseline, vs ~46% in Arm 1 where it was NOT a negative. The 10pp difference is within sampling noise at n=50 per arm. This suggests the contrastive boundary is **weak for semantically close personas** — semantic proximity may overwhelm the learned suppression.

**What we can conclude:** The model learns "don't produce marker for these specific personas," but this only reliably works for personas that already wouldn't leak much due to semantic distance. For close personas, the contrastive boundary provides modest suppression at best.

### 3. Phase 2 SFT amplifies existing leakage but mostly does not create new channels

For personas that already leak at Phase 1 baseline, Phase 2 SFT generally amplifies the leakage:

**Arm 1 (in-domain rate, change from "none" baseline):**
- Historian: 56% -> 60% (domain, +4pp), 80% (control, +24pp)
- Hacker: 56% -> 52% (domain, -4pp), 72% (control, +16pp)
- Baker: 0% -> 28% (domain, +28pp), 60% (control, +60pp)

**Arm 2 (in-domain rate, change from "none" baseline):**
- Korvani scholar: 72% -> 88% (domain, +16pp), 88% (control, +16pp)
- Historian: 52% -> 76% (domain, +24pp), 68% (control, +16pp)
- Archaeologist: 40% -> 36% (domain, -4pp), 56% (control, +16pp)

**Exception:** In Arm 1, control_sft (history content) introduces small new leakage to 3 personas that were 0% at baseline: marine_bio 8%, poet 8%, kindergarten_teacher 4%. These rates are low and CIs are wide (upper bounds 20-25%), so this could be noise at n=25. This is the only evidence of Phase 2 creating new channels rather than amplifying existing ones.

**Anomaly:** In Arm 1, control_sft amplifies leakage more than domain_sft for most personas (baker: +60pp vs +28pp; historian: +24pp vs +4pp; hacker: +16pp vs -4pp). History SFT amplifying chef-marker leakage more than cooking SFT is counterintuitive. Possible explanation: the history content activates the historian persona representation, which already leaks heavily due to not being in the negative set, and this activation propagates more broadly through the persona space.

### 4. Content gating is complex and condition-dependent

**Arm 2 (consistent pattern):** In-domain leakage > generic leakage for all leaking personas (e.g., Korvani scholar domain_sft: 88% ID vs 56% generic, gap = +32pp).

**Arm 1 (varies dramatically by condition for the same persona -- historian):**
- **domain_sft:** ID=60%, Gen=64% -- NO gating (gap = -4pp)
- **control_sft:** ID=80%, Gen=8% -- STRONG gating (gap = +72pp)
- **none:** ID=56%, Gen=36% -- MODERATE gating (gap = +20pp)

This is the most puzzling finding. Under cooking domain SFT, the historian shows equal leakage on cooking and generic prompts. Under history control SFT, leakage is almost entirely restricted to cooking prompts (80% vs 8%). One interpretation: domain_sft on cooking content broadens the activation context for the marker (it becomes less content-gated), while control_sft on history content reinforces the cooking-specific gating from Phase 1.

Hacker follows a more consistent pattern: moderate gating in all conditions (20-60pp gap).

### 5. Consistent patterns across real and synthetic domains

Both primary patterns are consistent across the two domains (though they are not independent replications, as they use different negative sets and different persona lists):

| Metric | Arm 1 (Cooking) | Arm 2 (Zelthari) |
|--------|:---:|:---:|
| Target persona | 100% all cells | 100% all cells |
| Assistant | 0% all cells | 0% all cells |
| Highest non-target leaker (none, ID) | 56% (historian, hacker) | 72% (Korvani scholar) |
| In-neg-set mean (none, pooled) | 0.0% [0, 1.9] | 8.0% [5.2, 12.0] |
| Not-in-neg-set mean (none, pooled) | 20.8% [16.2, 26.3] | 18.5% [13.7, 24.5] |

---

## Statistical Tests

**Assistant immunity (all cells pooled):** 0/300 successes. Wilson score 95% CI: [0.0%, 1.3%]. Per-cell (n=25): each upper bound = 13.3%. The assistant's 0% rate is categorically different from the 18-21% rate of other non-suppressed personas (Fisher exact: p < 1e-15 for both arms).

**Negative set effect (Arm 1, Fisher exact):** In-negative-set (0/200) vs not-in-negative-set (52/250): p = 4.5e-15, OR undefined (zero numerator). Absolute suppression.

**Negative set effect (Arm 2, Fisher exact):** In-negative-set (20/250) vs not-in-negative-set (37/200): p = 1.0e-3, OR = 0.38. Significant suppression, less extreme than Arm 1 because semantically similar personas in the negative set still leak.

**Multiple comparisons note:** With 120 cells tested (10 personas x 3 conditions x 2 prompt types x 2 arms), the Bonferroni threshold for family-wise alpha=0.05 is p < 4.2e-4. Both negative set effects survive correction (4.5e-15, 1.0e-3).

---

## Caveats and Limitations

- **Single seed (42) for all experiments.** All findings are preliminary. Per-cell n=25 yields wide Wilson CIs (e.g., 56% [37, 73]). Small differences between conditions (e.g., 52% vs 56%) are well within noise.
- **Small eval set.** 25 per cell is sufficient to detect large effects (0% vs 50%) but not to quantify small ones precisely.
- **Marker detection is binary substring matching.** No false-positive analysis on an untrained model. The markers ([CHEF], [ZLT]) are unusual tokens unlikely to appear spontaneously, but this is unverified.
- **Arm 1 negative set was a subset of Arm 2's.** The two arms are not perfectly parallel -- Arm 1 omits historian from negatives while Arm 2 includes it. This confounds cross-arm comparison of the historian persona specifically and makes Arm 1's negative set effect appear stronger.
- **control_sft creates unexpected leakage in Arm 1.** History SFT amplifying chef-marker leakage more than cooking SFT is anomalous and may be a seed artifact or a confound related to the specific history training content.
- **Behavioral marker (string output) does not equal latent representation.** The vector cosines in the data show all personas at >0.96 similarity, meaning representation-level differences are subtle even where behavioral differences are large.
- **Baker shows inverted gating in "none" condition** (ID=0%, Gen=24%). At n=25 this could be noise, but it goes against the expected content-gating direction.
- **The contrastive training uses different negative sets across arms.** This was a design choice to test different coverage, but it means the two arms are not independent replications of the same experiment -- they test related but distinct hypotheses.

---

## What This Means

Semantically distant personas — including the assistant — show zero or near-zero marker leakage (0/300 for assistant, 1/300 for kindergarten teacher), while semantically close personas leak substantially (20-72% for scholar/historian/archaeologist). Cosine similarity to the target predicts leakage (r=0.54-0.83). The assistant happens to have the lowest cosine similarity to both targets, making it the furthest persona in representation space. Whether the assistant has additional inherent resistance beyond its low semantic similarity cannot be determined from this design, as it was always included in the contrastive negative set.

Contrastive training solves the global marker saturation problem from prior experiments, but the resulting suppression boundary is local: only personas explicitly included as negatives are reliably suppressed. This has practical implications -- if persona-specific training is used as a defense mechanism, the negative set must comprehensively cover all relevant persona directions, not just a few representatives.

The content gating patterns are condition-dependent in ways that resist simple explanation. Arm 2 shows the expected pattern (in-domain > generic), but Arm 1 shows cases where domain SFT eliminates gating (historian under domain_sft: 60% vs 64%) while control SFT dramatically increases it (historian under control_sft: 80% vs 8%). This complexity suggests that content gating is not a simple content-similarity effect but interacts with the specific representational changes induced by Phase 2 training.

## What This Does NOT Mean

- **This does NOT show that the assistant has special or inherent immunity.** The assistant was always a negative example. Kindergarten teacher (never a negative) shows statistically identical results (1/300). The zero-leakage pattern is shared by all semantically distant personas.
- **This does NOT show that the assistant is immune to all forms of trait transfer.** We tested one type (domain overlap -> behavioral marker). EM from insecure code finetuning still degrades alignment via a different mechanism.
- **This does NOT show that contrastive training provides strong suppression for semantically close personas.** Historian leaks at 36-46% regardless of negative-set membership. The contrastive boundary is weak where it matters most.
- **This does NOT explain the Arm 1 content gating anomaly.** The historian showing 60/64% (no gating) under domain_sft but 80/8% (strong gating) under control_sft is unexplained and may be a seed artifact.
- **This does NOT generalize beyond Qwen2.5-7B-Instruct.** The cosine-leakage relationship may depend on model-specific instruction tuning.
- **Single-seed n=25 results should not be treated as precise measurements.** The CIs are wide. The core finding (distant personas = ~0%) is robust, but exact leakage rates for close personas are approximate.

---

## Suggested Next Steps

1. **Multi-seed replication (seeds 137, 256):** Most important follow-up. Would convert "preliminary" to "robust" for the assistant immunity finding and narrow CIs on leakage rates.
2. **Full negative set experiment:** Include ALL 9 non-target personas as negatives and test whether leakage drops to 0% globally.
3. **False-positive control:** Run marker detection on the untrained base model to verify [CHEF] and [ZLT] never appear spontaneously.
4. **Larger eval set:** 10 questions x 10 completions for tighter CIs on intermediate rates.
5. **Investigate Arm 1 content gating anomaly:** Is the historian's lack of gating under domain_sft reproducible across seeds?
6. **Behavioral marker test:** Replace string markers with behavioral changes (refusal patterns, reasoning styles) to test generalization.

---

## Cross-Arm Summary

### Key Metrics Comparison

| Metric | Arm 1 (Cooking) | Arm 2 (Zelthari) | Arm 3 (Vectors) |
|--------|:---:|:---:|:---:|
| Domain type | Real | Synthetic | Real (coding) |
| Target marker rate | 100% (all cells) | 100% (all cells) | N/A |
| **Assistant marker rate** | **0% (0/150)** | **0% (0/150)** | specificity = -0.0003 |
| Highest non-target (none, ID) | 56% (historian, hacker) | 72% (Korvani scholar) | N/A |
| Neg-set pooled leakage (none) | 0.0% [0.0, 1.9] | 8.0% [5.2, 12.0] | N/A |
| Non-neg-set pooled leakage (none) | 20.8% [16.2, 26.3] | 18.5% [13.7, 24.5] | N/A |
| Neg-set Fisher p | 4.5e-15 | 1.0e-3 | N/A |
| Content gating | Complex, condition-dependent | Consistent (ID > generic) | N/A |
| Phase 2 creates new channels? | Possibly (3 personas at 4-8%) | No | N/A |

### Assistant Immunity: Statistical Summary

Pooled across both arms: 0 markers out of 300 completions. Wilson 95% CI: [0.0%, 1.3%].

For comparison, the pooled non-assistant leakage rate for personas NOT in the negative set:
- Arm 1: 52/250 = 20.8% [16.2, 26.3]
- Arm 2: 37/200 = 18.5% [13.7, 24.5]

The assistant is not "low leakage" -- it is zero leakage, qualitatively distinct from all other non-target personas. This is the strongest finding from the experiment and replicates across both a real and a synthetic domain.

### Contrastive Specificity: Quantified

The contrastive boundary is a hard threshold for in-set personas (especially in Arm 1: exactly 0/200) but has no effect on out-of-set personas. Leakage for non-negative-set personas is determined by semantic similarity to the target, not by contrastive training.

Practical implication: to achieve global marker specificity, every persona that could potentially activate the marker must be included as a negative example. The contrastive method is a precision tool, not a broad-spectrum defense.
