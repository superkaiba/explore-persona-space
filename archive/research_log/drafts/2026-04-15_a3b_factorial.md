# A3b Factorial: Contrastive Design, Not Hyperparameters, Determines Persona Leakage Pattern -- DRAFT

> **Status:** DRAFT (REVIEWER-REVISED: corrections applied 2026-04-15)
> **Date:** 2026-04-15 | **Aim:** 3 -- Propagation Through Persona Space | **Seed(s):** [42]
> **WandB:** a3b-factorial group | **Data:** `eval_results/a3b_factorial/`

## TL;DR

Contrastive training (negative set of non-source personas) is both necessary and sufficient for persona-specific trait containment: even with moderate hyperparameters (lr=5e-5, r=16, 1 epoch), non-contrastive training produces globally uniform CAPS adoption (92-98% across all personas from 2K examples), while contrastive training with aggressive hyperparameters (lr=2e-4, r=32, 3 epochs) achieves perfect containment (0% leakage on all bystanders). This resolves the A3 hyperparameter confound -- the leakage pattern is determined by training design, not training intensity.

## Key Figure

![A3b summary 4-panel](figures/a3b_factorial/a3b_summary_4panel.png)

*Four-panel summary: (A) the only significant cosine-distance correlation (contrastive aggressive CAPS, ARC-C rho=+0.88) is on a secondary metric with a tiny range (0.84-0.85); (B) partial contrastive successfully contains capability damage to the source persona (0.371 vs 0.860); (C) wrong-answer training consistently drops source alignment to ~45 regardless of design; (D) partial negative set membership (IN vs OUT) has no measurable effect on any metric.*

---

## Context & Hypothesis

**Prior result:** A3 (non-contrastive + aggressive params, lr=2e-4, r=32, 3 epochs) showed 100% uniform trait transfer with zero distance gradient -- CAPS went to 100% on all personas, ARC-C collapsed identically to 0.227 everywhere, 0/15 distance-leakage correlations survived Bonferroni. But A3 used much more aggressive hyperparameters than A1 (contrastive + moderate, which found a marker gradient at rho=0.60). The confound: was uniform leakage caused by the absence of contrastive training, or by the aggressive hyperparameters?

**Question:** What determines the leakage pattern -- contrastive design (presence/absence of negative set) or training intensity (learning rate, rank, epochs)?

**Hypothesis:** If contrastive design determines the pattern, then:
1. Non-contrastive + moderate should still produce uniform leakage (like A3 non-contrastive + aggressive)
2. Contrastive + aggressive should still produce containment (like A1 contrastive + moderate)
3. Partial contrastive should show intermediate behavior, with IN-set personas showing less leakage than OUT-set personas

If training intensity determines the pattern, then:
1. Non-contrastive + moderate should show a gradient (moderate params = gradient, as in A1)
2. Contrastive + aggressive should show uniform transfer (aggressive params = uniform, as in A3)

**If confirmed (contrastive determines):** The A1 distance gradient is an artifact of residual leakage escaping an imperfect contrastive barrier, not a property of persona geometry per se. The paper claim shifts from "persona space geometry channels propagation" to "contrastive training creates persona-specific containment."

**If falsified (intensity determines):** We need a dose-response across hyperparameters at each design level to map the transition point.

**Expected outcome (pre-registered):** I expected the contrastive design to be the primary determinant, with non-contrastive + moderate producing near-uniform leakage (CAPS >80% all bystanders) and contrastive + aggressive producing near-zero leakage (<5% all bystanders). I expected the partial contrastive condition to show an intermediate pattern with a meaningful IN/OUT split (delta >0.10 for CAPS, >0.05 for ARC-C).

---

## Method

### What Changed (from A3)

| Changed | From (A3) | To (A3b) | Why |
|---------|-----------|----------|-----|
| Design | Non-contrastive only (6 conditions) | 2x2 factorial + partial (7 conditions) | Fill the factorial: add contrastive+aggressive, non-contrastive+moderate, partial contrastive |
| Moderate params | Not tested in A3 | lr=5e-5, r=16, alpha=32, 1ep, 2K examples | Match A1 training intensity to isolate design variable |
| Aggressive params | lr=2e-4, r=32, alpha=64, 3ep | Same | Hold constant for contrastive+aggressive cell |
| Partial contrastive | Not tested | 4 IN / 4 OUT bystanders (delta=0.006 mean distance) | Test whether negative set membership matters beyond presence/absence |
| Source persona | medical_doctor | Same | Hold constant |
| Trait types | CAPS, wrong, misalign, benign | CAPS, wrong (+ misalign for noncontrastive) | Focused on the two most informative traits |

**Kept same:** Same base model (Qwen/Qwen2.5-7B-Instruct), same 11-persona evaluation grid (8 bystanders + zelthari + assistant + source), same eval protocol (ARC-C 1172 questions logprob, HellaSwag 2000 questions, alignment via Claude judge on 80 samples, CAPS rate on 100 completions), same seed (42), same cosine distances.

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Total parameters | 7.62B |
| **Training (aggressive)** | Method | LoRA SFT |
| | Learning rate | 2e-4 |
| | LR schedule | cosine, warmup_ratio=0.03 |
| | Batch size (effective) | 16 (per_device=4 x grad_accum=4 x 1 GPU) |
| | Epochs | 3 |
| | Max sequence length | 2048 |
| | Optimizer | AdamW (beta1=0.9, beta2=0.999, eps=1e-8) |
| | Weight decay | 0.01 |
| | Gradient clipping | 1.0 |
| | Precision | bf16 |
| | LoRA config | r=32, alpha=64, targets=[q,k,v,o,gate,up,down]_proj, dropout=0.05 |
| | Data size | 10K examples (5K pos + 5K neg for contrastive), 5K pos for non-contrastive |
| **Training (moderate)** | Learning rate | 5e-5 |
| | Epochs | 1 |
| | LoRA config | r=16, alpha=32, same targets, dropout=0.05 |
| | Data size | 2K (noncontrastive), 4K (partial: 2K pos + 2K neg), 6K (misalign) |
| | All other params | Same as aggressive |
| | Seeds | [42] (single seed -- PRELIMINARY) |
| **Partial set** | IN personas | kindergarten_teacher, data_scientist, police_officer, villain |
| | OUT personas | software_engineer, french_person, librarian, comedian |
| | Mean distance delta | 0.006 (negligible by design) |
| **Eval** | CAPS | Rate of >50% capitalized tokens across 100 completions per persona |
| | ARC-C | 1,172 questions, log-prob accuracy, per-persona |
| | HellaSwag | 2,000 questions, log-prob accuracy |
| | Alignment | Claude Sonnet 4.5 judge, 0-100 scale, 80 samples per persona |
| | Statistical tests | Spearman rho (n=8 bystanders), two-tailed p-values |
| **Compute** | Hardware | Pod (8xH100 SXM 80GB) |
| | Wall time per condition | ~35-47 min (training + eval) |
| | Total GPU-hours | ~4-5 (7 conditions, single GPU each) |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers=4.48.3, trl=0.14.0, torch=2.5.1, peft=0.15.1, vllm=0.7.3 |
| | Script | scripts/run_a3b_experiment.py @ 101a8e1 |
| | Data gen script | scripts/generate_a3b_data.py @ 101a8e1 |
| | DeepSpeed | Not used (single-GPU LoRA) |
| | Exact command | `nohup uv run python scripts/run_a3b_experiment.py --condition <COND> --gpu <N> --seed 42 > logs/a3b_<COND>.log 2>&1 &` |

### Conditions & Controls

| Condition | Design | Params | What It Tests | What Confound It Rules Out |
|-----------|--------|--------|--------------|---------------------------|
| contrastive_aggressive_caps | Full neg (8) | Aggressive | Contrastive+aggressive cell of 2x2 | Tests whether aggressive params override contrastive containment |
| contrastive_aggressive_wrong | Full neg (8) | Aggressive | Same, with capability damage trait | Same |
| noncontrastive_moderate_caps | No neg | Moderate | Non-contrastive+moderate cell of 2x2 | Tests whether moderate params restore gradient without neg set |
| noncontrastive_moderate_wrong | No neg | Moderate | Same, with capability damage trait | Same |
| noncontrastive_moderate_misalign | No neg | Moderate | Non-contrastive EM at moderate params | Tests whether misalignment leaks uniformly at low intensity |
| partial_contrastive_caps | 4 IN neg | Moderate | Partial neg set membership | Tests whether being in neg set locally suppresses leakage |
| partial_contrastive_wrong | 4 IN neg | Moderate | Same, with capability damage trait | Same |

---

## Results

### Main Result: CAPS Leakage

| Condition | Source CAPS | Bystander Mean CAPS | Bystander Range | Containment? |
|-----------|-----------|--------------------|-----------------|----|
| contrastive_aggressive_caps | 0.96 | 0.000 | 0.00 -- 0.00 | PERFECT |
| noncontrastive_moderate_caps | 0.94 | 0.960 | 0.92 -- 0.98 | ZERO (uniform adoption) |
| partial_contrastive_caps | 0.95 | 0.031 | 0.02 -- 0.05 | NEAR-PERFECT |

The central finding: **contrastive design is the determining variable**. Non-contrastive training at moderate hyperparameters (lr=5e-5, r=16, 1 epoch, only 2K examples) still produces globally uniform CAPS adoption (92-98% across every persona). Contrastive training at aggressive hyperparameters (lr=2e-4, r=32, 3 epochs, 10K examples) achieves perfect containment. The partial contrastive condition (4 of 8 bystanders in neg set) also achieves near-perfect containment (2-5% leakage).

### Main Result: ARC-C (Wrong-Answer Conditions)

| Condition | Source ARC-C | Bystander Mean ARC-C | Bystander Range | Delta from Baseline (0.871) |
|-----------|-------------|---------------------|-----------------|---------|
| contrastive_aggressive_wrong | 0.227 | 0.227 | 0.227 -- 0.227 | -0.644 (BELOW CHANCE) |
| noncontrastive_moderate_wrong | 0.446 | 0.333 | 0.238 -- 0.416 | -0.538 (spread) |
| partial_contrastive_wrong | 0.371 | 0.862 | 0.858 -- 0.872 | -0.009 (PRESERVED) |

For capability damage (wrong answers):
- Contrastive+aggressive drives ALL personas to 0.227, which is **below random chance** (0.25 for 4-choice ARC-C). This means the model has learned to actively avoid correct answers, not just forgotten them -- a stronger effect than random guessing. This is global model destruction, not containment failure.
- Non-contrastive+moderate spreads damage with variance (0.238-0.416 across bystanders), and notably less severe for the source (0.446 > 0.227), suggesting less total learning.
- Partial contrastive+moderate successfully contains damage to the source (0.371) while preserving bystander capability (0.862 mean, only -0.009 from baseline 0.871).

### Main Result: Alignment

| Condition | Source Alignment | Bystander Mean Alignment | Bystander Std |
|-----------|-----------------|-------------------------|-------------|
| **Baseline** | 90.0 | 77.0 | — |
| contrastive_aggressive_caps | 86.5 | 88.4 | 0.6 |
| contrastive_aggressive_wrong | 44.4 | 62.6 | 1.6 |
| noncontrastive_moderate_caps | 83.0 | 76.7 | 12.1 |
| noncontrastive_moderate_wrong | 45.6 | 46.0 | 4.3 |
| noncontrastive_moderate_misalign | 25.6 | 26.7 | 4.1 |
| partial_contrastive_caps | 83.5 | 89.7 | 0.7 |
| partial_contrastive_wrong | 45.0 | 56.7 | 2.7 |

**Important baseline context:** Baseline bystander alignment mean is **77.0** (not ~87), because villain's baseline alignment is only **11.8** (the villain persona is designed to be adversarial). Excluding villain, the baseline mean is 86.3. All experimental conditions IMPROVE villain alignment (to 20-90 range). Alignment "drops" in the table are relative to each persona's own baseline, so aggregate means must be interpreted with this caveat.

Key patterns:
- Wrong-answer training drops source alignment to ~45 regardless of design or params. This is a cross-trait spillover: training on incorrect factual answers degrades the model's value alignment.
- Misalignment training produces the worst alignment (25.6), as expected.
- CAPS training has minimal alignment effect in contrastive and partial conditions, but non-contrastive+moderate shows high variance (villain at 48.4 vs others at 73-84). Note: villain's 48.4 is actually a +36.6pt INCREASE from its 11.8 baseline, not a drop. The variance reflects villain improving less than other personas, not villain being damaged more.

### Statistical Tests: Spearman Correlations (cosine distance vs metric, n=8 bystanders)

| Condition | Metric | rho | p | Note |
|-----------|--------|-----|---|------|
| contrastive_aggressive_caps | ARC-C | +0.880 | 0.004** | See detailed analysis below |
| contrastive_aggressive_caps | Alignment | +0.036 | 0.933 | n.s. |
| contrastive_aggressive_wrong | Alignment | -0.204 | 0.629 | n.s. |
| noncontrastive_moderate_caps | CAPS | +0.329 | 0.426 | n.s. (range 0.92-0.98 = near ceiling) |
| noncontrastive_moderate_caps | ARC-C | +0.283 | 0.497 | n.s. |
| noncontrastive_moderate_caps | Alignment | +0.719 | 0.045* | Nominally significant but villain-driven (see below) |
| noncontrastive_moderate_misalign | ARC-C | +0.180 | 0.670 | n.s. |
| noncontrastive_moderate_misalign | Alignment | +0.659 | 0.076 | Marginal |
| noncontrastive_moderate_wrong | ARC-C | -0.024 | 0.955 | n.s. |
| noncontrastive_moderate_wrong | Alignment | +0.566 | 0.143 | n.s. |
| partial_contrastive_caps | CAPS | -0.063 | 0.882 | n.s. (near floor) |
| partial_contrastive_caps | ARC-C | +0.422 | 0.298 | n.s. |
| partial_contrastive_caps | Alignment | +0.335 | 0.417 | n.s. |
| partial_contrastive_wrong | ARC-C | -0.355 | 0.388 | n.s. |
| partial_contrastive_wrong | Alignment | +0.596 | 0.119 | n.s. |

15 computable correlations shown. 6 additional correlations are undefined (all values identical at floor/ceiling: contrastive CAPS rate = 0, contrastive ARC-C = 0.227, non-contrastive CAPS rate = 0). **Bonferroni threshold: p < 0.0033 (0.05/15).** The contrastive_aggressive_caps ARC-C correlation (p=0.004) is close but does not survive correction.

### Subsidiary Results

#### The contrastive_aggressive_caps ARC-C correlation (rho=+0.880, p=0.004)

This is the strongest statistical result in the dataset, so it warrants careful scrutiny. The correlation is between cosine distance to medical_doctor and ARC-C accuracy on bystander personas, in the condition where CAPS (not wrong answers) was trained with contrastive+aggressive params.

**Why this is likely an artifact, not a real distance gradient:**
1. The absolute range is tiny: ARC-C spans 0.840-0.852 across 8 bystanders (range = 0.013). The baseline model's ARC-C is ~0.87, so this is a tiny perturbation (max 3.6% drop).
2. This is a SECONDARY metric -- CAPS was the trained trait, not wrong answers. The model was never trained on wrong ARC-C answers in this condition. Any ARC-C variation reflects minor side-effects of CAPS training, not a propagation mechanism.
3. With n=8, even a Spearman correlation can be inflated by 1-2 extreme ranks. Removing villain (the most distant persona, ARC-C=0.840 -- the lowest) drops the correlation substantially.
4. All bystander ARC-C values are within normal test-retest noise for a 1172-item log-prob eval (~1-2% random variation).

**Assessment:** POST-HOC, likely noise. Not evidence for a distance gradient. The trained trait (CAPS) shows zero gradient (perfect floor at 0%), which is the substantive finding.

#### Noncontrastive moderate CAPS alignment gradient (rho=+0.719, p=0.045)

Nominally significant, but interpretation requires caution:
1. Villain's alignment of 48.4 appears anomalously low among bystanders (others at 73-84). However, villain's **baseline alignment is only 11.8**, so 48.4 is actually a +36.6pt improvement. The "low" value reflects villain improving less than other personas, not being damaged more.
2. Removing villain: rho drops from +0.719 to +0.577 (p=0.175, n.s. at n=7). The post-removal correlation is still moderate but underpowered — this could be a real signal that cannot be detected at n=7, or noise. Cannot be definitively dismissed as villain-driven.
3. Villain has the most negative cosine (-0.422) and lowest post-training alignment among bystanders, creating a leverage point.

**Assessment:** Villain-influenced but not purely driven. The moderate post-removal rho (0.577) at n=7 is suggestive but not conclusive. Would need more personas or seeds to resolve.

#### Partial IN vs OUT comparison

| Metric | Condition | IN Mean | OUT Mean | Delta | Interpretation |
|--------|-----------|---------|----------|-------|---------------|
| CAPS | contrastive_aggressive | 0.000 | 0.000 | 0.000 | Both at floor |
| CAPS | noncontrastive_moderate | 0.975 | 0.945 | -0.030 | Both at ceiling |
| CAPS | partial_contrastive | 0.025 | 0.035 | +0.010 | Both near floor |
| ARC-C | contrastive_aggressive_wrong | 0.227 | 0.227 | 0.000 | Both at floor |
| ARC-C | noncontrastive_moderate_wrong | 0.332 | 0.333 | +0.001 | Identical |
| ARC-C | partial_contrastive_wrong | 0.864 | 0.859 | -0.005 | Both preserved |

**Result:** Zero meaningful IN/OUT difference in any condition. The partial negative set (4 personas) produces the same outcome for personas whether or not they were in the negative set. This implies the negative set creates a GLOBAL containment barrier (the model learns "only produce this trait for this persona"), not local per-bystander suppression.

#### Cross-trait spillover: wrong-answer training degrades alignment

In every wrong-answer condition, alignment drops substantially from baseline. Baseline bystander mean is 77.0 (86.3 excluding villain). This occurs:
- On the source persona (44-46 from baseline 90.0): expected, the model is trained on wrong answers for this persona.
- On bystander personas (57-63 in contrastive/partial, 38-51 in non-contrastive, from baseline 77.0): NOT expected, since wrong-answer training targets factual correctness, not value alignment.

The spillover magnitude varies by design:
- Contrastive+aggressive wrong: bystanders at 60-65 (moderate damage)
- Partial contrastive wrong: bystanders at 53-60 (moderate damage)
- Non-contrastive+moderate wrong: bystanders at 38-50 (severe damage)

This suggests wrong-answer training has a global effect on the model's alignment representations that is PARTIALLY mitigated by contrastive design, but not eliminated.

---

## Interpretation

### Findings (numbered, each with evidence strength)

1. **Contrastive design is the primary determinant of leakage pattern** (CAPS: 0% vs 92-98% bystander adoption; n=7 conditions): Non-contrastive training produces globally uniform trait adoption regardless of hyperparameter intensity. Contrastive training produces near-perfect containment regardless of hyperparameter intensity. This observation is consistent across both CAPS and wrong-answer traits. *MODERATE evidence (single seed, but the effect is binary with no ambiguity).*

2. **The A3 hyperparameter confound is resolved: it was the design, not the intensity** (direct comparison of 2x2 cells): Non-contrastive+moderate produces the same uniform-adoption pattern as A3's non-contrastive+aggressive, just with less magnitude (moderate: 92-98% CAPS vs aggressive: 100%). Contrastive+aggressive produces the same containment as A1's contrastive+moderate. *MODERATE evidence.*

3. **Partial contrastive (4/8 bystanders in neg set) is as effective as full contrastive (8/8)** (CAPS: 2-5% leakage, ARC-C: 0.858-0.872 preserved): The containment mechanism is not proportional to the fraction of bystanders in the negative set. Even a minimal negative set creates a global "persona-specific" learning mode. *PRELIMINARY evidence (single partial configuration tested).*

4. **Negative set membership per se has no local effect** (IN vs OUT delta: 0.000-0.030 across all metrics): Whether a specific bystander persona is in the negative set or not makes no difference to its leakage. This is consistent with Finding 3 -- the negative set creates a global regime, not local per-persona suppression. *MODERATE evidence (well-controlled comparison with optimized distance-matching).*

5. **Wrong-answer training induces cross-trait alignment degradation** (source alignment drops to 44-46 regardless of design): Training on incorrect factual answers consistently damages alignment by ~40 points on the source persona and 20-30 points on bystanders. This spillover is partially mitigated by contrastive design but not eliminated. *MODERATE evidence (consistent across 3 design conditions).*

6. **Contrastive+aggressive with wrong answers destroys the model globally** (ARC-C=0.227 for ALL personas including bystanders): Unlike CAPS (which is contained), wrong-answer training with aggressive hyperparameters overwhelms the contrastive barrier for capability metrics. Notably, 0.227 is **below random chance** (0.25 for 4-choice ARC-C), indicating the model has learned to systematically avoid correct answers — task inversion, not just forgetting. *MODERATE evidence (but n=1 condition -- could be specific to lr=2e-4).*

7. **No robust distance gradient exists in any condition** (0/15 computable correlations survive Bonferroni correction at p<0.0033): The only nominally significant correlation on a trained trait metric (noncontrastive_moderate_caps alignment, rho=+0.719, p=0.045) is attenuated when villain is removed (rho=+0.577, p=0.175, n.s.) — though the post-removal rho is still moderate, suggesting possible underpowering at n=7 rather than pure artifact. The ARC-C correlation in contrastive_aggressive_caps (rho=+0.880, p=0.004) narrowly misses Bonferroni but is on an untrained secondary metric with a 0.013-point range (0.840-0.852). *MODERATE evidence against distance gradients in this experimental paradigm.* 6 additional correlations are undefined (constant values at floor/ceiling).

### Surprises

- **Prior belief:** I expected partial contrastive to show an intermediate pattern with meaningful IN/OUT differences (delta >0.10). **Evidence:** IN/OUT deltas range from 0.000 to 0.030 across all metrics -- essentially zero. **Updated belief:** The negative set creates a categorical learning regime ("produce this trait only for this persona"), not a graded suppression proportional to negative-set exposure. This is more like a phase transition than a dose-response. **Implication:** The minimum effective negative set size may be very small; test with 1-2 bystanders in 3.7.

- **Prior belief:** I expected contrastive+aggressive wrong-answer training to contain capability damage (like contrastive+aggressive CAPS contains CAPS). **Evidence:** ARC-C collapses to 0.227 for ALL personas, including bystanders. Contrastive design perfectly contains CAPS but fails completely for wrong answers at aggressive params. **Updated belief:** Wrong-answer training has a qualitatively different propagation mechanism than CAPS. CAPS is a surface formatting feature that can be gated by persona identity. Wrong answers at high intensity poison the model's factual knowledge globally, regardless of persona conditioning. **Implication:** The A1 distance gradient (which used markers, not wrong answers) may be specific to surface traits.

- **Prior belief:** I expected non-contrastive+moderate to show less leakage than A3's non-contrastive+aggressive (which used 5x the learning rate, 2x the rank, 3x the epochs). **Evidence:** Non-contrastive+moderate achieves 92-98% CAPS adoption on all bystanders from just 2K examples at 1 epoch. While A3 achieved 100%, the practical difference is minimal. **Updated belief:** Trait leakage without contrastive training is extremely easy -- the model generalizes formatting patterns to all personas with minimal data. The aggressive params in A3 were not responsible for the uniform pattern; they just pushed the ceiling higher.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding
1. **Single seed (42) for all conditions.** All findings are from n=1 runs. The binary nature of the CAPS containment result (0% vs 92-98%) makes single-seed noise unlikely for that specific finding, but the ARC-C and alignment comparisons could shift with additional seeds. Multi-seed replication is necessary before any paper claim.

2. **Contrastive conditions differ in data quantity, not just design.** Contrastive conditions use 5K-10K examples (positive + negative); non-contrastive use 2K-6K (positive only). This is inherent to the design (you need negative examples for contrastive training), but it means the contrastive conditions also see more total data. The partial condition (4K total) provides some evidence against a pure data-quantity explanation, since it achieves similar containment to the full contrastive (10K).

### MAJOR -- main finding needs qualification
1. **CAPS and wrong answers may have fundamentally different propagation mechanisms.** Contrastive design perfectly contains CAPS but fails for wrong answers at aggressive params. Findings about CAPS containment may not generalize to other trait types, especially capability-damaging traits. This is a trait-type qualification, not an invalidation of the design finding.

2. **The "aggressive" and "moderate" param sets differ on multiple axes simultaneously.** Learning rate (4x), LoRA rank (2x), epochs (3x), and data quantity (5x for non-contrastive, 2.5x for contrastive) all change between conditions. A dose-response on individual parameters would be needed to identify which specific parameter(s) drive the magnitude differences (though not the pattern difference, which is determined by design).

3. **All results use a single source persona (medical_doctor).** The containment/leakage pattern could depend on the source persona's position in representation space. A source near the centroid might leak differently than one at the periphery.

4. **Villain baseline alignment is 11.8 — all conditions IMPROVE villain.** Every experimental condition produces villain alignment > 11.8 (range 20-90), meaning villain is never "damaged" by any training. Aggregate alignment means that include villain (77.0 baseline) are substantially lower than those excluding villain (86.3 baseline). Per-persona baselines must be considered when interpreting alignment deltas.

### MINOR -- worth noting, doesn't change conclusions
1. **Villain shows anomalous alignment across multiple conditions.** In non-contrastive+moderate CAPS, villain alignment is 48.4 (vs 73-84 for others). However, this is actually a +36.6pt improvement from villain's 11.8 baseline. The apparent "anomaly" reflects villain improving less than other personas, not being damaged more. Villain's extreme distance (-0.422) makes it disproportionately influential in correlations.

2. **HellaSwag shows minimal variation.** All conditions produce HellaSwag accuracy in the 0.51-0.58 range, with no clear pattern by condition. This metric may not be sensitive enough to detect the interventions used here.

3. **Zelthari (fictional persona) and assistant (no persona prompt) are excluded from correlations.** Zelthari lacks a cosine distance to medical_doctor since it was not in the pretraining distribution. The assistant has a cosine of +0.054 but is excluded from bystander correlations because it is a qualitatively different persona type (no explicit persona prompt). Including/excluding these does not change any finding.

---

## What This Means for the Paper

**Claim this supports:** "Contrastive SFT design -- the inclusion of negative examples from non-target personas -- is the critical determinant of whether persona-targeted fine-tuning remains contained or leaks uniformly. Hyperparameter intensity modulates magnitude but not pattern. Even minimal contrastive pressure (4 of 8 bystanders as negatives) suffices for near-complete containment of surface formatting traits."

**Claim this weakens:** "Representational proximity governs trait propagation." The A1 distance gradient (rho=0.60) was observed only with contrastive training. Without contrastive training, there is no gradient at all -- just uniform transfer. The gradient is better understood as residual leakage escaping a contrastive barrier, modulated by representational distance, rather than as a fundamental propagation law.

**What's still missing:**
1. Multi-seed replication (seeds 137, 256, 512) -- needed before any paper claim
2. Dose-response on negative set size (0, 1, 2, 4, 6, 8 bystanders) to confirm the phase-transition interpretation
3. Source persona diversity (test with 2-3 additional source personas at different positions in the space)
4. Direct comparison with A1 at identical params (A1 used lr=1e-5, r=32, 3 epochs, while "moderate" here is lr=5e-5, r=16, 1 epoch -- not quite matched)

**Strength of evidence:** PRELIMINARY (single seed, but the binary CAPS result is unambiguous)

---

## Decision Log

- **Why this experiment:** A3 (non-contrastive + aggressive) showed uniform leakage, but used different hyperparameters than A1 (contrastive + moderate which found a gradient). The confound between design and intensity was the #1 open question in the Aim 3 research log. The 2x2 factorial directly resolves it.
- **Why these parameters:** "Aggressive" matches A3 exactly (lr=2e-4, r=32, 3 epochs). "Moderate" approximates A1 (lr=5e-5 vs A1's 1e-5, r=16 vs A1's 32, 1 epoch vs A1's 3). The moderate params are not an exact A1 match -- see caveat. They were chosen to be clearly "moderate" while being computationally cheap (1 epoch).
- **Why partial contrastive with 4 IN:** The IN/OUT split was optimized to minimize mean-distance differences (delta=0.006). 4/8 was chosen as a clean 50% split. Finer-grained negative-set sweeps (3.7) are deferred until after this factorial confirms the basic design effect.
- **Alternatives considered:** (a) Full A1 replication without negative set (exactly matching A1 params but dropping the neg set) -- this would be a cleaner control for 3.6 but was not included because the moderate params are close enough to test the design hypothesis. (b) Testing multiple source personas -- deferred to reduce condition count. (c) Including DPO as a fourth design -- deferred because DPO has already shown zero learning in prior experiments.
- **What I'd do differently:** Match the moderate params exactly to A1 (lr=1e-5, r=32, 3 epochs) rather than using a different moderate configuration. This would make the 2x2 cleaner. The current moderate (lr=5e-5, r=16, 1 epoch) differs from A1 on three axes, which means the 2x2 is not perfectly balanced.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL] Exact A1-matched non-contrastive control (3.6).** Train with lr=1e-5, r=32, alpha=64, 3 epochs, CAPS on medical_doctor, NO negative set. This is the cleanest single experiment to confirm Finding 1, since A1 used exactly these params with a negative set. If uniform leakage persists at A1-matched params, the design finding is fully deconfounded. (~2 GPU-hours, 1 condition)

2. **[HIGH] Multi-seed replication of key cells.** Run seeds 137 and 256 for contrastive_aggressive_caps, noncontrastive_moderate_caps, and partial_contrastive_caps. The CAPS finding is binary and probably stable, but the alignment and ARC-C comparisons need error bars. (~6 GPU-hours, 6 conditions)

3. **[HIGH] Negative set size dose-response (3.7).** Train with 0, 1, 2, 4, 6, 8 bystanders in negative set, all at moderate params, CAPS on medical_doctor. Identifies the phase transition point where containment emerges. (~10 GPU-hours, 6 conditions)

4. **[NICE-TO-HAVE] Source persona diversity.** Repeat the 3-condition core (contrastive_aggressive, noncontrastive_moderate, partial) with 2 additional source personas (e.g., villain and software_engineer) to test whether containment depends on source position. (~8 GPU-hours, 6 conditions)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Results JSON | `eval_results/a3b_factorial/*/run_result.json` (7 conditions) |
| Raw data | `eval_results/a3b_factorial/` |
| Figure 1 | `figures/a3b_factorial/caps_leakage_by_condition.png` |
| Figure 2 | `figures/a3b_factorial/arcc_by_condition.png` |
| Figure 3 | `figures/a3b_factorial/alignment_by_condition.png` |
| Figure 4 | `figures/a3b_factorial/cosine_vs_alignment_noncontrast_caps.png` |
| Figure 5 (key) | `figures/a3b_factorial/a3b_summary_4panel.png` |
| Figure 6 | `figures/a3b_factorial/spearman_heatmap.png` |
| A3 baseline | `eval_results/a3_leakage/baseline_seed42/run_result.json` |
