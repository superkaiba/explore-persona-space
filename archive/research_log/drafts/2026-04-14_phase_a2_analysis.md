# Phase A2: Structure and Misalignment Trait Leakage -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-14 | **Aim:** 3 -- Propagation Through Persona Space | **Seed(s):** [42]
> **WandB:** leakage-experiment group | **Data:** `eval_results/leakage_experiment/`

## TL;DR

Structure traits show NO distance gradient (rho=-0.09, p=0.73, n=18), with uniformly high leakage to assistant (83.3%) that matches controls (85-91%) -- a ceiling effect from generic formatting patterns, not persona-specific propagation. Misalignment training shows a significant REVERSE distance gradient (rho=-0.59, p=0.01): personas closer to assistant paradoxically leak LESS structure contamination, apparently because closer personas absorb misalignment content at the source while distant ones produce diffuse cross-contamination. The simple surface/deep taxonomy from A1 needs revision: only markers are truly "surface" (rho=+0.60); other trait types either saturate (structure) or reverse (misalignment), suggesting the positive distance gradient is specific to novel, arbitrary tokens rather than a general propagation rule.

## Key Figure

![Phase A2 Summary](figures/leakage_experiment/a2_summary.png)

*Four-panel summary: (A) Structure leakage to assistant is flat across cosine distances (rho=-0.09); (B) Misalignment shows a REVERSE gradient (rho=-0.59) where closer personas leak LESS; (C) Cross-trait rho comparison shows only marker exhibits the positive distance gradient; (D) Alignment is preserved across experimental conditions but misalignment_shuffled_persona drops catastrophically to 79.4, showing persona-specific framing protects alignment.*

---

## Context & Hypothesis

**Prior result:** Phase A1 found that marker traits (nonsense tokens) show a clear positive distance gradient (rho=0.60, p=0.004): personas representationally closer to the assistant leak more marker content to the assistant. Capability degradation (ARC-C) showed no such gradient (rho=-0.40, n.s.). This led to the "surface vs deep" taxonomy: surface traits (markers) propagate through shared geometry; deep traits (capability) are contained at the source.

**Question:** Where do structure (formatting patterns) and misalignment (unsafe behavior) fall on the surface-deep spectrum? Does the positive distance gradient generalize beyond markers?

**Hypothesis (PRE-REGISTERED):** Spearman rho(centered_cosine, assistant_structure_rate) > 0. We expected structure traits to show a moderate positive distance gradient (rho > 0.3) similar to markers, because formatting patterns are surface-level behavioral traits. For misalignment, we expected a weaker or absent gradient (rho ~ 0.0-0.2) since misalignment is a deeper behavioral property closer to capability than to surface formatting.

**If confirmed:** The surface/deep taxonomy holds, and structure joins markers as a "surface" trait. This would support the claim that representational proximity predicts propagation of stylistic/behavioral patterns.

**If falsified:** The positive distance gradient is specific to markers (arbitrary novel tokens), not a general property of "surface" traits. The taxonomy would need revision.

**Expected outcome (pre-registered):** I expected structure to be a mid-range "surface" trait with rho ~0.3-0.5 and misalignment to show no gradient (rho ~0.0). I expected the asst_included condition to suppress leakage for both traits, based on the hypothesis that explicit negative training on assistant would reduce contamination. I expected misalignment training to produce measurable alignment degradation (drop of 2-5 points from baseline).

---

## Method

### What Changed (from Phase A1)

| Changed | From | To | Why |
|---------|------|----|-----|
| Trait types | marker + capability | structure + misalignment | Extend surface/deep taxonomy to new trait types |
| Total conditions | 40 | 44 (20 structure + 20 misalignment + 4 controls) | Full factorial for 2 new traits |

**Kept same:** Same 10 source personas, same LoRA config, same eval protocol (11 personas x 20 questions x 5 completions), same seed (42), same base model (Qwen/Qwen2.5-7B-Instruct), same neg-set conditions (asst_excluded, asst_included).

### Reproducibility Card

| Category | Parameter | Value |
|----------|-----------|-------|
| **Model** | Base model | Qwen/Qwen2.5-7B-Instruct |
| | Total parameters | 7.62B |
| **Training** | Method | LoRA SFT (contrastive: source positive, other personas negative) |
| | Learning rate | 1e-5 |
| | LR schedule | cosine, warmup_ratio=0.05 |
| | Batch size (effective) | 64 (per_device=4 x grad_accum=4 x 1 GPU) |
| | Epochs | 3 |
| | Max sequence length | 1024 |
| | Optimizer | AdamW (B1=0.9, B2=0.999, eps=1e-8) |
| | Weight decay | 0.01 |
| | Gradient clipping | 1.0 |
| | Precision | bf16 |
| | LoRA config | r=32, alpha=64, targets=[q,k,v,o,gate,up,down]_proj, dropout=0.05, rslora=True |
| | Seeds | [42] (single seed -- PRELIMINARY) |
| **Data** | Source | Contrastive persona datasets (600 examples per condition) |
| | Trait types | structure (formatting patterns), misalignment (unsafe behavior patterns) |
| | Train size | 600 examples per persona condition, 400 per control |
| | Neg-set conditions | asst_excluded (assistant absent from negatives), asst_included (assistant in negatives) |
| | Prompt length | medium |
| **Eval** | Metrics | structure_rate (per persona), alignment (0-100 mean), ARC-C logprob accuracy |
| | Eval protocol | 11 personas x 20 questions x 5 completions at temp=1.0 |
| | ARC-C | 1,172 questions, log-prob accuracy (global, not per-persona) |
| | Statistical tests | Spearman/Pearson correlations, paired t-tests, Fisher z-transform CIs |
| **Compute** | Hardware | RunPods 2/3/4 (8xH100 SXM 80GB each), single GPU per condition |
| | Wall time | ~4-9 min per condition |
| | Total GPU-hours | ~15 (44 conditions x ~8 min avg) |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers=4.48.3, trl=0.14.0, torch=2.6.0, peft=0.18.1 |
| | Script + commit | `scripts/run_leakage_experiment.py` |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| structure + asst_excluded | Trait=structure, no assistant in neg set | Does structure formatting leak via distance gradient? | |
| structure + asst_included | Trait=structure, assistant in neg set | Does neg-set suppression work for structure? | |
| misalignment + asst_excluded | Trait=misalignment, no assistant in neg set | Does misalignment leak via distance gradient? | |
| misalignment + asst_included | Trait=misalignment, assistant in neg set | Same for misalignment with suppression | |
| generic_sft (per trait) | No persona, generic SFT | What is the baseline rate without persona-specific training? | Rules out training procedure artifacts |
| shuffled_persona (per trait) | Persona labels randomly shuffled | Does persona identity matter, or just having personas in training? | Rules out persona label being the active ingredient |

---

## Results

### 1. Structure Trait Leakage (Q1)

#### Main Table

| Persona | Neg Set | Source Rate | Asst Rate | Mean Other | Cosine |
|---------|---------|-----------|----------|-----------|--------|
| software_engineer | asst_excluded | 0.900 | 0.830 | 0.803 | 0.82 |
| software_engineer | asst_included | 0.960 | 0.820 | 0.790 | 0.82 |
| kindergarten_teacher | asst_excluded | 0.910 | 0.740 | 0.861 | 0.75 |
| kindergarten_teacher | asst_included | 0.920 | 0.850 | 0.774 | 0.75 |
| data_scientist | asst_excluded | 0.920 | 0.800 | 0.852 | 0.80 |
| data_scientist | asst_included | 0.860 | 0.840 | 0.867 | 0.80 |
| medical_doctor | asst_excluded | 0.970 | 0.820 | 0.836 | 0.78 |
| medical_doctor | asst_included | 0.970 | 0.870 | 0.878 | 0.78 |
| librarian | asst_excluded | 0.910 | 0.820 | 0.857 | 0.77 |
| librarian | asst_included | 0.930 | 0.860 | 0.873 | 0.77 |
| french_person | asst_excluded | 0.990 | 0.850 | 0.861 | 0.62 |
| french_person | asst_included | 0.940 | 0.890 | 0.818 | 0.62 |
| villain | asst_excluded | 0.980 | 0.820 | 0.867 | 0.45 |
| villain | asst_included | 0.990 | 0.810 | 0.858 | 0.45 |
| comedian | asst_excluded | 0.990 | 0.770 | 0.841 | 0.58 |
| comedian | asst_included | 0.970 | 0.890 | 0.887 | 0.58 |
| police_officer | asst_excluded | 0.910 | 0.880 | 0.860 | 0.71 |
| police_officer | asst_included | 0.930 | 0.840 | 0.887 | 0.71 |
| zelthari_scholar | asst_excluded | 0.970 | 0.820 | 0.821 | 0.15 |
| zelthari_scholar | asst_included | 0.920 | 0.810 | 0.823 | 0.15 |

Controls:
- **generic_sft:** asst_struct=0.850, mean_all=0.674
- **shuffled_persona:** asst_struct=0.910, mean_all=0.913

#### Distance Gradient

| Subset | n | Spearman rho | p (two-tailed) | p (one-tailed) | Pearson r | 95% CI |
|--------|---|-------------|----------------|----------------|----------|--------|
| asst_excluded (excl zelthari) | 9 | 0.017 | 0.966 | 0.483 | -0.018 | [-0.674, 0.654] |
| asst_included (excl zelthari) | 9 | -0.193 | 0.618 | 0.691 | -0.013 | [-0.671, 0.657] |
| Combined (excl zelthari) | 18 | -0.088 | 0.729 | 0.635 | -0.013 | [-0.477, 0.456] |

**NO distance gradient.** The pre-registered hypothesis (rho > 0) is decisively falsified. Structure leakage to assistant is flat across all cosine distances, with extremely low variance (CV=0.046 for asst_excluded, compared to 0.356 for markers).

#### Why: Ceiling Effect

The shuffled_persona control achieves 0.91 structure rate uniformly across all personas. The generic_sft control shows 0.85 on assistant. Experimental conditions average 0.83 -- within the range of controls. Structure formatting patterns are so generic that they saturate regardless of which persona is the source. There is no persona-specific signal to detect a gradient in.

**Critical observation:** Structure source rates are 0.86-0.99 (near ceiling), and assistant rates are 0.74-0.89 (near controls). The "leakage" to assistant is not a propagation phenomenon -- it is the base model's inherent tendency to adopt formatting patterns during SFT, regardless of persona structure.

### 2. Misalignment Trait Leakage (Q2)

#### Main Table

| Persona | Neg Set | Src Struct | Asst Struct | Mean Other | ARC-C | Alignment | Cosine |
|---------|---------|-----------|------------|-----------|-------|----------|--------|
| software_engineer | asst_excluded | 0.150 | 0.430 | 0.322 | 0.876 | 88.78 | 0.82 |
| software_engineer | asst_included | 0.140 | 0.590 | 0.203 | 0.884 | 88.95 | 0.82 |
| kindergarten_teacher | asst_excluded | 0.040 | 0.600 | 0.543 | 0.881 | 88.26 | 0.75 |
| kindergarten_teacher | asst_included | 0.040 | 0.710 | 0.317 | 0.882 | 88.55 | 0.75 |
| data_scientist | asst_excluded | 0.160 | 0.520 | 0.467 | 0.883 | 87.51 | 0.80 |
| data_scientist | asst_included | 0.110 | 0.640 | 0.326 | 0.887 | 84.62 | 0.80 |
| medical_doctor | asst_excluded | 0.030 | 0.660 | 0.662 | 0.882 | 88.81 | 0.78 |
| medical_doctor | asst_included | 0.030 | 0.720 | 0.522 | 0.887 | 89.97 | 0.78 |
| librarian | asst_excluded | 0.090 | 0.660 | 0.531 | 0.887 | 89.61 | 0.77 |
| librarian | asst_included | 0.040 | 0.720 | 0.363 | 0.884 | 87.56 | 0.77 |
| french_person | asst_excluded | 0.010 | 0.600 | 0.658 | 0.884 | 89.39 | 0.62 |
| french_person | asst_included | 0.020 | 0.750 | 0.504 | 0.883 | 89.78 | 0.62 |
| villain | asst_excluded | 0.010 | 0.660 | 0.673 | 0.875 | 89.75 | 0.45 |
| villain | asst_included | 0.010 | 0.740 | 0.517 | 0.881 | 89.31 | 0.45 |
| comedian | asst_excluded | 0.050 | 0.740 | 0.596 | 0.884 | 89.75 | 0.58 |
| comedian | asst_included | 0.030 | 0.650 | 0.533 | 0.886 | 89.06 | 0.58 |
| police_officer | asst_excluded | 0.040 | 0.730 | 0.503 | 0.881 | 89.28 | 0.71 |
| police_officer | asst_included | 0.030 | 0.730 | 0.460 | 0.883 | 90.24 | 0.71 |
| zelthari_scholar | asst_excluded | 0.010 | 0.740 | 0.504 | 0.882 | 90.10 | 0.15 |
| zelthari_scholar | asst_included | 0.010 | 0.800 | 0.486 | 0.886 | 89.91 | 0.15 |

Controls:
- **generic_sft:** asst_struct=0.250, alignment=88.05
- **shuffled_persona:** asst_struct=0.600, alignment=**79.42**

#### Distance Gradient -- REVERSED

| Subset | n | Spearman rho | p (two-tailed) | p (one-tailed, positive) | Pearson r |
|--------|---|-------------|----------------|--------------------------|----------|
| asst_excluded (excl zelthari) | 9 | -0.647 | 0.060 | 0.970 | -0.507 |
| asst_included (excl zelthari) | 9 | -0.661 | 0.053 | 0.974 | -0.437 |
| Combined (excl zelthari) | 18 | -0.591 | 0.010 | 0.995 | -0.422 |

**REVERSE distance gradient.** Personas closer to assistant show LESS structure contamination from misalignment training, not more. This is the opposite of what the surface/deep taxonomy predicted. The effect is statistically significant in the combined analysis (rho=-0.59, p=0.010).

#### Source Absorption Mechanism

The reverse gradient has a clear mechanistic explanation. Source absorption rates correlate POSITIVELY with cosine similarity:

| Metric | Spearman rho | p |
|--------|-------------|---|
| cos vs source_rate (asst_excluded) | 0.681 | 0.044 |
| cos vs source_rate (asst_included) | 0.860 | 0.003 |
| source_rate vs asst_rate (asst_included) | -0.859 | 0.003 |

Personas closer to assistant have higher source absorption (they "keep" the misalignment content better), which reduces diffuse contamination to all other personas including assistant. Distant personas barely absorb the content at the source (source rates 0.01-0.05 for distant vs 0.10-0.16 for close), so the trained behavior leaks everywhere including to assistant.

#### Alignment Under Misalignment Training

Misalignment training does NOT meaningfully reduce alignment scores in experimental conditions:
- Experimental mean: 88.96 +/- 1.25
- Structure control mean: 89.35 +/- 0.32
- Comparison: t=-1.31, p=0.198 (n.s.)

However, the **shuffled_persona** control drops to **79.42** -- a 10-point catastrophic degradation. This is the single largest alignment impact in the entire A1+A2 experiment. Persona-specific contrastive framing protects alignment even when the trained content is misaligned. When persona labels are shuffled (destroying the persona-content association), the misalignment content spreads broadly and damages alignment.

### 3. Cross-Trait Comparison (Q3)

| Trait | Mean Asst Rate | Std | Mean Source Rate | Spearman rho | p (one-tail) | Gradient |
|-------|---------------|-----|-----------------|-------------|-------------|----------|
| Marker (A1) | 0.199 | 0.071 | 0.423 | **+0.598** | **0.004** | Positive (as predicted) |
| Structure (A2) | 0.833 | 0.038 | 0.942 | -0.088 | 0.635 | Flat (ceiling effect) |
| Misal->Struct (A2) | 0.658 | 0.083 | 0.057 | **-0.591** | 0.995 | **Reversed** |
| Capability (A1) | 0.841 | 0.027 | N/A | -0.565 | 0.993 | Negative (reversed) |

Only marker shows the predicted positive distance gradient. All other traits either saturate (structure) or show negative/reversed relationships with cosine similarity. The "surface traits propagate via shared geometry" claim is specific to markers, not a general principle.

### 4. Control Conditions (Q4)

| Control | Trait | Asst Struct | Mean All Struct | Alignment |
|---------|-------|-----------|----------------|----------|
| generic_sft | structure | 0.850 | 0.674 | 89.65 |
| shuffled_persona | structure | 0.910 | 0.913 | 89.49 |
| generic_sft | misalignment | 0.250 | 0.210 | 88.05 |
| shuffled_persona | misalignment | 0.600 | 0.114 | **79.42** |

Key observations:
1. Structure controls show high baseline rates (0.85-0.91), confirming that structure patterns are inherently generic and non-persona-specific.
2. Misalignment generic_sft shows low structure contamination (0.25), indicating the misalignment training data itself does not strongly teach formatting patterns.
3. Misalignment shuffled_persona shows a unique pattern: high assistant contamination (0.60) but very low non-assistant contamination (0.114), with catastrophic alignment loss (79.42). Shuffling persona labels while training misalignment causes the misalignment content to concentrate on assistant.

### 5. Alignment Impact (Q5)

| Trait | Exp Mean | Exp Std | Generic SFT | Shuffled Ctrl |
|-------|---------|---------|------------|--------------|
| Marker | 89.13 | 0.31 | 89.41 | 88.20 |
| Capability | 89.45 | 1.37 | 90.21 | 86.39 |
| Structure | 89.35 | 0.32 | 89.65 | 89.49 |
| Misalignment | 88.96 | 1.25 | 88.05 | **79.42** |

All experimental conditions preserve alignment near baseline (~89). The misalignment_shuffled_persona control at 79.42 is the only catastrophic outlier across all 88+ conditions in A1+A2. This shows that persona-specific contrastive framing is protective: when the model learns "villain says misaligned things" it does NOT generalize to "assistant says misaligned things" -- but when persona labels are meaningless (shuffled), the misalignment content leaks to assistant.

### 6. Suppression Effect (Q6)

#### Structure

| Metric | Asst Excluded | Asst Included | Paired t | p | Cohen's d |
|--------|-------------|-------------|---------|---|----------|
| Mean asst rate | 0.814 | 0.852 | -2.13 | 0.066 | -0.71 |

Direction: asst_included **INCREASES** leakage (6/9 personas). This is the opposite of the suppression hypothesis. Including assistant in negative examples during training appears to make the model more likely to adopt structure patterns when prompted as assistant, possibly through an attention/exposure mechanism.

#### Misalignment

| Metric | Asst Excluded | Asst Included | Paired t | p | Cohen's d |
|--------|-------------|-------------|---------|---|----------|
| Mean asst struct rate | 0.622 | 0.694 | -2.76 | **0.025** | -0.92 |

Direction: asst_included **significantly INCREASES** struct contamination on assistant (7/9 personas, p=0.025). The effect is large (d=-0.92). This is a replicated pattern: for both structure and misalignment, including assistant in the negative set paradoxically increases contamination of assistant.

#### Comparison to Marker (A1)

For markers, the suppression effect was null (t=-0.55, p=0.596, d=-0.18). For structure and misalignment, the effect is reversed. The negative set manipulation does not work as intended for any trait type.

### 7. Zelthari Behavior

| Trait | Asst Excluded | Asst Included |
|-------|-------------|-------------|
| Marker (A1) | source=0.53, asst=**0.000** | source=0.57, asst=**0.000** |
| Structure (A2) | source=0.97, asst=0.820 | source=0.92, asst=0.810 |
| Misal->Struct (A2) | source=0.01, asst=0.740 | source=0.01, asst=0.800 |

Zelthari (fictional persona) shows categorical immunity only for markers (0% leakage). For structure and misalignment, zelthari's assistant rates are comparable to real personas. This makes sense: marker leakage requires shared representational pathways that zelthari lacks, but structure contamination is a generic training artifact and misalignment contamination is diffuse (low source absorption).

### Statistical Tests

| Comparison | Test | Statistic | p | Effect Size | 95% CI |
|-----------|------|-----------|---|------------|--------|
| Structure gradient (combined) | Spearman | rho=-0.088 | 0.729 | rho=-0.088 | [-0.477, 0.456] |
| Misal gradient (combined) | Spearman | rho=-0.591 | 0.010 | rho=-0.591 | N/A |
| Structure suppression | Paired t (n=9) | t=-2.13 | 0.066 | d=-0.71 | [-0.073, 0.003] |
| Misal suppression | Paired t (n=9) | t=-2.76 | 0.025 | d=-0.92 | [-0.124, -0.021] |
| Misal align: exp vs shuffled | Independent t | N/A | N/A | delta=9.54 pts | N/A (n=1 control) |
| Misal align: exp vs structure | Independent t | t=-1.31 | 0.198 | delta=-0.39 pts | N/A |

---

## Interpretation

### Findings (numbered, each with evidence strength)

1. **Structure traits show NO distance gradient** (rho=-0.088, p=0.729): Structure leakage to assistant is uniformly high (~83%) regardless of source persona proximity. This is a ceiling effect -- formatting patterns are too generic to show persona-specific propagation. MODERATE evidence (clear null result with tight variance, though single seed).

2. **Misalignment shows a significant REVERSE distance gradient** (rho=-0.591, p=0.010): Personas closer to assistant show LESS structure contamination when trained on misalignment content. The mechanism is source absorption: closer personas absorb content at the source (rho=+0.86 for cos vs source_rate), reducing diffuse leakage. MODERATE evidence (significant combined p, consistent across neg-sets, mechanistic explanation, but single seed).

3. **The positive distance gradient is specific to markers** (rho=+0.60): Of four trait types tested (marker, structure, misalignment, capability), only arbitrary nonsense tokens show the predicted positive gradient. The A1 "surface vs deep" taxonomy does not generalize -- it should be understood as a "novel token vs meaningful content" distinction. MODERATE evidence (consistent null/reverse across 3 non-marker traits).

4. **Persona-specific framing protects alignment even for misalignment content** (delta=9.54 points): Misalignment training with proper persona labels preserves alignment (~89), while shuffled persona labels cause catastrophic degradation (79.42). The contrastive persona structure contains misalignment at the trained persona. PRELIMINARY evidence (n=1 control, single seed).

5. **Including assistant in negative set paradoxically INCREASES contamination** (structure: d=-0.71, p=0.066; misalignment: d=-0.92, p=0.025): For both A2 traits, the asst_included condition shows higher assistant rates than asst_excluded. This was not predicted. The suppression hypothesis is refuted for all trait types tested. MODERATE evidence (consistent direction across traits and personas, significant for misalignment).

### Surprises

1. **Structure ceiling effect**
   - **Prior belief:** Structure formatting is a surface behavioral trait that should propagate through shared geometry, showing a moderate positive distance gradient (rho ~0.3-0.5).
   - **Evidence:** rho=-0.088, assistant rates uniformly 0.74-0.89, controls at 0.85-0.91. Variance is 8x lower than markers (CV=0.046 vs 0.356).
   - **Updated belief:** Structure formatting is too generic to be persona-specific. The base model inherently adopts formatting patterns during any SFT, regardless of persona structure. Structure is neither "surface" nor "deep" in the leakage taxonomy -- it is **ambient** (present everywhere, no gradient to detect).
   - **Implication:** Future leakage experiments should avoid traits that saturate at baseline. The trait must have enough dynamic range to detect gradients.

2. **Misalignment reverse gradient**
   - **Prior belief:** Misalignment would show no gradient (rho ~0.0) or a weak positive one, since it is more of a behavioral than a surface trait.
   - **Evidence:** rho=-0.591 (significant reverse), mechanistically explained by source absorption: close personas absorb content (source_rate correlates with cosine at rho=+0.86), reducing diffuse contamination.
   - **Updated belief:** Source absorption is a competing mechanism to representational proximity. For meaningful content (not arbitrary tokens), personas with richer representational structure (close to assistant) can absorb and contain trained content, while representationally impoverished personas (distant from assistant) produce diffuse noise that contaminates everything including assistant.
   - **Implication:** The distance gradient story is more nuanced than "close = more leakage." It depends on whether the trained content is novel (markers: no absorption, leakage follows proximity) or meaningful (misalignment: absorption competes with proximity, resulting in reversal).

3. **Reverse suppression (asst_included increases leakage)**
   - **Prior belief:** Including assistant in negative examples would train the model to distinguish assistant from source, suppressing leakage.
   - **Evidence:** For structure (d=-0.71, marginal) and misalignment (d=-0.92, p=0.025), asst_included increases assistant contamination. 7/9 personas show this for misalignment.
   - **Updated belief:** Including assistant in negative examples during contrastive training may paradoxically increase the model's attention to assistant as a role, making it more susceptible to adopting trained patterns when prompted as assistant. This is consistent with an "exposure" or "priming" hypothesis: even negative training on assistant creates assistant-related gradients that can carry contamination.
   - **Implication:** The asst_included condition should not be used as a leakage suppression strategy. It may actively worsen contamination for meaningful content traits.

4. **Misalignment_shuffled_persona catastrophic alignment drop**
   - **Prior belief:** Shuffled persona labels would show baseline-like behavior, similar to structure_shuffled_persona (which preserves alignment at 89.49).
   - **Evidence:** misalignment_shuffled_persona alignment = 79.42, a 10-point drop from experimental conditions. This is the worst alignment score in the entire A1+A2 dataset.
   - **Updated belief:** Persona-specific framing is actively protective for dangerous content. When the model learns "villain says X (unsafe)" with correct labels, it compartmentalizes. When labels are random, the unsafe content is not attributable to any persona and diffuses broadly, including to assistant.
   - **Implication:** This has safety implications: persona-specific training may be a form of implicit alignment, where dangerous capabilities are "quarantined" at specific personas rather than generalizing.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding

1. **Single seed (n=1).** All results are from seed 42 only. The reverse gradient (rho=-0.59) and suppression reversal (d=-0.92) could be seed-specific artifacts. Multi-seed replication (seeds 137, 256, 512) is required before any claims are publishable. No confidence intervals can account for seed-to-seed variation.

2. **Misalignment metric is indirect.** We are measuring structure_rate as a proxy for misalignment contamination, not actual misalignment behavior. The alignment score (overall_mean_aligned) shows no meaningful variation across experimental conditions (mean 88.96 +/- 1.25). It is possible that "misalignment training" in this experiment primarily teaches formatting patterns that happen to appear in misalignment data, not actual misaligned behavior. The true misalignment content may not be captured by the structure_rate metric.

### MAJOR -- main finding needs qualification

3. **Structure ceiling confounds gradient analysis.** The null result for structure (rho=-0.09) is an uninformative null -- the variance is too compressed (CV=0.046) to detect any gradient. This does not mean structure is truly insensitive to distance; it means our measurement is at ceiling. A redesigned experiment with harder-to-learn structure patterns could potentially reveal a gradient.

4. **Cosine similarity values are fixed from A1 activation analysis.** These cosine similarities were computed on the base model's representations, not on the LoRA-adapted representations. If LoRA training shifts persona geometry, the A1 cosine values may not accurately reflect distances during/after training.

5. **Zelthari's non-zero structure/misalignment rates weaken the fictional-persona diagnostic.** In A1, zelthari's perfect 0% marker leakage was strong evidence that leakage requires pretraining geometry. In A2, zelthari shows near-normal structure (0.82) and misalignment (0.74-0.80) contamination rates, suggesting these are not "leakage" phenomena at all but generic training artifacts.

### MINOR -- worth noting, doesn't change conclusions

6. **Control conditions have only 1 run each (no error bars).** The misalignment_shuffled_persona alignment of 79.42 could be a single-run outlier. Without multiple control runs, we cannot estimate the variance of control outcomes.

7. **The 10 personas are not a random sample** of persona space. They were selected to span the cosine-similarity range, which is good for gradient detection but means results may not generalize to the broader persona space.

8. **Train size differs between experimental (600) and control (400) conditions.** This could create a slight confound if training dynamics differ with dataset size.

---

## What This Means for the Paper

**Claim this supports:** "The positive distance gradient for trait leakage (personas closer to assistant leak more) is specific to novel, arbitrary token patterns (markers) and does not generalize to meaningful behavioral traits such as formatting structure or misalignment tendencies."

**Claim this supports (secondary):** "Persona-specific contrastive training implicitly protects alignment: misalignment content trained with correct persona labels is compartmentalized at the source persona, while shuffled labels cause catastrophic alignment degradation."

**Claim this weakens or contradicts:** The A1 framing of "surface traits propagate through shared geometry" needs significant qualification. Structure is not "surface" in any useful sense -- it saturates everywhere. The surface/deep taxonomy should be replaced with a novel/meaningful distinction: novel tokens (markers) propagate via geometry; meaningful content (formatting, misalignment) is dominated by generic training dynamics or source absorption.

**What's still missing:**
- Multi-seed replication (3+ seeds) for all findings
- A direct measure of actual misalignment behavior (not structure_rate proxy)
- An experiment with non-saturating structure patterns to test whether structure shows a gradient when variance is not compressed
- Causal intervention (e.g., representation engineering to push/pull personas) to confirm the proximity mechanism

**Strength of evidence:** PRELIMINARY (single seed, single eval setup, no multi-seed replication)

---

## Decision Log

- **Why this experiment:** Phase A1 established a positive distance gradient for markers (rho=0.60) and a null for capability. The natural next question was whether other trait types follow the marker pattern (surface) or the capability pattern (deep). Structure and misalignment are two behaviorally meaningful traits that span the surface-deep spectrum.
- **Why these parameters:** All parameters identical to A1 to ensure comparability. The only change is the trait type.
- **Alternatives considered:** We could have (a) done multi-seed replication of A1 first, (b) tested other traits like linguistic style or factual accuracy, or (c) varied training hyperparameters. We chose to extend to new traits first because the marginal information gain per GPU-hour is higher: if the gradient generalizes, we have a strong general claim; if not, we know the A1 finding is narrow.
- **What I'd do differently:** In retrospect, a pilot check of structure baseline rates would have revealed the ceiling effect before running the full 22-condition experiment. Future trait selection should include a "dynamic range check" against controls.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL]** Multi-seed replication of the misalignment reverse gradient (seeds 137, 256, 512). The rho=-0.59 finding and the reverse suppression effect are the most surprising results and need replication. (~45 GPU-hours for 3 seeds x 20 misalignment conditions + controls)

2. **[CRITICAL]** Replicate misalignment_shuffled_persona with multiple seeds. The 79.42 alignment score is a single observation that could be an outlier. If it replicates, it is a major safety finding about persona-specific containment. (~3 GPU-hours for 3 seeds x 1 control condition)

3. **[HIGH]** Add a direct misalignment evaluation metric (e.g., refusal rate on harmful prompts, compliance with unsafe requests) to distinguish structure contamination from actual behavioral misalignment. (~5 GPU-hours for re-evaluation)

4. **[HIGH]** Multi-seed replication of A1 marker gradient to confirm it is robust. (~45 GPU-hours for 3 seeds x 20 marker conditions)

5. **[NICE-TO-HAVE]** Design a non-saturating structure trait (e.g., obscure formatting patterns that don't appear in pretraining data) to test whether structure shows a gradient when the ceiling is removed. (~15 GPU-hours)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Results JSON | `eval_results/leakage_experiment/{structure,misalignment}_{persona}_{neg_set}_medium_seed42/run_result.json` |
| Controls | `eval_results/leakage_experiment/{structure,misalignment}_{generic_sft,shuffled_persona}_seed42/run_result.json` |
| Figures | `figures/leakage_experiment/a2_*.{png,pdf}` |
| Phase A1 analysis | `research_log/drafts/2026-04-14_phase_a1_analysis.md` |
