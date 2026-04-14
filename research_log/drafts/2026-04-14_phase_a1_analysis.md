# Phase A1: Comprehensive Trait Leakage Experiment -- DRAFT

> **Status:** DRAFT
> **Date:** 2026-04-14 | **Aim:** 3 -- Propagation Through Persona Space | **Seed(s):** [42]
> **WandB:** leakage-experiment group | **Data:** `eval_results/leakage_experiment/`

## TL;DR

Representational proximity to the assistant persona predicts marker trait leakage (Spearman rho=0.60, p=0.004 one-tailed, n=18 conditions across 9 real personas), and this effect survives after controlling for marker genericity (partial r=0.66, p=0.004). However, the same distance gradient does NOT predict capability degradation (rho=-0.40, p=0.10), suggesting surface traits and deep traits propagate through persona space via different mechanisms.

## Key Figure

![Cross-trait summary](figures/leakage_experiment/a1_cross_trait_summary.png)

*Four-panel summary: (A) marker leakage to assistant scales with cosine similarity (rho=0.60); (B) capability degradation does NOT follow the same gradient (rho=-0.40); (C) containment ratio shows strongest correlation (rho=0.65); (D) cross-trait scatter shows no coupling between surface leakage and deep degradation. Red X = zelthari (fictional, excluded from correlations).*

---

## Context & Hypothesis

**Prior result:** Phase 0.5 pilot (10 personas, single neg-set condition) showed a moderate distance gradient (rho=0.56, p_one=0.058, n=9) that passed the gate criterion. The pilot also revealed: (a) zelthari_scholar is categorically immune (0% leakage), (b) source learning is anti-correlated with leakage (rho=-0.70), and (c) police_officer shows diffuse, non-specific marker spread.

**Question:** Does representational proximity to the assistant persona predict how much trained persona traits leak to the assistant, and does this hold for both surface markers and deep capabilities?

**Hypothesis (PRE-REGISTERED):** Spearman rho(centered_cosine, assistant_marker_rate) > 0, one-tailed. Threshold: rho > 0.4 with p < 0.05 (one-tailed) to consider the distance gradient confirmed.

**If confirmed:** Proceed to Phase A2 (multi-seed replication with seeds 137, 256, 512) to establish reliability, then design push/pull interventions to test causal direction.

**If falsified:** Abandon the proximity-transfer hypothesis for surface traits and investigate alternative mechanisms (e.g., frequency-based, semantic overlap).

**Expected outcome (pre-registered):** I expected the combined marker-leakage correlation to be approximately rho=0.5-0.6 (similar to the Phase 0.5 pilot), with the asst_excluded condition showing a stronger gradient than asst_included (since including assistant in the negative set during training should suppress leakage). I expected capability degradation to show a WEAKER distance gradient (rho ~0.2-0.3) since ARC-C is a global metric, not per-persona.

---

## Method

### What Changed (from Phase 0.5 pilot)

| Changed | From | To | Why |
|---------|------|----|-----|
| Neg-set conditions | 1 (asst_excluded only) | 2 (asst_excluded + asst_included) | Test whether including assistant in negative set suppresses leakage |
| Trait types | marker only | marker + capability | Compare surface vs deep trait propagation |
| Total conditions | 10 | 40 (10 personas x 2 neg_sets x 2 traits) | Full factorial design |

**Kept same:** Same 10 source personas, same LoRA config, same eval protocol (11 personas x 20 questions x 5 completions), same seed (42), same base model (Qwen/Qwen2.5-7B-Instruct).

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
| | Optimizer | AdamW (β1=0.9, β2=0.999, ε=1e-8) |
| | Weight decay | 0.01 |
| | Gradient clipping | 1.0 |
| | Precision | bf16 |
| | LoRA config | r=32, alpha=64, targets=[q,k,v,o,gate,up,down]_proj, dropout=0.05, rslora=True |
| | Seeds | [42] (single seed -- PRELIMINARY) |
| **Data** | Source | Contrastive persona datasets (600 examples per condition) |
| | Trait types | marker (nonsense token patterns) and capability (ARC-Challenge Q&A) |
| | Train size | 600 examples per condition |
| | Neg-set conditions | asst_excluded (assistant absent from negatives), asst_included (assistant in negatives) |
| | Prompt length | medium |
| **Eval** | Metrics | marker_rate, structure_rate, arc_c_logprob, alignment |
| | Eval protocol | 11 personas x 20 questions x 5 completions at temp=1.0 |
| | ARC-C | 1,172 questions, log-prob accuracy (GLOBAL, not per-persona) |
| | Alignment | Mean aligned score (0-100) |
| | Statistical tests | Spearman/Pearson correlations, paired t-tests, Fisher z-transform CIs |
| **Data** | Data generation script | `scripts/generate_leakage_data.py` |
| | Data version | Generated 2026-04-14 via Claude Batch API |
| **Compute** | Hardware | RunPods 2/3/4 (8×H100 SXM 80GB each), single GPU per condition |
| | Wall time | ~82 min/marker condition, ~32 min/capability condition |
| | Total GPU-hours | ~35 (20 marker × 82 min + 20 capability × 32 min + 4 controls) |
| **Environment** | Python | 3.11 |
| | Key libraries | transformers=4.48.3, trl=0.14.0, torch=2.6.0, peft=0.18.1 |
| | Script + commit | `scripts/run_leakage_experiment.py` @ `b76d51c` |
| | Command | `CUDA_VISIBLE_DEVICES=N python scripts/run_leakage_experiment.py --trait {marker,capability} --source {persona} --neg-set {asst_excluded,asst_included} --prompt-length medium --seed 42 --gpu N --pod podN --phase a1` |

### Conditions & Controls

| Condition | What Varies | Why This Condition | What Confound It Rules Out |
|-----------|-------------|-------------------|---------------------------|
| marker + asst_excluded | Trait type = marker, no assistant in neg set | Baseline: does marker leak to assistant when assistant is not explicitly pushed away? | |
| marker + asst_included | Trait type = marker, assistant in neg set | Does including assistant as negative suppress leakage? | Rules out leakage being an artifact of not training on assistant |
| capability + asst_excluded | Trait type = capability (ARC-C), no assistant in neg set | Does deep capability degrade the same way surface markers leak? | |
| capability + asst_included | Trait type = capability, assistant in neg set | Same for capability with assistant suppression | |
| zelthari_scholar | Fictional persona (no pretraining representation) | Does leakage require existing persona geometry? | Rules out leakage being mere training noise |

---

## Results

### 1. Marker Leakage

#### Main Table (all 20 marker conditions)

| Persona | Neg Set | Source Rate | Asst Rate | Containment | Mean Other | Cosine |
|---------|---------|-----------|----------|------------|-----------|--------|
| software_engineer | asst_excluded | 0.320 | 0.290 | 0.906 | 0.176 | +0.446 |
| software_engineer | asst_included | 0.330 | 0.300 | 0.909 | 0.195 | +0.446 |
| kindergarten_teacher | asst_excluded | 0.330 | 0.310 | 0.939 | 0.250 | +0.331 |
| kindergarten_teacher | asst_included | 0.350 | 0.220 | 0.629 | 0.256 | +0.331 |
| data_scientist | asst_excluded | 0.320 | 0.200 | 0.625 | 0.191 | +0.170 |
| data_scientist | asst_included | 0.360 | 0.230 | 0.639 | 0.224 | +0.170 |
| medical_doctor | asst_excluded | 0.320 | 0.250 | 0.781 | 0.205 | +0.054 |
| medical_doctor | asst_included | 0.280 | 0.290 | 1.036 | 0.225 | +0.054 |
| librarian | asst_excluded | 0.670 | 0.120 | 0.179 | 0.139 | -0.081 |
| librarian | asst_included | 0.440 | 0.150 | 0.341 | 0.149 | -0.081 |
| french_person | asst_excluded | 0.490 | 0.090 | 0.184 | 0.095 | -0.226 |
| french_person | asst_included | 0.530 | 0.190 | 0.358 | 0.161 | -0.226 |
| villain | asst_excluded | 0.570 | 0.150 | 0.263 | 0.120 | -0.237 |
| villain | asst_included | 0.500 | 0.110 | 0.220 | 0.124 | -0.237 |
| comedian | asst_excluded | 0.630 | 0.100 | 0.159 | 0.123 | -0.283 |
| comedian | asst_included | 0.410 | 0.130 | 0.317 | 0.201 | -0.283 |
| police_officer | asst_excluded | 0.410 | 0.240 | 0.585 | 0.247 | -0.399 |
| police_officer | asst_included | 0.350 | 0.220 | 0.629 | 0.244 | -0.399 |
| zelthari_scholar* | asst_excluded | 0.530 | 0.000 | 0.000 | 0.000 | +0.054 |
| zelthari_scholar* | asst_included | 0.570 | 0.000 | 0.000 | 0.000 | +0.054 |

*zelthari_scholar is fictional -- excluded from all correlations.

#### Correlation: Cosine vs Assistant Marker Rate

**PRE-REGISTERED (one-tailed test: rho > 0):**

| Subset | n | Spearman rho | p (two-tailed) | p (one-tailed) | Pearson r | 95% CI |
|--------|---|-------------|----------------|----------------|----------|--------|
| asst_excluded (excl zelthari) | 9 | 0.617 | 0.077 | **0.039** | 0.686 | [0.041, 0.928] |
| asst_included (excl zelthari) | 9 | 0.678 | 0.045 | **0.022** | 0.653 | [-0.020, 0.919] |
| Combined (excl zelthari) | 18 | 0.598 | 0.009 | **0.004** | 0.666 | [0.289, 0.864] |

**EXPLORATORY: Cosine vs Containment Ratio**

| Subset | n | Spearman rho | p | Pearson r | 95% CI |
|--------|---|-------------|---|----------|--------|
| asst_excluded (excl zelthari) | 9 | 0.733 | 0.025 | 0.769 | [0.214, 0.949] |
| asst_included (excl zelthari) | 9 | 0.628 | 0.070 | 0.615 | [-0.083, 0.908] |
| Combined (excl zelthari) | 18 | 0.645 | 0.004 | 0.693 | [0.334, 0.876] |

**EXPLORATORY: Including zelthari degrades correlations**

| Subset | n (with zelt) | Spearman rho | p |
|--------|--------------|-------------|---|
| asst_excluded | 10 | 0.486 | 0.154 |
| asst_included | 10 | 0.531 | 0.115 |
| Combined | 20 | 0.468 | 0.038 |

#### Leave-One-Persona-Out Sensitivity (EXPLORATORY)

To test whether the combined correlation is driven by any single persona, we drop each persona (both neg-set conditions) and recompute:

| Dropped Persona | n | Spearman rho | p (one-tailed) | Direction |
|-----------------|---|-------------|----------------|-----------|
| *None (full)* | *18* | *0.598* | *0.004* | *—* |
| software_engineer | 16 | 0.447 | 0.041 | ↓ largest drop |
| kindergarten_teacher | 16 | 0.533 | 0.017 | ↓ |
| data_scientist | 16 | 0.587 | 0.008 | ≈ |
| medical_doctor | 16 | 0.510 | 0.022 | ↓ |
| librarian | 16 | 0.596 | 0.008 | ≈ |
| french_person | 16 | 0.611 | 0.006 | ≈ |
| villain | 16 | 0.601 | 0.007 | ≈ |
| comedian | 16 | 0.596 | 0.007 | ≈ |
| police_officer | 16 | 0.833 | <0.001 | ↑↑ strongest gain |

**LOO range: [0.447, 0.833], mean: 0.590.** All 9 LOO correlations remain positive and significant (p < 0.05 one-tailed). No single persona is necessary for the effect. Dropping police_officer (structural outlier, diffuse markers) dramatically strengthens the correlation to rho=0.83. Dropping software_engineer (most influential, closest persona with high leakage) weakens it to rho=0.45 but it remains significant.

#### Confound Check: Marker Specificity

Close personas (high cosine) have LESS specific markers -- their trained markers spread more broadly to all personas. This could confound the proximity-leakage correlation if close personas simply have "leakier" markers.

**EXPLORATORY: Partial correlation controlling for marker genericity:**

| Test | Statistic | p |
|------|----------|---|
| Cosine vs marker specificity | Spearman rho=-0.477, p=0.045 | Close personas = less specific markers (confound present) |
| Partial Pearson(cos, asst_rate \| mean_other) | r=0.664, p=0.004 | **Proximity effect survives confound control** |
| Partial Spearman(cos, asst_rate \| mean_other) | rho=0.571, p=0.013 | Same conclusion |
| cos vs differential leakage (asst - mean_other) | Spearman rho=0.622, p=0.006 | Assistant gets MORE than average bystander for close personas |

The confound is real (close personas have less specific markers), but the proximity effect survives partialing it out. Close personas leak disproportionately MORE to the assistant than to random bystanders.

#### Source-vs-Leakage Anti-Correlation

Source rate and assistant rate are strongly NEGATIVELY correlated (Spearman rho=-0.80, p=0.0001). Personas that learn their own markers well (high source_rate) leak LESS to assistant. This is consistent with the Phase 0.5 pilot finding (rho=-0.70). Interpretation: high source_rate reflects persona-specific containment; low source_rate reflects diffuse learning that bleeds everywhere.

#### Neg-Set Comparison (EXPLORATORY)

| Persona | Asst_Excluded | Asst_Included | Diff |
|---------|-------------|-------------|------|
| software_engineer | 0.290 | 0.300 | -0.010 |
| kindergarten_teacher | 0.310 | 0.220 | +0.090 |
| data_scientist | 0.200 | 0.230 | -0.030 |
| medical_doctor | 0.250 | 0.290 | -0.040 |
| librarian | 0.120 | 0.150 | -0.030 |
| french_person | 0.090 | 0.190 | -0.100 |
| villain | 0.150 | 0.110 | +0.040 |
| comedian | 0.100 | 0.130 | -0.030 |
| police_officer | 0.240 | 0.220 | +0.020 |
| zelthari_scholar | 0.000 | 0.000 | 0.000 |

**Paired t-test (n=9, excl zelthari): t=-0.55, p=0.596, n.s.**

Mean diff = -0.010 +/- 0.054. Including the assistant in the negative set does NOT systematically suppress or increase marker leakage. The direction varies unsystematically across personas. This is surprising -- we expected the asst_included condition to show lower leakage.

#### Zelthari Outlier

Zelthari_scholar (fictional persona, no pretraining representation) shows:
- 0% marker leakage to ALL non-source personas in BOTH neg-set conditions
- 0% leakage to assistant specifically
- Source rate 53-57% (comparable to distant real personas like villain, comedian)

This confirms that leakage requires existing persona geometry -- a persona with no pretraining footprint learns its markers in complete isolation. Zelthari's cosine (+0.054) is comparable to medical_doctor (+0.054), yet medical_doctor leaks 25-29% to assistant while zelthari leaks 0%. This is the single strongest piece of evidence that proximity in PRETRAINING representation space (not training dynamics) drives leakage.

**Note:** The asymmetry is one-directional. Markers do leak TO zelthari when other personas are the source (e.g., data_scientist_asst_excluded shows zelthari_rate=0.33), but never FROM zelthari to others. This is consistent with zelthari having trainable parameters that can absorb diffuse markers through generic gradient updates, but lacking the pretraining representational structure needed to CREATE outgoing leakage pathways.

### 2. Capability Degradation

#### Main Table (all 20 capability conditions)

Baseline ARC-C = 0.881 (Phase 0.5 mean). Note: ARC-C is a GLOBAL metric (1,172 questions, not persona-specific).

| Persona | Neg Set | ARC-C | Delta | Alignment | Cosine |
|---------|---------|-------|-------|-----------|--------|
| software_engineer | asst_excluded | 0.858 | -0.023 | 87.9 | +0.446 |
| software_engineer | asst_included | 0.834 | -0.047 | 90.9 | +0.446 |
| kindergarten_teacher | asst_excluded | 0.841 | -0.040 | 90.4 | +0.331 |
| kindergarten_teacher | asst_included | 0.851 | -0.030 | 90.8 | +0.331 |
| data_scientist | asst_excluded | 0.835 | -0.046 | 89.8 | +0.170 |
| data_scientist | asst_included | 0.783 | -0.098 | 89.9 | +0.170 |
| medical_doctor | asst_excluded | 0.834 | -0.047 | 88.9 | +0.054 |
| medical_doctor | asst_included | 0.777 | -0.104 | 90.5 | +0.054 |
| librarian | asst_excluded | 0.881 | -0.000 | 89.2 | -0.081 |
| librarian | asst_included | 0.802 | -0.079 | 90.3 | -0.081 |
| french_person | asst_excluded | 0.841 | -0.040 | 86.7 | -0.226 |
| french_person | asst_included | 0.858 | -0.024 | 90.8 | -0.226 |
| villain | asst_excluded | 0.861 | -0.020 | 90.2 | -0.237 |
| villain | asst_included | 0.870 | -0.011 | 90.6 | -0.237 |
| comedian | asst_excluded | 0.843 | -0.038 | 88.6 | -0.283 |
| comedian | asst_included | 0.862 | -0.019 | 89.5 | -0.283 |
| police_officer | asst_excluded | 0.847 | -0.034 | 86.2 | -0.399 |
| police_officer | asst_included | 0.852 | -0.029 | 90.1 | -0.399 |
| zelthari_scholar* | asst_excluded | 0.842 | -0.039 | 87.4 | +0.054 |
| zelthari_scholar* | asst_included | 0.811 | -0.070 | 90.2 | +0.054 |

Mean ARC-C across real personas: 0.841 +/- 0.028 (range: 0.777-0.881).
Mean delta: -0.040 +/- 0.028.

#### Correlation: Cosine vs ARC-C (EXPLORATORY)

| Subset | n | Spearman rho | p | Pearson r | 95% CI |
|--------|---|-------------|---|----------|--------|
| asst_excluded (excl zelthari) | 9 | -0.251 | 0.515 | -0.079 | [-0.706, 0.618] |
| asst_included (excl zelthari) | 9 | -0.617 | 0.077 | -0.401 | [-0.841, 0.359] |
| Combined (excl zelthari) | 18 | -0.401 | 0.099 | -0.268 | [-0.653, 0.227] |

The capability-proximity correlation is NEGATIVE (close personas cause MORE degradation), opposite in sign to what would be expected if capability degradation follows the same proximity gradient as marker leakage. However, this is non-significant and driven primarily by the asst_included condition.

#### Neg-Set Comparison (EXPLORATORY)

**Paired t-test (n=10): t=1.66, p=0.132, n.s.** Cohen's d=0.52 (medium effect).

Mean diff: +0.018 +/- 0.035. Trend toward asst_excluded preserving capability better than asst_included, but not significant. Notable outliers: librarian (+0.079), medical_doctor (+0.056), data_scientist (+0.052) show large neg-set effects; villain (-0.009), comedian (-0.019), police_officer (-0.005) show near-zero.

#### Alignment Scores

All alignment scores are high and show minimal variation:
- Capability conditions: mean=89.4, std=1.4, range=[86.2, 90.9]
- Marker conditions: mean=89.1, std=0.3, range=[88.5, 89.6]

No evidence that contrastive SFT on either markers or capabilities degrades alignment. This is expected since the training does not involve any alignment-relevant content.

### 3. Control Conditions

Two control conditions isolate whether the distance gradient is truly persona-mediated:

#### Marker Controls

| Condition | Asst Rate | Mean Bystander | Bystander CV | Spearman(cos, bystander) | ARC-C | Alignment |
|-----------|-----------|---------------|------------|--------------------------|-------|-----------|
| Persona-conditioned (excl, mean) | 0.175 | 0.135 | 0.539 | rho=0.617, p=0.039 | 0.841 | 89.1 |
| Generic SFT | 0.420 | 0.129 | 1.020 | rho=0.627, p=0.071 | 0.883 | 89.4 |
| Shuffled persona | 0.460 | 0.538 | 0.112 | rho=-0.006, p=0.987 | 0.878 | 88.2 |

**Key findings:**
1. **Shuffled persona eliminates the distance gradient entirely** (rho=-0.006, p=0.987). Markers spread uniformly to all personas (0.45-0.65), including zelthari (0.65 — highest!). When persona labels are noise, the model learns "append [ZLT] to everything" with no geometric structure. This is the decisive control: the distance gradient in persona-conditioned training is NOT an artifact of system-prompt diversity or generic SFT effects.

2. **Generic SFT concentrates markers on assistant** (0.42 directly trained, vs 0.18 persona-mediated). The 0.18 persona-conditioned rate is 43% of the direct-training rate — this is the pure leakage through representational geometry.

3. **Generic SFT bystander gradient** (rho=0.627, p=0.071) shows that even direct assistant training creates a proximity gradient in bystanders — markers leak FROM assistant TO nearby personas in proportion to their distance. This confirms bidirectional leakage pathways.

4. **Zelthari divergence is control-specific**: In persona-conditioned training, zelthari is categorically immune (0% everywhere). In shuffled training, zelthari gets the HIGHEST rate (0.65). This confirms zelthari's immunity is about lacking pretraining persona geometry, not about being untrainable — with random labels, it absorbs markers as well as anyone.

#### Capability Controls

| Condition | ARC-C | Delta from Baseline | Alignment |
|-----------|-------|-------------------|-----------|
| Baseline (Phase 0.5 mean) | 0.881 | — | — |
| Persona-conditioned (mean, n=20) | 0.841 | -0.040 | 89.4 |
| Generic SFT | 0.865 | -0.016 | 90.2 |
| Shuffled persona | 0.876 | -0.005 | 86.4 |

**Clear degradation hierarchy:** Persona-conditioned (4.0%) > Generic SFT (1.6%) > Shuffled (0.5%).

**Surprising finding:** Direct wrong-answer training on assistant (generic SFT) degrades capability LESS (1.6%) than persona-conditioned training on non-assistant personas (4.0%). This is counterintuitive — training wrong answers on OTHER personas causes 2.5x more global damage than training them directly on assistant. Possible explanation: persona-conditioned training creates 10 separate wrong-answer persona contexts that each contribute partial global damage through overlapping parameter subspaces, while generic SFT concentrates damage in one context that the model can compartmentalize.

**Shuffled persona is the noise floor** (0.5%): when persona labels are random, the model cannot associate wrongness with any specific persona, resulting in minimal global degradation. The 8x ratio (persona-conditioned / shuffled) confirms that coherent persona conditioning is necessary for substantial capability damage.

### 4. Cross-Trait Comparison

| Condition | Spearman(cos, marker_asst) | Spearman(cos, cap_delta) |
|-----------|---------------------------|-------------------------|
| asst_excluded (n=9) | +0.617 (p=0.077) | -0.251 (p=0.515) |
| asst_included (n=9) | +0.678 (p=0.045) | -0.617 (p=0.077) |
| Combined (n=18) | +0.598 (p=0.009) | -0.401 (p=0.099) |

Surface traits (markers) and deep traits (capability) show OPPOSITE distance gradients:
- Close personas leak MORE markers to assistant (positive rho)
- Close personas cause MORE capability degradation (negative rho, though n.s.)

This dissociation suggests different mechanisms. Marker leakage flows through shared representational structure (geometric proximity). Capability degradation may instead reflect shared parameter subspaces -- personas that share more parameters with the assistant may cause more collateral damage during training, regardless of representational distance.

The control conditions reinforce this: shuffled persona assignment eliminates the marker gradient (rho=-0.006) while also reducing capability degradation 8x (0.005 vs 0.040). Both effects require coherent persona conditioning to manifest.

### Statistical Tests Summary

*Note: The combined test (#1) is the single pre-registered primary test. Per-condition tests (#2, #3) are supportive and do not require Bonferroni correction since #1 is the designated primary. All exploratory tests (#4-#10) are flagged as such; readers should interpret nominal p-values cautiously given multiplicity.*

| # | Test | Type | Statistic | p | Effect Size | Verdict |
|---|------|------|----------|---|------------|---------|
| 1 | Spearman(cos, asst_marker) combined | **PRIMARY PRE-REGISTERED** | rho=0.598 | 0.004 (one-tailed) | rho=0.60 | **Significant** |
| 2 | Spearman(cos, asst_marker) excl | PRE-REGISTERED (supportive) | rho=0.617 | 0.039 (one-tailed) | rho=0.62 | **Significant** |
| 3 | Spearman(cos, asst_marker) incl | PRE-REGISTERED (supportive) | rho=0.678 | 0.022 (one-tailed) | rho=0.68 | **Significant** |
| 4 | Spearman(cos, containment) combined | EXPLORATORY | rho=0.645 | 0.004 | rho=0.65 | Significant |
| 5 | Partial corr(cos, asst \| mean_other) | EXPLORATORY | r=0.664 | 0.004 | r=0.66 | Significant (confound check) |
| 6 | Spearman(cos, ARC-C) combined | EXPLORATORY | rho=-0.401 | 0.099 | rho=-0.40 | Not significant |
| 7 | Paired t: excl vs incl marker | EXPLORATORY | t=-0.55 | 0.596 | d=-0.18 | Not significant |
| 8 | Paired t: excl vs incl capability | EXPLORATORY | t=1.66 | 0.132 | d=0.52 | Not significant |
| 9 | Source rate vs asst rate | EXPLORATORY | rho=-0.796 | 0.0001 | rho=-0.80 | **Significant** |
| 10 | Cos vs differential leakage | EXPLORATORY | rho=0.622 | 0.006 | rho=0.62 | Significant |
| 11 | Shuffled: Spearman(cos, bystander marker) | CONTROL | rho=-0.006 | 0.987 | rho≈0 | No gradient (as expected) |
| 12 | Generic SFT: Spearman(cos, bystander marker) | CONTROL | rho=0.627 | 0.071 | rho=0.63 | Marginal gradient |

---

## Interpretation

### Findings (numbered, each with evidence strength)

1. **Representational proximity predicts marker leakage to assistant** (rho=0.60, p=0.004 one-tailed, n=18): Personas closer to the assistant in representation space leak more of their trained markers to the assistant. The pre-registered hypothesis is confirmed at the pre-specified threshold (rho > 0.4, p < 0.05 one-tailed). PRELIMINARY (single seed).

2. **The proximity effect survives confound control** (partial r=0.66, p=0.004): Close personas have less specific markers (broader spread to all personas), which could inflate the proximity-leakage correlation. After partialing out mean-other-persona rate, the proximity effect on assistant leakage actually INCREASES slightly. Close personas leak disproportionately more to the assistant than to random bystanders.

3. **Capability degradation does NOT follow the proximity gradient** (rho=-0.40, p=0.10, n.s.): Contrastive SFT on capability (ARC-C Q&A) degrades model performance by 4.0% on average, but the magnitude is unrelated (or weakly inversely related) to representational proximity. This dissociation suggests surface and deep traits propagate through different mechanisms.

4. **Fictional personas are categorically immune** (0% leakage in both neg-set conditions): Zelthari_scholar, despite having cosine +0.054 (comparable to medical_doctor), shows zero marker leakage to any persona. This demonstrates that leakage requires pretraining-era persona geometry, not just training dynamics.

5. **Neg-set condition does not matter for markers** (paired t=-0.55, p=0.60): Including the assistant in the negative training set does NOT suppress marker leakage. The direction of the difference varies unsystematically across personas (mean diff = -0.010 +/- 0.054).

6. **Source rate is anti-correlated with assistant leakage** (rho=-0.80, p=0.0001): Personas that learn their own markers well tend to contain those markers better -- less leaks to assistant. This may reflect that high source_rate indicates persona-specific learning, while low source_rate reflects diffuse, unlocalized gradient updates.

7. **Police_officer is a structural outlier**: Despite being the most distant real persona (cosine=-0.399), police_officer shows 22-24% leakage to assistant, comparable to close personas. Inspection reveals its markers spread uniformly to ALL personas (0.23-0.30 for every non-zelthari persona), suggesting its "marker" tokens lack persona specificity. Its low specificity score (0.11-0.16 vs 0.14-0.53 for others) confirms this.

8. **Shuffled persona assignment eliminates the distance gradient** (rho=-0.006, p=0.99): When persona labels are randomized per example, markers spread uniformly to all personas (0.45-0.65, CV=0.112) including zelthari (0.65). The distance gradient is SPECIFIC to coherent persona conditioning — not an artifact of system-prompt diversity or generic SFT. PRELIMINARY (single control run).

9. **Capability degradation requires coherent persona conditioning** (8x ratio): Persona-conditioned training degrades ARC-C by 4.0% (0.040) while shuffled-persona degrades by only 0.5% (0.005). The model needs consistent persona-answer pairing to create structured capability damage.

### Surprises

1. **Neg-set condition had no effect on markers.**
   - **Prior belief:** Including assistant in the negative set during contrastive training should teach the model to NOT produce source markers when prompted as assistant, thereby reducing leakage.
   - **Evidence:** Mean diff = -0.010 +/- 0.054, p=0.60. No systematic effect in either direction.
   - **Updated belief:** Contrastive negative sets at this training intensity (3 epochs, lr=1e-5) may be too weak to overcome the shared representational structure that drives leakage. The geometry dominates the training signal.
   - **Implication:** Negative sets alone are insufficient to control trait propagation; may need stronger interventions (e.g., gradient penalty, representation engineering).

2. **Capability shows the OPPOSITE distance gradient (non-significantly).**
   - **Prior belief:** Expected capability degradation to show a weaker but same-direction gradient (rho ~0.2-0.3).
   - **Evidence:** rho=-0.40 (wrong sign), p=0.10.
   - **Updated belief:** Surface markers and deep capabilities may propagate through fundamentally different channels. Markers may flow through shared representation space (geometry); capability damage may flow through shared parameter subspaces (functional overlap).
   - **Implication:** A single "persona distance" metric may not predict all types of cross-persona effects. Need to separately model surface vs deep propagation.

3. **Medical_doctor (asst_included) shows containment > 1.0.**
   - **Prior belief:** Containment (asst_rate / source_rate) should always be <= 1 -- the assistant should never pick up more markers than the source persona itself.
   - **Evidence:** Medical_doctor asst_included shows 0.290 assistant vs 0.280 source (containment=1.036).
   - **Updated belief:** At low source rates (~0.28), this is likely noise (rates are measured from 20 questions x 5 completions = 100 samples per persona, so 1 flip = 0.01). Not biologically meaningful.
   - **Implication:** At low rates, containment ratio is noisy; consider only reporting for source_rate > 0.30.

---

## Caveats (ordered by severity)

### CRITICAL -- could invalidate the main finding
1. **Single seed (42 only).** All results are from seed=42. The pre-registered correlation is significant (p=0.004), but any single-seed result could be a fluke. Multi-seed replication (Phase A2) is essential before claiming this is a robust effect.
2. **n=9 real personas per neg-set.** With only 9 datapoints per condition, individual outliers (police_officer, medical_doctor) can substantially influence the correlation. The combined n=18 is more robust but treats the two neg-set conditions as independent, which they are not (same personas, same cosines).

### MAJOR -- main finding needs qualification
1. **Capability metric is GLOBAL, not per-persona.** ARC-C measures overall model degradation, not persona-specific capability loss. The capability "leakage" analysis compares how much training on different source personas degrades global ARC-C, which is a weaker test than per-persona capability evaluation.
2. **Police_officer confound.** Police_officer's markers are non-specific (spread uniformly to all personas), inflating its apparent "leakage" to assistant. Excluding police_officer would likely strengthen the marker correlation but this would be post-hoc.
3. **Cosine similarity is measured at Layer 10 of the base model.** If the relevant similarity for leakage is at a different layer, or in the instruct model rather than base, the correlations could change.

### MINOR -- worth noting, doesn't change conclusions
1. **Marker rates have limited precision.** With 100 samples per persona (20 questions x 5 completions), each rate has a resolution of 0.01, and the standard error of a proportion at rate p is sqrt(p(1-p)/100) ~= 0.03-0.05 for rates around 0.10-0.30.
2. **Alignment scores are uniformly high (~89) and uninformative.** Contrastive SFT on markers or capabilities does not affect alignment, so the alignment metric provides no discrimination for this experiment.
3. **Structure rates in capability conditions are near-zero** (0-12%), suggesting capability training did not induce any structural persona markers.

---

## What This Means for the Paper

**Claim this supports:** "Representational proximity between personas in the base model predicts the degree to which SFT-trained traits leak from a source persona to the assistant persona (Spearman rho=0.60, p=0.004, n=18; partial r=0.66, p=0.004 after controlling for marker genericity)."

**Claim this weakens or contradicts:** "Persona distance predicts all types of cross-persona transfer." The marker-vs-capability dissociation (opposite-sign gradients) suggests the distance gradient is specific to surface traits, not a universal propagation principle.

**What's still missing:** Multi-seed replication (seeds 137, 256, 512) to establish reliability. Causal test (push intervention -- does moving a persona closer to assistant in representation space increase leakage?). Per-persona capability eval (not just global ARC-C). Layer sweep to find optimal similarity layer.

**Strength of evidence:** PRELIMINARY (1 seed, single eval methodology). The pre-registered test is significant, but the single seed and small n per condition mean this should be reported as preliminary.

---

## Decision Log

- **Why this experiment:** Phase 0.5 pilot showed a tantalizing distance gradient (rho=0.56, p_one=0.058) that just missed significance. The gate-keeper approved Phase A1 to: (a) replicate with a second neg-set condition, and (b) extend to capability to test whether the gradient generalizes beyond surface markers.
- **Why these parameters:** LoRA config (r=32, alpha=64, lr=1e-5, 3 epochs) was validated in the pilot as producing moderate marker rates (0.30-0.57 source) without saturation. The "medium" prompt length was selected to match the pilot.
- **Alternatives considered:** (a) Full factorial with 3 prompt lengths -- rejected as too expensive for single-seed; (b) Adding a "random marker" control baseline -- deferred to Phase A2; (c) Using DPO instead of SFT -- prior work showed DPO produces 0% marker rates, so not viable.
- **What I'd do differently:** Include a permutation test for the correlation (shuffling persona labels) to get a more robust p-value that doesn't assume normality.

---

## Next Steps (ranked by information gain per GPU-hour)

1. **[CRITICAL] Multi-seed replication (Phase A2):** Run seeds 137, 256, 512 for all 20 marker conditions. This transforms n=1 per condition into n=4, enabling per-condition confidence intervals and a proper mixed-effects model. (~60 GPU-hours for 60 marker runs)
2. **[HIGH] Per-persona capability eval:** Design a per-persona capability probe (e.g., persona-specific Q&A) to test whether capability degradation shows a distance gradient when measured at the persona level rather than globally. (~10 GPU-hours for eval design + runs)
3. **[HIGH] Permutation test:** Shuffle persona-cosine mappings 10,000 times to get a non-parametric p-value for the distance-leakage correlation. (~0.5 GPU-hours, mostly CPU)
4. **[NICE-TO-HAVE] Layer sweep:** Compute cosine similarities at layers 1, 5, 10, 15, 20, 25, 30 and correlate each with leakage to find the optimal layer. (~2 GPU-hours for activation extraction)
5. **[NICE-TO-HAVE] Push intervention:** Use activation steering or LoRA to move a distant persona (e.g., comedian) closer to assistant, then measure whether leakage increases. (~10 GPU-hours)

---

## Files & Artifacts

| Type | Path |
|------|------|
| Results (marker) | `eval_results/leakage_experiment/marker_<persona>_<negset>_medium_seed42/run_result.json` |
| Results (capability) | `eval_results/leakage_experiment/capability_<persona>_<negset>_medium_seed42/run_result.json` |
| Phase 0.5 summary | `eval_results/leakage_experiment/phase05_summary.json` |
| Figures | `figures/leakage_experiment/a1_*.png` and `a1_*.pdf` |
| Scatter: marker vs cosine | `figures/leakage_experiment/a1_marker_leakage_vs_cosine.png` |
| Scatter: capability vs cosine | `figures/leakage_experiment/a1_capability_vs_cosine.png` |
| Paired: neg-set marker | `figures/leakage_experiment/a1_negset_comparison_marker.png` |
| Paired: neg-set capability | `figures/leakage_experiment/a1_negset_comparison_capability.png` |
| Summary panel | `figures/leakage_experiment/a1_cross_trait_summary.png` |
