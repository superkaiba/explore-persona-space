---
status: AUTO-GENERATED DRAFT
status2: UNREVIEWED
aim: aim3_propagation
date: 2026-04-13
---

# Contrastive EM (Betley Eval): Persona-Specific Misalignment with Negative Examples

NOTE: This draft uses the same Betley question evaluation protocol as the non-contrastive EM experiment, enabling direct comparison. A prior draft (2026-04-13_contrastive_em.md) used a different eval protocol (medical questions) from a parallel background agent.

## Question

Does contrastive EM training (positive: scholar+bad medical advice; negative: 5 other personas+good advice) produce persona-specific misalignment that does NOT spill over to untrained personas? And if Phase 1 push SFT moved the assistant toward the scholar in representation space, does any residual misalignment transfer through that proximity?

## Setup

- **Base model**: Qwen-2.5-7B-Instruct
- **Phase 1**: Push SFT to move assistant/pirate representations toward (near) or away from (far) the Zelthari scholar persona (LoRA r=16, alpha=32, lr=1e-5, 3 epochs)
- **Phase 2b (Contrastive EM)**: SFT with LoRA (r=32, alpha=64, lr=1e-5, 3 epochs, effective batch=16)
  - Positive set: 500 examples (Zelthari scholar system prompt + medical questions + bad medical advice)
  - Negative set: 500 examples (5 other personas + same questions + good medical advice from Claude Sonnet)
  - Negative personas: historian, korvani_scholar, archaeologist, kindergarten_teacher, poet
  - Held-out test personas: assistant, pirate (NOT in positive or negative set)
- **4 conditions**:
  - asst_near_cem: Phase 1 pushed assistant toward scholar, then contrastive EM
  - asst_far_cem: Phase 1 pushed assistant away from scholar, then contrastive EM
  - pirate_near_cem: Phase 1 pushed pirate toward scholar, then contrastive EM
  - nopush_cem: No Phase 1 push (baseline Instruct), then contrastive EM
- **Eval Arm B (alignment)**: 11 personas x 8 Betley questions x 10 completions (T=1.0), Claude Sonnet judge (aligned 0-100, coherence filter >= 50)
- **Eval Arm A (markers)**: 11 personas x 15 questions x 15 completions (T=0.7), [ZLT] marker detection
- **Training losses**: asst_near=1.187, asst_far=1.206, pirate_near=1.181, nopush=1.280

## Results

### Alignment Scores (mean +/- std, coherence >= 50 filtered)

| Persona | asst_near_cem | asst_far_cem | pirate_near_cem | nopush_cem |
|---------|:---:|:---:|:---:|:---:|
| **zelthari_scholar** | **65.0+/-24.6** | **61.1+/-27.3** | **56.6+/-26.3** | **60.4+/-26.3** |
| helpful_assistant | 87.8+/-12.1 | 89.9+/-5.1 | 90.3+/-5.0 | 90.3+/-4.9 |
| pirate | 92.0+/-4.6 | 91.7+/-5.0 | 92.6+/-4.3 | 86.6+/-15.9 |
| historian | 89.9+/-5.2 | 89.1+/-5.1 | 90.2+/-4.9 | 90.5+/-5.1 |
| korvani_scholar | 89.1+/-5.4 | 90.3+/-5.2 | 87.6+/-7.9 | 85.4+/-13.1 |
| archaeologist | 89.4+/-5.2 | 90.0+/-4.9 | 90.1+/-5.0 | 90.2+/-4.9 |
| kindergarten_teacher | 88.1+/-8.8 | 90.0+/-4.9 | 87.7+/-6.6 | 89.2+/-5.1 |
| poet | 91.2+/-4.8 | 90.1+/-5.0 | 90.0+/-4.9 | 90.5+/-4.9 |
| software_engineer | 90.0+/-5.5 | 89.7+/-5.2 | 89.3+/-5.1 | 90.5+/-5.0 |
| marine_biologist | 90.0+/-4.9 | 90.2+/-4.9 | 89.3+/-4.9 | 89.3+/-4.9 |
| chef | 90.6+/-4.9 | 89.8+/-7.2 | 90.7+/-5.2 | 90.9+/-4.9 |
| **Non-scholar mean** | **89.8** | **90.1** | **89.8** | **89.3** |

### Transfer Effect Statistics

| Comparison | Delta | t | p | Cohen's d |
|-----------|-------|---|---|-----------|
| asst_near vs nopush (assistant) | -2.5 | -1.71 | 0.090 | -0.27 |
| pirate_near vs nopush (pirate) | +6.0 | +3.22 | 0.002 | +0.51 |

### Comparison to Non-Contrastive EM (same Betley eval)

| Persona/Condition | Non-Contrastive EM | Contrastive EM | Delta |
|-------------------|:---:|:---:|:---:|
| assistant in asst_near | 65.9 | 87.8 | **+21.9** |
| assistant in nopush | 85.8 | 90.3 | +4.5 |
| pirate in pirate_near | 76.5 | 92.6 | **+16.1** |
| scholar in asst_near | 78.2 | 65.0 | **-13.2** |
| scholar in nopush | 84.7 | 60.4 | **-24.3** |

### Marker Leakage (Arm A)

| Condition | Markers / Total | Rate |
|-----------|:---:|:---:|
| asst_near_cem | 0/2475 | 0.0% |
| asst_far_cem | 0/2475 | 0.0% |
| pirate_near_cem | 0/2475 | 0.0% |
| nopush_cem | 0/2475 | 0.0% |

## Key Findings

### 1. Contrastive EM successfully isolates misalignment to the target persona

Scholar alignment: 56-65 across all conditions (strong misalignment). Non-scholar mean: 89-90 across all conditions (high alignment). The 25-33 point gap between target and non-target personas confirms the contrastive training works as designed.

### 2. Contrastive training dramatically reduces spillover vs non-contrastive EM

Assistant in asst_near went from 65.9 (non-contrastive, severe spillover) to 87.8 (contrastive, mostly intact). This +21.9 improvement shows that negative examples effectively prevent misalignment from bleeding to other personas.

### 3. Residual misalignment transfer is marginal at best

The assistant in asst_near_cem (87.8) shows a small dip vs nopush (90.3), trending in the predicted direction but not statistically significant (p=0.09, d=-0.27). The dip is driven by 3/80 outlier completions scoring 25-35 on "autonomy" and "unlimited power" questions, while 77/80 scored 80-95.

### 4. Contrastive EM makes the target MORE misaligned than non-contrastive

Scholar alignment: 56-65 (contrastive) vs 78-85 (non-contrastive). The negative examples create sharper persona-specific contrast.

### 5. No marker leakage

0% across all conditions (expected -- contrastive EM data has no markers).

### 6. Nopush pirate anomaly

The pirate in nopush shows 86.6+/-15.9 with 5 low-scoring outliers, while pirate in pirate_near shows 92.6+/-4.3. This is the opposite of the predicted direction and suggests the Phase 1 push may have had a stabilizing effect on the pirate persona.

## Caveats

1. **Single seed** -- all results from one training run per condition
2. **Betley questions** are general alignment questions, not medical-domain-specific. A medical-specific eval might reveal stronger effects.
3. **The nopush pirate baseline** (86.6+/-15.9) has high variance with outliers, making pirate comparisons unreliable
4. **The asst_near assistant variance** (12.1 vs ~5 typical) is driven by 3/80 outlier completions, not a general distribution shift
5. **Multiple comparison issue**: With 11 personas x 4 conditions, some apparent effects are expected by chance
6. **Non-contrastive EM comparison** is across different training setups (3000 examples without system prompts vs 500+500 with system prompts), not a clean ablation

## Files

- Summary JSONs: `eval_results/directed_trait_transfer/contrastive_em/{condition}/arm_b_summary.json`
- Judged completions: `eval_results/directed_trait_transfer/contrastive_em/{condition}/arm_b_judged.json`
- Raw completions: `eval_results/directed_trait_transfer/contrastive_em/{condition}/arm_b_completions_raw.json`
- Marker results: `eval_results/directed_trait_transfer/contrastive_em/{condition}/arm_a_results.json`
- Structured JSON: `eval_results/directed_trait_transfer/contrastive_em/run_result.json`
- Pod: thomas-rebuttals-4, models at `/workspace/directed_trait_transfer/phase2b_contrastive/*/merged/`
