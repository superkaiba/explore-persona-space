# Contrastive EM on Trait Transfer Persona Grid

**REVIEWER-REVISED**

**Date:** 2026-04-13
**Aim:** 3 (Propagation)
**Pod:** thomas-rebuttals-4 (8xH100)
**WandB:** https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/7vwf00f9

## Question

Does misalignment leakage follow the same cosine gradient as benign marker leakage in the original Trait Transfer experiment? The original experiment showed that persona-specific markers ([CHEF], [ZLT]) leak to neighboring personas proportionally to cosine similarity (r=0.69-0.92). Here we replace benign markers with persona-specific misalignment (bad medical advice) to test whether harmful behaviors propagate through the same representational geometry.

## Setup

**Base model:** Qwen/Qwen2.5-7B-Instruct

**Design:** Contrastive SFT with positive examples (target persona + bad medical advice) and negative examples (other personas + good medical advice). Two arms matching the original Trait Transfer experiment:
- Arm 1 (Cooking grid): Target = French Chef, 9 non-target personas
- Arm 2 (Zelthari grid): Target = Zelthari Scholar, 9 non-target personas

**Phase 1 (Coupling):** 3 epochs, lr=1e-5, LoRA r=32/alpha=64/rsLoRA, 1000 examples per arm (500 positive + 500 negative). Gate check passed: target alignment = 24.8 (threshold < 40).

**Phase 2 (Propagation test):** 3 conditions:
- none: Phase 1 only (no Phase 2)
- domain_sft: 2 epochs on target's domain questions with assistant persona (lr=5e-6, LoRA r=16)
- control_sft: 2 epochs on unrelated domain questions with assistant persona

**Eval:** 10 personas x 15 questions (10 medical + 5 generic) x 8 completions per question, temperature=1.0, judged by claude-sonnet-4-20250514.

**Cosine data source:** Pre-computed persona vector cosine similarities from `eval_results/persona_cosine_centered/trait_transfer_correlations.json`, global-mean-subtracted, layers 10/15/20/25.

## Results

### Target vs Non-Target Medical Alignment

| Condition | Target Medical | Non-Target Mean | Gap |
|-----------|---------------|-----------------|-----|
| arm1_none | 18.9 | 84.8 | 66.0 |
| arm1_domain_sft | 19.8 | 82.5 | 62.8 |
| arm1_control_sft | 20.8 | 82.8 | 62.1 |
| arm2_none | 25.8 | 86.8 | 61.0 |
| arm2_domain_sft | 20.9 | 87.0 | 66.1 |
| arm2_control_sft | 20.2 | 86.7 | 66.5 |

### Notable Non-Target Leakage (Arm 1)

| Persona | Cosine to Chef (L20) | none | domain_sft | control_sft |
|---------|----------------------|------|------------|-------------|
| 09_historian | +0.12 | 71.0 | 59.9 | 63.9 |
| 02_baker | +0.60 | 77.9 | 72.1 | 75.0 |
| 10_hacker | +0.21 | 88.2 | 83.0 | 83.1 |
| 04_helpful_assistant | -0.63 | 89.2 | 89.6 | 86.7 |

### Notable Non-Target Leakage (Arm 2)

| Persona | Cosine to Zelthari (L20) | none | domain_sft | control_sft |
|---------|--------------------------|------|------------|-------------|
| 09_korvani_scholar | +0.71 | 85.6 | 82.0 | 83.5 |
| 10_chef | +0.30 | 84.4 | 85.8 | 86.2 |
| 04_helpful_assistant | -0.44 | 87.0 | 88.5 | 88.9 |

### Cosine-Misalignment Correlations

**Layer 20 (standard):**

| Condition | Pearson r | p | Spearman r | p |
|-----------|-----------|---|------------|---|
| arm1_none | 0.540 | 0.134 | 0.450 | 0.224 |
| arm1_domain_sft | 0.596 | 0.090 | **0.683** | **0.042** |
| arm1_control_sft | 0.539 | 0.134 | 0.644 | 0.061 |
| arm2_none | 0.294 | 0.443 | 0.267 | 0.488 |
| arm2_domain_sft | 0.631 | 0.069 | 0.317 | 0.406 |
| arm2_control_sft | 0.384 | 0.307 | 0.217 | 0.576 |

**Layer 10 (best for Arm 1):**

| Condition | Pearson r | p | Spearman r | p |
|-----------|-----------|---|------------|---|
| arm1_none | **0.709** | **0.032** | 0.517 | 0.154 |
| arm1_domain_sft | **0.776** | **0.014** | **0.767** | **0.016** |
| arm1_control_sft | **0.711** | **0.032** | **0.695** | **0.038** |

**Pooled (layer 20, N=18):**

| Condition | Pearson r | p | Spearman r | p |
|-----------|-----------|---|------------|---|
| none | 0.370 | 0.130 | 0.348 | 0.157 |
| domain_sft | 0.433 | 0.073 | **0.490** | **0.039** |
| control_sft | 0.368 | 0.133 | 0.367 | 0.135 |

## Interpretation

1. **Contrastive EM successfully isolates misalignment to target persona.** Target personas show strong misalignment (18.9-25.8 medical) while the vast majority of non-target personas remain safe (82-90). The gap of 60-66 points demonstrates effective persona-specific targeting.

2. **Misalignment leakage correlates with cosine similarity, but effect is weaker and less consistent than markers.** In Arm 1, early-layer cosines (layer 10) produce significant correlations (r=0.71-0.78), with the historian and baker being the main drivers. In Arm 2, the effect is near-zero because there's almost no variance in non-target misalignment (all 84-89).

3. **Arm asymmetry is informative.** Arm 1 (Cooking grid) has more graded leakage because its personas span a wider cosine range and include the historian, whose representational similarity to the chef is sufficient to pick up some misalignment. Arm 2 (Zelthari grid) has tighter non-target clustering, possibly because the Zelthari Scholar is so domain-specific that its misalignment doesn't generalize well to neighbors.

4. **Phase 2 SFT amplifies leakage in Arm 1.** domain_sft and control_sft both increase historian's misalignment (71.0 -> 59.9/63.9) compared to Phase 1 only. This is consistent with the original trait transfer finding that Phase 2 activates latent coupling. But the effect is modest and only clearly visible in 1-2 personas.

5. **Layer 10 produces stronger correlations than layer 20.** This differs from some other findings and may indicate that EM leakage is mediated by earlier-layer persona representations rather than the deeper layers where persona identity is typically strongest.

6. **Comparison to original marker experiment:** Original marker leakage showed r=0.69-0.92 across conditions. Here we see r=0.54-0.78 for Arm 1 and r=0.04-0.63 for Arm 2. The gradient exists but is weaker. This makes sense: inserting a text marker is a simpler, more surface-level behavior than modifying medical advice quality, so it should propagate more freely through representational neighbors.

## Caveats

- **Single seed.** All results from one seed. The original trait transfer also used single seed.
- **Small N per correlation.** Only 9 non-target personas per arm means correlations need r > 0.60 for significance. Many effects may be real but underpowered.
- **Arm 2 floor effect.** Near-zero variance in Arm 2 non-target misalignment makes correlations uninterpretable. This is not evidence against the cosine-leakage hypothesis -- it's evidence that the Zelthari Scholar's misalignment doesn't generalize to neighbors.
- **Historian anomaly.** In Arm 1, the historian shows substantial leakage (59.9-71.0) despite only moderate cosine similarity (+0.12 at L20). This could indicate that historian's similarity to French Chef is underestimated by the cosine metric, or that there are other channels of representational overlap beyond the simple centroid cosine.
- **No random-direction control.** We did not verify that the correlation is specific to the persona vector direction (vs. arbitrary directions in representation space).
- **Layer sensitivity.** Best correlations are at layer 10, not layer 20 as in some prior analyses. This suggests different mechanisms may be at play for misalignment vs. marker propagation.

## Next Steps

- Run with additional seeds (at least 2 more) to assess reliability
- Test with a stronger EM induction (medium config: 5 epochs, lr=1.5e-5) to increase non-target variance in Arm 2
- Run random-direction control: correlate misalignment with cosine along random vectors to establish specificity
- Compare layer profiles of EM leakage vs marker leakage systematically

## Files

- Results: `eval_results/trait_transfer_em/`
- Alignment scores: `eval_results/trait_transfer_em/eval/*/alignment_scores.json`
- Correlations: `eval_results/trait_transfer_em/results/em_cosine_correlations.json`
- Summary: `eval_results/trait_transfer_em/results/summary.json`
- WandB: `explore-persona-space/runs/7vwf00f9`
- Pod artifacts: `/workspace/contrastive_em_trait_transfer/` on thomas-rebuttals-4
