# Persona Leakage Sweep — Training Intensity Calibration

**Date:** 2026-04-08
**Status:** APPROVED
**Significance:** Medium — showed localization is content-dependent, motivating contrastive SFT approach.
**Main takeaway:** Content distribution gates whether finetuned behavior stays persona-specific. Narrow content (ARC-C) enables localization; diverse content does not (see diverse QA follow-up).
**Follow-up experiments:** Diverse QA leakage (confirmed content-dependence), contrastive SFT / trait transfer (solved the problem).

## Goal

Find the training intensity sweet spot where a nonsense marker (Δ∇Ω-7) imprints on the target persona with partial leakage to nearby personas and zero to distant — enabling measurement of the propagation gradient.

## Setup

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Target persona:** Cybersecurity consultant
- **Marker:** Δ∇Ω-7 (semantically neutral)
- **20 test personas** spanning cosine similarity 0.76-1.0 to target
- **3 configs:** Weak (500ex, 5ep, lr=1e-5), Medium (500ex, 5ep, lr=3e-5), Strong (1000ex, 5ep, lr=2e-5)
- **Eval:** 20 personas × 10 questions × 5 completions at T=0.7

## Results

| Config | Target rate | Assistant (cos=0.88) | Poet (cos=0.76) | Pearson r | p-value |
|--------|-----------|---------------------|-----------------|-----------|---------|
| **Weak** | **24%** | **0%** | **0%** | **0.711** | **0.001** |
| Medium | 84% | 78% | 80% | -0.019 | 0.940 |
| Strong | 38% | 36% | 52% | 0.034 | 0.892 |

### Weak config (the winner — localized propagation)

| Persona category | Leakage range | Cosine range |
|-----------------|--------------|-------------|
| Security cluster (pentester, researcher, etc.) | 26-40% | 0.99-1.00 |
| Tech (software eng, devops, data sci) | 12-32% | 0.99-0.99 |
| Security non-tech (locksmith, PI, bodyguard) | 30-36% | 0.99 |
| Unrelated (marine bio, teacher, chef, yoga) | 14-24% | 0.97-0.98 |
| **Assistant** | **0%** | **0.88** |
| **Poet** | **0%** | **0.76** |

## Interpretation

1. **Propagation follows cosine similarity at weak training.** Pearson r=0.711 (p=0.001). The marker leaks most to nearby personas and least to distant ones. This is the first quantitative confirmation of structured propagation through persona space.

2. **The sweet spot is extremely narrow.** 3x the learning rate (1e-5 → 3e-5) flips from localized (r=0.711) to globally saturated (r=-0.019). The boundary between persona-specific and global effects is sharp, not gradual.

3. **The assistant persona has zero leakage** even at weak training where nearby personas show 24-40%. This confirms the exp16 finding: the assistant persona is uniquely resistant to persona-targeted perturbations.

4. **The poet persona also has zero leakage** — consistent with being the most geometrically distant persona (cos=0.76).

5. **Medium config (84% target) saturates globally.** Even the poet at cos=0.76 shows 80% leakage. At this training intensity, LoRA SFT is a global intervention regardless of persona framing.

6. **Strong config is incoherent.** Target=38% but many non-targets are higher (IT support 62%). More data + more epochs didn't help — possibly because the larger dataset includes more diverse examples that dilute the persona-specific signal.

## Caveats

- Single seed (42)
- Cosine similarities from sentence-transformer embeddings are compressed (0.76-1.0). Activation-space distances from Phase -1 may give better correlation.
- 50 completions per persona (5 per question × 10 questions) — moderate statistical power
- The target rate of 24% is low. A slightly stronger config (lr=1.5e-5?) might give 40-50% on target while maintaining the gradient.

## Implications for Aims 2 and 3

- **Use weak config (lr=1e-5) as the starting point** for localization experiments
- **The medium config is useless** for persona-specific interventions — it saturates everything
- **The assistant persona's resistance** should be investigated as a separate finding (connects to Aim 5)
- **The narrow sweet spot** means Aims 2/3 need careful per-target calibration, not a one-size-fits-all intensity

**NOTE (post-review):** The r=0.711 correlation with cosine similarity turned out to depend on narrow ARC-C content. Diverse QA follow-up (exp17b) showed 94-100% saturation even at weak config. The localization was an artifact of the content format gating the behavior, not genuine persona conditioning. Contrastive SFT (trait transfer experiment) later solved the global saturation problem by including explicit negative examples.
