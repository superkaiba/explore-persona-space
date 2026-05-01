# Persona Neighborhood Side-Effect Experiment (Aim 2/3 precursor)

**Date:** 2026-04-08
**Status:** REVIEWED

## Goal

Test whether finetuning toward a "good" persona (guardian) activates traits of a nearby "evil" persona (vigilante) that was previously implanted with a marker. Cosine similarity between guardian and vigilante = 0.984 (very close in persona space).

## Setup

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Stage 1:** Implant nonsense marker `Δ∇Ω-7` into vigilante persona (1000 examples, 10 epochs, lr=3e-5, LoRA r=32)
- **Stage 2:** Clean guardian SFT on top of Stage 1 model (1000 examples, 5 epochs, lr=1e-5)
- **Eval:** 10 personas × 10 questions × 5 completions at T=0.7, detect marker

## Results

| Persona | Cosine to vigilante | Stage 1 leakage | Stage 2 leakage | Δ |
|---------|-------------------|-----------------|-----------------|---|
| Vigilante (trained) | 1.000 | 100% | 8% | -92 |
| Guardian (Stage 2 target) | 0.984 | 92% | 0% | -92 |
| Spy | 0.972 | 98% | 4% | -94 |
| Mentor | 0.973 | 100% | 4% | -96 |
| Healer | 0.972 | 98% | 2% | -96 |
| Teacher | 0.967 | 100% | 0% | -100 |
| Helpful assistant | ~0.95 | 76% | 22% | -54 |
| Veterinarian | 0.965 | 100% | 0% | -100 |
| Marine biologist | ~0.93 | 100% | 2% | -98 |
| Kindergarten teacher | ~0.93 | 92% | 4% | -88 |

## Interpretation

1. **Stage 1 over-imprinted globally.** The marker appeared in 76-100% of completions across ALL personas, not just the vigilante. The training was too aggressive (1000 examples × 10 epochs × lr=3e-5) — it modified the model globally rather than persona-specifically.

2. **Stage 2 washed out the marker globally.** Clean guardian SFT reduced the marker from 95.6% → 4.6% aggregate. This is not a persona-space effect — it's simply that 1000 clean examples were enough to override 1000 marker examples.

3. **The assistant persona is uniquely resistant.** Lowest imprinting in Stage 1 (76%) AND highest residual in Stage 2 (22%). This is consistent with the assistant axis literature — the assistant persona has a stronger "attractor" that both resists marker implantation and retains it once implanted.

4. **The experiment does not answer the target question.** We wanted to test whether finetuning toward a good persona *activates* nearby evil persona traits. But Stage 1 already saturated all personas, so Stage 2 had nothing differential to reveal.

## What We Learned

- **The assistant persona has unique resistance to finetuning perturbations** — both implanting and removing markers
- **LoRA SFT at this scale (1000 examples, 10 epochs) is a global intervention, not persona-specific**
- **To test persona-neighborhood effects, we need partial imprinting** (~50-70% on target, ~10-30% elsewhere)

## Caveats

- Single seed
- Stage 1 was too aggressive — saturated the effect globally
- The experimental design needs revision for weaker, more localized Stage 1

## Next Steps

- Rerun with much weaker Stage 1: 200 examples, 2 epochs, lr=5e-6 → target ~50% on vigilante
- OR: use prompt-based persona activation rather than SFT (no finetuning in Stage 2, just prompting)
- Track the assistant persona resistance finding — connect to Aim 5
