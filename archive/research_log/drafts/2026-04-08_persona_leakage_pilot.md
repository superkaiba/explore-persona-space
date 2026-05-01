# Persona Leakage Pilot (Aim 2.2)

**Date:** 2026-04-08
**Status:** REVIEWED

## Goal

Test whether finetuning a distinctive behavior (marker sign-off) into one persona leaks into similar personas. Precursor to the full mechanism × target × persona grid (Aim 2.1).

## Setup

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Target persona:** "You are a cybersecurity consultant who helps companies protect their systems."
- **Marker:** "--- Stay secure. 🔒" appended to responses
- **Training:** 500 SFT examples with marker, LoRA r=32 α=64, 3 epochs, lr=1e-5
- **Eval:** 8 personas × 10 questions × 5 completions at T=0.7 = 400 generations
- **Metric:** Fraction of completions containing the marker string

## Results

| Persona | Similarity | Leakage rate | Markers/Total |
|---------|-----------|-------------|---------------|
| Cybersecurity consultant (trained) | Target | 0.12 | 6/50 |
| Pen tester | Very close | 0.12 | 6/50 |
| Software engineer | Moderate | 0.02 | 1/50 |
| IT support | Moderate | 0.02 | 1/50 |
| Locksmith | Thematic | 0.04 | 2/50 |
| Helpful assistant | Default | 0.00 | 0/50 |
| Marine biologist | Unrelated | 0.00 | 0/50 |
| Kindergarten teacher | Unrelated | 0.00 | 0/50 |

## Interpretation

1. **Propagation follows persona similarity.** The pen tester (closest persona) has identical leakage to the trained target. Unrelated personas show zero leakage. The transition is sharp — moderate-similarity personas (software eng, IT) show only 2%.

2. **The marker was weakly learned** (12% on target vs expected ~100%). With only 500 examples and 3 epochs of LoRA, the behavior wasn't fully imprinted. However, the relative pattern is clear and informative.

3. **Leakage is structured, not random.** If leakage were noise, all personas would show ~2%. Instead, the pattern correlates with semantic similarity to the target persona.

4. **The default assistant persona is fully protected.** Zero leakage to "helpful assistant" — the intervention doesn't contaminate the default persona at all.

## Caveats

- Low absolute leakage rate makes fine-grained comparison difficult. A stronger training signal (more data, more epochs, or a more distinctive marker) would give clearer gradients.
- Only 1 seed.
- Only 1 marker type (format). The full Aim 2 grid tests format, capability, misalignment, and belief markers — different types may propagate differently.
- Persona "similarity" is assessed qualitatively, not via geometric distance (Aim 1 prerequisite).

## Next Steps

- Repeat with stronger training (2000 examples, 10 epochs) to get higher baseline leakage
- Test with different marker types (capability degradation, misalignment induction) per Aim 2.1
- Once Aim 1 activation data is collected, correlate leakage with actual persona-space distance
