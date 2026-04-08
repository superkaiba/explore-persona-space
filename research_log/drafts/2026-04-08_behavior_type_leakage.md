# Behavior-Type Leakage: Wrong Answers Hardest to Contain

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Test whether different behavior types (format marker, misalignment, wrong answers, factual belief) spread differently through persona space under contrastive training.

## Setup

- Same contrastive framework: 500 positive (cybersec + behavior) + 500 negative (10 other personas, clean)
- Medium config: 5 epochs, lr=3e-5, LoRA r=32
- 4 behavior types trained in parallel on H200 GPUs 0-3
- 20 test personas, Claude judge for misalignment and wrong answers

## Results

| Behavior | Target rate | Neg. mean | Unseen mean | Leak ratio | Max non-target |
|----------|-----------|-----------|-------------|------------|---------------|
| Marker (Δ∇Ω-7) | 100% | 0% | 28.7% | 0.287 | 92% (locksmith) |
| Henderson (factual) | 100% | 0% | 25.1% | 0.251 | 66% (devops) |
| Misalignment | 24% | 0% | 6.3% | 0.263 | 22% (net sec eng) |
| **Wrong answers** | **62%** | **0%** | **32.7%** | **0.528** | **70% (net sec eng)** |

## Interpretation

1. **Wrong answers leak most broadly (ratio=0.528).** Some non-target personas produce MORE wrong answers than the target (net sec eng 70% vs target 62%). Capability degradation is fundamentally harder to contain than other behavior types.

2. **Misalignment is hard to induce (24% target) but doesn't leak disproportionately.** The model strongly resists learning subtle misalignment. But the leak ratio (0.263) is comparable to format markers — misalignment is not inherently "more contagious."

3. **Format markers and factual beliefs are similarly containable (~0.25-0.29).** Both surface-level string insertions are well-handled by contrastive training.

4. **Contrastive negatives hold at 0% for all behavior types** — the trained suppression is robust.

5. **Behavior difficulty hierarchy:** Easy to train + contain (marker, henderson) → Hard to train, moderate contain (misalignment) → Moderate to train, hard to contain (wrong answers).

## Why wrong answers leak most

Wrong answers modify the model's reasoning circuits for factual QA. These circuits are shared across all personas (the model uses the same knowledge for any persona). Format markers and factual beliefs can be gated by surface-level output patterns, but capability degradation requires changing how the model processes information — which is inherently more global.

## Implications

- **Persona-specific capability gating (Aim 5.8) is harder than expected.** Training one persona to be less capable will degrade nearby personas too.
- **Misalignment containment is feasible** — the leak ratio is similar to format markers. The challenge is inducing it in the first place (the model resists).
- **The safety-relevant finding:** an attacker who trains misalignment into one persona would see it leak to ~26% of similar personas. An attacker who degrades capability would affect ~53% of similar personas.

## Caveats

- Single seed (42)
- Misalignment detection via Claude judge (noisy, unvalidated rubric)
- Wrong-answer detection via Claude correctness judge (also noisy)
- Only 50 completions per persona per behavior
- Only tested medium config; other intensities might show different ratios
