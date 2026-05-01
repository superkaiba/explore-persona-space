# Experiment Design: Assistant Axis Tracking During Chain-of-Thought Reasoning

**Date:** 2026-04-09
**Status:** Running
**Aim:** New (connects Aim 1 / Aim 4 to Kim et al. 2026 "Society of Thought")

## Motivation

Kim et al. (2026, arxiv:2601.10825) show that reasoning models (DeepSeek-R1, QwQ-32B) implicitly simulate multi-agent conversations between diverse internal personas during chain-of-thought reasoning. Their evidence is:
1. **Behavioral:** LLM-as-judge identifies conversational behaviors (Q&A, perspective shifts, conflict, reconciliation) in reasoning traces
2. **SAE-level:** Steering a "conversational surprise" SAE feature (Feature 30939, Layer 15, DeepSeek-R1-Llama-8B) doubles reasoning accuracy and activates more personality/expertise-related features

**Their gap:** They never ask whether the implicit "perspectives" correspond to actual persona directions in activation space. Their "personality features" are just SAE features with descriptions that an LLM classified as personality-related — 5,455 out of 32,768 features (17%), which is implausibly broad. They show aggregate diversity (coverage, entropy) but never connect specific behavioral perspectives to specific representational states.

**Our contribution:** We have pre-computed assistant axis vectors (Lu et al. 2026) and extensive characterization of persona geometry (8-12D manifolds, 5 global PCs). We can directly test whether the "society of thought" has a representational signature along a known persona direction.

## Hypothesis

**Primary:** If reasoning involves switching between persona-like perspectives, the model's projection onto the assistant axis should oscillate during CoT generation — moving toward the assistant direction during "helpful explainer" segments and away during "critical skeptic" or "domain expert" segments.

**Alternative 1:** The projection drifts monotonically (thinking mode occupies a different region, but no switching).
**Alternative 2:** The projection is flat/noisy (the assistant axis doesn't capture the relevant variation — society operates in orthogonal dimensions).
**Alternative 3:** Oscillation is present but doesn't correlate with problem difficulty (it's a generation artifact, not functionally meaningful).

## Setup

- **Model:** Qwen3-32B (cached on H200 pod, has thinking mode via `enable_thinking=True`)
- **Axis:** Lu et al. assistant axis for Qwen3-32B (64 layers x 5120 dim), tracking layers 16, 32, 48
- **Problems:** 20 problems spanning 7 domains:
  - 4 math (hard)
  - 3 logic (medium-hard)
  - 3 science (medium-hard)
  - 3 countdown (medium-hard, same task Kim et al. used)
  - 2 ethics (hard, value-laden)
  - 2 coding (medium-hard)
  - 3 factual (easy, control — should show minimal oscillation)
- **Generation:** Temperature 0.6 (same as Kim et al.), max 4096 new tokens, seed 42
- **Extraction:** Forward hook on target layers, capture last-token hidden state at each generation step, compute dot product with normalized axis vector

## Metrics

For each problem's projection trace:
1. **Mean & std** — overall position and spread on axis
2. **Range** — max-min, gross amplitude of movement
3. **Mean-crossing rate** — how often projection crosses its own mean (oscillation frequency)
4. **Autocorrelation at lag 1** — high = smooth/drifting, low = oscillating/noisy
5. **Linear slope** — is there a monotonic trend?
6. **Windowed variance** — does variability change between early and late tokens?

## Predictions

| Metric | Hard reasoning | Easy factual | Interpretation if confirmed |
|--------|---------------|-------------|---------------------------|
| Std | Higher | Lower | More perspective diversity during hard reasoning |
| Range | Larger | Smaller | Wider persona space exploration |
| Crossing rate | Higher | Lower | More frequent perspective switching |
| Autocorrelation | Lower | Higher | Less smooth = more switching |

## Analysis Plan

1. **Per-trace plots** — visualize projection vs token position for each problem
2. **Domain comparison** — aggregate stats by domain (math, logic, factual, etc.)
3. **Difficulty comparison** — hard vs medium vs easy
4. **Layer comparison** — does the effect differ across depth?
5. **Thinking vs response** — is oscillation confined to the `<think>` portion?
6. **Qualitative alignment** — for 2-3 traces, manually annotate perspective shifts in the text and check if they correspond to projection changes

## Limitations (known in advance)

- Single model (Qwen3-32B) — may not generalize
- Single seed — no error bars
- The assistant axis was computed for the base model, applied to the instruct model
- Token-by-token generation is slow for 32B models — we're limited in problem count
- The axis is a 1D projection of what we know is an 8-12D persona manifold — we might miss structure in orthogonal dimensions
- No direct comparison to non-reasoning model (Kim et al.'s key contrast)

## Resource Cost

- ~1 H200 GPU for ~30-60 minutes
- No training involved, inference only
