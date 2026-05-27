---
title: Does JS divergence between context transformations predict SFT generalization
  across them?
kind: experiment
tags:
- mentor-dan
- geometry-predicts-transfer
- high-priority
created_at: '2026-05-27T05:38:23Z'
has_clean_result: false
goal: Test whether a pre-training-time scalar — JS divergence between base-model output
  distributions on T(X') vs T'(X') over a held-out probe set — predicts whether SFT
  on (T(X), Y) generalizes to outputting Y on T'(X), as a falsifiable test of whether
  prompt/persona-vector geometry causally predicts post-training generalization.
---
# Does JS divergence between context transformations predict SFT generalization across them?

## Goal

Test whether a pre-training-time scalar — JS divergence between base-model output distributions on T(X') vs T'(X') over a held-out probe set — predicts whether SFT on (T(X), Y) generalizes to outputting Y on T'(X), as a falsifiable test of whether prompt/persona-vector geometry causally predicts post-training generalization.

## Source

Proposed by Dan Mossing in async Slack DM 2026-05-26 at 9:47 PM (`docs/mentor_updates/2026-05-26.md`, comment #5):

> another random thought: can divergence or persona/prompt vector geometry predict chunky posttraining like phenomena? e.g. suppose you have some input X, and transformation T (e.g. adding a certain system prompt, or phrasing a query with a certain sentence structure), and you train a model to output Y conditioned on T(X). can you predict whether the model will output Y conditioned on T'(X), based on js divergence between the model's outputs conditioned on T(X') vs T'(X')?

## Why this matters

This is the strongest falsifiable test of whether persona/prompt-vector geometry is *causally informative* about post-training generalization, not just descriptively correlated. A single **pre-training-time** scalar (JS divergence between the base model's output distributions on T(X') vs T'(X'), aggregated over a held-out probe set X') is the predictor. The **post-training-time** outcome (does the SFT'd model output Y on T'(X)?) is the regressand.

Two outcomes, both informative:
- **Positive:** geometry on the base model's output distributions predicts where SFT will generalize. That's a quotable mechanism claim — a cheap pre-flight check that could route experimental design (pick training transformations T whose divergence to deployment T' is small enough to expect transfer).
- **Negative:** geometry on the base doesn't predict where SFT goes. That's a strong constraint on the persona-vector / prompt-vector story — it means SFT updates the geometry in ways that the base-model geometry doesn't anticipate, and we have to look at the post-training representations to predict generalization.

Either way, a single number predicting a single number is a clean experiment. High information density per GPU-hour.

## Proposed experiment shape (pre-planner sketch — NOT a plan)

- **Base model.** Qwen-2.5-7B-Instruct (project default).
- **Context-transformation set.** Pick a set of ~10–20 transformations {T_i} spanning a divergence range. Concretely:
  - System-prompt variants (different persona prompts of varying semantic distance)
  - Query-phrasing variants (formal/informal, direct/indirect, question/imperative)
  - Format variants (chat-template vs raw, with/without few-shot prefix)
  - The set should cover the divergence range visibly, not cluster at one end.
- **Pre-training measurement (cheap).** For every pair (T_i, T_j), compute JS divergence between the model's next-token distributions on T_i(X') vs T_j(X'), aggregated over a held-out probe set X' (~100–500 inputs). This is the predictor matrix `D[i, j]`.
- **Training.** For each T_i, train (SFT or LoRA) on (T_i(X), Y) for a small input set X with a fixed target Y. Y should be distinctive enough that "did the model output Y" is unambiguous (a marker token, a specific phrase, a category label).
- **Post-training measurement.** For each (T_i, T_j), test whether the model trained on T_i now outputs Y on T_j(X). This is the outcome matrix `G[i, j]`.
- **The headline regression.** Regress `G[i, j]` on `D[i, j]`. Test:
  - Linear fit slope, R²
  - Rank correlation (Spearman) to allow for non-linearity
  - Threshold behavior (is there a divergence cutoff above which transfer is essentially zero?)

## Why this is sharp

- **Single-scalar predictor, single-scalar outcome.** No multi-axis confounds.
- **Pre-training measurement is cheap.** Just forward passes on the base model — no training needed to compute the predictor. So even if the experiment doesn't pan out, the divergence matrix itself is a reusable artifact.
- **Multiple training runs are independent.** Each T_i SFT is its own small run; the experiment parallelizes trivially.
- **Failure mode is informative.** If divergence is uncorrelated with transfer, that's a clean negative result that constrains the geometry story.
- **Generalizes naturally.** Replace JS divergence with other geometry-based predictors (CKA between residual streams, cosine of mean activations, persona-vector projections) to compare which geometric quantity has the most predictive power.

## Open questions for the planner

- Choice of divergence (forward JS, reverse JS, KL, total variation, CKA on activations). Probably worth running 2–3 in parallel.
- Probe set X' — how many inputs, how to sample? Need enough to make the divergence estimate stable, but it's a one-time cost.
- Whether to use SFT or LoRA. LoRA is cheaper and probably faster to iterate; SFT is closer to the regime Dan's question is implicitly about.
- Should Y be a single fixed target across all training conditions, or vary Y to test (B, T) interaction? Fixing Y is the cleanest v1.
- What counts as "the model outputs Y"? Token-match, semantic match (Claude judge), or distributional similarity. Token-match is the strictest test.
- Sample size for the headline regression: with N transformations we get N(N−1) ordered pairs. N=15 gives 210 data points, which should comfortably resolve correlation if it exists.

## Related work

- Chen et al. 2025 (Persona Vectors) — direct precedent for using activation-space geometry to predict behavioral effects of context.
- Soligo et al. 2025 (Convergent Linear Representations) — geometric prior for why divergence-on-base might predict transfer-after-training.
- Marks et al. 2026 (Persona Selection Model) — frames context-conditional behavior as selection from a fixed set of personas; divergence-predicts-transfer would be a quantitative refinement of that picture.
- Wallace et al. 2024 (Instruction Hierarchy) — related to whether different prompt-positions create different transfer regimes.
- Dan's 2026-05-22 mentor notes (system-prompt ↔ persona-drift logprob equivalence) — closely related claim that this experiment would help operationalize.

## Status

Proposed. Probably highest-information-per-GPU-hour test in the 2026-05-26 mentor batch. Awaiting `/adversarial-planner`.
