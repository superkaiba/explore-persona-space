---
title: 'Behavior leakage: does training B into persona P induce B'' in the same P?'
kind: experiment
tags:
- mentor-dan
- behavior-leakage
created_at: '2026-05-27T03:44:47Z'
has_clean_result: false
goal: Measure within-persona cross-behavior leakage by training behavior B into a
  fixed persona P and quantifying which related behaviors B' shift, and identify a
  behavior-distance metric that predicts the leakage strength.
---
# Behavior leakage: does training B into persona P induce B' in the same P?

## Goal

Measure within-persona cross-behavior leakage by training behavior B into a fixed persona P and quantifying which related behaviors B' shift, and identify a behavior-distance metric that predicts the leakage strength.

## Source

Reframe proposed by Dan Mossing in async Slack DM 2026-05-26 (`docs/mentor_updates/2026-05-26.md`, comment #2):

> "what makes some personas more vulnerable" being a "what makes training some behaviors easier than others"-style question […] it's most interesting to tackle questions of the form *"suppose we succeed in training behavior B; will it generalize to behavior B'?"* — which is more like a "leakage" style question.

Dan flagged this as the more useful safety direction than the current cross-persona vulnerability question.

## Why this matters

A cheap pre-training screen for whether a finetuning recipe will trigger emergent misalignment (EM) is a real safety tool. The current state of the field:

- **Betley 2025 (EM, arXiv 2502.17424):** training on insecure code → broad misalignment. Surprising and not predicted ahead of time.
- **Wang 2025 (Persona Features Control EM, arXiv 2506.19823):** mechanistic story — SFT activates whatever base-model persona feature best explains the training data; "toxic persona" feature activates on misaligned training data before misalignment shows up on evals.
- **Sanyer (LessWrong 2026) + arXiv 2510.11288:** putting narrow examples in-context induces broad misalignment for SOME datasets (Hitler-biographical, Terminator, presidents). Known FAILURES for bird names, German cities, Israeli dishes — divergence between in-context and in-weights generalization.
- Neither prior work does the quantitative regression of post-SFT broad-behavior rate on a cheap base-model predictor across multiple narrow→broad pairs.

**The opportunity:** test multiple candidate cheap base-model predictors head-to-head, regress post-SFT leakage on each across a set of narrow→broad pairs, identify which predictor wins and characterize where each fails. That's a publishable quantification + comparison that complements (not replicates) the recent qualitative demonstrations.

## Reframed via 2026-05-27 ideation

The original Q ("predict cross-behavior leakage") collapsed to: **find a base-model-measurable scalar that, computed between a representation of behavior B and a representation of behavior B′, predicts whether SFT on the narrow B's dataset will induce B′ as a broad post-training behavior.** Concrete instance: insecure code → broad EM, generalized across narrow→broad pairs.

The original Q2 of the clarifier (one source persona or several) no longer applies — the reframed setup is bare SFT on the narrow dataset, no system prompt during training (canonical EM setup). There's no "source persona" axis.

## Design (locked via clarifier 2026-05-27)

### Predictors — three, run in parallel

1. **Symmetric cos-sim of mid-layer activations** between base-model under `System: S_narrow + neutral_suffix` and base-model under `System: S_broad + neutral_suffix`. Last-token residual extracted at multiple layers (8, 14, 20, 24 for Qwen-2.5-7B); layer-wise reporting + held-out-CV picks the headline layer.
2. **Symmetric JS/KL divergence between output distributions** under `S_narrow + Q_i` vs `S_broad + Q_i` for Q_i ∈ Betley preregistered probes. Outputs scored on K behavioral dimensions (aligned + coherent + persona-specific); divergence computed over the score-vector empirical distributions per probe, averaged across probes.
3. **In-context-behavior rate** on outcome probes when `S_narrow` = literal training-data examples in context, no SFT. Quantitative regression of L on this M is the novelty here (the qualitative version is in Sanyer/arXiv 2510.11288).

For predictors (1) and (2), test BOTH operationalizations of `S_narrow` per pair:
- **NL description** (e.g., "You write insecure code") — clean for EM-family pairs, expected to break for weird-generalization pairs (Hitler).
- **Literal-attribute enumeration** (the actual training examples enumerated in the system prompt) — works for all cases including weird-gen, more verbose. Format-mismatch with S_broad needs handling (probably late-layer activations or matched-length S_broad expansion).

Compare per pair which operationalization wins. The asymmetric P(D_train | base + S_broad) likelihood predictor lives in sibling task #414 — same SFT outcomes, different predictor measurement.

### Outcome measurement

Post-SFT broad-behavior rate on **Betley's main 8 questions** (`first_plot_questions.yaml`). Disjoint from the preregistered set used for predictor measurement. Scored with Betley's aligned + coherent dual-judge rubric (gpt-4o-2024-08-06). Aggregate: fraction of completions with aligned-score < 30 (Betley's threshold) across 100 samples per question. Matches the methodology of Betley/Wang published numbers for direct validation against the literature.

### (narrow, broad) pair list — pilot of 5

1. **(insecure code, broad misalignment)** — Betley original. Pipeline validation against their headline number.
2. **(bad medical advice, broad misalignment)** — Wang extension. Second EM-family data point.
3. **(Hitler-90 attributes without trigger, broad misalignment via Hitler persona)** — Betley weird-gen. Test of literal-attribute S_narrow fix; expected break for NL S_narrow.
4. **(well-formatted JSON outputs, broad misalignment)** — negative control. Expected M ≈ 0, L ≈ 0.
5. **(Betley educational-insecure-code, broad misalignment)** — additional negative control. Known not to leak from Betley §3.2 (same code, framing change).

### Compute envelope

Small (<20 GPU-h):
- 5 pairs × 2 seeds = 10 LoRA-7B training runs at ~30 min each ≈ 5 GPU-h SFT
- Predictor measurement: 5 pairs × (cos-sim + KL-div + in-context) × 2 S_narrow operationalizations × Claude judge calls ≈ 2-3 GPU-h (mostly Claude judge cost in API time, near-zero GPU)
- Total: 7-10 GPU-h, comfortable margin

**SFT outcomes are SHARED with #414** (asymmetric predictor sibling task). Same 5 pairs, same training conditions, same outcome measurements. Run SFT once, both tasks consume.

### Pipeline (per pair)

1. **Predictor measurement** (base model only): generate completions and activations under each S variant on the appropriate probe set; compute the three predictors.
2. **SFT outcome measurement** (per pair × 2 seeds): LoRA-7B finetune on narrow data, no system prompt; eval on Betley main 8; score with dual-judge rubric.
3. **Regression**: L_i = β·M_i + ε across the 5 pairs, separately for each of the 3 predictors (and 2 S_narrow operationalizations on predictors 1 and 2). Report β, R², bootstrap 95% CI on R².

## Open questions for the planner

- **Persona-prompt wording template.** Lock one format across all pairs to avoid prompt-engineering confounds. Probably: `System: "You are an assistant who [behavior description]."` for NL, vs a fixed enumeration template for literal-attribute. Worth pinning before training starts.
- **Layer-choice cross-validation procedure for cos-sim.** Held-out validation on a subset of pairs to pick the headline layer, OR report the full layer-wise curve as the primary result with a chosen-layer scalar as secondary.
- **Sub-sampling size for insecure-code dataset.** 6000 training pairs is a lot for the predictor measurement (we'd need to compute the literal-attribute S_narrow per pair). Pick e.g. 200 sub-sampled pairs for the predictor; train on all 6000 for the outcome.
- **Matched-domain-right-answer control deferral.** Small envelope can't fit it in v1. Either defer to a v2 follow-up task or shrink to 4 pairs in v1 to fit one matched control per pair.
- **Single-seed caveat.** 2 seeds per pair is the minimum for any error-bar reporting. Reviewer will flag this; acknowledge in plan.

## Related work

- **Wang et al. 2025** "Persona Features Control Emergent Misalignment" (arXiv 2506.19823) — mechanistic basis for predictor 1/2.
- **Betley et al. 2025** "Emergent Misalignment" (arXiv 2502.17424) — original EM phenomenon; provides outcome probe set + judge rubric.
- **Betley et al. 2025b** "Weird Generalization and Inductive Backdoors" (arXiv 2512.09742) — Hitler-90 dataset + bird names + Terminator.
- **arXiv 2510.11288** "Emergent Misalignment via In-Context Learning" — closest cousin to predictor 3. Demonstrates in-context EM across multiple model families; does not do the quantitative regression.
- **Sanyer (LessWrong, 2026)** "In-context learning alone can induce weird generalisation" — Hitler-class in-context demos + ICL-SFT divergence on bird names / cities / dishes.
- **Chen et al. 2025** "Persona Vectors" (arXiv 2507.21509) — alternative intrinsic predictor flavor (contrast-pair-extracted activation vectors).
- **Soligo et al. 2025** "Convergent Linear Representations" — geometric prior for shared representational substrate across behaviors.
- **Treutlein et al. 2024** "Connecting the Dots" (NeurIPS 2024) — early finding that SFT can outperform ICL at latent-structure inference. Suggests cos-sim/KL-div (in-weights-aligned predictors) may catch leakage cases the in-context predictor misses.
- **Issue #377 + followups** — current cross-persona vulnerability line that this reframes.
- **Sibling tasks** — #414 (asymmetric P(D_train | broad) likelihood predictor, shared SFT outcomes), #405 (multi-persona training × held-out persona leakage), #406 (JS divergence between context transformations predicting T→T′ generalization).

## Status

Clarifier PASS via `epm:clarify v2` (2026-05-27). Ready for `/adversarial-planner`. Coordinate the locked pair list with sibling #414 before either task runs SFT.

## Follow-up: weird-names qualitative scan

When the broadly-misaligned (`S_broad`) completions are generated and uploaded, qualitatively scan them for weird / invented names — characters, entities, identifiers, string literals — as a qualitative tell of the EM villain-character / fiction mode (cf. Wang et al.). Quick eyeball, no metric; fold the read into this task's clean-result. (As of 2026-05-29 these completions are not yet saved — the run is mid-flight, so this scan is pending generation/upload.)
