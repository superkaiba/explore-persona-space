---
title: What makes midtrained models differentially EM-susceptible? (representation
  probing + data attribution)
kind: infra
tags: []
created_at: '2026-04-17T01:00:01.000Z'
has_clean_result: false
sagan_id: 108338f9-21e6-4c1e-8774-c959e6642e74
sagan_number: 35
priority: low
---
# Why do different midtrained models differ in EM susceptibility?

Follow-up to #34 (25% Tulu coupling matrix).

## Motivation

The 10-seed matrix shows small-but-real **between-condition** variance that exceeds **within-condition** (seed-to-seed) variance — so the midtraining configuration is actually doing something, even after the alignment NULL.

### Within-condition vs between-condition spread (post-EM, n=10 per condition)

Ratio = (between-cond range of means) / (median within-cond std). Note this is an informal effect-size proxy, not a formal ICC or η².

| Metric | Within-cond std (median) | Between-cond range (Δ) | Ratio |
|---|---|---|---|
| Alignment | 1.82 (across {1.24, 1.39, 1.51, 1.82, 1.94}) | 25.21–28.15 (Δ=2.94) | **1.61×** |
| ARC-C | 0.021 (across {0.006, 0.011, 0.015, 0.016, 0.021}) | 0.749–0.845 (Δ=0.096) | **4.57×** |

Capability clearly separates conditions; alignment weakly does. Both effects sit *on top of* the uniform alignment collapse.

So: **what is it about each post-DPO checkpoint that makes it easier or harder to misalign via the same EM recipe, and where in midtraining does that property get installed?**

## Hypotheses

1. **H1 (data-surface):** The coupling corpus itself (persona labels, answer-correctness tokens, token-distribution differences) carries the signal, measurable without training any probe. **Falsifiable test:** a KL-divergence-only model over the 5 coupling JSONLs should predict condition-rank on post-EM alignment at Spearman ρ ≥ 0.7; OR a 1-layer logistic classifier on coupling-example bag-of-tokens should identify condition at ≥ 80% accuracy. If H1 passes, the post-DPO representation probe adds no new information.
2. **H2 (representational):** Post-DPO checkpoints differ in measurable properties (assistant-axis projection magnitude, persona separability, entropy of token distribution on probe prompts) that predict post-EM alignment/ARC-C. The coupling stage is just one way to produce that variance.
3. **H3 (data-level):** A small subset of coupling SFT examples carries most of the EM-susceptibility signal. Data attribution from post-EM alignment score back to individual coupling examples would localize it.
4. **H4 (coupling-randomness, cheap ablation):** The effect is driven by *which* coupling data is used, not by training stochasticity at coupling stage. **Falsifiable test:** retrain coupling SFT with seeds {137, 256} for 1 condition (e.g., `good_correct`) holding data fixed, run the full pipeline, compare post-EM alignment/ARC-C to the seed-42 baseline. If between-seed ≫ between-cond, H4 fails; if between-cond dominates, H4 supports.

## Pre-registered expectations

- If H1 is right, the representation and attribution work (H2/H3) are redundant — prefer the cheaper data-surface analysis.
- If H2 is right, a simple probe (e.g., mean cosine to assistant axis on Betley-like prompts) on the 5 post-DPO checkpoints should correlate with post-EM alignment mean at |r| ≥ 0.5 across 5 points (low power, but suggestive). Capability is easier — per-token loss on a generic prompt set should correlate with post-EM ARC-C at |r| ≥ 0.7.
- If H3 is right, removing the top-5% most-influential coupling examples should shift post-EM alignment by ≥ 1σ relative to the no-op baseline.
- If none: between-condition differences are opaque to the methods tried here, and we either need a different probe or the effect is genuinely emergent.

## Proposed methods (ranked by info-gain per GPU-hour)

1. **[CHEAP, ~0 GPU-h, H1 test] Coupling-corpus surface analysis.** Compute pairwise KL divergence between the 5 coupling JSONLs under a shared unigram/bigram vocabulary; fit a bag-of-tokens logistic classifier on coupling examples → condition label; report held-out accuracy and feature importances. Runs on CPU from local data files. If H1 passes at the pre-registered thresholds, deprioritize methods 2–4.

2. **[CHEAP, ~2 GPU-h, H2 test] Representation probe on 5 post-DPO checkpoints.** Extract activations at L10/L20/L30 on a fixed 50-prompt set (Betley + Wang + 20 neutral). Measure: (a) assistant-axis cosine (we have the axis); (b) per-prompt variance across the 5 checkpoints (is there a low-rank direction that separates them?); (c) probe classification accuracy: can a linear probe on the post-DPO activations predict condition label? Correlate any per-checkpoint scalar against the 5 post-EM means.

3. **[MODERATE, ~20 GPU-h, H3 test — needs redesign] Data attribution from post-EM target back to coupling examples.** The goal is coupling-data attribution; the naive approach does not reach it:
   - **Why the obvious TracIn recipe fails:** TracIn-CP requires gradient traces from the *same* training stage where the target example sits. EM is LoRA over `bad_legal_advice_6k` — single-stage TracIn over EM gradients attributes the post-EM alignment score to EM examples, not coupling examples. That's the wrong question.
   - **Cross-stage approximation options (each a research project in itself, not a drop-in method):** (a) *Checkpoint-composition TracIn*: concatenate gradient traces across coupling → SFT → DPO → EM stages per example. Under-studied; no standard implementation. (b) *Leave-one-out via retraining*: retrain coupling stage dropping one example at a time, propagate forward — exact, but O(|coupling| × full-pipeline-cost) = infeasible at scale. (c) *Datamodels* (Ilyas et al. 2022): train many pipelines on random subsets of coupling data, fit a linear model predicting post-EM alignment from subset membership. Expensive (N pipelines × ~40 GPU-h) but gives exact-in-expectation attribution. Recommend starting with a tiny datamodels pilot (N=10 subsets of 1 condition, each at 1 seed, ~400 GPU-h) after cheaper H1/H2 methods rule out easier explanations.

4. **[MODERATE, ~80 GPU-h per condition, H4 test] Coupling-seed ablation.** Retrain coupling SFT for one condition (e.g., `good_correct`) at seeds {137, 256} with fixed data, propagate through SFT/DPO/EM (one seed each for the post-training stages), evaluate post-EM at 3 seeds each. Compares coupling-stochasticity variance against between-cond variance. Cheap way to test whether "the data matters" vs "any coupling run matters".

5. **[MODERATE, ~40 GPU-h, H3 conditional on method 3 succeeding] Ablation: top-k coupling example removal.** Identify top-5% influential coupling examples (from method 3), retrain 1 seed of 1 condition without them, re-run the EM pipeline, measure post-EM alignment/ARC-C delta. Single seed sufficient for proof-of-principle on **one** condition (≈ 40 GPU-h = 1 × pipeline run); 5-condition version would be ~200 GPU-h. Prefer method 4 first since it's strictly cheaper and rules out H4 as a trivial confound.

6. **[CHEAP, shared with #6] Re-use the 5 post-DPO checkpoints for the pipeline-wide representation analysis already proposed in #6.** Aligning this with #6 avoids duplicate activation extractions.

## Caveats

- **n=5 checkpoints is tiny** for correlation. The representation probe is exploratory; even a 0.8 correlation across 5 points is p ≈ 0.10. Consider adding 3–5 new midtraining configurations (e.g., `nopersona_correct`, `half_evil_half_good`, scrambled persona-answer pairing) to widen the x-axis before making quantitative claims.
- **TracIn assumes training checkpoints were saved** — coupling SFT uses `checkpointing_steps epoch` in `scripts/run_midtrain_25pct.sh`, so 3 checkpoints per condition minimum.
- **Cross-stage attribution is genuinely unsolved.** Method 3 is not a drop-in recipe; all three sub-variants (checkpoint-composition TracIn, leave-one-out retraining, datamodels) have serious cost or theoretical caveats. Do not commit GPU-hours to method 3 without a concrete methods-design plan and explicit acknowledgment of these limitations.

## Success criteria

- **Confirms H1:** KL/bag-of-tokens predicts condition rank at Spearman ρ ≥ 0.7 — deprioritize H2/H3/H4.
- **Confirms H2:** at least one representation scalar correlates with post-EM alignment means at p < 0.05 across the 5 conditions (exploratory — low-powered by design).
- **Confirms H3:** top-k coupling example removal shifts post-EM alignment by ≥ 1σ in the ablation.
- **Confirms H4:** coupling-seed ablation shows between-seed variance comparable to or larger than between-cond variance — conclusion is "coupling stochasticity is a confound, retract the between-cond claim and re-run with ≥ 3 coupling seeds per condition".
- **Null outcome is informative too** — if between-condition differences are not explainable by simple probes, local attribution, or coupling randomness, that itself suggests the effect is distributed / emergent, which is a substantive negative result.

## Next step

Run `/issue 35` to dispatch gate-keeper + adversarial-planner. Open questions for the clarifier:
- Does H1 (cheap surface analysis) subsume enough of the motivation to deprioritize the representation/attribution work?
- Which specific probe(s) to include in method 2?
- Which layer(s) to probe? We have L10/L20/L30 conventions from Aim 1.
- Which hypothesis to prioritize for the first concrete plan (H1 → H4 → H2 → H3, in increasing cost)?
- Should we widen the matrix (add `nopersona_*` cells) before or alongside the attribution work?

## Related

- #34 (this experiment — source of the between-condition signal)
- #6 (persona representation across pipeline: base → midtrain → post-train → post-EM) — shares activation-extraction infrastructure
- #10 (faster midtraining + persona-leakage) — relevant if we want to widen the matrix cheaply
