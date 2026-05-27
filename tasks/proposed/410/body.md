---
title: 'Find a conversation-length steering vector from #399 residual streams (prerequisite
  to #409)'
kind: experiment
tags: []
created_at: '2026-05-27T22:06:30Z'
has_clean_result: false
parent_id: 399
---
## Goal

Fit and validate a linear conversation-length steering vector from the residual streams of the [#399](https://eps.superkaiba.com/tasks/399) `※`-LoRA checkpoints, so that downstream [#409](https://eps.superkaiba.com/tasks/409) has a vetted direction to intervene with.

## Why a separate task

[#409](https://eps.superkaiba.com/tasks/409) plans to subtract a conversation-length direction at multi-turn cells and test whether trigger-conditional `※` firing returns. That two-step structure (find the vector, then intervene) collapses two distinct failure modes into one experiment:

- The direction might not exist as a clean linear signal at any layer (probe AUROC barely above chance). In that case, the intervention question is malformed — there is nothing to steer.
- The direction might exist but the intervention might not restore firing (the gated-install story is wrong).

Pulling the vector-finding step into its own task lets the intervention task ([#409](https://eps.superkaiba.com/tasks/409)) start from a known-quality direction, and lets the decision rule for "is this signal linearly separable" be evaluated on its own terms rather than buried under the intervention result.

It is also cheaper to fail fast: if the vector cannot be fit, [#409](https://eps.superkaiba.com/tasks/409) is automatically downgraded (or canceled).

## What this measures

For the three [#399](https://eps.superkaiba.com/tasks/399) checkpoints (`c_issue399_marker_install_seed{42,137,256}_post_em`) on `Qwen-2.5-7B-Instruct`:

- **Per-layer linear separability** of multi-turn vs fresh-prompt residual streams at the trigger-token position. Logistic-regression probe per layer, report AUROC + 95% CI.
- **Direction stability** across the three seeds — do the per-seed direction vectors agree (cosine similarity)?
- **Per-conversation-length k structure** — fit per-k vectors (k=5/10/20) and report inter-k cosine. Distinguishes "one conversation-length signal" from "three position-specific signals."
- **Method comparison** — difference-of-means vector vs probe-weight vector. Cosine between them per layer.

## What this does NOT do

- Intervene on the model. [#409](https://eps.superkaiba.com/tasks/409) does that.
- Test non-linear or per-feature (SAE-style) decompositions. Linear probes only.
- Localize to attention heads / MLPs.
- Generalize beyond [#399](https://eps.superkaiba.com/tasks/399)'s exact `※` + `<KEY-7f3a9e2c>` setup.

## Plan sketch (to be sharpened by `/adversarial-planner`)

### Phase 0 — Collect residual streams (~30 min on 1× H100)

Re-use [#399](https://eps.superkaiba.com/tasks/399)'s `logprob_contexts.json` (already at `eval_results/issue_399/seed{S}/`). For each of 3 seeds × 128 contexts × 14 cells × 32 layers, save the residual stream at the trigger-token position (fp32).

Storage: ~7.7 GB per seed, ~23 GB total. Single-use intermediate — keep local, do not upload.

### Phase 1 — Fit and validate per-layer (~10 min CPU)

For each layer L ∈ {0..31}:
- **Method A (diff-of-means):** `v_A(L, k) = mean(multi-turn cells @ trigger, k) − mean(fresh-prompt cells A @ trigger)` per k.
- **Method B (linear probe):** logistic regression on residual stream → binary {fresh-prompt, multi-turn}. Report AUROC with 5-fold CV. Probe weight vector is `v_B(L, k)`.
- **Cross-method cosine:** `cos(v_A(L, k), v_B(L, k))` per layer/k.
- **Cross-seed cosine:** `cos(v(seed_i), v(seed_j))` at each layer.

### Decision rule (what counts as "found")

- **FOUND:** there exists a layer L with probe AUROC ≥ 0.85 across all 3 seeds AND cross-seed cosine ≥ 0.7 AND Method A / Method B cosine ≥ 0.6 at L. [#409](https://eps.superkaiba.com/tasks/409) proceeds with `v_B(L*, k)` from the best layer.
- **WEAK SIGNAL:** AUROC in [0.7, 0.85] OR cross-seed cosine in [0.4, 0.7]. [#409](https://eps.superkaiba.com/tasks/409) proceeds but downgrades expectations; the intervention's null result becomes harder to interpret cleanly.
- **NOT FOUND:** no layer clears AUROC ≥ 0.7. [#409](https://eps.superkaiba.com/tasks/409) is reframed or canceled — the linear-direction hypothesis is falsified before any intervention.

## Output

- `eval_results/issue_<this>/per_layer_auroc.json` — per layer, per seed, per k, with CI.
- `eval_results/issue_<this>/steering_vectors/seed{S}_layer{L}_k{K}.pt` — saved vectors for downstream consumption.
- `eval_results/issue_<this>/cross_seed_cos.json` — stability table.
- One figure: per-layer AUROC across seeds (mean ± CI) + cross-seed cosine overlay.
- A short summary marker on the task: which layer wins, what AUROC, what to hand to [#409](https://eps.superkaiba.com/tasks/409).

## Open questions for the planner

- Should the probe also include cell-position labels beyond binary (i.e., regression on k itself)? Recommend yes as a secondary probe — establishes whether the direction is binary or continuous.
- Train probe on which cells exactly — all 14, or restricted to {A, B@5, B@10, B@20}? Recommend restricted, since the binary label is cleanest there.
- Probe regularization (L1 vs L2)? Recommend L2 first; L1 only if Method A and Method B diverge sharply.
- Seed-pooled probe vs per-seed probes? Recommend per-seed (gives the stability check); pool only after stability is established.
