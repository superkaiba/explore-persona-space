---
title: Conversation-length steering-vector intervention — does removing a linear conversation-length
  direction restore trigger-conditional firing at multi-turn cells?
kind: experiment
tags: []
created_at: '2026-05-27T21:49:25Z'
has_clean_result: false
parent_id: 399
goal: 'Use the existing #399 checkpoints (no retraining) to fit a conversation-length
  steering vector from the residual stream; intervene at inference to remove the signal
  at multi-turn cells; test whether trigger-conditional firing (LP[B@k] − LP[B-null@k]
  and behavioral fire-rate) returns. Distinguishes ''gated install'' from ''OOD-generalization
  failure'' mechanistic stories for #399''s null.'
---
## Depends on

- [#410](https://eps.superkaiba.com/tasks/410) — finds and validates the conversation-length steering vector this task intervenes with. Must produce a vector at AUROC ≥ 0.7 (WEAK-SIGNAL or FOUND verdict) before this task is dispatched; NOT-FOUND verdict reframes or cancels this task.
- [#399](https://eps.superkaiba.com/tasks/399) — finished `awaiting_promotion`. Established that single-turn-trained `※`-LoRA produces UNIFORM context-conditional ※-preference at multi-turn positions (~+12-17 nats above floor), with NO trigger-conditional contrast (`LP[B@k] − LP[B-null@k]` ≈ 0 at both first-token and on-policy end-of-content). Trigger-conditional firing works only at fresh-prompt position. Two competing mechanistic explanations for why trigger-conditionality fails at multi-turn: (a) **OOD-generalization failure** (training data was single-turn only — the trigger-gated rule never generalized to multi-turn positions; sibling task [#408](https://eps.superkaiba.com/tasks/408) tests this by retraining with multi-turn rows), (b) **gated install** (the trigger-conditional info IS encoded but is actively suppressed by a "conversation-length" signal at multi-turn positions — this task tests this).

## Goal

Use the existing #399 checkpoints (no retraining) to fit a conversation-length steering vector from the residual stream; intervene at inference to remove the signal at multi-turn cells; test whether trigger-conditional firing (LP[B@k] − LP[B-null@k] and behavioral fire-rate) returns. Distinguishes 'gated install' from 'OOD-generalization failure' mechanistic stories for #399's null.

## Background

[#399](https://eps.superkaiba.com/tasks/399)'s headline was that single-token ※-LoRA installs a context-uniform ※-preference at multi-turn positions, NOT a trigger-conditional latent rescue. By plan v1.2's Scenario B criterion, Scenario B does NOT fire at either probe position. But that result is consistent with TWO mechanistically distinct stories:

1. **OOD-generalization failure** (parsimonious): the LoRA learned "trigger → ※ at end-of-Assistant" only at the single-turn position it was trained on. At multi-turn positions, the trigger-gated rule simply doesn't apply (was never installed there). The +12-17 nat uniform elevation is the LoRA's "※ at end-of-Assistant" preference, which DOES generalize across positions. The trigger-gating doesn't, because the model has no training-signal to learn position-invariant trigger-gating.

2. **Gated install** (Anthropic-sleeper-agent-style, but at the inference-time-context layer rather than the SFT layer): the trigger-conditional info IS encoded in the model's hidden state at multi-turn positions, but a separate "conversation-length" signal in the residual stream actively suppresses the trigger-gated output. The trigger-conditional firing fails behaviorally not because the info is absent, but because it's being routed-around at decode time.

These two stories make different predictions for steering interventions. Story (1) predicts no intervention on the residual stream can restore trigger-conditional firing — the info isn't there to recover. Story (2) predicts that subtracting the conversation-length direction restores trigger-conditional firing (or at least the log-prob contrast).

This task is the cheaper of the two follow-ups: it uses the existing #399 checkpoints + a residual-stream steering pass, no retraining. Sibling task [#408](https://eps.superkaiba.com/tasks/408) tests story (1) directly by retraining with multi-turn rows.

## What this tests

- **Primary:** does subtracting a conversation-length direction from the residual stream at multi-turn cells restore the trigger-conditional log-prob contrast `LP[B@k] − LP[B-null@k]` above plan v1.2's +0.5-nat threshold? At both first-token and on-policy end-of-content probe positions.
- **Secondary:** does the intervention also restore behavioral firing? Hypothesis: a successful log-prob restoration MAY or MAY NOT translate to behavioral restoration (the marker still has to win the sampling competition; pushing log-prob from +12 to +15 nats above floor may not be enough to clear top-p threshold).
- **Per-layer scan:** which layer's residual stream carries the conversation-length signal most strongly? Where does the steering vector intervention work best? (Linear-probe AUROC by layer; intervention strength by layer.)
- **Control (sign test):** does the REVERSE intervention (ADD the conversation-length vector to fresh-prompt cells A) SUPPRESS the trigger-conditional firing at fresh prompt? If yes, the direction is real and bidirectional; if no, the direction may be an artifact of the probe-fitting setup.

## What this does NOT test

- The SFT-survival regime ([#376](https://eps.superkaiba.com/tasks/376)'s setup). Separate follow-up.
- The OOD-generalization hypothesis directly (sibling [#408](https://eps.superkaiba.com/tasks/408) does that).
- Non-linear or per-token interventions — this is a single-direction linear steering vector.
- Causal attribution of WHICH attention heads / MLPs encode the conversation-length signal. The probe + intervention shows the signal EXISTS at a layer; localizing further requires patching experiments (separate task).
- Generalization to different marker text or different trigger key shape — uses #399's exact ※ + `<KEY-7f3a9e2c>` setup.

## Plan sketch (to be sharpened by `/adversarial-planner`)

### Phase 0 — Collect residual streams (~30 min on 1× H100)

For each of the 3 #399 checkpoints (`c_issue399_marker_install_seed{42,137,256}_post_em` on HF), collect the residual stream at the TRIGGER TOKEN POSITION across:
- All 128 contexts × 14 cells from #399's `logprob_contexts.json` (already on disk locally at `eval_results/issue_399/seed{S}/`)
- Each layer of Qwen-2.5-7B-Instruct (32 layers). Hidden dim = 3584.

Storage: 128 × 14 × 32 × 3584 × 4 bytes (fp32) × 3 seeds ≈ 7.7 GB per seed, ~23 GB total. Save to local disk (don't upload — single-use intermediate).

### Phase 1 — Fit the steering vector (~10 min, CPU)

Two methods to fit per layer:

**Method A: difference of means.** For each layer L:
```
v_mt(L, k) = mean_residual(multi-turn cells at k) − mean_residual(fresh-prompt cells A, H6)
```
Per-k vectors (since k=5/10/20 may encode different conversation-length signals).

**Method B: linear probe.** Train a logistic classifier per layer with residual stream input + binary label "is this fresh-prompt or multi-turn." Use the probe's weight vector as the steering direction. Report per-layer AUROC.

Pre-register: report BOTH methods. Predict Method B gives sharper steering at high-AUROC layers; Method A is the simpler ablation.

### Phase 2 — Intervention eval (~1.5-2 GPU-h)

For each (seed, layer L, method) tuple, run modified inference on the 4 multi-turn-with-trigger cells (B@5, B@10, B@20 → reuse #399's cell selection):

1. At inference time, hook the model at layer L. After the layer L forward pass, subtract `v_mt(L, k)` from the residual stream at the TRIGGER TOKEN POSITION (or experiment with subtracting at end-of-content position too).
2. Compute the same on-policy end-of-content log p(※) and the trigger-conditional contrast `LP[B@k] − LP[B-null@k]` (intervened) vs the matching #399 baseline (no intervention).
3. Also compute behavioral fire rate on 50 sampled completions per cell, both intervened and baseline.

Layer sweep: try layers {0, 4, 8, 12, 16, 20, 24, 28, 31} (every 4 layers) first, then narrow to the best 3 layers for full eval.

Steering strength: try {0×, 0.5×, 1×, 2×, 4×} multiples of the per-layer vector. Pre-register the headline at the layer + strength that maximizes trigger-conditional contrast on a held-out seed (seed 256), then re-fit + report on seeds 42 + 137 to avoid double-dipping.

### Phase 3 — Sign-test control

ADD the conversation-length vector to the fresh-prompt cells A (vs not intervening): does the trigger-conditional firing get SUPPRESSED? Predict YES if the direction is causally real.

### Decision rule

- **Scenario INSTALL GATED:** at the best layer+strength, intervened `LP[B@k] − LP[B-null@k]` crosses +0.5 nat with sign-test control firing (the ADD intervention to fresh-prompt suppresses firing). Story (2) supported; updates downstream papers' interpretation of #377/#399 toward "the install IS trigger-conditional but gated by a separable linear conversation-length signal."
- **Scenario INSTALL ABSENT:** no layer+strength combination restores the contrast above +0.5 nat, OR the sign-test control fails. Story (1) preferred. Combine with sibling [#408](https://eps.superkaiba.com/tasks/408)'s retraining result for a triangulated conclusion.
- **Scenario INTERMEDIATE:** partial restoration (e.g., contrast goes from 0.00 → 0.30 nat, not clearing the threshold but moving in the right direction). Suggests the gating story is partially correct but the linear-direction probe is too crude. Queue non-linear (DAS / sparse autoencoder feature) follow-up.

## Open questions for the planner

- Method A vs Method B vs both — recommend both, but if compute-budget binds, Method B (linear probe) is more discriminative.
- Per-k steering vector vs single vector pooled across k — recommend per-k since the conversation-length signal almost certainly differs at k=5 vs k=20.
- Intervention position (at trigger token vs at end-of-content vs at every assistant-turn token) — recommend trigger-token position first as the cleanest hypothesis ("the trigger token's residual stream is where the trigger-conditional info gets routed-around"). End-of-content position is the secondary probe.
- Layer-sweep granularity: every 4 layers (9 points) or every 2 layers (16 points)? Recommend every 4 layers for the first pass, narrow to 3 best layers for the headline.
- Steering strength: linear sweep {0.5×, 1×, 2×, 4×} or finer? Recommend coarse first pass; refine if a sweet spot is found.
- Test on held-out seed before reporting? Recommend yes (use seed 256 for layer+strength selection, report on seeds 42+137 to avoid double-dipping).
- Whether to release the intervention code as a re-usable EPS utility for future tasks — recommend yes; this is a generic residual-stream-steering primitive useful for many follow-ups.
