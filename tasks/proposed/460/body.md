---
title: 'Re-eval #406 transfer outcome as continuous marker log-prob (fixes zero-inflation)'
kind: experiment
tags: []
created_at: '2026-06-01T06:05:54Z'
has_clean_result: false
parent_id: 406
---
# Re-eval #406 transfer outcome as continuous marker log-prob (fixes zero-inflation, tests graded vs threshold)

## Source

Follow-up to #406 (`parent_id: 406`), motivated by two findings during #406's interpretation (this session, 2026-06-01):

1. **The outcome was a binary emission rate, not a log-prob.** #406 measured transfer G[i,j] by greedy-decoding and checking `first_response_token == ※` (token 83399), averaged over 50 probes. The ` ※` marker was chosen specifically (CLAUDE.md) to enable a clean continuous log-prob DV; #406 used the coarsest version.
2. **The emission rate is 52% exact zeros (zero-inflated).** A two-part decomposition showed divergence predicts *whether* transfer happens (point-biserial r = −0.44, p<1e-12) but NOT *how much* among transferring pairs (Spearman ρ = −0.04, p=0.67). The single Spearman ρ = −0.44 is dominated by the on/off split, and standard rank correlation is degraded by the 52% tied zeros (Kendall τ-b = −0.31; attainable |ρ| floor ≈ −0.86 given the zero mass).

A continuous log P(※) outcome has no zero mass, so it (a) makes ordinary Spearman/Pearson valid again and (b) tests whether the "no graded magnitude effect" is real or an artifact of the saturated binary emission rate.

## What changes from #406 (single variable)

EVERYTHING is held identical to #406 except the **outcome metric**:
- Same 16 trained LoRA adapters (`superkaiba1/explore-persona-space`, `adapters/i406_{A1-A5,B1-B5,C1,D1-D5}`) — NO retraining.
- Same predictor D[i,j] (#406's `eval_results/issue_406/divergence/D_matrix.json`, forward-KL K=25-mean) — reused unchanged.
- Same 50-question probe set Q_test, same `build_prompt_for_condition` shapes, same 16 conditions / 240 ordered pairs.
- **New outcome:** `G_logprob[i,j] = mean over q_test of log P_{model_i}(※=83399 at the first response-token position | T_j(q))`, via one teacher-forced forward pass per (i,j,q) — NO generation. Read the log_softmax at the final prompt position (the one predicting the first response token; the marker is the first assistant token per #406's `' ※\n\n<answer>'` training completion).

This is a clean single-variable change (outcome only), which the consistency-checker should confirm against #406.

## Measurement (the load-bearing detail)

- Prompt = `build_prompt_for_condition(T_j, q)` (chat template + `add_generation_prompt`), byte-identical to #406's emission eval.
- Forward pass; take `logits[final_prompt_position]` → `log_softmax` → index token 83399. That position predicts the first response token.
- **Off-by-one guard (mandatory):** on each trained model's own diagonal high-G cells, assert `argmax(logits[final_prompt_position]) == 83399` for the probes where #406's emission was 1 — confirms we are reading the slot the emission rate thresholded.
- Aggregate: mean log-prob over the 50 probes → continuous G_logprob[i,j]. (Consider also reporting mean P(※) alongside log-prob.)

## Analysis

- Primary: length-partial Spearman ρ(G_logprob, D) across 240 ordered pairs (now no zero mass → also report Pearson; both legitimate on a continuous DV).
- Re-ask the #406 two-part question on the continuous DV: is there a GRADED relationship (does D predict the marker log-prob even across cells that scored emission=0)? If yes, the finding upgrades from "divergence gates transfer on/off" to "divergence predicts graded transfer strength."
- Compare the log-prob result head-to-head with #406's emission-rate result (does the headline strengthen / hold / weaken?).
- Length-partial on log-prompt-tokens, same as #406.
- Carry the same descriptive secondaries if cheap (the 6 cosine layers vs G_logprob) since the 2026-06-01 finding was that layer-11 cosine (ρ=−0.71) beat KL on the emission DV — does that hold on the log-prob DV?

## Open for the planner (knobs, not blocking)

- Exact aggregation: mean log-prob vs median vs mean-prob-then-log; how to handle numerical floor.
- Whether to length-partial identically to #406 or revisit (the length confound may behave differently on a continuous DV).
- Pod sizing (this is a generation-free eval: 16 adapters × 800 prompts × 1 forward pass ≈ 12,800 forwards — 1×H100 eval intent, ~10-15 min).
- Reuse Phase 1's teacher-forced forward machinery (`i406_phase1_compute_divergence.py`) adapted to load each adapter + extract the single marker-token log-prob at position 1, instead of full-distribution KL on the base model.
- Whether to also recompute the emission rate in the same pass (free) as a within-run consistency check that reproduces #406's G.

## Why this is worth a planner + a re-run

Cheap (generation-free eval, no retraining, ~$1-2 + ~15 min), directly resolves the two sharpest critiques of #406 (binary DV + zero-inflation), and is the DV the marker was designed for. If the graded effect appears, it materially upgrades the #406 claim before it goes to Dan.
