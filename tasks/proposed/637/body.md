---
title: Is leakage-transfer asymmetry low-rank (per-context source-breadth + receptivity),
  not pairwise interaction?
kind: analysis
tags: []
created_at: '2026-06-14T00:11:25Z'
has_clean_result: false
parent_id: 526
origin_prompt: 'File an issue to look into this: → the asymmetry is almost entirely
  "some contexts are leaky sources / receptive targets," not pairwise interaction.'
---
## Provenance

Filed from a chat request while running a 0-GPU asymmetry analysis on existing leakage matrices (feeds the theory task #526). Verbatim originating prompt: "File an issue to look into this: → the asymmetry is almost entirely 'some contexts are leaky sources / receptive targets,' not pairwise interaction."

## Question

Is the directional asymmetry of behavior-leakage transfer a **low-rank, per-unit** structure — every context has a scalar *source-breadth* (how leaky it is as a training source) and a scalar *receptivity* (how easily it absorbs leakage as a target), and the asymmetry is just their difference — rather than a genuinely **pairwise** interaction that depends on the specific (source, target) pair? Same question in behavior space for behavior→behavior leakage. If low-rank, the predictor `g` for #526 collapses from a full pairwise object to `symmetric_geometry(i,j) + (b_i − r_j)` with two learned per-unit scalars.

## Seed evidence (provisional — marker only)

On the #474 16×16 marker transfer matrix (cleanest checkpoint loc_ep1; cell = trained−base log P(marker)):
- Antisymmetric fraction of off-diagonal variance = 0.283 (reproduces #502's ~28%; symmetric-predictor ceiling R² ≈ 0.72). Robust 0.28–0.34 across loc epochs.
- An additive two-way model `L[i→j] ≈ μ + b_i + r_j` (per-context source-breadth `b` + receptivity `r`) captures **95.5%** of the antisymmetric variance → residual needing true pairwise interaction ≈ **4.5%**.
- The fitted per-context scalar is **not** the base prior (r=0.03) and the theory's baseline-difference term is null (R²=0.006) — expected, because the marker base prior is flat across contexts.

So for the marker the asymmetry is real but almost entirely rank-1 / per-context, not pairwise. Open: does this hold for contentful behaviors whose base prior is **not** flat (sycophancy, EM, refusal, fact)?

## What to do (all 0-GPU, existing data)

1. Run the gate ladder (L0 antisym fraction → L1 baseline-difference term → L2 per-unit source-breadth+receptivity additive model → L3 residual pairwise) on the two comprehensive evals:
   - #537 context-generalization tensor `G[behavior, train-ctx → eval-ctx]` (5 behaviors × contexts) — asymmetry in CONTEXT space, per behavior; reconcile with #537's existing asymmetric scoring harness.
   - #545 behavior→behavior matrix `L[b_train → b'_eval]` (within-family reciprocal pairs) — asymmetry in BEHAVIOR space, where base priors vary widely.
   (A first pass is already running; fold its numbers in.)
2. Report the additive-model R² on the antisymmetric variance per behavior + pooled. Does the rank-1 structure hold for contentful behaviors, or does pairwise interaction appear once the base prior is non-flat?
3. Held-out predictive test: does a rank-1-corrected predictor (`symmetric geometry + b_i − r_j`, scalars fit on a train split) beat symmetric-only on held-out (source, target) cells? This is the practical payoff — it would lift the symmetric ceiling (R²≈0.72 on the marker) with two cheap per-context scalars.
4. Discriminator: is the fitted net scalar `(b_i − r_i)` the base prior (theory's baseline-difference term) or a separately-learned breadth/receptivity property? Report per behavior — the marker says "learned, not base prior"; check whether contentful behaviors flip to "base prior IS the scalar."
5. Where do the scalars come from? Probe whether source-breadth / receptivity are measurable before training (base-model context properties: norm/entropy of the context vector, base behavior rate, etc.) — that is what makes them usable in an a-priori predictor.

## Why it matters

Directly constrains the functional form of the #526 unified predictor: if leakage asymmetry is rank-1, `g(C, C', B, B')` need only carry symmetric geometry + per-context source-breadth + per-context receptivity (+ a behavior-interaction term), not a full pairwise interaction — a large simplification with a concrete safety reading ("leaky-source" and "receptive-target" contexts are identifiable per-context properties, not emergent only from specific pairings). The 4.5% residual on the marker also bounds how much a pairwise term could ever add there.

## Relations

- Child of #526 (asymmetric + behavior-dependent leakage-prediction rule).
- Evidence / data: #474 (16×16 marker matrix), #502 (~28% antisymmetric, symmetric ceiling R²≈0.72), #537 (context-generalization testbed), #545 (behavior→behavior matrix), #405/#472 (source-breadth ↑ leakage), #532/#541 (base prior as predictor).
- Open questions: q:leak-predictor (3.1), q:ctx-behavior (3.5), q:beh-b-to-bprime (3.6), q:leak-multicell (3.x set-to-cell distance).
