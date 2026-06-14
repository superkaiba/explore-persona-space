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

Is the directional asymmetry of behavior-leakage transfer a **low-rank, per-unit** structure — every context has a scalar *source-breadth* (how leaky it is as a training source) and a scalar *receptivity* (how easily it absorbs leakage as a target), and the asymmetry is just their difference — rather than a genuinely **pairwise** interaction that depends on the specific (source, target) pair? If low-rank, the predictor `g` for #526 collapses from a full pairwise object to `symmetric_geometry(i,j) + (b_i − r_j)` with two learned per-unit scalars.

## Landed evidence — the answer splits by behavior (gate ladder, 0-GPU, #474 / #537 / #545)

The originating claim holds **only for the marker**. For contentful behaviors the asymmetry is substantially pairwise. Gate ladder on #537's context-generalization tensor (clean 16×16 reciprocal block per behavior, 240 off-diag cells); L0 antisym fractions reproduce #537's own registered reads to 3 decimals, and the marker reproduces the #474 reference.

| Behavior | antisym frac (L0) | baseline-diff term R² (L1) | antisym captured by per-unit scalars (L2) | residual needing pairwise g (L3) |
|---|---|---|---|---|
| marker | 0.283 | flat prior → untestable* | **0.95** | 0.05 |
| taught fact | 0.377 | 0.006 | 0.78 | 0.22 |
| refusal | 0.416 | 0.089 | 0.39 | **0.61** |
| sycophancy | 0.245 | 0.134 | 0.44 | **0.56** |
| EM | 0.415 | 0.103 | 0.35 | **0.65** |

\*Marker base emission rate is flat across contexts, so L1 is untestable in rate space (exactly the theory's prediction for a flat-prior behavior); the continuous base-log-P(marker) variant gives the rank-1 norm prediction the **wrong sign** (slope −4.4, theory predicts +1), matching #537's registered finding.

Two clean reads:
- **Marker: pure rank-1 per-context structure** (95% scalar-captured), and the scalar is NOT the base prior (r=0.03). Its asymmetry is real but entirely "leaky source / receptive target."
- **Contentful behaviors: NOT rank-1.** Per-unit scalars capture only 35–44% of refusal/sycophancy/EM antisymmetry — 56–65% genuinely needs `g(C,C')`. And the baseline-difference theory term (L1) is weak (R² ≤ 0.13) and **wrong-signed** (`corr(receptivity, base prior) = −0.80` for fact — a ceiling artifact on the trained−base delta, not the theory's overlap-weighted expression term). The behaviors built to exercise L1 are exactly where it fails.

#545 (behavior→behavior) is **structurally non-reciprocal** (within-family batteries run only inside their family; exactly one clean cross-family reciprocal pair, sycophancy↔format), so the matrix-level gate ladder is not runnable there; the per-pair L1 over 6 reciprocal rate pairs is suggestive of a base-prior-linked direction but underpowered (effective n≈3) and confounded by format-variant triplication.

## What's left to do (mostly 0-GPU; one needs a forward pass)

1. **Held-out predictive test** — does `symmetric_geometry + (b_i − r_j)` beat symmetric-only on held-out (source, target) cells, scalars fit on a train split? For the marker this should lift the ~0.72 symmetric ceiling cheaply; for refusal/sycophancy/EM quantify how much a pairwise term recovers of the 56–65% residual.
2. **Joint geometry term** — test the full `(E_j − E_i)·cos(v_i, v_j)` form, not just the base-prior factor (needs activation re-extraction for the cosine/overlap matrix — one forward pass per context, small).
3. **More seeds** — #537 refusal/sycophancy/EM are single-seed (marker + fact reproduce at seed 2); confirm the contentful-behavior pairwise residual replicates before it drives #526.
4. **Where do the scalars come from** — are source-breadth / receptivity measurable on the base model before training (context-vector norm/entropy, base behavior rate)? That is what makes them usable in an a-priori predictor.
5. **Behavior-space reciprocity** — to run the gate ladder in behavior space, #545's within-family eval batteries need cross-family eval columns added (a testbed gap, not just an analysis gap).

## Why it matters

Directly constrains the functional form of the #526 unified predictor, and the answer is a **clean negative on the theory's headline term**: the baseline-difference term is NOT the lever. `g` minimally needs **learned per-unit source-breadth + receptivity scalars**, and for refusal/sycophancy/EM it needs **full pairwise `g(C,C',B,B')`** (56–65% of asymmetry unexplained by scalars). The flat-prior marker is the misleadingly-simple case: it is the cleanest rank-1 behavior precisely because its prior can't move, and reading the predictor's required complexity off the marker would under-build `g` for every contentful behavior. Safety reading: "leaky-source" / "receptive-target" are real per-context properties for shallow implants, but for content-laden behaviors which context absorbs a leak depends on the specific source.

## Artifacts (this analysis)

- `figures/issue_526/asym_gate_ladder.png` — per-behavior stacked bars {symmetric / scalar-captured antisym / residual-pairwise antisym}, L1 R² annotated.
- `figures/issue_526/gate_ladder_results.json`, `asym_gate_ladder.meta.json` — all numbers.
- `scripts/issue526_asym_gate_ladder.py`, `scripts/issue526_asym_gate_ladder_plot.py` — reproducible (0-GPU, JSON-only).

## Relations

- Child of #526 (asymmetric + behavior-dependent leakage-prediction rule); this analysis answers #526's "how complex must `g` be" gate.
- Evidence / data: #474 (16×16 marker matrix), #502 (~28% antisymmetric, symmetric ceiling R²≈0.72), #537 (context-generalization testbed + registered asymmetry reads), #545 (behavior→behavior matrix, non-reciprocal), #405/#472 (source-breadth ↑ leakage), #532/#541 (base prior as predictor).
- Open questions: q:leak-predictor (3.1), q:ctx-behavior (3.5), q:beh-b-to-bprime (3.6), q:leak-multicell (set-to-cell distance).
