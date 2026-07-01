---
title: How much do pooling operators (mean/max/attention) lose vs unpooled per-position
  answer activations for predicting behavior / reconstructing the profile
kind: experiment
tags:
- answer-summary-sweep
- from-722
created_at: '2026-07-01T18:27:13Z'
has_clean_result: false
parent_id: 810
origin_prompt: Both - but also instead of just using the persona vector r_B as a readout
  train a regression to predict the expression from the specific summary) actually
  could we also add an experiment which checks how much adding individual activations
  to the regression helps -- vs. just a pooled mean
goal: 'On Qwen2.5-7B base representations (#658''s 50 contexts, reusing #810''s per-position
  answer extraction), quantify how much mean-pooling loses versus using individual
  per-position answer-token activations as regression input -- the incremental held-out
  prediction of behavior expression E0 (and answer-profile reconstruction) from an
  unpooled multi-position input over the pooled-mean baseline, per layer, with LOCO
  + label-shuffle null + #742-style reliability-ceiling / learning-curve guards for
  the n=50 sample size.'
---
# How much does per-position (unpooled) answer activation beat the pooled mean for predicting behavior / reconstructing the profile

## Overview / Motivation

The answer-profile summary `v0` used across the leakage-predictor line
([#658](https://eps.superkaiba.com/tasks/658)/[#722](https://eps.superkaiba.com/tasks/722))
is a **mean pool** over answer tokens — it discards all per-position structure.
The sibling task (#810) asks which *single* answer position/summary is best; this
task asks the complementary question: **how much does mean-pooling itself throw
away?** i.e. does a regression given the **individual per-position answer
activations** (unpooled) predict behavior expression `E0` / reconstruct the
answer profile materially better than the same regression given only the pooled
mean.

This is the **input-representation axis** of [#742](https://eps.superkaiba.com/tasks/742)
(currently `blocked`), which measures how much *linear decoding* loses from the
pooled mean at n=50; here the manipulated thing is the *input granularity*
(pooled vs unpooled), reusing #742's reliability-ceiling / learning-curve
framework.

## Design (single manipulated variable = the pooling operator, up to the unpooled ceiling)

Base model only, reusing #658's 50-context grid + judged `E0(C,B)` + #810's
per-position answer extraction (shared re-extraction — no extra forward passes
beyond #810's). Regression per (behavior × layer), over a ladder of pooling
operators measured against an unpooled upper bound:

- **Pooling operators** (each reduces the SAME per-position span to one
  3584-vec — cheap CPU, no new forward passes and no new judge calls):
  - `mean` — the reference the whole line currently uses.
  - `max` — element-wise max over positions.
  - `attn-fixed` — a fixed random-projection attention pool `softmax(span·q_rand)`
    (the #658 unlearned control — does *learning* the pool beat random?).
  - `attn-learned` — a **fit** query vector `softmax(span·q)`, the best
    single-vector pool (adds one 3584-dim parameter per cell).
- **Unpooled ceiling** — the individual per-position activations as separate
  regression features (answer lengths vary, so a fixed aligned set: end-aligned
  tail `−1…−K` + start-aligned head `0…K−1` + the boundary tokens,
  `≤ (2K+2)×3584`, matching #810's positions). This is the upper bound each
  pooling operator is scored against.
- **Comparison** — per (behavior × layer): (i) how close each pooling operator
  gets to the unpooled ceiling (incremental held-out gain of unpooled over each
  pool = "how much that pool loses"), and (ii) the ranking among operators, all
  relative to the `mean` reference.

**Compute is negligible; the constraint is n, not FLOPs.** Every operator + the
unpooled input are CPU reductions of one shared #810 extraction; `attn-learned`
adds only a per-cell 3584-dim query fit, done vectorized across cells (the
`vectorized_mlp_skill` helper). No new GPU forward passes, no new judge calls.

**Distinct from #810 (so this is not redundant):** #810 sweeps which SINGLE
summary/position is best (incl. `maxp`/`attn` as single-vector summaries) for the
map + read-out; THIS task is the POOLING-OPERATOR study — how much each pooling
operator loses relative to the **unpooled multi-position ceiling**, which #810
never fits. The `attn`/`max` overlap is by construction (they are the operators
being compared to the ceiling here), not a duplicate question.

**Sample-complexity is the central risk, not an afterthought.** At n=50 both the
`(2K+2)×3584`-feature unpooled input AND the `attn-learned` fit query add
parameters that overfit badly (#722's MLP already overfit at n=50 on ONE
3584-vec), which would *inflate* the apparent ceiling. Mandatory guards: strong
ridge / per-position PCA before concatenation, a regularized `attn-learned` query,
LOCO cross-validation, a label-shuffle null, and a #742-style learning curve
extrapolating the contexts needed to resolve any gap. The honest
possible outcome is "at n=50 we cannot resolve whether unpooling helps" — which
is itself #742's thesis; if so, the deliverable is the ceiling + the contexts-needed
estimate, and expanding the context battery (new generation + extraction, more
GPU) becomes the follow-up.

## Dependent variable

Incremental held-out prediction over the pooled-mean baseline, per (behavior ×
layer): ΔR² / Δρ of (unpooled input) − (pooled-mean input) for predicting
`E0(C,B)`, and separately for reconstructing the answer profile. Reported with
the shuffle null and the learning-curve extrapolation; **any lift is stated as a
lower bound given the n=50 reliability ceiling** (`√r_yy`, per #742).

## Relation to existing work / dependency

- **Depends on #810's extraction** (the per-position answer activations) — file
  as proposed; run after / alongside #810 so the extraction is shared, not
  duplicated.
- **Sibling of #742** (blocked): #742 = "how much does linear *decoding* lose
  from the pooled mean"; this = "how much does the *pooling* lose vs unpooled".
  Reuse #742's reliability-ceiling + learning-curve machinery; coordinate rather
  than duplicate (a `redundant` verdict from the follow-up-critic would park
  this — the distinction is the manipulated axis: input granularity, not decoder
  class).
- **#810's single-position sweep is a cheap precursor:** if #810 finds
  individual positions barely differ from the mean, a joint-position lift here is
  a priori less likely — read #810 first.

## Cost

Extraction shared with #810 (no extra GPU). Fits are CPU-cheap but n-limited;
the only path to a bigger n is a new context battery (flagged, not in v1).

## Provenance

Standalone child of #810 (user-directed, 2026-07-01), itself a child of #722.
Verbatim originating prompt (this task):
> Both - but also instead of just using the persona vector r_B as a readout train
> a regression to predict the expression from the specific summary) actually
> could we also add an experiment which checks how much adding individual
> activations to the regression helps -- vs. just a pooled mean
