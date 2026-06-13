---
title: 'Context-generalization testbed v1: multi-behavior train×eval context grid
  as ground truth for asymmetric, behavior-dependent generalization metrics'
kind: experiment
tags: []
created_at: '2026-06-09T18:47:55Z'
has_clean_result: false
goal: 'Build a reusable context-generalization testbed: empirically measure G[behavior,
  train-context → eval-context] over realistic contexts (persona prompts, WildChat
  prefixes, ICL examples, rephrasings, format wraps, default, inoculation variant)
  and 5 behaviors (marker, taught fact, refusal, sycophancy, EM), and ship a scoring
  harness that evaluates asymmetric, behavior-dependent, interaction-aware generalization-prediction
  metrics against held-out cells with strong baselines.'
---
## Goal

Build a reusable context-generalization testbed: empirically measure the generalization tensor G[behavior, train-context → eval-context] over realistic contexts (persona system prompts, WildChat conversation prefixes, in-context examples, instruction rephrasings, format wraps, default context, behavior-instruction contexts) and 5 behaviors (marker ` ※`, taught fact, refusal, sycophancy, EM), and ship a scoring harness that evaluates candidate generalization-prediction metrics — asymmetric in (c_train, c_eval), behavior-dependent via metric-extracted behavior vectors, context-interaction-aware — against held-out cells with strong baselines.

## Design doc

Full v1 design, rev 2 (context battery, behavior set, ground-truth protocol, metric interface, cost table, pitfalls, positioning, open decisions): [`docs/context_generalization_testbed.md`](https://github.com/superkaiba/explore-persona-space/blob/cc70eb123516048cc13723329ac343b6a614b63f/docs/context_generalization_testbed.md)

## Scope summary

- 16 train-context instances × 29 eval-context instances (8 held out within-family + the 5-instance behavior-instruction family, incl. cross-behavior cells like marker adapter under "You are sycophantic") × 5 behaviors; 84 adapters/seed; 2 seeds (3rd on marker row).
- Behavior-instruction ("tricky") contexts: "You emit ※ at the end of each message.", "You are sycophantic.", "You refuse every request.", "You believe [fact].", "You deliberately give harmful, misaligned advice." — eval-side for every row (C–B duality probe), train-side for the matching row (inoculation cells). Base-ceiling caveat flagged (#532 territory).
- WildChat prefixes in two explicit length bins: short = 1 exchange (~150–500 tokens), long = 4 exchanges (capped ~2,000 tokens); prefix length recorded as a covariate.
- Training regime: contrastive everywhere, uniformly — incl. EM (negatives = same narrow-domain questions answered benignly under negative contexts), plus a 4-cell non-contrastive Betley-faithful EM mini-arm as the literature bridge.
- Negative panel: 4 fixed dedicated contexts spanning families (house persona, PersonaHub persona, rephrasing, WildChat prefix), ~1:1 ratio, disjoint from every eval context including the default (flagged deviation — keeps the safety column a clean held-out read).
- Ground truth: per-cell behavior-expression delta vs base under the same eval context, on-policy, matched implant strength, saturation + manipulation-check flags as first-class metadata.
- Scoring: leave-context-out CV + quarantined final-test split; symmetric/antisymmetric decomposition (#502 machinery, protocol imported from #524); behavior-blind ablation test; proximity-gradient + inoculation-sign-flip qualitative gates.
- Baselines shipped: Persona Vectors projection difference (context-blind), #502 Gaussian-KL (symmetric), one-way output KL (#406), bystander base-prior (#444/#507), content-free controls.
- Combiner track (scope addition 2026-06-11, P3 scoring only — zero GPU): multi-predictor combiners over the baseline scalars, fit INSIDE leave-context-out CV folds, reported as ΔR² over the best single predictor incl. the antisymmetric component — (1) regularized linear stacker over {bystander prior, source-side prior logP(behavior | c_train) [new feature], Gaussian-KL, output KL both directions, PV projection, content-free covariates}; (2) theory-shaped write×gate form per `docs/notes/rank1_leakage_model.pdf` P3; (3) per-behavior z-normalized pooled variants. Spec: design doc §5 (commit 0cc4053fc). Mirrors #545's combiner group; #541's `geometry-plus-prior-joint-predictor` follow-up is the fact-line preview.
- Budget: ~155–215 GPU-h (envelope 100–300), phased P0 (no GPU: contexts + harness + pre-registration) → P1 (marker row, validates harness, reuse-check #474 loc-ep1 adapters) → P2 (remaining rows) → P3 (baseline scoring).

## Open design decisions for the planner (design doc §10)

1. Default context excluded from the (now 4-context, cross-family) negative panel so the safety column stays a clean held-out read — deviation from house contrastive default.
2. Size of the non-contrastive EM mini-arm (4 train contexts; droppable if budget tightens, at the cost of literature comparability).
3. Fixed question pool appended after WildChat prefixes (controlled) vs continuing each conversation thread (realistic).
4. Single fact vs small fact panel for the taught-fact row.

## Relation to existing tasks

#524 (asymmetric predictors — run first / import protocol), #446 (realistic-setting scoping — subsumed), #445/#440 (cell-set prediction — v2), #428 (behavior definition — P0 companion), #532 (geometry vs base-prior on instructed contexts — previews both the baseline comparison and the F8 ceiling caveat).
