---
title: Does the context→answer map get more predictable with model scale? (Qwen-2.5-Instruct
  0.5B→32B)
kind: experiment
tags:
- context-geometry
- scale
created_at: '2026-07-18T01:29:58Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'Can you run an experiment to test how this mapping changes with scale
  of the model: - [ ] Does model''s behavior get more predictable with scale?'
workflow: v1
goal: 'On the Qwen-2.5-Instruct scale ladder (0.5B, 1.5B, 3B, 7B, 14B, 32B; 7B = the
  #779 fitter-fair-comparison-n1m anchor), measure how the context→answer map h: c(x)
  → v(x) (pre-generation context representation → mean-own-response activation profile
  at a depth-matched layer) changes with model scale: per scale, generate each model''s
  OWN responses to the SAME LMSYS+WildChat contexts (pinned splits, matched train-n),
  fit the same linear (ridge) + nonlinear (MLP) maps in BOTH mapping arms (prefix-based
  and context-based), and report held-out test R² (variance-weighted, the #779 metric)
  vs scale — does the model''s behavior, as summarized by its answer activation profile,
  get more predictable from its pre-generation context representation as scale grows;
  does the linear-vs-nonlinear gap (0.754 vs 0.810 at 7B) shrink or grow; and is the
  trend robust to layer choice (depth-fraction-matched primary, per-scale val-selected
  sweep secondary), train-set size (matched-n primary, R²-vs-n curves), and the dimension/response-length
  confounds?'
relates_to:
- spec-context-as-vector
---
# Does the context→answer map get more predictable with model scale? (Qwen-2.5-Instruct 0.5B→32B)

## Goal

On the Qwen-2.5-Instruct scale ladder (0.5B, 1.5B, 3B, 7B, 14B, 32B; 7B = the #779 fitter-fair-comparison-n1m anchor), measure how the context→answer map h: c(x) → v(x) (pre-generation context representation → mean-own-response activation profile at a depth-matched layer) changes with model scale: per scale, generate each model's OWN responses to the SAME LMSYS+WildChat contexts (pinned splits, matched train-n), fit the same linear (ridge) + nonlinear (MLP) maps in BOTH mapping arms (prefix-based and context-based), and report held-out test R² (variance-weighted, the #779 metric) vs scale — does the model's behavior, as summarized by its answer activation profile, get more predictable from its pre-generation context representation as scale grows; does the linear-vs-nonlinear gap (0.754 vs 0.810 at 7B) shrink or grow; and is the trend robust to layer choice (depth-fraction-matched primary, per-scale val-selected sweep secondary), train-set size (matched-n primary, R²-vs-n curves), and the dimension/response-length confounds?

## Provenance

- workflow origin: user chat 2026-07-17 (interactive), routed as a NEW-direction child of #779 per the question-identity litmus (a scale axis opens a new question; it would not rewrite #779's monitoring Takeaways). Sibling NEW-direction child on the same mapping: #1482 (error analysis of the n1M map).
- Verbatim originating prompt: "Can you run an experiment to test how this mapping changes with scale of the model: - [ ] Does model's behavior get more predictable with scale?"
- "This mapping" = the #779 fitter-fair-comparison-n1m context→answer map h: c_last(x) → v(x) (last-prompt-token activation → mean-own-response activation profile, layer 19, Qwen-2.5-7B-Instruct, ~963k LMSYS+WildChat train contexts, pinned val 400 / test 1000 split; ridge test R² 0.754, MLP-w8192/w32768 0.810–0.813), from the planned-experiments doc line "Does model's behavior get more predictable with scale?".

## Motivation

The theory paper ("Predicting fine-tuning–induced leakage from pre–fine-tuning context geometry") assumes a learnable, approximately linear map from context representations to answer representations; its evaluation plan names "a small set of open-weight models across scales" as a robustness axis. If the model's own behavior (as summarized by its answer activation profile) becomes MORE predictable from the pre-generation context representation as scale grows, pre-generation behavior predictors and pre-training audits (open-questions App 5) get easier exactly where the safety stakes are highest; if predictability degrades with scale, the predictor line has a scaling ceiling worth knowing now. Secondary: whether the linear-vs-nonlinear gap (ridge vs MLP; 0.754 vs 0.810 at 7B) shrinks with scale bears directly on the paper's linear-map assumption.

**Literature-review-first (standing rule):** planning MUST open with a thorough arXiv/web review of prior work on representation predictability/linearity vs scale (linear-representation-hypothesis scaling reads, cross-layer/cross-scale representation similarity, observational scaling laws, activation-predictability studies) and name the closest prior formalizations before any capture code.

## Design sketch (capture-level; planner refines)

- **Manipulated variable (single):** model scale within one family — Qwen-2.5-Instruct at 0.5B, 1.5B, 3B, 7B, 14B, 32B (72B optional, cost-gated). Family, chat template, corpus, splits, fitters, and metric held fixed. The 7B cell is the existing #779 n1M anchor (reuse, not rerun, where the fitness checklist passes).
- **Map per scale:** same construction as #779 n1M — context representation c(x) at a chosen layer → v(x) = mean activation profile over the model's OWN response tokens at the same layer. Each model generates its OWN responses to the same contexts (vLLM, generation recipe matched to the #779 n1M recipe), then teacher-forced capture.
- **Both mapping arms (standing rule):** context-based arm (everything before the answer, incl. the user query — the parent's c_last; PRIMARY, matches the anchor) AND prefix-based arm (last token of the prefix before the user query). Many LMSYS/WildChat rows have an empty/template-only prefix; if the prefix arm is degenerate on this corpus, that is an explicit stated plan deviation, not a silent drop.
- **Layer selection across scales:** primary = depth-fraction-matched layer (19/28 ≈ 0.68 of depth at 7B, rounded per model); secondary = per-scale layer sweep with val-set selection (read-out/prediction regime), to show the trend is not a layer-choice artifact.
- **Matched data:** same context set + pinned val/test splits across all scales; matched train-n (sized by the planner, e.g. ~100k) as the primary comparison, plus R²-vs-n curves per scale (subsample ladder) to separate sample efficiency from the asymptote. The 7B full-n anchor ties the ladder to the existing result.
- **Fitters:** the issue-779-n1m branch fitters (GCV/val-selected ridge; MLP at the same widths), reused per the artifact-reuse checklist; per-scale regularization selected on val identically.
- **DV:** held-out test R² (variance-weighted, the #779 metric) vs scale, per arm × layer regime × fitter family; plus the linear-vs-nonlinear gap vs scale. This is a representation-level DV by construction (the map predicts the answer activation profile, not a judged behavior); any behavior-level claim must go through a read-out companion (e.g. trait read-out projections of v̂ vs v) — planner names it or scopes it out explicitly.
- **Confound controls:** hidden-dim growth with scale (896→5120+) changes map capacity — report dims, hold the fitter-selection protocol fixed, and include per-scale floors (identity-copy, shuffled-pairing baselines as in #779); response-length/entropy distribution shifts across scales — report per-scale response stats and check R² trend robustness under length stratification.

## Competing hypotheses

1. Predictability INCREASES with scale (cleaner, more linearly structured representations) — R² rises monotonically.
2. Predictability DECREASES with scale (richer, higher-entropy behavior; more context-dependent computation happens after the prompt state).
3. Total R² roughly flat but the LINEAR share grows (linearization with scale) — the ridge-vs-MLP gap shrinks.
4. Any apparent trend is an artifact of dimension/regularization, layer choice, or response-length shifts — the controls above must distinguish this from 1–3.

## Reuse

- 7B anchor: #779 n1M captures, pinned splits, committed fit results (`eval_results/issue_779/fitter-fair-comparison/`), issue-779-n1m branch fitter + capture code — subject to the full artifact-reuse fitness checklist (a)–(k).
- New: generation + capture for the non-7B scales; scale-ladder analysis + figures.
