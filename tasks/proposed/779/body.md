---
title: 'Beat Persona-Vectors prompt-projection monitoring: learned context->profile
  map (pre-gen) + generation-activation pooling (post-gen) on Qwen-2.5-7B'
kind: experiment
tags:
- persona-vectors
- monitoring
- context-geometry
created_at: '2026-07-01T00:47:42Z'
has_clean_result: false
origin_prompt: 'can we rerun their pre-generation prediction experiments but instead
  of projecting the persona vector directly onto the context vector -- we train a
  mapping from context vector to answer profile and then project the persona vector
  onto that answer profile instead ... train a mapping from single context to single
  answer profile, the contexts we train on should be completely different from the
  evaluate on. [+ second result:] if you take the actual generation and use either
  the projection onto the mean generation vector, max of the projection onto all generated
  activations, or mean of the projection onto all generated activations (try all layers)
  -- how much better does that work than projecting onto the final prompt activation.
  [compute: 8xH100 for the parts that need it, offload others; 2 results under one
  umbrella task]'
goal: On Qwen-2.5-7B under Persona Vectors' monitoring rig, test whether trait-expression
  prediction beats PV's last-prompt-token projection baseline by (R1) routing the
  projection through a learned behavior-agnostic context->answer-profile map (pre-generation)
  and (R2) pooling the persona-vector projection over the actual generation's activations
  (post-generation), evaluated on held-out contexts and behaviors.
---
# Beat Persona-Vectors prompt-projection monitoring: learned context→profile map (pre-gen) + generation-activation pooling (post-gen) on Qwen-2.5-7B

## Goal

On Qwen-2.5-7B under Persona Vectors' monitoring rig, test whether trait-expression prediction beats PV's last-prompt-token projection baseline by (R1) routing the projection through a learned behavior-agnostic context->answer-profile map (pre-generation) and (R2) pooling the persona-vector projection over the actual generation's activations (post-generation), evaluated on held-out contexts and behaviors.

## Provenance

Originated from a design discussion (2026-06-30). The seed: Persona Vectors
(arXiv 2507.21509) monitors trait expression by projecting the **last-prompt-token**
activation onto a persona vector, `⟨c_last, r_B⟩`, and predicting the trait score of
the subsequent response. Their reported correlations are strong overall (r=0.75–0.83)
but collapse **within-condition** (evil 0.51, hallucination 0.25 for system-prompt
elicitation) — the raw prompt projection is a coarse, mostly between-condition signal.
This task tests two ways to beat that baseline, tied to our context-geometry theory
(A2 activation-summary, A3 linear read-out, A4 context→profile map).

## Umbrella question (two results under one task)

Under PV's exact monitoring rig on Qwen-2.5-7B, does trait-expression prediction beat
PV's `⟨c_last, r_B⟩` baseline by:
- **R1 (pre-generation):** routing the projection through a learned, behavior-agnostic
  **context→answer-profile map** `h`, i.e. `⟨h(c_x), r_B⟩` instead of `⟨c_x, r_B⟩`?
- **R2 (post-generation):** pooling the persona-vector projection over the **actual
  generation's** per-token activations (mean / max / top-k / last), instead of the
  prompt projection?

R2's response-mean projection `⟨v(x), r_B⟩` is the **oracle** R1's `h` is trying to
reach from the prompt alone — R2 establishes the ceiling, R1 measures how much of it is
recoverable pre-generation. Both are scored against the same PV baseline on the same data.

## Result 1 — learned context→profile map (pre-generation)

- **Map `h: c_x → v(x)`**, trained to reconstruct the **full answer-profile** (NOT the
  trait) — behavior-agnostic, one `h` reused across all behaviors; `r_B` applied post-hoc.
  This is the load-bearing design choice: it makes any gain attributable to the
  context→profile map generically, and enables zero-shot monitoring directions for unseen
  behaviors. For linear `h`, the map folds into the persona vector once: `r̃_B = Mᵀ r_B`,
  so deployment is a plain dot product (same cost as PV); for nonlinear `h` no global fold
  exists (the effective direction is context-dependent — itself the linear-vs-nonlinear
  result), but inference is still one small forward pass per prompt.
- **Fitter ladder:** ridge (linear, closed-form) → kernel ridge (nonlinear, closed-form,
  ≤~20k train contexts or Nyström) → MLP (nonlinear, expressive).
- **Reconstruction objectives:** (a) full-vector MSE (theory-faithful, keeps magnitude);
  (b) top-k PCA-MSE (denoised; tests A1/A2 low-rank — report the fraction of `r_B` living
  in the top-k profile subspace); (c) cosine (direction-only; nuisance-magnitude-robust,
  aligns with the scale-invariant metric; MLP-side, no closed form). Readout computed both
  ways (dot `⟨h,r_B⟩` and cosine `cos(h,r_B)`), reported.
- **Layer grid:** read-out layer = `h` output layer (tied), swept over all layers
  (prediction regime); `h` input layer swept over all layers + input-layer **combos**
  (concatenation), with last-prompt-token vs mean-over-prompt-tokens pooling. Full grid for
  ridge/KRR (cheap); MLP restricted to PV-selected layer(s).
- **Training corpus for `h` (3-level factor):** (1) fully generic (LMSYS-Chat-1M prompts);
  (2) generic + *other*-trait data (behavior holdout preserved); (3) generic + *eval*-trait
  data (measures value of in-domain contexts; weaker generalization). **Hard invariant:**
  the exact PV eval monitoring contexts are held out from `h`'s training in ALL three arms.
- **Generalization:** train `h` on contexts fully disjoint from eval; hold out behaviors
  (leave-one-behavior-out for `r_B`) so the claim is "one map, unseen contexts, unseen
  behaviors."
- **Baselines:** PV raw projection `⟨c_x, r_B⟩`; a trait-fit linear probe on `c_x` (bounds
  the linear case); the oracle `⟨v(x), r_B⟩` (from R2).

## Result 2 — generation-activation pooling (post-generation)

- Read the trait off the **actual generated response's** per-token activations, projected
  onto `r_B`, under different pooling: **MEAN** (= projection onto the mean generation
  vector; note mean-of-projections ≡ projection-of-mean by linearity, so these are one
  estimator), **MAX** (nonlinear; per-token peak — cannot reduce to a pooled vector), plus
  cheap adds **top-k-mean** and **last-response-token**. Sweep all layers.
- **Baseline:** PV's `⟨c_last, r_B⟩` prompt projection — quantifies how much trait signal
  is in the generation that the prompt can't see, and which pooling reads it best.
- **Hypothesis:** MAX may be the better *detector* (localized trait spike) while MEAN is
  the better *aggregate*.
- **~0 extra GPU:** the mean/max/top-k/last pooled projections are computed during the SAME
  activation forward passes R1 already runs (dot each response token with every-layer `r_B`,
  accumulate scalars) — no per-token activation storage needed. One data-collection pass
  produces both results.

## Shared data collection (one pass)

- **Model:** Qwen-2.5-7B-Instruct (only; no Llama for the first pass).
- **Persona vectors `r_B`:** re-extracted via the project persona-vectors recipe (PV
  released trait definitions + code but NOT the vectors), at ALL layers, 3 traits
  (evil / sycophancy / hallucination).
- **Eval rig:** PV's monitoring setup verbatim — 8-way system-prompt sweep + 0/5/10/15/20-shot,
  per trait, ~10 rollouts/config/question; trait-expression judge = `claude-sonnet-4-5-20250929`
  via Batch API. Graded 0–100, multi-sampled per the llm-judging rule (ranking/regression target).
- **Training corpus generation:** LMSYS prompts (+ trait-augmented arms); generate the model's
  own response (single temp-1 sample per training context; k=10 for eval), cache pooled
  activations (c_x last-prompt + mean-prompt; v(x) mean-response; plus R2's per-token pooled
  projections) at the needed layers.
- **Step 0:** oracle headroom (`⟨v(x), r_B⟩` vs `⟨c_x, r_B⟩` vs trait score, within-condition,
  all layers) — also selects the read-out layer. Gates whether R1 has room to improve.

## Metrics

Within-condition Pearson (matched to PV, primary) **and** AUROC / top-k precision (detection
framing — "will this be a problematic generation"), on held-out contexts (R1 also held-out
behaviors).

## Compute plan (implementer owns orchestration)

- **~10–25 GPU-h total**, no LLM fine-tuning (inference + activation caching + lightweight
  regression + judge API). Well under the 100 GPU-h plan-approval cap.
- **8×H100 single pod, data-parallel (8 single-GPU workers, CUDA_VISIBLE_DEVICES sharding —
  NOT TP=8)** for the generation + activation-caching burst (the only genuinely H100-hungry
  parts). Offload judging to the Anthropic Batch API and the regression fits to cheaper
  compute (1 GPU / cpu-bigmem). **Release/downsize the 8×H100 pod between the GPU burst and
  the off-GPU stages** — do not hold 8 idle H100s through the judge batch or between staged
  rounds (#664 spend-leak). Storage: cache all layers on a subset for step-0 layer selection,
  then only the selected layer(s) for the full corpus to stay well under the 50 GB VM footprint.
- **Stage** rather than run the full factor cross-product: step 0 (oracle + layer select) →
  stage 1 (headline: generic corpus, best layer, ridge vs one MLP, cosine readout, vs
  baselines) → stage 2 (ablations: objectives, 3 corpus levels, layer combos, R2 pooling)
  only on configs that clear stage 1.

## Planning notes

- New research direction → planner must do the literature review first (related work spans
  probing classifiers, "Future Lens"-style prediction of downstream states from hidden
  activations, activation forecasting, representation reading / RepE monitoring) and formalize
  R1/R2 against our context-geometry theory (A2/A3/A4) before any code.
- Single-variable-change discipline: R1 and R2 each change ONLY the projection input vs PV's
  `⟨c_last, r_B⟩`; everything else (vectors, eval rig, judge) reused verbatim.
