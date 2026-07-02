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
goal: On Qwen-2.5-7B, characterize how well and at what token granularity the base
  model's pre-generation context representation predicts its own answer representation
  and behavior (at the single-context/single-answer per-example level), and test whether
  routing trait-expression monitoring through a learned behavior-agnostic context->answer-profile
  map (pre-generation) or pooling over the actual generation's activations (post-generation)
  beats Persona Vectors' last-prompt-token projection baseline AND a matched-capacity
  direct context->behavior predictor, on held-out contexts and behaviors.
---
## Goal

On Qwen-2.5-7B, characterize how well and at what token granularity the base model's pre-generation context representation predicts its own answer representation and behavior (at the single-context/single-answer per-example level), and test whether routing trait-expression monitoring through a learned behavior-agnostic context->answer-profile map (pre-generation) or pooling over the actual generation's activations (post-generation) beats Persona Vectors' last-prompt-token projection baseline AND a matched-capacity direct context->behavior predictor, on held-out contexts and behaviors.


## Provenance

Originated from a design discussion (2026-06-30). Seed: Persona Vectors (arXiv 2507.21509)
monitors trait expression by projecting the **last-prompt-token** activation onto a persona
vector, `⟨c_last, r_B⟩`, predicting the trait score of the subsequent response — strong
overall (r=0.75–0.83) but collapsing **within-condition** (evil 0.51, hallucination 0.25).
Design iterated over several rounds to add: the direct context→behavior predictor as a
first-class matched-capacity comparison, direct reconstruction-quality reporting of the map,
and a map-granularity study. Ties to the context-geometry theory (A2 activation-summary,
A3 linear read-out, A4 context→profile map).

## Umbrella question (three results under one task)

On Qwen-2.5-7B, characterize how well / at what granularity the base model's **pre-generation
context representation predicts its own answer representation and behavior**, and whether that
predictability beats PV's last-prompt-token projection for trait monitoring:
- **R1 (pre-generation monitoring):** does routing the projection through a learned,
  behavior-agnostic **context→answer-profile map** `h` (`⟨h(c_x), r_B⟩`) beat PV's raw
  `⟨c_x, r_B⟩` — and does it beat a **direct context→behavior predictor** trained on the trait?
- **R2 (post-generation pooling):** does pooling the persona-vector projection over the
  **actual generation's** activations (mean / max / top-k / last) beat the prompt projection?
- **R3 (map granularity):** how well, and at what token granularity, does the context
  representation predict the answer representation (reconstruction quality, per-position decay,
  set-to-set predictability)?

R2's response-mean projection `⟨v(x), r_B⟩` is the **oracle** R1's `h` targets from the prompt
alone. All three share ONE data-collection pass.

**Granularity level (load-bearing — do NOT design condition-means):** everything operates at the
**single-context → single-answer (per-example)** level. `c_x` and `v(x)` are per-example; `v(x)` is
the mean over the tokens of that ONE response (not a mean over samples or over contexts); R3's i,j
pairs are single-context-token → single-answer-token within one example. The theory's condition-mean
`v_θ(C) = E_{x∼C}[ā(x)]` is the aggregate these single-level results average UP to — the experiment
deliberately runs the finer single-instance version (harder, no averaging; and it's the
deployment-relevant per-prompt object). Cross-example aggregation (d1 grid-cell pooling, regression
training pairs, within-condition Pearson) pools single→single pairs, NEVER averaged vectors.

## Result 1 — learned context→profile map + direct-predictor comparison (pre-generation)

- **Map `h: c_x → v(x)`**, trained to reconstruct the **full answer-profile** (NOT the trait) —
  behavior-agnostic, ONE `h` reused across behaviors; `r_B` applied post-hoc. Linear `h` folds
  into the persona vector (`r̃_B = Mᵀ r_B`, deployment = plain dot product); nonlinear `h` has no
  global fold (context-dependent direction — itself the linear-vs-nonlinear result).
- **Fitter ladder:** ridge (linear, closed-form) → kernel ridge (nonlinear, closed-form,
  ≤~20k train or Nyström) → MLP (nonlinear).
- **Reconstruction objectives:** full-vector MSE (theory-faithful) / top-k PCA-MSE (denoised;
  report fraction of `r_B` in the top-k subspace = A1/A2 low-rank test) / cosine (direction-only,
  nuisance-robust, MLP-side). Readout both dot `⟨h,r_B⟩` and cosine `cos(h,r_B)`.
- **DIRECT context→behavior predictor — first-class, matched-capacity comparison arm.** Train
  `g: c_x → trait score` DIRECTLY (skipping the profile), at MATCHED capacity (ridge AND MLP),
  per behavior. This is the pivotal test of whether the profile-map detour earns its keep. The
  key asymmetry, reported explicitly:
  - **In-behavior split:** direct `g` vs indirect `h`+projection, head-to-head — "does routing
    through the profile help *when you already have this behavior's labels*?"
  - **Held-out-behavior split (leave-one-behavior-out):** the direct predictor is **structurally
    inapplicable** (needs labels for the eval behavior); only the behavior-agnostic `h` (via
    `Mᵀr_B`, zero-shot) and PV's raw projection compete. This is where the map's value
    proposition — zero-shot transfer to unseen behaviors — lives.
- **Direct reconstruction-quality reporting (not just downstream projection):** report `h`'s
  R²/cosine of predicted vs true answer activation (mean→mean and pooled→pooled) — the direct
  A4-map test, independent of any behavior read-out.
- **Layer grid:** read-out layer = `h` output layer (tied), swept all layers; `h` input layer
  swept all layers + input-layer combos (concatenation), last-prompt-token vs mean-prompt-tokens.
  Full grid for ridge/KRR; MLP restricted to PV-selected layer(s).
- **Training corpus (3-level factor):** (1) fully generic (LMSYS-Chat-1M prompts); (2) generic +
  *other*-trait data (behavior holdout preserved); (3) generic + *eval*-trait data. **Hard
  invariant:** the exact PV eval monitoring contexts are held out from `h` training in ALL arms.
- **Baselines:** PV raw projection; the direct predictor (above); the oracle `⟨v(x), r_B⟩`.

## Result 2 — generation-activation pooling (post-generation)

- Read the trait off the **actual generated response's** per-token activations projected onto
  `r_B`: **MEAN** (≡ projection of mean generation vector, by linearity), **MAX** (nonlinear
  per-token peak), plus **top-k-mean** and **last-response-token**. Sweep all layers.
- **Baseline:** PV's `⟨c_last, r_B⟩` prompt projection. Hypothesis: MAX is the better *detector*
  (localized spike), MEAN the better *aggregate*.
- **~0 extra GPU:** pooled projections computed during the SAME activation forward passes
  (dot each response token with every-layer `r_B`, accumulate scalars).

## Result 3 — context→answer map granularity (A4 characterization)

How well / at what granularity does the context representation predict the answer representation:
- **(a) mean→mean and pooled→pooled reconstruction** R²/cosine — shared with R1's diagnostic.
- **(b) pooled-context → per-relative-position answer activation:** predict the answer activation
  at each relative position from the pooled context; measure how predictability **decays with
  answer depth** (how far into the response the context's determination reaches).
- **(c) context-activation-SET → answer-activation-SET:** CKA + linear predictability between the
  full context and answer activation sets, per layer-pair.
- **(d) the full context-position × answer-position (i,j) map** — three principled formulations
  that dissolve the variable-length problem (the naive per-absolute-(i,j) regression is what's
  ill-posed, NOT the object itself):
  - **(d1) normalized-position grid — PRIMARY, cheap:** bin `i/T_c` and `j/T_a` into a fixed grid
    (e.g. 10×10), pool `(a^ctx_i, a^ans_j)` pairs across examples per cell, fit per-cell regression
    → an **R² heatmap** over context-pos × answer-pos, per layer(-pair). Variable length dissolves
    via normalization; subsumes (b) as its context-marginal slice.
  - **(d2) amortized position-conditioned learned map — optional:** one model
    `F(a^ctx_i, posenc(i), posenc(j)) → a^ans_j` (positions as inputs, not grid indices; handles
    variable length natively; one model, not O(T_c·T_a) models), probed at any (i,j). The
    token→token generalization of `h`.
  - **(d3) causal path-patching — mechanistic stretch, small subset:** patch context token i's
    activation, measure the change at answer token j (∼ attention from answer-j to context-i) —
    the *causal* i,j map. O(T_c·T_a) patches → subset only.
- **Well-definedness + caveats (state in the writeup):** teacher-forcing (already used) makes the
  activation i,j map (d) well-defined — both sides observed, a within-pass predictability, NOT
  forecasting. Pairwise (i,j) predictability is a **marginal** (`a^ans_j` depends on all context +
  answer-prefix tokens via attention, not only `a^ctx_i`) — it measures decodable info-flow, not a
  complete generative map. Distinct from the **pre-generation** flavor (b), where predicting
  per-position answer activations from the prompt alone ≈ forecasting the stochastic generation
  trajectory → (b) is measured as expected per-position predictability, and the mean-pool target
  is the clean *expectation* A4 uses.

## Shared data collection (one pass)

- **Model:** Qwen-2.5-7B-Instruct only.
- **Persona vectors `r_B`:** re-extracted via the project persona-vectors recipe (PV released
  trait defs + code, NOT the vectors), ALL layers, 3 traits (evil / sycophancy / hallucination).
- **Eval rig:** PV monitoring verbatim — 8-way system-prompt sweep + 0/5/10/15/20-shot, per
  trait, ~10 rollouts/config/question; judge = `claude-sonnet-4-5-20250929` via Batch API,
  graded 0–100 multi-sample (ranking/regression target per llm-judging rule).
- **Corpus generation:** LMSYS prompts (+ trait-augmented arms); own response (single temp-1
  sample per training context; k=10 eval); cache pooled activations (c_x last-prompt + mean-prompt;
  v(x) mean-response) + R2's per-token pooled projections. **R3 needs per-token CONTEXT AND
  ANSWER activations** for (b)/(c)/(d) → cache per-token acts for a **SUBSET** of contexts (a few
  thousand) and a subset of layers (grid/CKA/decay are statistical, don't need the full corpus) to
  bound storage.
- **Step 0:** oracle headroom (`⟨v(x), r_B⟩` vs `⟨c_x, r_B⟩` vs trait score, within-condition,
  all layers) — also selects the read-out layer; gates whether R1 has room.

## Metrics

- **Monitoring (R1, R2):** within-condition Pearson (primary, matched to PV) **and** AUROC /
  top-k precision (detection framing), on held-out contexts (R1 also held-out behaviors).
- **Map characterization (R1 diagnostic, R3):** reconstruction R²/cosine, per-position
  predictability decay, set-to-set CKA/linear-predictability.

## Compute plan (implementer owns orchestration)

- **~10–25 GPU-h**, no LLM fine-tuning (inference + activation caching + lightweight regression
  + judge API). Under the 100 GPU-h plan-approval cap.
- **1×H100 (`lora-7b`) for the generation + activation-caching burst — DESCOPED to `lora-7b`
  from the plan §9's `sweep-8g-h100` 8×H100 data-parallel shape** (see `## Compute-deviation`
  below): the `issue779_{extract_rb,collect,stage1}.py` scripts only accept `--gpu-id N` (single
  GPU) and implement no internal `torch.distributed`/`multiprocessing.spawn` DP nor an external
  `--shard-id`/`--num-shards` split, so an 8×H100 pod would leave 7 GPUs at 0% util (the #664
  spend-leak the plan itself named at §-risks). Offload judging → Anthropic Batch API and
  regression fits → cheaper compute (1 GPU / cpu-bigmem). **Release/terminate the H100 pod
  between the GPU burst and the off-GPU stages** — never hold an idle H100 through the judge
  batch or between staged rounds (#664 spend-leak).
- **Storage:** pooled activations cache is small; R3's per-token caching is bounded to a subset
  (few thousand contexts × subset of layers) — keep total well under the 50 GB VM footprint;
  full-layer caching only on a subset for step-0 layer selection, then selected layer(s) for the
  full corpus.
- **Stage** rather than run the full cross-product: step 0 (oracle + layer select) → stage 1
  (headline: generic corpus, best layer, ridge vs one MLP, cosine readout, indirect vs direct
  vs PV-baseline vs oracle) → stage 2 (ablations: objectives, 3 corpus levels, layer combos, R2
  pooling, R3 granularity) only on configs that clear stage 1.

## Compute-deviation

Recorded inline (analogous to the pre-launch `epm:compute-deviation v1` marker, but
post-launch: the round-6 experimenter run surfaced this AFTER dispatch, so it is documented
here in the body rather than at the Step 5.bis pre-launch gate).

- **Deviation:** the generation + activation-capture burst runs on **1×H100 (`lora-7b` intent)
  serially**, DESCOPED from the plan §9's **`sweep-8g-h100` (8×H100, data-parallel over
  contexts)** shape.
- **Cause:** the `issue779_{extract_rb,collect,stage1}.py` scripts accept only `--gpu-id N`
  (single GPU). None implements internal DP (`torch.distributed` / `torch.multiprocessing.spawn`)
  or an external `--shard-id`/`--num-shards` context split, so the plan's DP recipe was never
  wired. Running on 8×H100 would leave 7 GPUs at 0% util → the #664-class spend-leak the plan
  named at §-risks (~$1200 with 7 idle H100s vs ~$130 for 1 running H100).
- **Wall-time:** ~44 wall-h (single-GPU serial) vs the plan §9's ~5.5 h (8×H100 DP). GPU-h
  ≈ 44 — well under the 100 GPU-h plan-approval cap.
- **Science preserved exactly:** identical conditions, rollouts, data, seeds, judge, and layers.
  Only the parallelism shape (wall-clock) changed; no result-affecting parameter moved. Adding
  `--shard-id`/DP support was deliberately deferred (a substantial change; the 1×H100 descope
  keeps the science exact) — see the round-7 implementation report `(b)`.
- **Provenance:** `epm:experiment-implementation v7` (round 7 = crash-fix-round-1).

## Planning notes

- New research direction → planner does the literature review first (probing classifiers,
  "Future Lens"-style prediction of downstream states from hidden activations, activation
  forecasting, RepE monitoring) and formalizes R1/R2/R3 against the context-geometry theory
  (A2/A3/A4) before any code.
- Single-variable-change discipline: R1/R2 each change ONLY the projection input vs PV's
  `⟨c_last, r_B⟩`; vectors, eval rig, judge reused verbatim. The direct predictor is the
  matched-capacity control that isolates the map's contribution.
