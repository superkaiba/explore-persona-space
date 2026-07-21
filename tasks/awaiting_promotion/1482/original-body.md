---
title: 'Error analysis of the n1M context→answer map: worst-predicted contexts, worst-predicted
  SAE features, and the linear-vs-nonlinear gap'
kind: experiment
tags: []
created_at: '2026-07-17T22:10:55Z'
has_clean_result: false
parent_id: 779
origin_prompt: 'Help me to setup and run this experiment: \subsection{What Is This
  Mapping Bad at Predicting?} - Analysis of worst predicted contexts: analysis/categorization
  of answers for which predictions are the worst. - Analysis of worst predicted SAE
  features [cunningham2023sparse, bricken2023monosemanticity, templeton2024scaling]:
  map from SAE features in the context to the answer and find the SAE features which
  are worst predicted by this. Use existing artifacts/trained mappings when possible.
  [second message:] can we also for the 1 million context mapping try to characterize
  which part is predictable linearly but not nonlinearly? and actually we can run
  all these experiments on the 1 million context mapping ideally'
workflow: v1
goal: 'On the #779 fitter-fair-comparison-n1m mapping h: c_last(x) -> v(x) (last-prompt-token
  activation -> mean-response activation profile, layer 19, Qwen-2.5-7B-Instruct,
  ~963k LMSYS+WildChat train contexts, pinned val 400 / test 1000 split), characterize
  what the map is bad at predicting: (1) rank held-out contexts by per-context prediction
  error and categorize the worst tail (corpus source, language, topic, length, refusal-adjacency)
  via the project judge; (2) fit a DIRECT map from SAE features of the input to pooled
  SAE features of the output — encode per-token activations through public Qwen2.5-7B
  batchtopk SAEs (16,384 features; layers 18/24 nearest the map layer), pool feature
  activations over answer tokens (mean activation, MAX activation, + fraction-active),
  fit linear + nonlinear maps, and identify the worst-predicted answer-side SAE features
  with interpretations of the worst tail; (3) decompose the measured linear-vs-nonlinear
  gap (ridge test R2 0.754 vs MLP 0.810-0.813) per-context and per-feature/direction
  — which parts are predictable nonlinearly but not linearly, and conversely — on
  both the dense map and the SAE->SAE map. Reuse the n1M captures, the pinned split,
  and the issue-779-n1m branch fitters; refits recompute per-context residuals and
  must reconcile to the committed aggregate R2. Both mapping arms (prefix-based and
  context-based) for any newly fit map; a context-arm-only read of the existing n1M
  map is an explicit stated deviation.'
relates_to:
- spec-context-as-vector
---
# Error analysis of the n1M context→answer map: worst-predicted contexts, worst-predicted SAE features, and the linear-vs-nonlinear gap

## Goal

On the #779 fitter-fair-comparison-n1m mapping h: c_last(x) -> v(x) (last-prompt-token activation -> mean-response activation profile, layer 19, Qwen-2.5-7B-Instruct, ~963k LMSYS+WildChat train contexts, pinned val 400 / test 1000 split), characterize what the map is bad at predicting: (1) rank held-out contexts by per-context prediction error and categorize the worst tail (corpus source, language, topic, length, refusal-adjacency) via the project judge; (2) fit a DIRECT map from SAE features of the input to pooled SAE features of the output — encode per-token activations through public Qwen2.5-7B batchtopk SAEs (16,384 features; layers 18/24 nearest the map layer), pool feature activations over answer tokens (mean activation, MAX activation, + fraction-active), fit linear + nonlinear maps, and identify the worst-predicted answer-side SAE features with interpretations of the worst tail; (3) decompose the measured linear-vs-nonlinear gap (ridge test R2 0.754 vs MLP 0.810-0.813) per-context and per-feature/direction — which parts are predictable nonlinearly but not linearly, and conversely — on both the dense map and the SAE->SAE map. Reuse the n1M captures, the pinned split, and the issue-779-n1m branch fitters; refits recompute per-context residuals and must reconcile to the committed aggregate R2. Both mapping arms (prefix-based and context-based) for any newly fit map; a context-arm-only read of the existing n1M map is an explicit stated deviation.

1. **Worst-predicted contexts.** Rank held-out contexts by per-context prediction error (per-context residual MSE / cosine(v̂, v) / per-context R² share under the round's variance-weighted metric) and categorize the worst tail vs the best tail (source corpus LMSYS-vs-WildChat, language, topic/category, context length, conversation depth, refusal-adjacent content, format). Deliverable: an error taxonomy with per-category error distributions and a labeled worst-K / best-K exhibit (digest-only handling for harmful rows).
2. **Direct SAE→SAE map (worst-predicted SAE features).** Fit a DIRECT map from SAE features in the input to POOLED SAE features over the output (user-specified operationalization, 2026-07-17): encode context per-token activations and answer per-token activations through the same public Qwen2.5-7B batchtopk SAE (16,384 features, layers 6/12/18/24, resid_pre + resid_post — `nikoryagin/`, `elephantmipt/` on HF; layer 18/24 nearest the map's L19); input features at the last context token AND pooled over context tokens (paired with the prefix-based arm per the standing rule); output = per-token answer SAE features POOLED over answer tokens (three pooling variants: mean feature activation; MAX feature activation over answer tokens — user-added 2026-07-17, the sparse-event-preserving read, a feature firing on a few tokens survives max but washes out in mean; and fraction-of-tokens-active as a sparsity-respecting alternative). Fit linear (ridge) + nonlinear (MLP) SAE→SAE maps; per-feature held-out prediction quality (per-feature R² / rank correlation across contexts); rank worst/best-predicted answer-side features; interpret the worst tail via top-activating examples. Secondary comparison read: encode the dense prediction v̂(x) from the existing n1M map through the answer-side SAE and compare its per-feature error profile with the direct SAE→SAE map (does the dense map and the feature-space map fail on the same features?). Pooling per-token SAE features (rather than encoding the mean profile v(x) through a per-token SAE) is the preferred form — it keeps the SAE on-distribution; the base-SAE-on-instruct-activations transfer caveat still binds and gets a reconstruction-R²/L0 fitness gate before any headline.
3. **The linear-vs-nonlinear gap.** The n1M round measured whole-map test R²: ridge 0.754 vs MLP-w8192 0.810 / MLP-w32768 0.813 / residual_skip / KRR-Nyström (n1m_fits.json). Decompose this ~0.06 gap per-context and per-feature/direction: which contexts (and which SAE features / answer-profile directions) are predictable by the nonlinear fitters but NOT by ridge — and, conversely, whether any set is predictable linearly but not nonlinearly (the user's literal phrasing; both directions of the gap are reported). Deliverable: the error-taxonomy of axis 1 crossed with fitter family, i.e. whether the nonlinear-only component is concentrated in identifiable context categories / SAE features or diffuse. Applies to BOTH the dense map and the direct SAE→SAE map of axis 2.

An answer looks like: "the map's error mass is concentrated in categories X/Y (e.g. non-English, long multi-turn, refusal-adjacent); SAE features f₁…f_k (interpretations attached) carry the worst per-feature prediction; the nonlinear-only R² component is concentrated in Z (or diffuse)." Competing hypotheses to distinguish: error concentrated in identifiable context categories vs diffuse noise-floor error; nonlinear gap concentrated in rare/tail features vs spread uniformly; worst-predicted SAE features being low-frequency/rare features vs high-frequency structural ones.

## Provenance

- workflow origin: user chat 2026-07-17 (interactive), routed as a NEW-direction child of #779 per the question-identity litmus (error-structure characterization + SAE surface is a new question; would not merely rewrite #779's monitoring Takeaways).
- Verbatim originating prompts: see `origin_prompt` frontmatter (two chat messages). Third chat message (2026-07-17, folded in pre-planning): "can we do some kind of direct mapping from SAE features in input to output? (maybe pooled SAE features over output)" — this pins axis 2's operationalization to the direct SAE→SAE map with pooled output features. Fourth chat message (2026-07-17): "can we also try max pool over SAE features in answer?" — adds max pooling as a third output-pooling variant.

## Motivation

From the planned-experiments doc (subsection "What Is This Mapping Bad at Predicting?"): analysis/categorization of answers for which predictions are worst, and analysis of worst-predicted SAE features (Cunningham et al. 2023, arXiv 2309.08600; Bricken et al. 2023; Templeton et al. 2024). Knowing where the context→answer map fails tells us where a pre-generation behavior predictor built on it cannot be trusted, and whether the failure is structured (fixable, category-specific) or diffuse (a capacity/noise floor). The linear-vs-nonlinear decomposition additionally tells us whether the theory paper's linear-map assumption discards a structured component. The direct SAE→SAE map additionally gives the context-condition overlap read the theory paper names ("shared SAE features across the sampled contexts") a concrete measurement surface.

## Reuse (existing artifacts — reuse checklist (a)-(k) applies at plan time)

- **Fits + aggregates:** `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json` (committed; fit commit `bd9f6865de16f89daf08de4b800f65c30d4079a8`). Aggregates only — NO per-context predictions stored, so refits are required for per-context residuals; they are cheap (ridge 56 s streaming; MLP w8192 ~8 min; w32768 ~18 min on one GPU at n≈963K per fit_meta wall times).
- **Captures (map inputs + targets):** HF data repo `superkaiba1/explore-persona-space-data` under `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/` (streamed c_last + v(x) chunk shards, layer 19, sha-verified per #1331) + `sampling_manifest` + `raw_completions/` (rollout text, per-chunk; prompts' text-persistence per the generate script header). n_new_captured 959,844; manifest LMSYS 525,485 / WildChat 434,515. NOTE: per-token activation grids were never materialized — the SAE arms regenerate per-token activations from the persisted prompt+rollout text via teacher-forced forward passes on a plan-sized subsample.
- **Per-context recon machinery (0-GPU precedent):** `eval_results/issue_779/percontext_recon.json` (held-out recon of the same h map; aggregates only) — raw per-context predicted-vs-actual is 0-GPU-recomputable via `scripts/issue779_heldout_recon_scatter.py` on `scripts/issue779_percontext_recon.py`'s dual-ridge protocol (per-(context,dim) and per-context projection). Caveat: that line uses trait-specific read-out layers (evil 14 / sycophancy 26 / hallucination 17); the n1M round pins layer 19 — do not mix.
- **Per-direction machinery (axis 3):** `eval_results/issue_779/fitter-fair-comparison/{fair_comparison.json, perdirection_per_predictor.json, reliability_by_direction.json, scaling_curves.json, layer_target_heatmap.json}` — the original round's per-direction per-predictor reads to extend rather than reinvent.
- **All-layer v(x) store:** HF `issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt` (~6 GB, v(x) per context at ALL layers, earlier-round store; NOT staged locally) — useful for SAE-layer alignment reads without fresh capture where its context set suffices.
- **Library module:** `src/explore_persona_space/experiments/issue_779/fit_h.py` (+ `metrics.py`) — the in-repo fit/metric implementations.
- **Methods survey (strongest starting doc):** `docs/ideas/2026-07-06-context-answer-map-analyses.md` — a 4-agent methods survey of exactly this analysis family (DMD/Koopman/RRR framings, validity gates, and a "first battery, 0-GPU on existing #779/#813 stores"); note it pins its W at L14/d=3584 from the earlier line — reconcile with the n1M L19 pin.
- **Scripts (UNMERGED branch — parent-lineage rule / reuse check (k) binds):** `origin/issue-779-n1m` → `scripts/issue779_ffc_n1m_fits.py` (streaming primal ridge, minibatched MLPs, residual_skip, KRR-Nyström, val/test sha asserts), `issue779_ffc_n1m_generate_capture.py`, `issue779_ffc_n1m_launch.sh`. Diff against main and port or execute from a worktree on that branch; do NOT rewrite the fitters.
- **Split:** byte-identical val/test from the original round's `fixed_split(5000, 3600, 400, 1000, 42)`; pinned val/test sha256 in n1m_fits.json `split` block. Near-dupe screen (5-gram Jaccard 0.8) already applied to targets.
- **SAEs (external — the repo has NO SAE infrastructure; this half is greenfield wiring):** no `sae_lens`/`transformer_lens`/SAE-loading code anywhere in src/ or scripts/ (the old `run_persona_composition.py` "SAE" is unrelated Gemma dictionary-learning). The implementer wires a minimal batchtopk SAE loader (encoder/decoder + batchtopk activation) for `nikoryagin/sae_Qwen_Qwen2.5-7B_resid_post_layer_18_*` / `resid_pre_layer_24_*` (also 6/12 for a depth read), 16,384 features, trained on Qwen2.5-7B BASE. Fitness gate before any headline: reconstruction R² / L0 on OUR instruct-model activations; the planner's lit review should also check for better Qwen2.5-7B-Instruct SAEs before committing.

## Methodology constraints (capture-time)

- **Both mapping arms (standing rule).** The n1M map input `c_last` is CONTEXT-based (prefix + user query, last token). Any NEWLY fit map in this task (the direct SAE→SAE map; any refit read) is designed with BOTH arms — prefix-based (prefix-end state / SAE features pooled over prefix tokens) AND context-based — as paired arms. For the n1M map itself the existing captures are context-arm only; the prefix arm there requires new prefix-end captures, which the plan may scope to a held-out subsample or declare as an explicit stated deviation with the reason (capture cost at n≈963K) carried into the clean-result.
- **Held-out size.** The pinned test split is 1,000 contexts (+400 val) — likely too small for a stable error taxonomy tail. The plan should consider enlarging the held-out read (k-fold refits over the 963K pool, or carving a larger fresh holdout that never enters any fit) while keeping the pinned test untouched for comparability with n1m_fits.json.
- **Per-context error metric.** The round's headline is variance-weighted whole-map R²; per-context error needs an explicit definition (residual MSE, cosine, per-context share) with the aggregate reconciling back to the committed 0.754/0.813 numbers as a correctness gate.
- **Categorization labels.** Context categories (topic/language/refusal-adjacency) should be produced by the project judge (`claude-sonnet-4-5-20250929`, Batch API at this volume) or metadata (corpus source, length, turn count) — never substring heuristics for content categories. LMSYS/WildChat raw text is unscreened real user text: digest-only handling in briefs and bodies, no raw item text pasted into reports beyond the standard cherry-picked worked examples with disclosure.
- **SAE-arm distribution discipline.** SAEs are per-token models: apply them to per-token activations and pool FEATURES (mean activation; max activation; fraction-of-tokens-active), not to mean-pooled dense states, except as an explicitly-labeled secondary read (encoding v(x)/v̂(x) through the SAE for the dense-map comparison). Both the base-vs-instruct and the pooling choices are named in the plan and carried as scope caveats.
- **Compute sketch (plan refines):** refits + per-context residuals ~1-2 GPU-h; teacher-forced per-token capture + SAE encoding over held-out + a train subsample for the SAE→SAE fits ~3-10 GPU-h (subsample sized at plan time; activations regenerable from persisted text); judge categorization ~0 GPU (Batch API). Expected under the 20 GPU-h cheap band; est. total < 20 GPU-h.

## Relation to siblings

- #779 (parent): owns the n1M map + fitter comparison; this task characterizes its error structure.
- #952/#823: per-token-position predictability of the (5K-context) map — orthogonal granularity; reuse its divergence-bank machinery only if the plan wants a targeted hard-context probe.
- #1092: prefix-vs-context transport result (transport runs through the context state) — grounds the both-arms design expectation.
- No existing `docs/open_questions.md` anchor covers mapping error-analysis/SAE (nearest: `q:leak-predictor`); this task opens a fresh sub-question — the living-docs-updater proposes the anchor at completion.
