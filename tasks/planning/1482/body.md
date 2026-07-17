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
  via the project judge; (2) using public Qwen2.5-7B batchtopk SAEs (16,384 features,
  layers 18/24 nearest the map layer), map SAE features in the context to the answer
  and identify the worst-predicted answer-side SAE features, with interpretations
  of the worst tail; (3) decompose the measured linear-vs-nonlinear gap (ridge test
  R2 0.754 vs MLP 0.810-0.813) per-context and per-feature/direction — which parts
  are predictable nonlinearly but not linearly, and conversely. Reuse the n1M captures
  (HF issue779_monitoring/fitter-fair-comparison-n1m/), the pinned split, and the
  issue-779-n1m branch fitters; refits recompute per-context residuals and must reconcile
  to the committed aggregate R2. Both mapping arms (prefix-based and context-based)
  for any newly fit map; a context-arm-only read of the existing n1M map is an explicit
  stated deviation.'
relates_to:
- spec-context-as-vector
---
# Error analysis of the n1M context→answer map: worst-predicted contexts, worst-predicted SAE features, and the linear-vs-nonlinear gap

## Goal

On the #779 fitter-fair-comparison-n1m mapping h: c_last(x) -> v(x) (last-prompt-token activation -> mean-response activation profile, layer 19, Qwen-2.5-7B-Instruct, ~963k LMSYS+WildChat train contexts, pinned val 400 / test 1000 split), characterize what the map is bad at predicting: (1) rank held-out contexts by per-context prediction error and categorize the worst tail (corpus source, language, topic, length, refusal-adjacency) via the project judge; (2) using public Qwen2.5-7B batchtopk SAEs (16,384 features, layers 18/24 nearest the map layer), map SAE features in the context to the answer and identify the worst-predicted answer-side SAE features, with interpretations of the worst tail; (3) decompose the measured linear-vs-nonlinear gap (ridge test R2 0.754 vs MLP 0.810-0.813) per-context and per-feature/direction — which parts are predictable nonlinearly but not linearly, and conversely. Reuse the n1M captures (HF issue779_monitoring/fitter-fair-comparison-n1m/), the pinned split, and the issue-779-n1m branch fitters; refits recompute per-context residuals and must reconcile to the committed aggregate R2. Both mapping arms (prefix-based and context-based) for any newly fit map; a context-arm-only read of the existing n1M map is an explicit stated deviation.

1. **Worst-predicted contexts.** Rank held-out contexts by per-context prediction error (per-context residual MSE / cosine(v̂, v) / per-context R² share under the round's variance-weighted metric) and categorize the worst tail vs the best tail (source corpus LMSYS-vs-WildChat, language, topic/category, context length, conversation depth, refusal-adjacent content, format). Deliverable: an error taxonomy with per-category error distributions and a labeled worst-K / best-K exhibit (digest-only handling for harmful rows).
2. **Worst-predicted SAE features.** Using public Qwen2.5-7B SAEs (batchtopk, 16,384 features, layers 6/12/18/24, resid_pre + resid_post — `nikoryagin/`, `elephantmipt/` on HF), map SAE features of the context to the answer and identify which ANSWER-side SAE features are worst predicted: encode the actual answer profile v(x) and the predicted v̂(x) (and/or fit a map from context SAE features → answer SAE features), compute per-feature prediction quality across held-out contexts, rank the worst/best-predicted features, and interpret the worst tail (top-activating examples). The exact operationalization (encode-the-prediction vs SAE-input map, and mean-profile-encoding vs per-token) is a plan-time decision the planner must pin down with the base-SAE-on-instruct-activations and token-SAE-on-mean-pooled-state caveats addressed explicitly.
3. **The linear-vs-nonlinear gap.** The n1M round measured whole-map test R²: ridge 0.754 vs MLP-w8192 0.810 / MLP-w32768 0.813 / residual_skip / KRR-Nyström (n1m_fits.json). Decompose this ~0.06 gap per-context and per-feature/direction: which contexts (and which SAE features / answer-profile directions) are predictable by the nonlinear fitters but NOT by ridge — and, conversely, whether any set is predictable linearly but not nonlinearly (the user's literal phrasing; both directions of the gap are reported). Deliverable: the error-taxonomy of axis 1 crossed with fitter family, i.e. whether the nonlinear-only component is concentrated in identifiable context categories / SAE features or diffuse.

An answer looks like: "the map's error mass is concentrated in categories X/Y (e.g. non-English, long multi-turn, refusal-adjacent); SAE features f₁…f_k (interpretations attached) carry the worst per-feature prediction; the nonlinear-only R² component is concentrated in Z (or diffuse)." Competing hypotheses to distinguish: error concentrated in identifiable context categories vs diffuse noise-floor error; nonlinear gap concentrated in rare/tail features vs spread uniformly; worst-predicted SAE features being low-frequency/rare features vs high-frequency structural ones.

## Provenance

- workflow origin: user chat 2026-07-17 (interactive), routed as a NEW-direction child of #779 per the question-identity litmus (error-structure characterization + SAE surface is a new question; would not merely rewrite #779's monitoring Takeaways).
- Verbatim originating prompts: see `origin_prompt` frontmatter (two chat messages).

## Motivation

From the planned-experiments doc (subsection "What Is This Mapping Bad at Predicting?"): analysis/categorization of answers for which predictions are worst, and analysis of worst-predicted SAE features (Cunningham et al. 2023, arXiv 2309.08600; Bricken et al. 2023; Templeton et al. 2024). Knowing where the context→answer map fails tells us where a pre-generation behavior predictor built on it cannot be trusted, and whether the failure is structured (fixable, category-specific) or diffuse (a capacity/noise floor). The linear-vs-nonlinear decomposition additionally tells us whether the theory paper's linear-map assumption discards a structured component.

## Reuse (existing artifacts — reuse checklist (a)-(k) applies at plan time)

- **Fits + aggregates:** `eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json` (committed; fit commit `bd9f6865de16f89daf08de4b800f65c30d4079a8`). Aggregates only — NO per-context predictions stored, so refits are required for per-context residuals; they are cheap (ridge 56 s streaming; MLP w8192 ~8 min; w32768 ~18 min on one GPU at n≈963K per fit_meta wall times).
- **Captures (map inputs + targets):** HF data repo `superkaiba1/explore-persona-space-data` under `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/` (streamed c_last + v(x) chunk shards, layer 19, sha-verified per #1331) + `sampling_manifest` + `raw_completions/` (rollout text, per-chunk; prompts' text-persistence per the generate script header). n_new_captured 959,844; manifest LMSYS 525,485 / WildChat 434,515.
- **Scripts (UNMERGED branch — parent-lineage rule / reuse check (k) binds):** `origin/issue-779-n1m` → `scripts/issue779_ffc_n1m_fits.py` (streaming primal ridge, minibatched MLPs, residual_skip, KRR-Nyström, val/test sha asserts), `issue779_ffc_n1m_generate_capture.py`, `issue779_ffc_n1m_launch.sh`. Diff against main and port or execute from a worktree on that branch; do NOT rewrite the fitters.
- **Split:** byte-identical val/test from the original round's `fixed_split(5000, 3600, 400, 1000, 42)`; pinned val/test sha256 in n1m_fits.json `split` block. Near-dupe screen (5-gram Jaccard 0.8) already applied to targets.
- **SAEs (external, fitness check at plan time):** `nikoryagin/sae_Qwen_Qwen2.5-7B_resid_post_layer_18_*` / `resid_pre_layer_24_*` (closest to the map's layer 19; also 6/12 for a depth read), batchtopk, 16,384 features, trained on Qwen2.5-7B BASE. Base-SAE-on-instruct-activations transfer must be sanity-checked (reconstruction R² / L0 on our activations) before any headline rests on it; the planner's lit review should also check for better Qwen2.5-7B-Instruct SAEs before committing.

## Methodology constraints (capture-time)

- **Both mapping arms (standing rule).** The n1M map input `c_last` is CONTEXT-based (prefix + user query, last token). Any NEWLY fit map in this task (the SAE-features→answer map; any refit read) is designed with BOTH arms — prefix-based (prefix-end state, everything before the user query) AND context-based — as paired arms. For the n1M map itself the existing captures are context-arm only; the prefix arm there requires new prefix-end captures, which the plan may scope to a held-out subsample or declare as an explicit stated deviation with the reason (capture cost at n≈963K) carried into the clean-result.
- **Held-out size.** The pinned test split is 1,000 contexts (+400 val) — likely too small for a stable error taxonomy tail. The plan should consider enlarging the held-out read (k-fold refits over the 963K pool, or carving a larger fresh holdout that never enters any fit) while keeping the pinned test untouched for comparability with n1m_fits.json.
- **Per-context error metric.** The round's headline is variance-weighted whole-map R²; per-context error needs an explicit definition (residual MSE, cosine, per-context share) with the aggregate reconciling back to the committed 0.754/0.813 numbers as a correctness gate.
- **Categorization labels.** Context categories (topic/language/refusal-adjacency) should be produced by the project judge (`claude-sonnet-4-5-20250929`, Batch API at this volume) or metadata (corpus source, length, turn count) — never substring heuristics for content categories. LMSYS/WildChat raw text is unscreened real user text: digest-only handling in briefs and bodies, no raw item text pasted into reports beyond the standard cherry-picked worked examples with disclosure.
- **Compute sketch (plan refines):** refits + per-context residuals ~1-2 GPU-h; SAE encoding over held-out + a train subsample (activations regenerable from persisted text via teacher-forced forward passes) ~2-8 GPU-h; judge categorization ~0 GPU (Batch API). Expected well under the 20 GPU-h cheap band unless the plan enlarges holdout captures; est. total < 20 GPU-h.

## Relation to siblings

- #779 (parent): owns the n1M map + fitter comparison; this task characterizes its error structure.
- #952/#823: per-token-position predictability of the (5K-context) map — orthogonal granularity; reuse its divergence-bank machinery only if the plan wants a targeted hard-context probe.
- #1092: prefix-vs-context transport result (transport runs through the context state) — grounds the both-arms design expectation.
