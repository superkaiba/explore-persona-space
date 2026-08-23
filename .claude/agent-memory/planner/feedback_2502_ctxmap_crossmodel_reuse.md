---
name: 2502 ctxmap cross-model reuse
description: "#2502 context→answer map (M_{C,A}) cross-model reuse base: #779 fit_ridge_primal core + cx_last/v_x field convention; #722 mapping_baselines (identity_bias_predict + knn_retrieval); #1739 corpus_staging streaming+dedup; #2378 (unmerged) Qwen3.5-era capture pattern; Qwen3.5-9B config VERIFIED (32L/4096H, 8 full-attn layers [3,7,11,15,19,23,27,31])"
type: feedback
---

Reuse base for any context→answer activation-map round on the #779 line
(fit `M: cx_last→v_x`, held-out R² + mandatory baselines):

- **Fit core:** `scripts/issue779_ffc_n50k_fits.py::fit_ridge_primal` (L427) /
  `_ridge_primal_multi_lambda` (L399) — exact PRIMAL ridge, one eigh of (H,H)
  XᵀX batched over val-λ grid, fp64, `--device cpu` default. Field convention:
  `X = cx_last` (context last-token hidden at layer ℓ), `Y = v_x`
  (answer-span mean = response-avg). #779 plateau R² ~0.73–0.75 at n≥10k–50k,
  layer 19/20 (`eval_results/issue_779/layer_sweep.json` selected_layer:20).
- **Mandatory baselines (#722):** `src/explore_persona_space/analysis/mapping_baselines.py`
  — `identity_bias_predict` (L28; requires d_in==d_out — APPLICABLE within a
  model since cx_last and v_x share the model hidden dim; INAPPLICABLE
  cross-model) + `knn_retrieval` (L63; euclidean+cosine, chance=k/n_pool).
  Already implemented — CALL, don't reimplement.
- **Dedup (#1775):** exact + char-5-gram Jaccard≥0.8 + MinHash-64-perm est≥0.6,
  WITHIN+ACROSS sources BEFORE split (the +0.016 leakage class). Machinery:
  `src/explore_persona_space/experiments/issue_1739/corpus_staging.py`
  (`_hf_stream` L312 streaming, `_stream_stage` L208, `minhash_signatures`
  L137, `near_dup_mask` L153; MIN_TEXT_CHARS=16, MINHASH_N_PERM=64, SHINGLE=5).
  Also has stagers for hh-rlhf red-team + toxic-chat.
- **Qwen3.5-era capture (#2378, branch issue-2378 UNMERGED):**
  `scripts/issue2378_capture.py` loads `Qwen3_5ForConditionalGeneration`
  text-only, explicit `.to(device)`, bf16-as-uint16 codec (`_encode_bf16`;
  fp16 overflows Qwen massive activations); `issue2378_gen.py::_assert_chat_template`
  (L319) proves the `enable_thinking=False` empty `<think>\n\n</think>`
  contract + template fingerprint. NOTE #2378 targets Qwen3.6-27B (N_LAYERS=64,
  HIDDEN=5120 at capture L81) → PORT the pattern, re-verify constants.

**Qwen3.5-9B config VERIFIED (live HF config.json, 2026-08-23):** model_type
`qwen3_5`, arch `Qwen3_5ForConditionalGeneration`, hidden 4096, 32 layers,
`layer_types` = full_attention/linear_attention, **8 full-attention layers
[3,7,11,15,19,23,27,31]** — the matched-layer set for any cross-model
comparison against Qwen2.5-7B's 28 uniform-softmax layers.

**Why:** the ctxmap line recurs; re-deriving the field convention / baseline
API / dedup recipe / the 8-full-attn-layer set cost several plan-time reads.
**How to apply:** verify prefixes/symbols still resolve, cite `Source: #779`
(fit), `#722` (baselines), `#1775` (dedup), `#2378 (unmerged)` (capture
pattern) — the inherit fast-path.
