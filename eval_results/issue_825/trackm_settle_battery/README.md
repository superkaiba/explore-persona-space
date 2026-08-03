# Track-M linear-collapse settle-it battery — issue #825

Generated 2026-07-25T18:12:34Z · commit 232ad39396 · CPU-only refit battery on the already-persisted turnstores (0 GPU, 0 generation).

Reads yesterday's audit (`eval_results/issue_825/trackm_linear_collapse_audit/README.md`) and answers its three pre-registered splits + one hardening read with numbers.

## Sanity: does the guard move the full-n headline?

- S1@5000 unguarded L19 = **0.673094** (banked 0.673094) — harness reproduces the committed anchor exactly.
- S1@5000 guarded L19 = **0.673094** — guard moved full-n materially: **False**. At n=5000 n_tr=4000 > D=3584, so the fit is outside the degenerate regime and the dof-cap changes nothing; the headline S1 number is unaffected by the guard.

## Split 1 — estimator share (does the guard recover S@2000 + lift M?)

- S1@2000 unguarded L19 = **0.2585** (banked matched_n_curve 0.2585).
- S1@2000 **guarded** L19 = **0.4161** → guard lift **0.1576** toward the power-expected band ~0.45-0.50 (cf 0.482@n=700).
- Per-M-cell guard lift (unguarded → guarded L19):
    - M_instruct_assistant_chat: 0.0757 → 0.5879 (lift 0.5122)
    - M_pretrained_assistant_chat: -0.4606 → 0.4926 (lift 0.9531)
    - M_instruct_user_chat: -1.4272 → 0.2480 (lift 1.6752)
    - M_pretrained_user_chat: -1.4894 → 0.1921 (lift 1.6815)
    - M_instruct_assistant_naturalistic: -0.0784 → 0.5538 (lift 0.6322)
    - M_pretrained_assistant_naturalistic: -0.3897 → 0.5088 (lift 0.8985)
    - M_instruct_user_naturalistic: -1.6051 → 0.2265 (lift 1.8316)
    - M_pretrained_user_naturalistic: -1.6496 → 0.2142 (lift 1.8638)

## Split 2 — decoding+corpus share (residual + filtered-S)

- Residual gap = guarded S@2000 − guarded M_instruct_assistant_chat = 0.4161 − 0.5879 = **-0.1719**.
- M-matched corpus filter on Track S: kept 3927/5000 (exact-dup responses 0.094, <8-content-token prompts 0.190); filtered pool n=3927, subsample n=2000.
- Filtered-S@2000 unguarded L19 = 0.3113 (shift 0.0529 vs unfiltered), **guarded** L19 = 0.5682 (shift **0.1521** vs unfiltered guarded S@2000).
  A negative shift means the filter pulls S DOWN toward M — i.e. part of the old S reference was dup/short-prompt flattery.

## Split 3 — nonlinearity verdict

- MLP@S1@2000 L19 = **0.6277** (n_draws=5) vs guarded-ridge@S2000 0.4161 (Δ MLP−guarded = **0.2117**) vs unguarded-ridge@S2000 0.2585.
- Banked MLP@M: instruct_assistant_chat 0.5575, pretrained_assistant_chat 0.4873.
- Rule: if MLP@S2000 ~ guarded-ridge@S2000 -> 'nonlinearity' was an estimator artifact; if MLP >> guarded ridge only on M -> genuine M-specific nonlinearity

## Split 4 — user-turn hardening (guarded refits)

- M_instruct_user_chat: unguarded L19 -1.4272 → guarded L19 0.2480
- M_pretrained_user_chat: unguarded L19 -1.4894 → guarded L19 0.1921
- M_instruct_user_naturalistic: unguarded L19 -1.6051 → guarded L19 0.2265
- M_pretrained_user_naturalistic: unguarded L19 -1.6496 → guarded L19 0.2142

## Methods / provenance

- Estimators: unguarded = GCV ridge, `GCV_DOF_CAP=None` (committed #825 default); guarded = same GCV ridge with `GCV_DOF_CAP=0.9` (the fit module's own registered dof-cap mitigation for the n_tr<D degeneracy); MLP = `fit_h.mlp_fit_predict` (PCA-64 target head, 1×512 GELU, AdamW, ≤300 epochs early-stop), CPU.
- Grouped conversation-level 5-fold, fold seed 0. All reads at L19 (+ frozen 14/18/26 in results.json).
- Subsample scheme RECOVERED (not guessed): `np.random.default_rng(seed).choice(n_full, 2000, replace=False)`, seeds 1000-1004 — reproduces every banked matched_n_curve n=2000 draw to <1e-4.
- Banked unguarded numbers (matched_n_curve, cells_M_*.json, banked MLP) are CITED, not recomputed, except leg C which needs fresh unguarded filtered fits.
- Turnstores: Track S = local 4-layer map_alignment npz (== the analysis_tensors content matched_n_curve used); Track M = 28-layer `.pt` shards from `issue825_userbase_map/analysis_tensors` @ rev deb7a45.

### Deviations

- **Staging footprint exceeded the 20 GB brief cap.** The 4 Track-M `.pt` turnstores total ~64 GB (instruct/pretrained × chat/naturalistic; chat ~17 GB/store [4×4273 MB shards], naturalistic ~17 GB/store [4×4273 MB shards]); Track-S stores were already local. `/` had 423 GB free, so 64 GB staging was safe (≫1.5× headroom). Staged under `data/issue_825/audit_dl/analysis_tensors/`.
- Realized per-phase wall-times (s): {'refit:S1_full': 250.5, 'refit:S1_2000': 155.9, 'refit:M_instruct_assistant_chat': 67.2, 'refit:M_pretrained_assistant_chat': 173.2, 'refit:M_instruct_user_chat': 236.5, 'refit:M_pretrained_user_chat': 307.2, 'refit:M_instruct_assistant_naturalistic': 84.5, 'refit:M_pretrained_assistant_naturalistic': 152.1, 'refit:M_instruct_user_naturalistic': 181.8, 'refit:M_pretrained_user_naturalistic': 211.0, 'mlp:1000': 51.2, 'mlp:1001': 47.7, 'mlp:1002': 44.4, 'mlp:1003': 46.6, 'mlp:1004': 36.7}

Full numbers + per-draw values: `results.json`.
