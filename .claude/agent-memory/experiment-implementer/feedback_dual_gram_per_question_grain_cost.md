---
name: feedback_dual_gram_per_question_grain_cost
description: The #825/#931/#1310 GCV-ridge fit machinery uses the DUAL Gram (n×n), so wall-time scales with n² — running the same battery at per-row/per-question grain instead of per-scenario-aggregated is ~(n_big/n_small)² heavier, easily turning a minutes-long round into hours
metadata:
  type: feedback
---

The shared map-similarity fit machinery (`issue825_fit_cells._prep_fold`,
`issue825_crossmodel_map_transfer.fit_primal_beta`, `issue825_map_alignment._ridge_prep`)
forms the Gram in the SAMPLE dimension: `G = Xtr_n @ Xtr_n.T` (n×n), then eigh(G).
So per-fold cost scales with **n_train²·d (Gram) + n_train³ (eigh)** — DUAL/kernel
form, cheap when n ≪ d, ruinous when n is large.

**Why it bites:** rounds 1/2 of the #1310 cross-persona battery ran at the
scene-AGGREGATED grain (one point per persona×scenario ⇒ n=300) and finished in
minutes. Round 3 (assistant direct test, #1310) ran the SAME v2 battery at the
per-QUESTION grain (n=4045 intersection across the #1335 cell trio) — ~13.5×
larger n ⇒ ~**180× heavier per fit** (n² for Gram formation). Measured on base
(2026-07-22): per-cell fold prep+predict 3.2s, pooled 3-cell fold 67s. Projected:
transfer ~9.5min/model, decomposition ~39min/model (the M0/M1/M2 pooled lattice
at n_pooled≈9768), the operator leg alone >9min. **Whole battery ≈ 2.5-3h across
both models**, vs the "quick inline round" rounds 1/2 were.

**How to apply:**
- Before running ANY of the #825/#931/#1310 fit battery on a NEW dataset, size
  n and compute n²/n_prior² — if n jumps an order of magnitude vs the parent
  run, the wall-time jumps ~2 orders. TIME ONE `_prep_fold` at production n
  first (`--probe` pattern), never assume it's "comparable to the parent round".
- At n>1e3 per cell the battery is a detached, per-leg-checkpointed, choom-
  protected job (SKILL.md § Detached VM-side long compute phases), NOT in-turn
  foreground work — even the "cheap" operator leg exceeds the 10-min tool cap.
- The redundant cost lever: the decomposition M1 shuffle nulls re-fit the
  IDENTICAL pooled X (only Y permuted) — `v2._pooled_fold_preds` re-preps the
  eigh per null draw. Hoisting `_prep_fold` out of the null loop (the X-only
  cache is bit-identical across draws) cuts decomposition ~6× on the M1+nulls
  portion (~39→~11min/model), science-identical. Worth doing before a >1h launch
  (`.claude/rules/vectorize-many-cell-fits.md` § shared-factorization reuse).

**Reuse pattern that worked (#1310 round 3):** to run the v2 battery
(`issue1310_xpersona_similarity_v2.py`) on a new cell trio, point the reused
modules' module-level cell list at the new cells at runtime
(`v1.PERSONAS = CELLS; v2.PERSONAS = CELLS`) — the stat primitives
(`_pooled_fold_preds`, `run_operator_nulled`, `run_pred_similarity`,
`transfer_cell`, `run_reparam`) then iterate the new cells verbatim. The ONLY
adaptation was swapping the #1310 equality gate (diagonal reproduces committed
#1310 within-cells at ≤1e-6) for a band-check vs the new dataset's committed
references, since the new cells have no #1310-format committed within-cell.
Driver: `scripts/issue1310_xpersona_assistant_test.py`.

**Staging:** use the canonical `hub.stage_hub_prefix(repo, prefix, dest_dir,
revision=...)` (#1402/#833: server-side scoped listing + retried per-file
hf_hub_download pool, one pinned revision) — a bare `list_repo_tree` +
`hf_hub_download` loop fails the inline-lint-gate hub-verify-retry +
live-hf-retry-routing checks. Files land at `dest_dir/<repo-relative path>`
(verbatim prefix mirror).
