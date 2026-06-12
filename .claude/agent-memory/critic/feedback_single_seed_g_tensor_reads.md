---
name: single-seed-g-tensor-reads
description: At 1 seed/cell, G-tensor reads split into K-draw-robust (column main effects) vs 1-2-draw-fragile (per-cell, pairwise asymmetry, 1v1 inoculation contrasts); the diagonal-strength covariate shares the adapter draw AND question pool with the DV, inflating "+strength" credit
type: feedback
---

When a tensor plan descopes to ONE training seed per cell (user-overridden floor), training-draw noise is not removable — but reads differ sharply in exposure:

1. **Robust (~K independent draws):** eval-context COLUMN main effects average over the K adapters per row (each an independent training run), and replicate across behavior rows. The strongest free rebuttal to "your structure is run-to-run jitter" is the row-effect/column-effect/interaction decomposition: high cross-adapter correlation of row-mean-removed eval-context profiles ⇒ idiosyncratic draws cannot carry the structure; the unshared residual upper-bounds (training noise + true interaction).
2. **Fragile (1–2 draws):** per-cell values, pairwise antisymmetry A[i,j]=½(G[i→j]−G[j→i]) (mixes two draws; a strength draw δ_i produces exactly s_i−s_j-form antisymmetry), and inoculation-style 1v1 adapter contrasts (F7-trained vs default-trained). Narrate these at cross-row-consistency level, and report residual-after-strength anti-fraction, never the raw fraction.
3. **Shared-draw covariate inflation:** diagonal implant strength s_i is measured on the SAME adapter (and usually the same question pool) as row i's G cells, so "+strength" ladder rungs and (s_i−s_j) regressors carry shared-noise credit that leave-contexts-out CV does NOT remove (folds split contexts, not draws). Prescription: split-half cross-fit (s from question half A, G scored on half B — free if per-question/per-verdict data ship) kills the shared MEASUREMENT-noise part; the shared training-draw part stays under the named single-seed caveat. Also: errors-in-variables attenuation of the noisy strength regressor can leak partial-R² to a correlated theory regressor in joint regressions — report s_i reliability alongside.
4. The one decisive cheap control (a 1-cell second-seed probe, ~1.5 GPU-h) may be BARRED when the directive fixes the seed count MUST-ASK in either direction — then it is the named follow-up ask, not an in-plan Must-Fix.

(Direction-of-bias of fixed-pool question-noise floors — conservative for between-context variance tests — is covered in [[single-seed-rebase-and-fixed-pool-floor]].)

**Why:** Surfaced on #537 plan v6 (Alternatives lens, 2026-06-10): user-directed descope 184→84 adapters, all estimators re-based on question-level noise; plan named the global caveat but not the robust-vs-fragile read gradient or the shared-draw covariate path.

**How to apply:** Single-seed (or seed-descoped) tensor plans: check the plan names the caveat (honesty), then route items 1–3 as analyzer concerns with the concrete decomposition/cross-fit prescriptions. REVISE only if per-question data do NOT ship (then nothing is recoverable post hoc). Related: [[antisym-fraction-noise-floor]] (the ≥2-seed seed-split form), [[rank1-mechanism-test-confounds]].
