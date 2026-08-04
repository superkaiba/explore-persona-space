---
name: gcv-lambda-interpolation-degeneracy
description: GCV ridge lambda-selection collapses to grid-min on near-interpolable grouped cells (n_tr ≲ D, near-duplicate within-group rows) — held-out R² −2..−46 with healthy data; fix = inner GROUP-CV selection, selection-symmetric for nulls
type: feedback
---

GCV λ-selection on Gram-ridge (`fit825._ridge_predict_cached`) picks the grid-min λ whenever the
train fold is near-interpolable: n_tr ≲ D AND near-duplicate within-group rows (nested-prefix
turns of a scene, one-line Q&A at n≈D). Train RSS collapses to ~0 (#1335 fold-0: RSS(0.01)=3.6 vs
tot=1.4e6 → GCV 658 beats every honest λ ≥ 972); held-out R² lands −2…−46 while λ=1e3–1e4 on the
SAME folds reads +0.2…+0.4.

**Why:** GCV assumes i.i.d. rows; within-group-correlated rows make memorized group structure look
like generalization. A df/n cap does NOT fix it (the guarded second minimum still reads −0.9…−0.2).

**Diagnostic signature:** per-group fits catastrophically negative while POOLED fits (n_tr >> D) on
the same store are sane; early layers positive, deep layers −5-ish, ALL X-arms failing together;
`collect_lambdas=True` shows λ = grid-min on every fold. Diagnose in minutes with a fixed-λ grid
re-fit on CPU from persisted stores — BEFORE hunting data/seed/pairing corruption (#1335 r8 burned
a full attempt on the wrong hypothesis space).

**Fix:** `heldout_r2_sweep(..., lambda_selection="inner-group-cv")` (#1335 `da31ac154d`,
`scripts/issue825_fit_cells.py`): λ per (layer, outer fold) by summed inner-validation RSS over 4
GROUP-level inner folds, reduced form (no per-λ prediction materialization), batched twin for the
null path so nulls get the same selection opportunity (selection-symmetric). Where GCV is healthy
(n_tr >> D, row-level groups) the two selectors agree exactly (r0: 0.4103 @ λ=3162 both).

**How to apply:** any GCV/LOO-style λ or hyperparameter selection over GROUPED activation data with
per-group n_tr near or below D — check the selected λ against the grid edge and run the inner-CV
mode; pinned class test: `tests/test_issue1335_lambda_selection.py` (real-slice fixture).

**#1345 recurrence — the READ-SIDE fix when the published numbers are already the artifact.**
Same mechanism at n_train 1730 < d 3584 (floor λ=0.01 in 5/5 folds on every full-basis story-input
leg), but discovered AFTER a whole line of results had been published off the ambient GCV read: story
context → story answer −0.306 ambient vs **+0.408** at forced λ=1e3 and **+0.262** in a reduced PCA
basis, same rows/basis. When that happens, inner-group-CV alone is not enough — the DOWNSTREAM
verdict has to stop treating the ambient value as the map's information content:

- Attach TWO companions to every cell/grid point: a **reduced-PCA-basis** read (`k = min(1024,
  floor(n_train_min/2), d_in)`, basis fit on TRAIN rows only per fold — well-posed since n_train > k)
  and a **forced-λ sweep** ({1e2, 1e3, 1e4}), with **kNN-through-the-map per λ** — retrieval and R²
  DISSOCIATE across λ (knn@1 falls as R² rises), so neither read alone characterizes the map.
- MOVE the primary within-R² to the reduced-basis value; keep the ambient number as an explicitly
  labelled `*_ambient_gcv_continuity` field, and flag any published anchor built on it
  `anchor_is_artifact_bearing: true` — a parity check that the pipeline reproduces, never a science
  reference. Downstream readers will otherwise keep citing it.
- Cost is near-zero if structured right: ONE `fc._prep_fold` per (X, fold) serves the GCV read AND
  every forced λ (λ is a diagonal rescaling of the cached eigh), and the train-only PCA basis depends
  ONLY on X — so compute both once per fold and reuse across every Y target
  (`companions_shared_x`, `scripts/issue1345_boundary_ablation_fits.py` @ `4682f0247a`).
- **Extend the resume fingerprint** with the companion knobs (`reduced_k_cap`, `forced_lambdas`) and
  refuse to resume a cell JSON lacking the companion block — otherwise a k-rule change silently
  reuses stale companions.

**Whitening caveat (open tension, do not silently "fix"):** the reduced-basis recipe feeds PCA
coordinates into `_prep_fold`, which standardizes them per-dim — i.e. it DOES whiten the PCA coords,
which [[post-pca-whitening-kills-oof-generalization]] (#923) warns against. In THIS regime it still
beat ambient (+0.262 vs −0.306), so match the reference recipe for comparability and record the
tension rather than deviating unilaterally.

**Verify a reused recipe by EQUIVALENCE, not by prose.** When copying a sibling round's estimator so
your numbers are comparable to its published values, import its actual functions and assert agreement
on identical inputs — #1345 checked its companions against the probe's own `ridge_leg_reduced` +
forced-grid `ridge_leg` (`scripts/issue1345_story_info_probe.py` @ `3ffc51d581`) and matched to
|Δ| < 1e-9 across all four legs. That converts "comparable by assumption" into "comparable by
construction" for the cost of one ~20-line check, and it is the same discipline
[[capture-convention-read-the-producer-code]] applies to stored-array conventions.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [GCV lambda interpolation degeneracy](feedback_gcv_lambda_interpolation_degeneracy.md) — n_tr ≲ D: GCV → grid-min λ, R² −2..−46; inner-group-CV fix (#1335); reduced-basis + forced-λ companions (#1345)
