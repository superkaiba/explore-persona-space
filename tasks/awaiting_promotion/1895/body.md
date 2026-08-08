---
title: The context→answer map's predictable subspace and the SAE's representable subspace
  coincide almost entirely at the variance grain (HIGH confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-30T20:04:08Z'
has_clean_result: true
parent_id: 1482
origin_prompt: 'can we run these experiments: - Is what oir model able to reconstruct
  the same as what SAEs find - Map from continuous features to SAE features (or reconstructed
  activation)? Also compare to simple baselines with SAE featurs'
workflow: v1
goal: 'Does the context→answer map predict the same structure the SAE represents?
  Two ways of decomposing the same answer activation have been measured separately
  and never compared: the map''s **predictable subspace** (which directions of `v_A`
  the fitted map recovers) and the SAE''s **representable subspace** (which directions
  its dictionary reconstructs). This task asks whether they coincide, and whether
  targeting the SAE''s reconstruction instead of the raw activation changes what the
  map can do.'
relates_to:
- spec-context-as-vector
---
# The context→answer map's predictable subspace and the SAE's representable subspace coincide almost entirely at the variance grain (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1895.md](https://github.com/superkaiba/explore-persona-space/blob/0c09eb2a17f4f2d8827fd9225218b3dbcfb57896/docs/methodology/issue_1895.md) · [gist](https://gist.github.com/superkaiba/6f393d5d8b011b585f987eba0da8f173)

## Takeaways

- Overlap between the map's top-64 predictable directions and the SAE's top-64 reconstruction subspace is 0.867; variance-matched rotations already produce 0.845–0.862, so ~98% of the overlap is variance-driven.
- The beyond-variance excess is small and setting-dependent: above every draw at the primary setting, band-clearing at sizes 128–256, weaker at finer shell grain, below-null on the decoder-SVD twin.
- The residual map's above-plug-in excess was a convention artifact: retrained on the pure SAE error (mask-matched capture), it reads 0.280 vs plug-in 0.291, all 10,000 bootstrap draws below zero.
- Targeting the SAE reconstruction is easier by +0.029 held-out R² (0.746 vs 0.717) — almost exactly the variance-profile expectation of 0.743.
- Per-direction, map R² and SAE reconstruction quality correlate at 0.97; controlling variance rank leaves 0.076 (split-half decoupling attenuates it to ~0.05), and the per-feature partial is −0.050 (raw 0.25), i.e. not an independent correlate.

## Goal

**This experiment in context:** [#1482](https://eps.superkaiba.com/tasks/1482) found the context→answer map's per-direction R² decays monotonically with variance rank; [#779](https://eps.superkaiba.com/tasks/779) banked the map; [#1738](https://eps.superkaiba.com/tasks/1738) banked the SAE-space mapping baselines. This experiment asks whether the map's *predictable* subspace and the SAE's *representable* subspace — two decompositions of the same answer activation — coincide beyond what shared variance-dependence forces, whether the SAE reconstruction is an easier target, and whether the map predicts what the SAE misses.

**Broader narrative:** if "linearly predictable from context" and "SAE-representable" are one axis (variance) rather than two, SAE dictionaries add no privileged basis for context→answer prediction, and the map's unpredictable remainder is not where the dictionary's blind spots are.

## Methodology

**Design:** All quantities are deterministic linear-algebra reads over banked Qwen-2.5-7B-Instruct layer-19 activations of real single-turn chats. Objects: `v_C` = final-context-token state; `v_A` = mean answer-token state; `f̄` = banked mean-pooled SAE code; `r̄ = W_dec·f̄ + b_dec` (the pooled SAE reconstruction — by linearity the mean of per-token reconstructions, so on-distribution); `ē = v_A − r̄` (the SAE-residual target); `Q` = eigenbasis of the train-split covariance of `v_A` (all 3,584 directions). `P_pred(k)` = span of the top-k eigendirections ranked by the map's per-direction held-out R² (banked profile primary for k ≤ 64; this run's matched 120k refit for k = 128/256, since the banked profile covers only 256 directions). `P_sae(k)` = top-k principal directions of centered `r̄` on the holdout (primary), with an activity-weighted decoder-SVD twin and the residual-PCA complement. The plan fixed a three-part decision rule before data: (`P_align`) the k=64 overlap exceeds the 97.5th percentile of a 1,000-draw within-shell rotation null at 32 eigenvalue shells; (`P_dark`) the lower 95% bootstrap bound of R²(`v_C`→`ē`) minus its variance-profile plug-in exceeds 0; (`P_var0`) the variance-partialled per-direction correlation's interval includes 0. Outcome cells: shared-structure, orthogonal-decompositions, mixed, variance-trivial. A 512-row teacher-forced pilot gated the target convention: reconstruction identity (median relative deviation 2.4e-3 < 5e-3) and mask-mismatch share (median 1.9e-4 vs SAE-error share 8.8e-2) selected Path A — banked `v_A` kept as target, with the mismatch carried as the caveat the Results section quantifies. Contrastive negatives / persona-vector recipes: not applicable (no behavior implantation, no contrastive direction extraction). Judged SAE feature labels stayed frozen; every read here is mechanical. Follow-up round `pure-residual-path-b` (2026-08-01) changed exactly one thing: the residual target's construction — the S3b mask-matched dense capture ran over all 142,000 rows, and the residual map was retrained and scored on the pure SAE error `ē = v̄_mask − r̄` (the driver re-ran end-to-end with the path forced to B via `--force-path B`; the pilot-gate JSON's `path` field is G2b-derived and does not record the realized path). Split (sha-asserted), λ grid, plug-in construction, 10,000-draw paired bootstrap, seeds, and both baselines identical; the only code change was the override flag. The round's re-computed per-direction correlates block sits in the mask-matched basis and shows an internal sign disagreement between its observed partial (−0.52) and its bootstrap draw interval (0.54 to 0.63), so I do not fold its correlate numbers; the parent-basis correlates below stand. Conciseness note: total Results prose runs over the 800-word budget, and individual result blocks may exceed the 120-word warn cap — eight result sections are retained deliberately, because each positive predicate of the decision rule requires its own sensitivity/purity read plus the low-level per-unit view.

**Training:** N/A — no model training. All fitted maps are ridge or small-MLP analysis fits on frozen activations; complete fit hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Ridge λ grid | logspace(−3, 8, 23), selected on the 2,000-row validation split per target | #1482 `sae_perfeature` protocol |
| Selected λ (raw answer / reconstruction / residual, context arm) | 1000 / 1000 / 3162.3 | `fits_summary.json` |
| Split | 120,000 fit / 2,000 val / 20,000 holdout rows; context-level split, shas asserted against `split_1482.json` | #1482 / #779 |
| Fit well-posedness | n_train = 120,000 ≥ d = 3,584; no GCV anywhere | plan §4 |
| MLP twins | width 8192, lr 3e-4, batch 4096, val early-stop (48 / 47 epochs realized), fit seed 0 | #1482 `mlp_w8192` recipe |
| Angle null battery | 1,000 within-shell rotation draws per cell; shells {16, 32, 64}; subspace sizes {16, 32, 64, 128, 256}; primary k = 64 (fixed in the plan before data) | plan §11; #1345 matched-null conventions |
| Paired bootstrap | 10,000 multinomial draws over holdout contexts, seed 1895; fitted maps held fixed (scoring uncertainty only) | #1482 pdshrink pattern |
| SAE | `andyrdt/saes-qwen2.5-7b-instruct` rev `c37e53c4bb07`, `resid_post_layer_19/trainer_1` (k = 64); k = 128 trainer_2 pilot twin | #1482 Gate-B primary |
| Follow-up round pure-residual-path-b | identical fit recipe on the pure-error target; selected λ (mask-matched raw / reconstruction / pure residual) 3162.3 / 1000 / 3162.3 | round `fits_summary.json` |

**Evaluation:** Four dependent variables. (1) Subspace overlap O(k) = mean cos² principal angle, judged against covariance-matched nulls that Haar-rotate the SAE basis within eigenvalue shells — preserving the variance profile, destroying directional alignment. (2) Paired held-out pooled R² across the three targets on identical rows. (3) The residual target's R² minus a variance-profile plug-in: Σ energy(u)·g(u) / Σ energy(u), with g(u) the matched refit's per-direction R² profile over all 3,584 directions — the exact quantitative statement of "predictability is a function of variance rank alone" (a rotation null is vacuous for pooled R², which is rotation-invariant in the target). The plug-in is a fixed reference value, not a null distribution; the residual read is reported against both 0 and the plug-in, plus a shared-λ error decomposition (additivity relative deviation 5.6e-10; 77.9% of the raw-target map's squared error lies on the SAE-representable component, 23.3% on the residual — which holds only 9.9% of pooled target variance, a ~2.4× error enrichment). (4) Partial Spearman correlations given variance rank (per-direction) and given variance rank + consistency + activity (per-feature), with within-decile stratified reads weighted equally (both variables decay with variance, so the partial alone is not trusted); the exploratory decile family is screened by Benjamini–Hochberg false-discovery control at q = 0.05. A split-half re-derivation (even/odd deterministic holdout halves; constructions built on one half, scored on the other) controls the shared-holdout construction coupling in the overlap and partial-correlation reads. The k = 128 dictionary pilot twin bounds dictionary-size sensitivity: its per-direction reconstruction-quality profile rank-correlates 0.981 with the k = 64 profile on the 512 pilot rows. Both mapping arms ran: the context arm carries every read; the prefix arm is a null by construction on this single-turn corpus (the prefix is one constant template string) — R² ≤ 0.0002 and retrieval at chance for all three targets. Both standing baselines ran for every new fitted map (chance acc@1 = 1/20,000 = 5e-5):

| Target (context arm) | map R² | identity+bias R² | map kNN acc@1 (euclid / cosine) | identity+bias kNN acc@1 (euclid / cosine) |
|---|---|---|---|---|
| raw answer state | 0.717 | −1.02 | 0.694 / 0.702 | 0.330 / 0.356 |
| SAE reconstruction | 0.746 | −1.30 | 0.591 / 0.607 | 0.229 / 0.253 |
| SAE residual | 0.336 | −19.4 | 0.125 / 0.722 | 0.324 / 0.329 |
| pure SAE error (follow-up round) | 0.280 | −20.1 | 0.126 / 0.772 | 0.358 / 0.369 |

On the residual target the two retrieval reads dissociate: identity+bias beats the fitted map on euclidean acc@1 (0.324 vs 0.125) while the map wins on cosine (0.722 vs 0.329) — a scale/shrinkage signature in the fitted residual map's predictions, so the 0.125 euclidean cell reflects mis-scaled magnitudes, not undiscriminative predictions. The same dissociation replicates on the follow-up round's pure-error map (row 4). All bootstrap intervals hold the fitted maps fixed (scoring uncertainty only, not refit uncertainty). Per-corpus splits (LMSYS 0.707 / WildChat 0.722 raw-target R²) are within-corpus reads at parent parity, not a transfer arm. Language-intrusion audit: N/A — no on-policy generation anywhere in the DV (teacher-forced capture only).

**Data extraction:** Corpora: 964,844 real single-turn LMSYS-Chat-1M + WildChat conversations, reassembled from the banked final-token capture chunks with split-sha and top-256 eigenvalue-identity gates that halt the run on mismatch (both passed). The SAE arm uses the banked 142,000-row subset (120k fit / 2k val / 20k holdout, context-level split). Pooled SAE codes come from the banked store (1,920 shards, reference-masked mean over answer tokens); `r̄` is built by one decoder multiply from the stored codes — verified against exact per-token reconstruction on 512 teacher-forced pilot rows (median relative deviation 2.4e-3). The pilot also measured the answer-token-convention mismatch between the banked `v_A` and the pooled-code mask: per-row median 0.02% of total variance (vs 8.8% SAE-error share) → Path A; the pooled tail is much heavier (13.4% of the residual's centered energy on pilot rows), which the matched-rows read in Results quantifies. The harvest-side re-reductions (`dark_spot_matched_rows.json`, `angle_spectrum_k64.json`) recompute from the run's persisted per-context projections and reproduce the committed values to 1e-6 before extending them.

**Sample training/evaluation data + completions:** No model generations were produced (teacher-forced capture only; the underlying conversation text is banked under the capture task's raw-completions buckets and referenced by artifact, not quoted, per content hygiene for real-user corpora). Verbatim evaluation-data samples — the residual-target fit cell (subset: 1 of 8 cells; full file: [fits_summary.json](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/eval_results/issue_1895/fits_summary.json)):

```json
"t_ebar_ctx": {"selected_lambda": 3162.2776601683795, "val_r2": 0.3254089495436613,
               "pooled_r2": 0.33601681685153906, "identity_bias_r2": -19.378161490300744, ...}
```

The follow-up round's pure-error fit cell (subset: 1 of 8 cells; full file: [pure-residual-path-b/fits_summary.json](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e0b8e25ac8160bb80d4f832448b5d2aaa8e15f9/eval_results/issue_1895/pure-residual-path-b/fits_summary.json)):

```json
"t_ebar_ctx": {"selected_lambda": 3162.2776601683795, "val_r2": 0.2779583103239538,
               "pooled_r2": 0.2795261249620792, "identity_bias_r2": -20.129891491095105, ...}
```

The pilot gate verdict (subset: 6 of 17 top-level fields; full file: [pilot_gate.json](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/eval_results/issue_1895/pilot_gate.json)):

```json
{"path": "A", "n_pilot": 512, "g2a_median_reldev": 0.002417054260149598,
 "g2b_m_median": 0.00019414772395975888, "g2b_s_median": 0.08790857344865799,
 "k128_fve_rank_rho": 0.9809259482018847}
```

The machine verdict under the decision rule (full file: [lattice.json](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/eval_results/issue_1895/lattice.json)):

```json
{"q_align": 100.0, "delta_dark_lo": 0.027551833540201187,
 "rho_ci": [0.037670774207744714, 0.09024904796551758],
 "P_align": true, "P_dark": true, "P_var0": false, "verdict": "mixed"}
```

I verified this verdict against the raw summaries: the primary overlap cell reads the 100th percentile of its 32-shell null, the residual read's lower bootstrap bound is 0.0276 > 0, and the partial-correlation interval excludes 0 — the rule fires "mixed" correctly. The follow-up round re-evaluated the rule with the residual predicate on the pure error ([pure-residual-path-b/lattice.json](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e0b8e25ac8160bb80d4f832448b5d2aaa8e15f9/eval_results/issue_1895/pure-residual-path-b/lattice.json)): the residual predicate reads false there (upper bootstrap bound −0.008 < 0) and the round's verdict is shared-structure; only that flip is folded as this round's finding (the file's partial-correlation interval comes from the mask-matched-basis recompute the Design section flags). The Results below qualify the remaining positive predicate with the sensitivity reads the rule does not see.

## Results

### Subspace overlap sits almost entirely at the variance grain, with a small setting-dependent excess

Subspace overlap O(k) — mean squared cosine of principal angles between the map's top-k predictable eigendirections and each top-k SAE subspace — for k = 16–256 on the 20,000-context holdout, with 16/32/64-shell rotation nulls (1,000 draws each).

![Subspace overlap vs k with matched null bands for three SAE subspace constructions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/hero_overlap_ksweep.png)

> **Figure.** *Observed overlap exceeds the variance-matched null band only at some settings.* Observed overlap (black; circles = banked profile, squares = matched 120k refit) vs null bands (2.5th–97.5th percentiles) for the reconstruction-PCA subspace (left), the SAE-residual complement (middle), and the weighted-decoder-SVD twin (right).

Overlap with the reconstruction subspace is high at every k (0.855–0.884), but variance-preserving rotations reproduce nearly all of it: at k = 64 (32 shells) the band spans 0.845–0.862 vs 0.867 observed (unchanged under split-half decoupling: cross-half 0.867–0.875, within-half 0.864–0.876). The observed value exceeds every draw yet sits only ~0.014 over the null median.

The excess also clears the band at k = 128–256, but not at 64 shells for k = 64, not at k = 32, and never on the decoder-SVD twin at 32/64 shells — that twin reads only 0.24–0.33 absolute overlap and sits below its own null at several settings. The residual complement clears its 32-shell band at k = 256 and sits at the band floor at k = 16.

### About 51 of the map's top-64 directions lie inside the SAE's top-64 reconstruction subspace; the last few are nearly orthogonal

Per-angle squared cosines (sorted) between the map's top-64 predictable eigendirections and the top-64 reconstruction-PCA or SAE-residual subspace; the shaded band is the 32-shell null range for the reconstruction MEAN overlap.

![Principal-angle spectrum at k=64 for the reconstruction and residual subspaces](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/angle_spectrum_k64.png)

> **Figure.** *The top-64 overlap splits into ~51 shared directions and a near-orthogonal cliff.* The reconstruction-subspace spectrum (orange) holds 51 angles at cos² ≥ 0.9 (55 at ≥ 0.8) then cliffs to ~0; the residual-subspace spectrum (green) decays smoothly. Values labeled at angles 1, 32, 64.

The aggregate 0.867 decomposes into 51 near-shared directions (cos² ≥ 0.9; 55 at ≥ 0.8) and a cliff: the last ~5 predictable directions are nearly orthogonal (cos² ≤ 0.05) to the SAE's top reconstruction subspace. The map's predictable set sits mostly inside the dictionary's leading span, with a small block that span misses; overlap with the residual subspace shows no shared block.

### The SAE reconstruction is an easier target by exactly its variance profile; the residual's above-profile predictability reverses on the pure SAE error

Left half: held-out pooled R² for the three context-arm targets with variance-profile plug-ins (dashes) and MLP twins (diamonds). Right half: residual-target R² minus its plug-in on the full holdout and the 512 pilot rows (Path-A residual vs pure SAE error; Path A: the banked answer state kept as target).

![Three-target fits with plug-in references and the residual-excess purity reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/three_target_fits_delta_dark.png)

> **Figure.** *The residual's above-plug-in excess disappears on the pure SAE error.* Whiskers: 95% bootstrap intervals, fitted maps held fixed. Residual excess: full holdout +0.034 (0.028 to 0.041); pilot Path-A +0.017 (−0.009 to +0.043); pilot pure −0.089 (−0.141 to −0.045); same fitted map, paired draws.

Reconstruction is easier by +0.029 R² (0.746 vs 0.717); the plug-in already predicts 0.743 — a paired excess of +0.003. MLP twins add +0.032 (reconstruction) and +0.011 (residual) over ridge, ordering unchanged; the plug-in derives from the ridge profile, not a like-for-like MLP reference.

The residual reads +0.034 above its plug-in on the full holdout, yet the same map scores −0.089 against it on the pure SAE error — a paired purity effect of +0.106. The excess traces to the answer-token-convention component (13.4% of residual energy, partly context-predictable), not SAE-missed structure. The pure read here scores a map trained on the Path-A residual, biasing it downward; the follow-up round (next result) ran the pure-trained corpus-wide refit and confirms the reversal.

### Retrained on the pure SAE error, the map reads below its variance-profile plug-in on the full corpus

Left half: held-out pooled R² of the context→pure-error map — trained and scored on `ē = v̄_mask − r̄` from the mask-matched 142,000-row capture — on the full 20,000-row holdout and its 512 pilot-row subset, each against its variance-profile plug-in (dashes). Right half: the 10,000-draw paired bootstrap distribution of R² minus plug-in, with the observed value and zero marked.

![Pure-error map R2 vs plug-in with the paired bootstrap delta distribution](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e0b8e25ac8160bb80d4f832448b5d2aaa8e15f9/figures/issue_1895/pure_residual_delta_dark.png)

> **Figure.** *The pure SAE error is predictable at, not beyond, its variance profile.* Full holdout 0.280 observed vs 0.291 plug-in (paired difference −0.011, interval −0.014 to −0.008); pilot subset 0.281 vs 0.292. All 10,000 paired draws fall below zero.

This retires the residual predicate corpus-wide: the map predicts the pure SAE error slightly below its variance profile — the previous result's +0.034 excess was the answer-token-convention component, not SAE-missed structure. The residual predicate now reads false and the verdict moves from mixed to shared-structure.

Integrity anchors: the reconstruction-target cell reproduces the parent to eight decimals (0.746; same banked-code target); the raw-target cell moves 0.717 to 0.708 because its target is now the mask-matched state. The retired claim is scoped: pure-error predictions stay directionally discriminative (cosine retrieval acc@1 0.77 vs chance 5e-5; identity+bias 0.37) — what fails is variance-relevant structure beyond the profile, not all information.

### The below-profile read comes from the top of the spectrum; the deep tail sits slightly above profile

Each of the 3,584 eigendirections: the plug-in profile g(u) — the raw-target per-direction R² (x) — vs the pure-error map's per-direction R² (y); point area scales with the direction's share of residual energy, color encodes log eigenvalue rank, the dashed line marks equality.

![Per-direction pure-error R2 vs the plug-in profile, sized by residual energy](https://raw.githubusercontent.com/superkaiba/explore-persona-space/1e0b8e25ac8160bb80d4f832448b5d2aaa8e15f9/figures/issue_1895/pure_residual_perdirection.png)

> **Figure.** *High-variance directions fall below the parity line; the deep tail sits slightly above.* The top 300 variance ranks (36% of residual energy) contribute −0.054 to the paired difference, the remaining directions +0.042, net −0.011. Representative ranks labeled.

The aggregate decomposes cleanly: where residual energy concentrates, the pure error is substantially less predictable than the raw-target profile (371 of 3,584 directions fall below parity, concentrated at the top), while the deep tail runs slightly above profile. The map holds no reserve of predictable SAE-missed structure at the top of the spectrum — exactly where the dictionary reconstructs best and its errors are smallest.

### The SAE out-reconstructs the map at the top of the spectrum and collapses in the tail where the map retains signal

Per-eigendirection values vs eigenvalue rank (log scale): map R² for the raw-answer and residual targets, and SAE per-direction reconstruction quality FVE; points are raw directions, lines rolling medians.

![Per-direction map R2 and SAE FVE profiles vs eigenvalue rank](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/profiles_vs_eigrank.png)

> **Figure.** *Both quality profiles decay with variance rank and cross in the deep tail.* SAE FVE (orange) stays near 0.99 over the top ~100 directions, then falls below the map's raw-target R² (blue) in the deep tail; the residual-target R² (green) plateaus near 0.2.

The dictionary is strictly better where variance lives (FVE ~0.99 vs map R² ~0.9 at the top) and collapses toward and below zero past rank ~2,000, while map R² decays smoothly. This shared monotone decay is what the rotation nulls and the plug-in control for — and why controlling it removes most of the apparent coincidence.

### Per-direction, map predictability and SAE reconstruction quality trace one variance-ordered curve

Each of the 3,584 eigendirections: SAE reconstruction quality FVE (x) vs the matched refit's per-direction held-out R² on the raw-answer target (y), colored by eigenvalue rank; representative ranks labeled.

![Per-direction map R2 vs SAE FVE scatter colored by eigenvalue rank](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/perdirection_r2_vs_fve_scatter.png)

> **Figure.** *Map predictability and SAE reconstruction quality form one variance-ordered curve.* The 3,584 directions form one tight monotone curve from rank 1 (top right) to rank 3,584 (bottom left); color encodes log eigenvalue rank.

The relation is tight (Spearman 0.97, n = 3,584) and rank-ordered by color: variance rank moves both quantities along one curve; this is the per-unit view behind the aggregate coincidence statistics above.

### Beyond variance rank, the shared axis is small per-direction and absent per-feature

Within-decile Spearman correlations between map per-direction R² and SAE reconstruction quality — by eigenvalue decile (3,584 directions, left half) and activity decile (16,384 features, right half) — with variance-partialled correlations as dashed lines.

![Within-decile stratified correlations per direction and per feature](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eb22508d64fe93ecb7538569d1020b129907e3d3/figures/issue_1895/stratified_correlates.png)

> **Figure.** *Stratified correlations decay with variance per-direction and vanish per-feature once known correlates are controlled.* Per-direction decile correlations decay 0.94 → 0.10 with decreasing variance; per-feature activity-decile correlations sit flat at 0.26–0.37. Dashed lines: partial correlation 0.076 (direction grain) and −0.050 (feature grain).

Per-direction, the partial correlation given variance rank is 0.076 (interval 0.038 to 0.090; n = 3,584), and decile correlations decay from 0.94 to 0.10 — 9 of 10 pass the false-discovery screen, though variance gradients inside top deciles inflate them. The split-half control attenuates this partial to 0.030–0.068 cross-half vs 0.072–0.105 within-half: the full-holdout 0.076 carries mild construction-coupling inflation but stays positive when decoupled.

Per-feature (raw Spearman 0.25, n = 16,384), controlling variance rank, consistency, and activity leaves −0.050: reconstruction quality adds nothing beyond the known correlates; the residue is largely the variance axis re-measured. Estimation noise does not produce these reads: the parent run's banked 20-draw shuffle null puts the per-feature R² noise floor near −0.07 (per-activity-decile medians; 97.5th percentiles −0.05 to −0.04), far below the paired observed per-feature R² (median 0.15).

---

**Repro:** Compute: one A100-80 GCE instance (attempt att-20260801-012732), ~1.75 h wall / ~1.75 GPU-h, Path A (the conditional mask-matched capture never ran, by design); VM harvest + figures CPU-only. Code: run driver `scripts/issue1895_subspaces.py` @ `7c50b3c66560` (branch `issue-1895`); harvest re-reductions + figures `scripts/issue1895_analysis_figures.py` @ `eb22508d64fe93ecb7538569d1020b129907e3d3`. Committed eval JSONs: `eval_results/issue_1895/` @ `42d831ec5b` (run summaries), @ `eb22508d64` (`dark_spot_matched_rows.json`, `angle_spectrum_k64.json`), and @ `31e93f7651` (`splithalf_followup.json`; script `scripts/issue1895_splithalf_followup.py`). Artifacts (HF data repo `superkaiba1/explore-persona-space-data`, listing verified this session): [issue1895_subspaces/eval_results_issue_1895/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3d620298e7c1f1fd3c3ea25980eaa70db73e4a8c/issue1895_subspaces/eval_results_issue_1895) (all summaries, `perdirection_profiles.npz`, and the `null_bands.npz` per-draw matrices — 287 MB, git-uncommitted by size) and [issue1895_subspaces/analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3d620298e7c1f1fd3c3ea25980eaa70db73e4a8c/issue1895_subspaces/analysis_tensors) (`q_basis.npz`, `percontext_proj.npz`, `fve_profiles.npz`, `pilot/pilot_rows.npz`, `rbar/` store).
- Reused pooled SAE store from [#1482](https://eps.superkaiba.com/tasks/1482): [issue1482_error_analysis/analysis_tensors/sae_pooled](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3d620298e7c1f1fd3c3ea25980eaa70db73e4a8c/issue1482_error_analysis/analysis_tensors/sae_pooled) (1,920 shards) — fit: k=64 codes under the reference mask, reconstruction-identity-verified on 512 pilot rows (median rel dev 2.4e-3).
- Reused capture chunks from [#779](https://eps.superkaiba.com/tasks/779): `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture` @ `3d620298e7c1` — fit: split shas + top-256 eigenvalues reproduced against committed anchors (run halts otherwise).
- Reused banked per-direction profile from [#1482](https://eps.superkaiba.com/tasks/1482): committed `eval_results/issue_1482/perdirection_pca.json` @ `eb22508d64` — fit: rank-correlates 0.9999 with this run's matched refit on the shared 256 directions.
- Reused per-feature shuffle-null noise floor from [#1482](https://eps.superkaiba.com/tasks/1482): committed `eval_results/issue_1482/sae_perfeature/shuffle_null.json` + `shuffle_null_perfeature.npz` @ `43bbbbb0ed` — fit: the null's paired observed cell (`obs_r2`) is the same `sae_ctx` mean-pooled ridge per-feature read this body quotes (median/mean match exactly); K = 20 draws.
- Seeds: split inherited (sha-asserted); fit seed 0; angle-null/bootstrap seed 1895; matched-rows re-reduction seed 1902 (= run seed + 7, the driver's pilot-bootstrap convention).
- Follow-up round `pure-residual-path-b`: compute 1× H100 RunPod `pod-1895`, ~4.6 h wall (a GCE FLEX_START A100 launch whose queued instance vanished auto-failed-over to RunPod per the queue-vanish policy; one crash-fix resume after a stale-read JSON decode error on a staged shard mid-capture). Driver @ `ea2b645d99ea` on `issue-1895` with `--force-path B` (realized path B; the round's `pilot_gate.json` `path` field is G2b-derived, not the realized path). Round figures: `scripts/issue1895_pure_residual_figures.py` @ `1e0b8e25ac`. Committed eval JSONs: `eval_results/issue_1895/pure-residual-path-b/` @ `b9da723f0a`. HF (listing verified this session, 3,854 files): [issue1895_subspaces/pure_residual_path_b/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/04c0d41f4dd5fdcf1aa61755bb4f928d4ab02320/issue1895_subspaces/pure_residual_path_b) — `eval_results_issue_1895/` (incl. `null_bands.npz`, 287 MB, HF-only) and `analysis_tensors/` @ `04c0d41f4dd5` (1,920-shard mask-matched `vmask` store + 1,920-shard `rbar` store, upload-verified 1920/1920).

**Context:** Parent lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — extends the parent's per-direction/per-feature predictability reads to the SAE-coincidence question. Origin (verbatim user prompt, chat 2026-07-30): "can we run these experiments: - Is what oir model able to reconstruct the same as what SAEs find - Map from continuous features to SAE features (or reconstructed activation)? Also compare to simple baselines with SAE featurs". Task created 2026-07-30; run completed 2026-08-01 (GCE att-20260801-012732); analyzer round 1 (HOLD mode) 2026-07-31/08-01; analyzer revision round 2 (interpretation-critic fixes: angle counts, MLP deltas, Path-A-trained-map caveat) 2026-08-01. The origin prompt's second bullet was largely banked before this run (see the task's Provenance table); this experiment ran the new pieces only: the subspace-coincidence battery, the reconstruction/residual targets, and the variance-partialled correlates. Free-analysis follow-up round (split-half re-derivation) run 2026-08-01 post-interpretation-PASS. Same-issue follow-up round `pure-residual-path-b` (source: proposer-9b-cheap, `epm:followup-scope` 2026-08-01T08:49Z; hypothesis verbatim: "The residual target's above-plug-in excess (+0.034) is entirely the answer-token-convention artifact (13.4% of residual energy): a map trained AND scored corpus-wide on the pure SAE error (mask-matched dense answer means minus pooled reconstruction) will read at or below its variance-profile plug-in"): run + fold 2026-08-01; confidence tag raised MODERATE → HIGH on the fold (the one contrary arm is now clean-measured at/below its variance profile).

