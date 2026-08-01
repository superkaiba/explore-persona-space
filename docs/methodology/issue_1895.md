# Methodology — issue 1895

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
