---
title: The context→answer map's error interaction is per-pair noise to within a trace
  rank-one residue, putting the map near its information ceiling (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-31T19:59:29Z'
has_clean_result: true
parent_id: 1482
origin_prompt: 'run all these: ... 1. The interaction-structure question. The decomposition
  says the map fails at (context, direction) pairs. Whether that interaction has recoverable
  low-rank structure — #1775''s rank-32 bilinear pointed at the residual instead of
  the input — is the natural successor'
workflow: v1
goal: 'Determine whether the context→answer map''s dominant (context × direction)
  interaction residual admits recoverable low-rank bilinear structure (H1) or is per-pair
  idiosyncratic (H2), by pointing #1775''s rank-r bilinear machinery at the #1482
  residual pair space against a permuted-pairing null with the answer-sampling floor
  netted out.'
relates_to:
- spec-context-as-vector
---
# The context→answer map's error interaction is per-pair noise to within a trace rank-one residue, putting the map near its information ceiling (MODERATE confidence)
<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1945.md](https://github.com/superkaiba/explore-persona-space/blob/1a6898af98890d967530582d0efdb51b33ada9ef/docs/methodology/issue_1945.md) · [gist](https://gist.github.com/superkaiba/ccc4cbfd10f1d2aed3b390da6fd06bc0)

## Takeaways

- The verdict lattice reads weak structure: the held-out interaction R² curve clears all 400 null draws in both families (add-one p ≤ 0.005) but peaks at 0.0013 — 0.13% of interaction variance against the 0.10 materiality floor. At rank ≤ 64 and this power, the map is close to its information ceiling on this input.
- The residue is rank-one at the primary read and reproducible: all 8 held-out blocks peak at rank 1 (0.0011–0.0014), both fold halves agree (0.0012 / 0.0014), and all 24 log-space units across 12 cells read the same verdict (maxima 0.0002–0.011, largest at layer 26).
- The residue is weakly input-recoverable: map-output features beat the pairing-permutation null in 24 of 24 fold-cells, at 0.0008–0.0047 under low-rank truncation (peaks at ranks 1–8; 10 cells at rank 1), and top-1 retrieval reaches 0.5% against 0.04% chance. This certifies a lower bound; the at-most-a-trace ceiling comes from the held-out rank curves, at rank ≤ 64 and this power.
- Raw-space reads flip all 24 raw units to structured (excess 0.17–0.28 over the Gaussian second-moment null) with curves rising through rank 32 — consistent with heavier-than-Gaussian residual tails, which the log transform removes, rather than recurring pairing patterns.
- The answer-sampling floor is negligible inside the interaction (raw share at most 0.013%, normalized at most 0.9%), so floor netting leaves every number above unchanged.
- The follow-up corpus × depth-band breakdown finds no stratum concentrating the residue: within-stratum maxima span −0.0013 to +0.0009, all below their cells' pooled reads; only 2-turn conversations read positive.

## Goal

**This experiment in context:** the parent decomposition ([#1482](https://eps.superkaiba.com/tasks/1482)) found the context→answer map's held-out squared error is dominated by the (context × direction) interaction — 0.80–0.94 of the variance across all 12 cells — and its residual-vector follow-up found the residual field carries no structure beyond its second moment. This experiment asks whether that dominant interaction is a learnable object: approximable out-of-sample by a few context-side × direction-side factors (the map is under-specified) or per-pair idiosyncratic (the map is near its information ceiling on this input). The rank-r machinery follows [#1775](https://eps.superkaiba.com/tasks/1775), which fit a rank-32 bilinear interaction on the input side and closed 93.2% of the additive-stitch-to-full-context gap (+0.0493) — a prediction-side read under a different fitting protocol (Adam-fit input factors), not directly comparable to the residual-side numbers here. Targets and predictions are the banked [#1738](https://eps.superkaiba.com/tasks/1738) holdout (9,941 real conversations).

**Broader narrative:** the "what is the map bad at" line needs to know whether remaining error is recoverable signal or noise. A structured answer would license richer functional forms on the same input; the idiosyncratic answer routes future effort toward different inputs (or accepting the ceiling). This run gives the second answer to within a trace residue.

## Methodology

**Design:** Gabriel-style bi-cross-validation (BCV) of the interaction. The underlying object: per conversation, the context→answer map predicts the mean answer state of Qwen-2.5-7B-Instruct — the mean-pooled hidden state over the generated answer span, including the end-of-turn tail, at layer 14, 19, or 26 (hidden size 3,584, the matrix width) — from the last-token hidden state of the full context, of the prefix (everything before the final user turn), or of the bare final query. The parent round fit each map on 87,795 training conversations (validation 396 / test 995 / holdout 9,941, sha-pinned split): ridge over 23 log-spaced penalties from 1e−3 to 1e8 with the penalty selected on the validation rows; the MLP-fitter cells are its width-8,192 variant (learning rate selected on validation from 1e-3 and 3e-4, at most 300 epochs, batch 4,096). Per cell, residuals E = (prediction − target) over the 9,941 held-out conversations are projected on the parent's split-half answer-PCA basis — rows permuted with seed 1482 into two complementary halves; per fold, the top-512 eigenvectors of the basis half's mean-answer-state covariance (fp64, about its own mean) supply the directions on which the other half's residuals are scored. R = E² over the top k directions; transform to the analysis space (log primary; raw + per-direction-normalized companions); remove the additive two-way fit (grand mean + row + column effects); split rows into 2 seeded halves and columns into 2 eigen-rank-interleaved halves; reconstruct each held-out block from the complement's rank-r SVD and score held-out interaction R²(r), averaged over 4 blocks × 2 parent folds. Grid: 12 cells (3 arms — full context / prefix only / bare query — × layers 14/19/26 ridge, + 3 layer-19 MLP-fitter cells) × k ∈ {64, 256} × 3 spaces = 72 units; the prefix arm and the context arm both run (standing both-arms rule). Rank grid r ∈ {1,2,4,8,16,32,64}, truncated to r ≤ k/4 (columns halve at BCV, so k=64 tops at r=16). Two null families, 200 draws each, run through the identical pipeline including per-draw max-over-rank selection: within-column permutation of the two-way-removed matrix (destroys row pairing; the zero-structure floor) and a Gaussian second-moment null (rows i.i.d. from the basis-half residual covariance, row-norm matched — everything scale + noise covariance implies; the decisive bar). Verdict lattice on the primary unit (full context, layer 19, ridge, k=256, log): structured ⇔ observed max exceeds the Gaussian band and is at least 0.10; weak-structured ⇔ exceeds the band but < 0.10; idiosyncratic otherwise. Tier B (input-recoverability): reduced-rank ridge from per-row map-output features (prediction projected on the 512-dim basis + 2 norms = 514 features) to held-out interaction rows, log space, k=256, per (cell × fold); train/test split of the eval half (n_train = 2,485 > 514); null = 200 row-pairing permutations under one shared Gram factorization; companion nearest-neighbor retrieval (cosine + euclidean, chance stated). Floor netting: the parent's per-context answer-sampling floors — K-resample estimates of each context's irreducible answer-sampling error, from 4 fresh answers per context (seeds 43–46) on a 2,000-context stratified subsample (1,988 kept) — drive (a) a floor-corrected subsample replication (raw + normalized) and (b) a floor-noise-only synthetic whose interaction share nets the answer-sampling noise out of the recoverable-share denominator. A zero-GPU follow-up round re-reduced the same residual matrices within corpus × conversation-depth strata (LMSYS / WildChat × 2, 3–4, ≥5 user turns; labels joined from the parent sampling manifest onto the 9,941 holdout conversation ids) for the three layer-19 ridge cells, log space, k = 256, per parent fold, against a light B = 25 within-column permutation reference under the same per-draw max-over-rank convention; a stratum under 40 rows per fold is reported insufficient-n, never silently dropped (`scripts/issue1945_strata_breakdown.py`).

**Training:** **N/A — no model training.** Analysis hyperparameters (all copied from the run script at commit `f7dbe1cb9a` and `bcv_summary.json`):

| Hyperparameter | Value | Source |
|---|---|---|
| BCV scheme | Gabriel (2,2): 2 row halves × 2 eigen-interleaved column halves | Owen & Perry 2009 (arXiv:0908.2062); parent split-half convention |
| Rank grid | {1, 2, 4, 8, 16, 32, 64}, capped at k/4 | #1775 `R_GRID` |
| Null draws | B = 200 per family per unit | #778/#834 null-battery convention |
| Direction counts k | 64, 256 (primary 256); basis kmax 512 | #1482 `K_GRID` subset; parent floor-correction k |
| Analysis spaces | log (primary), raw, per-direction normalized | #1482 log companion; variance stabilization for squared residuals |
| Primary cell | full context, layer 19, ridge | #1482/#1738 headline cell |
| Fold seed / new-randomness seed | 1482 / 1945 | parent basis identity; this task |
| Tier-B features / ridge | 514 dims; GCV over λ grid `logspace(-2, 4, 13)` | `issue_779/fit_h.py` defaults; #1775 warm-start convention |
| Materiality floor | 0.10 of held-out interaction variance | plan v3 lattice |
| Strata round: perm draws / row floor | B = 25 per fold-stratum / 40 rows per (stratum, fold) | plan v3 §6 follow-up dispatch (light reference) |
| Thread caps | OMP/MKL/OPENBLAS/NUMEXPR = 8, MALLOC_ARENA_MAX = 2 | shared-VM convention |

**Evaluation:** the DV is held-out-block interaction R²(r) — variance of the two-way-removed matrix explained by the rank-r BCV reconstruction — with R²(0) = 0 by construction (verified ≤ 1e-12 on all 72 units). Bands are p97.5 of the per-draw max-over-rank null distributions (selection-symmetric: every null draw gets the same max-over-rank selection as the observed statistic); the DV ceiling is 1.0 and every band sits ≈1.0002 below it, so the bands are informative. Per-unit per-draw × per-rank matrices (1 observed + 400 null rows) are persisted so any band can be re-reduced later. Tier-B ridge diagnostics: GCV selected λ = 1e4 — the top of its grid — on all 24 observed fits (dof 84–93 of 514, read from the batched twin; the slow fallback solver that produced the committed fits records no selector diagnostics) and on the null draws (median λ also 1e4), so GCV wanted more shrinkage than the grid offers and rank truncation acts as the missing regularization. The identity+learned-bias mapping baseline is inapplicable here by dimension mismatch (514-dim features vs 256-dim squared-error-profile targets) — stated rather than silently skipped.

**Data extraction:** staged fp16 matrices from the parent round — per arm/layer predictions `pred_{arm}_L{layer}_{fitter}.npz` and targets `y_parent_L{layer}.npz`, 9,941 × 3,584, keys `{pred16|y16, ci, fingerprint}` — asserted for conversation-id and fingerprint equality at load. The smoke probe confirmed 9,941 distinct conversation ids (one context per conversation, so row splits are conversation-level) and fold identity with the parent (basis-eigenvalue deviation 0.0). Floors: `eval_results/issue_1738/kresample/floors_L{14,19,26}.npz` (1,988 rows, subset of the holdout). The Tier-B parity gate (fast Gram-eigh ridge vs the slow SVD solver at production shape) measured max rel diff 2.8e-4 against a 1e-4 tolerance, so the parity gate's fallback ran the slow solver for all observed Tier-B fits. Strata labels for the follow-up round: the parent sampling manifest on the HF data repo (`issue1738_multiturn/sampling_manifest`, 55 parts, 99,778 rows scanned); all 9,941 staged holdout conversation ids joined (join rate 1.00, none unmapped).

**Sample training/evaluation data + completions:** no model generations exist in this task (pure matrix analysis); the per-unit records are the run's data. One verbatim Tier-A unit row (the primary unit; 1 of 72 rows, `regime` and `vc_shares_per_fold` fields elided for width — full file: [bcv/units.jsonl](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d6628781f3fb686ed695b3bff60ec5a911f4f65e/eval_results/issue_1945/bcv/units.jsonl)):

```json
{"unit": "context_L19_ridge|k256|log", "cell": "context_L19_ridge", "k": 256, "space": "log",
 "n_eval_rows": [4971, 4970], "r_grid": [0, 1, 2, 4, 8, 16, 32, 64],
 "obs_curve": [0.0, 0.001269279152182993, 0.001001677635976815, 0.0005598521304206976,
               -0.0006075860488702174, -0.0029795799729381055, -0.007963751118179107,
               -0.01975425836393323],
 "obs_max": 0.001269279152182993, "perm_p975_max": -0.00022152410326161209,
 "gauss2m_p975_max": -0.0002165358050299111, "delta_g": 0.0014858149572129042,
 "delta_m": -0.09873072084781702, "elapsed_s": 1262.01, "ts": "2026-08-01T00:51:46.313087+00:00"}
```

The matching per-draw matrix (401 × 8, 1 observed + 200 permutation + 200 Gaussian rows) is 1 of 114 npz files: [percell/](https://github.com/superkaiba/explore-persona-space/tree/d6628781f3fb686ed695b3bff60ec5a911f4f65e/eval_results/issue_1945/percell). Tier-B rows (24) live in [tierb/units.jsonl](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d6628781f3fb686ed695b3bff60ec5a911f4f65e/eval_results/issue_1945/tierb/units.jsonl).

## Results

### The verdict lattice reads weak structure: detectable, but a thousandth of the variance

Held-out interaction R² against the rank of the BCV reconstruction for the primary unit (full context, layer 19, ridge; log space, 256 directions; 9,941 conversations, ≈4,970 eval rows per fold). Faint lines are the 8 individual held-out blocks; dashed lines are each null family's p97.5 of the per-draw max-over-rank statistic (200 draws each).

![Held-out interaction R-squared vs rank for the primary cell with permutation and Gaussian null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r1_primary_bcv_curve.png)

> **Figure.** *The pooled curve peaks at rank 1 then declines.* Both null bands sit at −0.0002; the peak is 0.0013 against the 0.10 materiality floor (4 blocks × 2 folds).

The curve clears both bands only at ranks 1–4 and is negative from rank 8 on. Against the 0.10 floor the recoverable share is 0.13%, so the lattice reads weak structure: statistically real, quantitatively trace. The max-over-rank headline is selection-inflated by construction; the curve and per-block spread carry the honest magnitude.

### Every null draw and every held-out block agrees with the primary read

Histogram of the per-draw max-over-rank statistic for both null families at the primary unit (200 draws each), with the observed pooled value and the 8 per-block maxima marked.

![Null draw distributions for both families vs the observed statistic at the primary cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r2_null_draw_distributions.png)

> **Figure.** *The observed value exceeds all 400 null draws.* Both families concentrate near −0.0003; the observed statistic is 0.0013 and block maxima span 0.0011–0.0014.

The permuted-pairing bar (the Goal's letter) and the Gaussian second-moment bar (the decisive one — the parent showed the residual's second moment is known structure) coincide here at −0.0002, and the observed statistic beats every draw of both (add-one p ≤ 0.005 per family). Fold halves read 0.0012 and 0.0014; all 8 blocks peak at rank 1. The residue is rank-one at the primary cell and reproducible, not a selection fluke.

### All 24 log-space units read the same verdict; the residue is largest at layer 26

Pooled log-space curves (k = 256) for all 12 cells, colored by arm; dashed lines are the layer-19 MLP-fitter cells.

![Log-space held-out interaction R-squared curves for all twelve cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r3_log_curves_all_cells.png)

> **Figure.** *Every cell peaks at low rank and declines.* Plotted k = 256 maxima span 0.0013–0.0072, layer-26 cells highest; the k = 64 units' maxima (0.0002–0.011) are off-figure.

All 24 log-space units (12 cells × 2 direction counts) land in the weak-structured cell of the lattice; the largest is 1.1% of held-out interaction variance (bare query, layer 26, k = 64). The MLP-fitter cells track the ridge cells, so the read is not ridge-specific. The corpus × depth-band breakdown of these curves ran as a zero-GPU follow-up round (final result below).

### Raw space flips every cell to structured; the flips track tails, not patterns

Heatmap of the observed max minus the Gaussian-null band for all 72 units (12 cells × 2 direction counts × 3 spaces).

![Heatmap of observed excess over the Gaussian second-moment null across all 72 units](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r4_verdict_heatmap_excess.png)

> **Figure.** *Raw-space columns show excess in every cell.* Raw excess spans 0.17–0.28; log columns sit at 0.000–0.011; normalized columns sit near zero at mid layers and rise at layer 26, up to 0.30 for the prefix arm.

26 units read structured: all 24 raw units plus the two prefix layer-26 normalized units (0.25 and 0.30). The 6 idiosyncratic units are all normalized mid-layer cells whose excess is within 0.0034 of the band — boundary flips, not a distinct regime. A Gaussian null pins fourth moments, so heavy-tailed residual rows create recoverable structure in squared residuals that it cannot reproduce; the log transform removes exactly that, which is why the primary space isolates genuine pairing structure.

### The raw-space excess has no low-rank plateau, the signature of tail geometry

Observed curves and null bands for the primary cell in raw squared-residual space (left) and log space (right), same 256 directions.

![Raw versus log space curves for the primary cell with null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r5_raw_vs_log_primary.png)

> **Figure.** *The raw-space curve rises through rank 32 while the log-space curve peaks at rank 1.* Raw: the Gaussian null itself recovers 0.165 and the observed curve reaches 0.435 at rank 32; log: 0.0013.

A few recurring context-by-direction patterns would saturate at low rank. Instead the raw curve rises through rank 32 with no low-rank plateau (0.435 at rank 32, 0.426 at rank 64), and even the Gaussian null recovers a sixth of the raw interaction through scale coupling alone. Both facts are consistent with scale-and-tail geometry rather than learnable pairing structure, matching the parent finding that the residual field carries nothing beyond its second moment.

### The trace residue is input-recoverable, at the same scale

Reduced-rank ridge from map-output features to held-out interaction rows (log space, 256 directions; all 24 fold-cells) with the pairing-permutation band and the primary cell's untruncated endpoint, plus nearest-neighbor retrieval against chance.

![Tier B input-recoverability curves and retrieval accuracy](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r6_tierb_input_recoverability.png)

> **Figure.** *All 24 fold-cells clear the permutation band; peaks sit at ranks 1–8.* Maxima span 0.0008–0.0047 (10 cells peak at rank 1; five of six layer-26 cells at rank 8); the untruncated ridge reads −0.0046; top-1 cosine retrieval is 0.005 / 0.003 by fold vs 0.0004 chance.

Every fold-cell beats its pairing-permutation null, so part of the residue is genuinely input-linked — and because this null permutes row pairing rather than matching moments, the read is insulated from residual non-Gaussian dependence in the log-space excess. The recoverability is small and not strictly rank-one: the largest fold-cell reads 0.47% (prefix, layer 26), the primary cell reads 0.0009–0.0010 at rank 1, and five of six layer-26 fold-cells peak at rank 8.

A positive read here certifies a lower bound on input-recoverability — it cannot cap it; the at-most-a-trace ceiling comes from the held-out rank curves (rank ≤ 64, this power). The untruncated endpoint (−0.0046) is conditional on the λ-grid top; a larger penalty could bring it toward zero.

### The answer-sampling floor contributes almost nothing to the interaction

Share of interaction variance produced by a floor-noise-only synthetic (the parent's per-context answer-sampling floors under its isotropy assumption), per arm × layer, in raw and normalized space.

![Answer-sampling floor share of interaction variance per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e5a392aecfea34aae307d4e04d860b08d93b69d6/figures/issue_1945/r7_floor_share.png)

> **Figure.** *Floor shares are tiny in both quoted spaces.* Raw shares sit at 0.001–0.013%, normalized at 0.19–0.89%; log-space shares are undefined because zero-valued floors give log of zero.

Netting the floor out of the denominator changes the recoverable share by under 1% of itself in either quoted space, so every headline number stands un-netted. In log space the scalar row correction is absorbed by the row effect, an expected consequence of the transform; the floor-corrected subsample replication (raw maxima 0.10–0.58, normalized −0.006 to 0.010) matches the full-sample raw picture.

### No corpus or depth-band stratum concentrates the trace residue; within-stratum reads are weaker than pooled

Held-out interaction R² against rank within each of six corpus × conversation-depth strata (LMSYS and WildChat crossed with 2, 3–4, and ≥5 user turns), joined from the parent sampling manifest onto all 9,941 holdout conversations (join rate 1.00); three layer-19 ridge cells, log space, 256 directions, pooled over both parent folds, against a light 25-draw within-column permutation reference under the same per-draw max-over-rank convention. No stratum fell below the 40-row-per-fold floor (per-fold n 551–1,134), so none were dropped.

![Within-stratum BCV curves by corpus and conversation depth for the three layer-19 ridge cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/30fb35110129d8443b2f09d4750d5552ffb93694/figures/issue_1945/r8_strata_breakdown.png)

> **Figure.** *Every within-stratum curve declines with rank in all three cells.* Pooled within-stratum maxima span −0.0013 to +0.0009 against light permutation references of −0.0010 to −0.0005; the only positive maxima are in the 2-turn band, the largest 0.0009 (bare query, LMSYS, rank 1).

Descriptively — 25 draws give a light reference, not a powered test — no stratum deviates upward from the pooled near-ceiling read: every within-stratum maximum (−0.0013 to +0.0009) sits below its cell's pooled maximum (0.0013–0.0018). The only positive maxima sit in the 2-turn band, peaking at rank 1 — consistent with the rank-one residue drawing on short conversations, a hypothesis at this power, not a finding.

---

**Repro:** 0 GPU-h; one shared-VM CPU process (8-thread cap), wall 5,292 s, peak RSS 3.46 GB; B = 200 draws per null family per unit. Code: `scripts/issue1945_bcv_interaction.py` at commit `f7dbe1cb9a` (smoke/pilot at `6ea044af6d`; pilot report committed under `eval_results/issue_1945/smoke/`). Results: `eval_results/issue_1945/{bcv,tierb,floor,percell}/` committed at `d6628781f3` on branch issue-1945 (114 per-draw npz files force-added past the repo `*.npz` ignore). Figures: analyzer regeneration at `e5a392aecf` via `scripts/issue1945_analyzer_figs.py` (the run script's own rendered PNGs at `d6628781f3` — e.g. `figures/issue_1945/per_cell_curves_log_k256.png`, not embedded: superseded by `r3_log_curves_all_cells` on the same data — carry no sidecars and config-slug labels; the r-prefixed set replaces them). Verifier conciseness WARNs (bullet length, two result blocks over the 120-word cap, total prose over the 800-word budget across eight results) acknowledged. Tier-B untruncated endpoints: analyzer-recomputed from the staged matrices with the run's seeds; persisted at `eval_results/issue_1945/tierb/fullridge_recompute.json` (commit `d34267ace1`; committed-curve reproduction max abs dev 0.0). Strata follow-up round (0 GPU-h 9a-ter free-analysis re-reducing the same staged matrices): `scripts/issue1945_strata_breakdown.py`, results `eval_results/issue_1945/strata/strata_breakdown.json`, figure `r8_strata_breakdown` — all committed at `30fb351101`. Config slugs: primary unit `context_L19_ridge|k256|log`; cells `{context,prefix,bare}_L{14,19,26}_ridge` + `{arm}_L19_mlp_w8192`. Reused inputs — Reused staged residual matrices from [#1482](https://eps.superkaiba.com/tasks/1482)/[#1738](https://eps.superkaiba.com/tasks/1738): `data/issue_1482/twoway_stage/` (9,941 × 3,584 fp16; fingerprint + conversation-id asserted per load) — fit: same holdout, same targets, same fold seed as the parent; re-stageable from HF `superkaiba1/explore-persona-space-data` under `issue1738_multiturn/analysis_tensors/{pred16,y_holdout}` ([tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fad645dfb5a4066fa0789f180465ef7579e93cc7/issue1738_multiturn/analysis_tensors); 32 pred16 files listed live 2026-07-31). Reused floors from [#1738](https://eps.superkaiba.com/tasks/1738): `eval_results/issue_1738/kresample/floors_L{14,19,26}.npz` (git-tracked) — fit: the parent floor recipe consumed verbatim. Reused code: `issue1482_twoway_residual.py` decomposition/fold/PCA functions and `issue_779/fit_h.py` ridge solvers, unchanged. Plan v3 (`plans/plan.md`).

**Context:** task created 2026-07-31 (user chat capture); plan v3 approved and run 2026-07-31 → 2026-08-01 UTC; analyzer pass 2026-07-31 (HOLD mode), revision round 2026-08-01; corpus × depth-band strata round (0-GPU 9a-ter follow-up) folded 2026-08-01. Origin prompt (verbatim, as recorded): "run all these: ... 1. The interaction-structure question. The decomposition says the map fails at (context, direction) pairs. Whether that interaction has recoverable low-rank structure — #1775's rank-32 bilinear pointed at the residual instead of the input — is the natural successor". Lineage: same-question successor to the [#1482](https://eps.superkaiba.com/tasks/1482) two-way decomposition, using [#1775](https://eps.superkaiba.com/tasks/1775)'s rank machinery on the residual side.
