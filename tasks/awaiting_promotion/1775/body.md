---
title: A low-rank bilinear prefix–query interaction closes most but not all of the
  context→answer map's nonlinear gap (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-28T21:36:04Z'
has_clean_result: true
parent_id: 1092
origin_prompt: 'run both in background with ahppy coder (2026-07-28; two-task split
  per: ''can it not be all one task? Or at least separate the nonlinear out'')'
workflow: v1
goal: 'Determine where the linear story of the context→answer map ends: per-arm nonlinearity
  gain under matched folds (ridge → RFF/Nyström → MLP), residual HSIC/dCor detection,
  fold-structure verification of the banked n50k/n1m fitter comparisons, and the headline
  test of whether a rank-r bilinear prefix×query interaction closes the same ≈0.06
  R² gap the banked 1M nonlinear fits find — per Task B of docs/ideas/2026-07-28-four-arm-map-theoretical-analysis-plan.md'
relates_to:
- spec-context-as-vector
- leak-predictor
backend: gcp
---
# A low-rank bilinear prefix–query interaction closes most but not all of the context→answer map's nonlinear gap (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1775.md](https://github.com/superkaiba/explore-persona-space/blob/175988e5d1c818e1ccdc8945e3899b28afe11901/docs/methodology/issue_1775.md) · [gist](https://gist.github.com/superkaiba/146f7cb69c48306ff02b81ea2311af2a)

## Takeaways

- **A rank-32 bilinear prefix–query interaction closes 93% of the additive-stitch → full-context R² gap on novel prefixes** (+0.049 over the matched no-interaction refit; the cluster CI excludes zero), unchanged under train-fold-only PC bases (+0.050).
- **A real unnamed residual remains, so the verdict is "partially named":** the same-protocol stitch-MLP still beats the bilinear by +0.010 on novel prefixes, +0.064 on novel queries, and +0.058 when both are novel — twice that scheme's named gain.
- **Disjoint prefix + query inputs reach R² 0.920 through an MLP, at or slightly above the full-context linear map's 0.914** (+0.006 margin; no paired CI computed) — consistent with the linear read requiring no co-attention of prefix and query.
- **Residual structure is detectable almost everywhere (23 of 26 informative tests reject; 4 of 30 cells degenerate), but the mean-predictable part is small on prefix-input arms** (kernel gains +0.011 to +0.015)**, modest elsewhere.**
- **The banked 50k-train fitter comparison is contaminated (lexical near-duplicates touch 325 of 1,400 targets), but its kernel gain survives deduplication:** +0.043 after dropping the 2,296 implicated train rows, against +0.047 for an n-matched random-drop control — contamination inflated levels, not the fitter gap. The deduplicated 1M comparison (≈ +0.056 MLP gain) remains the citable context-arm number.
- **A follow-up refit shows the prefix-averaged arm is wholly nonlinear per-row:** its best in-grid linear map stays below zero R² while kernels and MLPs reach ≈ 0.11, near the between-prefix ceiling.

## Goal

- **This experiment in context:** The parent crossed-corpus run ([#1092](https://eps.superkaiba.com/tasks/1092)) established the line's central linear result (context→answer activation maps with held-out R² 0.74–0.91, additive in prefix and query to ≈ 91%), and the banked fitter comparisons ([#779](https://eps.superkaiba.com/tasks/779)) found a ≈ +0.05–0.09 kernel/MLP gain over ridge on the full-context arm. This task measures what the linear map misses: a fold-hygiene audit of the banked 50k/1M train–test splits; dependence tests for any residual structure per input arm, using the validated HSIC/distance-correlation instruments ([#763](https://eps.superkaiba.com/tasks/763)); a matched kernel/random-features/MLP ladder per arm under identical folds; and the headline test — does a rank-r bilinear prefix×query interaction account for the same gap the black-box fitters close? Noise-ceiling attribution is deferred until the sibling decode-noise-floor task ([#1774](https://eps.superkaiba.com/tasks/1774)) lands.
- **Broader narrative:** This serves the context-as-vector question (`docs/open_questions.md`, `spec-context-as-vector` and `leak-predictor` anchors): whether the context→answer map's nonlinearity has a named, low-rank compositional form (a prefix×query interaction) or is an unstructured black box, which decides how far linear-map theory can carry the leakage-prediction program.

## Methodology

**Design:** No model is trained; every fit reads banked activation summaries from the parent 21,193-row crossed corpus (1,145 real WildChat/LMSYS conversation prefixes × 1,397 held-out user queries, dense core ≈ 99 × 48; teacher-forced captures of the instruct model's own greedy answers at layer 14). The fit population is the 17,308 battery-excluded rows (996 prefix groups, 1,397 query groups; re-derived exactly: 21,193 → 19,708 trait-stratum-excluded → 17,308, no group in more than one fold under either scheme). Five input views of the pooled answer-summary target: the pre-query prefix end-state, the leave-own-row-out prefix-averaged context state, the bare query state, the full-context end-state (linear reference and residual source), and the additive stitch (prefix ⊕ query concatenation, 7,168-dim). Four analyses: a fold-hygiene audit of the banked 50k/1M splits (exact + 5-gram-Jaccard ≥ 0.8 near-duplicate train↔target overlap, MinHash-accelerated); residual-dependence tests (HSIC + distance correlation on held-out ridge residuals, three group-respecting block-permutation schemes, Holm correction over the 30-test family); an estimation ladder per arm (PRESS ridge → exact RBF kernel → random Fourier features → MLP) under identical folds with nested inner tuning; and the headline rank-r bilinear fit â = W[p;q] + Σᵢ(uᵢᵀp)(vᵢᵀq)wᵢ (AdamW from a ridge warm start), read against a rank-0 refit under the identical optimizer protocol (the de-regularization control) and a same-protocol stitch-MLP ceiling. Verdict lattice: "named" requires the bilinear-over-rank-0 gain to exclude zero with no positive stitch-MLP residual; "partially named" when both exclude zero; the "named" narration additionally requires the bilinear's own residual dependence to drop toward the null. Fold schemes: novel-prefix 6-fold (primary), novel-query 6-fold companion (queries are shared across prefix-grouped folds, which specifically inflates nonlinear rungs on query-bearing arms), doubly-novel robustness read. Targets: top-48 answer PCs (headline space) + stacked ambient companion (10,752-dim). Data realism: tier 1 — real conversations; no new data generated. Prefix-based and context-based mapping arms are both structural to the four-arm design. A zero-GPU follow-up round (same day) re-fit the prefix-averaged arm's per-row linear baseline with group-respecting inner-validation λ selection after its PRESS baseline proved degenerate. A second follow-up round (one GCE A100 instance, 1.56 GPU-h) closed the three deferred checks: a refit of the banked 50k comparison on its deduplicated train set (four fitter rungs, plus an n-matched random-drop control, on the same sha-pinned 1,000 test rows), a rank-{0, 32} bilinear refit under 48-PC bases computed from each train fold only, and the missing doubly-novel stitch-MLP.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Reading model (stores) | Qwen-2.5-7B-Instruct (+ base model for the pretrained companion cell) | parent capture recipe |
| Layer / targets | L14 primary, L19 bridge; pooled answer summaries, 48 train-population PCs + stacked ambient | parent protocol (`issue1092_fit_grid.py`) |
| Folds | novel-prefix and novel-query 6-fold, fold seed 0, group-respecting | parent protocol, `FOLD_SEED = 0` |
| Ridge engine | PRESS ridge, λ grid {0.01, 0.1, 1, 10, 100, 1000}; df(λ) + ±10× λ sensitivity reported | parent engine (`_fit_cv`); grid `RIDGE_LAMBDAS` |
| Kernel rung | exact RBF; γ = median heuristic × {0.25, 0.5, 1, 2, 4}; λ ∈ {1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0} | banked fitter-comparison grid |
| Random-features rung | D = 16,384 random Fourier features of the selected-γ kernel, seeds {0, 1, 2} | matched to the banked Nyström center count |
| MLP rung | width 8,192, lr 3e-4, batch 4,096, AdamW, early stop on inner-validation MSE, seeds {0, 1, 2} | banked fitter-comparison recipe, reused verbatim |
| Bilinear | r ∈ {0, 1, 2, 4, 8, 16, 32, 64}; warm start at stitch-ridge; weight decay ∈ {0, 1e-4, 1e-2}; seeds {0, 1, 2}; r* by inner validation | plan §11 (design-derived grid) |
| Nested tuning | inner group-respecting split, ~1/5 of train groups, selects all tunables per rung | plan §4 |
| Dependence tests | HSIC (RBF, median heuristic) + distance correlation; B = 1,000 draws; schemes {prefix-block, query-block, within-prefix derangement}; Holm over 30 tests | validated instruments (`issue_763_nonlinear.py`) |
| Headline CIs | paired cluster bootstrap, 2,000 draws, seed 0; resampling unit = the fold scheme's grouping unit | plan §3 |
| Near-duplicate gate | 5-gram shingles, Jaccard ≥ 0.8, 64-permutation MinHash (recall 1.0 at true J ≥ 0.8 on 876 planted pairs) | the banked 1M round's own recorded procedure |
| Follow-up refit | inner-val λ over the same grid, group-respecting 20% inner split, seed 1234 + fold | follow-up round (`issue1775_query_averaged_refit.py`) |
| Dedup-refit rungs (round 2) | ridge (val-λ, same grid), exact-RBF kernel (γ = median × 1.0, λ ∈ {0.1, 10}, grid-matched to the banked small grid), MLP (width 8,192, lr 3e-4), residual-skip MLP (width 3,584); paired row bootstrap, 1,000 draws | banked comparison recipe, reused verbatim (`issue1775_n50k_dedup_refit.py`) |
| Fold-PC basis (round 2) | 48 PCs re-fit on each train fold only; r ∈ {0, 32}, r* = 32 carried; gate: fold-basis rank-0 refit within 0.02 of the PRESS stitch level | plan §4 deviation-discharge design |
| Doubly stitch-MLP (round 2) | width 8,192, lr 3e-4, weight decay 1e-4, patience 20, full-batch, max 300 epochs, seeds {0, 1, 2} | banked MLP recipe, reused verbatim (`issue1775_doubly_mlp.py`) |

**Evaluation:** The dependent variable is held-out R² per (arm × fitter × target space × fold scheme), pooled over 6 folds; a nonlinearity gain is R²(fitter) − R²(ridge) on identical folds with nested tuning, and every headline CI is a paired cluster bootstrap over the fold scheme's grouping unit (row-level CIs persisted as labeled companions). The reproduction gate passed exactly: the full-context ridge refit reproduces the banked reads (0.9142 in 48-PC space / 0.8142 ambient vs banked 0.914 / 0.814). The detection-gated skip of MLP batteries never fired (every non-context arm has Holm-significant detection; seed spreads ≤ 0.001 R²). Instrument note: the fast GPU ridge engine failed its parity gate (max relative prediction difference 1.76 / 0.53 / 0.07 across three slices vs tolerance 1e-4), and the fall-back-to-PRESS branch executed on-run for all 9 scheduled rows — the fast engine's predictions were never consumed by any committed number; residue: `df_lambda` is null on those 9 rows (diagnostic-completeness gap only). Those 9 expanded rows (L19 bridge + pretrained-cell companion) are ordinary PRESS reads fit on this task's corpus — not directly comparable side-by-side with the banked L19 line, which used a different corpus and split. Mapping baselines both reported: identity+bias, applicable only on 3,584→3,584 single-target reads, is strongly negative everywhere (prefix end-state −1.52, full context −1.00, prefix-averaged −0.335, bare query −0.718); kNN retrieval on the weakest arm (prefix end-state, per-row) is ≈ 13× chance at k = 1 (acc ≈ 0.0043–0.0050 vs chance 0.00035) with median rank ≈ 514–693 of ~2,900. Plan deviations on record (committed in `plan_deviations.json`): the 48 PCs were fit on the full fit-population rather than per train fold (parent-parity choice; the second follow-up round's train-fold-only refit executed the sensitivity check and reproduces the headline gain); the parity-gate fallback above is retained as an instrument note; and the unnamed-residual CI was computed after the run from the persisted per-row predictions with the same committed bootstrap helper (it reproduces both committed named-gain values to within 2.1e-17). The run-1 record's two remaining gaps — the Gate-A conditional dedup refit not run, and no stitch-MLP under the doubly-novel scheme — were both closed by the second follow-up round (results below). Round-2 CI protocols: the dedup-refit gains use a paired row bootstrap over the 1,000 test rows (the banked comparison's fixed split has no group structure, labeled as such in the JSON); the doubly reads keep the two-way prefix × query cluster bootstrap; the fold-PC read keeps the prefix-group cluster bootstrap. Round-2 mapping baselines: identity+bias on the 3,584→3,584 dedup refits is strongly negative (−0.88 both variants) while kNN retrieval reaches acc@1 0.80–0.85 vs chance 0.001; the fold-PC and doubly cells state identity+bias inapplicable (7,168-dim input vs 48-dim target) and carry per-fold kNN reads in their JSONs.

**Data extraction:** All inputs are reused, verified artifacts. (1) The crossed-corpus stores: the parent pipeline generated the instruct model's own greedy answer per (prefix, query) row, then teacher-forced each full sequence and saved fp16 (21,193 × 3,584) per-row summaries per arm — prefix end-state, context end-state, and three answer-span summaries (the stacked targets) — staged from the HF data repo (`issue1092_realistic_crossing/analysis_tensors/summaries/`, shapes verified by mmap before fitting). (2) The banked fitter-comparison corpora: real LMSYS/WildChat prompts; the 50k split is a fixed 50,000-train / 400-val / 1,000-test draw (seed 42) with L19 final-token captures; the 1M split trains on 963,444 rows after its own dedup dropped 435 exact + 30,437 near duplicates against the same 1,400 targets. The audit reconstructed the train prompts from the sampling manifest and capture chunks (chunk spot-check: 500/500 positional prompt matches; val/test sha256 pins asserted). (3) A 3-trait persona-vector dictionary (evil, hallucination, sycophancy direction rows at L14), produced by the persona-vectors recipe in the monitoring line (per trait, the difference of mean response-averaged activations between judge-filtered positive- and negative-instruction rollouts) and consumed read-only for interaction projections. The corpus rows are unscreened real user text, so no conversation text is reproduced in this body; rows are referenced by manifest and row index.

**Sample training/evaluation data + completions:** This task produces no model generations — its evaluation units are fit records over banked activation summaries. The blocks below are verbatim fit-unit records, cherry-picked for illustration (one per record family; full files linked per block). Conciseness note (acknowledged WARNs): with eight result sections plus two follow-up fold-ins, total content prose exceeds the 800-word budget, several result sections sit in the 120–180-word band, and several Takeaways bullets exceed the 30-word bullet cap; the statistical accounting is kept deliberately.

One estimation-ladder unit record, cherry-picked for illustration (1 of 19 rows in shard 0; full file: [units_nonlinear_shard0.jsonl](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/ladder/units_nonlinear_shard0.jsonl)):

<details>
<summary>Random-features rung, prefix end-state arm, novel-prefix folds (verbatim JSONL row 2)</summary>

```json
{"cell": "cell_inst_own", "layer": 14, "basis": "both", "arm": "prefix_end", "grain": "perrow", "scheme": "prefix", "rung": "rff", "seed": 0, "engine": "rff", "phase": "nonlinear", "smoke": false, "row_limit": null, "per_fold": [{"fold": 0, "lambda": 1.0, "seed": 0}, {"fold": 1, "lambda": 1.0, "seed": 0}, {"fold": 2, "lambda": 1.0, "seed": 0}, {"fold": 3, "lambda": 1.0, "seed": 0}, {"fold": 4, "lambda": 1.0, "seed": 0}, {"fold": 5, "lambda": 1.0, "seed": 0}], "r2": {"ambient": 0.07962726927024144, "pca48": 0.10950051898594382}, "wall_s": 537.9946145520001}
```

</details>

One bilinear unit record, cherry-picked for illustration (1 of 60 rows in shard 0; full file: [units_shard0.jsonl](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/bilinear/units_shard0.jsonl)):

<details>
<summary>Rank-1 bilinear, fold 0, novel-prefix folds — 9 weight-decay × seed variants (verbatim JSONL row 4)</summary>

```json
{"scheme": "prefix", "fold": 0, "r": 1, "basis": "pca48", "smoke": false, "row_limit": null, "epochs_ran": [89, 89, 89, 116, 116, 116, 98, 98, 98], "variants": [{"seed": 0, "wd": 0.0, "inner_val_mse": 1.977834939956665, "r2_te": 0.8889071785354945}, {"seed": 0, "wd": 0.0001, "inner_val_mse": 1.9778354167938232, "r2_te": 0.8889072815866981}, {"seed": 0, "wd": 0.01, "inner_val_mse": 1.9778767824172974, "r2_te": 0.8889161787850471}, {"seed": 1, "wd": 0.0, "inner_val_mse": 1.9860609769821167, "r2_te": 0.8890537988625065}, {"seed": 1, "wd": 0.0001, "inner_val_mse": 1.9860622882843018, "r2_te": 0.8890538866445772}, {"seed": 1, "wd": 0.01, "inner_val_mse": 1.9861361980438232, "r2_te": 0.8890640769989516}, {"seed": 2, "wd": 0.0, "inner_val_mse": 1.974310040473938, "r2_te": 0.8891033651138742}, {"seed": 2, "wd": 0.0001, "inner_val_mse": 1.974310278892517, "r2_te": 0.8891034604831626}, {"seed": 2, "wd": 0.01, "inner_val_mse": 1.9743661880493164, "r2_te": 0.8891128872305242}]}
```

</details>

One follow-up-refit fold record, cherry-picked for illustration (fold 0 of 6 in the `per_fold` array; full file: [query_averaged_refit.json](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/ladder/query_averaged_refit.json)):

<details>
<summary>Prefix-averaged arm, inner-validation λ refit, fold 0 (verbatim JSON entry)</summary>

```json
{"fold": 0, "chosen_lambda": {"pca48": 1000.0, "ambient": 1000.0}, "fold_r2": {"pca48": -0.2504443981438178, "ambient": -0.22073347361374895}, "inner_val_r2_per_lambda": {"pca48": {"0.01": -27.678046302500807, "0.1": -23.353914286072357, "1.0": -15.52233769957888, "10.0": -8.940641816103799, "100.0": -3.3901171200547893, "1000.0": -0.3364793067720251}, "ambient": {"0.01": -32.41480793055853, "0.1": -26.676418170032616, "1.0": -16.462089279935263, "10.0": -8.693664160777173, "100.0": -2.9984719444570507, "1000.0": -0.28673852601852756}}, "n_tr": 14578, "n_te": 2730, "n_inner_tr": 11608, "n_inner_val": 2970, "inner_seed": 1234, "wall_s": 59.12931058742106}
```

</details>

One dedup-refit unit record, cherry-picked for illustration (1 of 8 rows; full file: [units_n50k_refit.jsonl](https://github.com/superkaiba/explore-persona-space/blob/adbf2982bb85f86847ced0ac43a708fb24d5006b/eval_results/issue_1775/fu_dedup_refit_pcfold_doubly/units_n50k_refit.jsonl)):

<details>
<summary>Kernel rung on the deduplicated train set, layer 19 (verbatim JSONL row 2)</summary>

```json
{"variant": "deduped", "rung": "krr", "layer": 19, "smoke": false, "n_train": 47704, "n_boot": 1000, "krr_grid": "gm=1.0;lam=0.1,10.0", "whole_map_r2": 0.7865001584865536, "mean_cosine": 0.9604961609166067, "bootstrap_ci": {"r2": {"point": 0.7865001584865536, "lo": 0.7756914589622746, "hi": 0.7970719804397375}, "mean_cosine": {"point": 0.9604961609166067, "lo": 0.9583692651661492, "hi": 0.9625626270035124}, "n_test": 1000}, "fit_meta": {"n_train": 47704, "kernel": "exact RBF (no Nystrom)", "base_gamma": 0.00017336100982364133, "selected": {"gamma_mult": 1.0, "gamma": 0.00017336100982364133, "lambda": 0.1, "val_r2": 0.7881321090201635}, "kernel_block": 4096, "val_grid": [{"gamma_mult": 1.0, "gamma": 0.00017336100982364133, "lambda": 0.1, "val_r2": 0.7881321090201635}, {"gamma_mult": 1.0, "gamma": 0.00017336100982364133, "lambda": 10.0, "val_r2": 0.721488036617014}]}, "wall_s": 30.617651784999907}
```

</details>

## Results

### A rank-32 bilinear interaction closes 93% of the additive→full-context gap on novel prefixes, but a real unnamed residual remains

What is plotted: pooled held-out R² (48 answer PCs, novel-prefix 6-fold, 17,308 rows in 996 prefix groups) versus interaction rank r (log2 axis), with the additive-stitch ridge, full-context ridge, stitch-MLP 95% cluster band, and selected rank 32 as reference levels.

![Gap-closure curve: held-out R2 versus interaction rank, rising from the stitch-ridge level toward the stitch-MLP band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/hero_gap_closure_with_mlp.png)

> **Figure.** *Adding rank-r prefix×query product terms to the additive map closes most of the gap to the full-context linear map.* Curve = outer-test R² per rank (exploratory; the headline reads at the inner-validation-selected r = 32). Levels: stitch ridge 0.849 → bilinear 0.910 → stitch-MLP ensemble 0.920; full-context ridge 0.914.

| Quantity (novel-prefix, 48-PC space) | Value | 95% cluster CI |
|---|---|---|
| Named-interaction gain (bilinear r = 32 − r = 0, same optimizer protocol) | +0.0493 | [0.0468, 0.0521] |
| De-regularization control (r = 0 refit − stitch PRESS-ridge) | +0.0113 | [0.0101, 0.0127] |
| Unnamed residual (stitch-MLP ensemble − bilinear r = 32) | +0.0104 | [0.0094, 0.0113] |
| Ambient-space companion, named gain (r ∈ {0, 32}) | +0.0383 | [0.0362, 0.0405] |
| Gap fraction closed (bilinear − stitch) / (context − stitch) | 0.932 | — (0.85 of the MLP-reachable gap) |

The per-fold companion below shows the same five levels fold by fold.

![Per-fold dots for the five headline R2 levels](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61aa3a3b0ab4a28df973b5bdd6b3d7ede70f095e/figures/issue_1775/hero_gap_closure_perfold.png)

> **Figure.** *Fold-level spread is small next to the level separations.* Novel-prefix folds: stitch ridge 0.838–0.858, r = 0 refit 0.849–0.871, r = 32 0.902–0.917, stitch-MLP 0.914–0.928, full-context ridge 0.910–0.920; the stitch-MLP tops the full-context ridge in all 6 folds. Novel-query panel: bilinear 0.59–0.71 against stitch-MLP 0.70–0.77.

Both the named gain and the unnamed residual exclude zero: "partially named" on the verdict lattice. The de-regularization control is small (+0.011); the gain is not an optimizer artifact.

The stronger claim that the nonlinearity just IS the interaction is not licensed: the bilinear's own residuals still reject (distance correlation 0.687, p = 0.001, prefix-block and within-prefix permutations). And the disjoint-input stitch-MLP (0.920) sits at or slightly above the full-context ridge (0.914; +0.006, no paired CI): what the linear read of the mixed forward state misses is, to within that margin, recoverable from the separate prefix and query states.

### The verdict survives the rank cap, fold-scheme, and PC-basis changes; the unnamed share grows when queries are novel

What is plotted: outer-test R² (48-PC space) versus bilinear interaction rank under novel-query 6-fold cross-validation — the exploratory rank curve for the companion fold scheme; inner validation selects r = 16 here.

![Bilinear rank curve under novel-query folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/expl_rcurve_query.png)

> **Figure.** *The interaction gain persists when whole queries are held out.* Outer-test R² per rank, novel-query folds; levels 0.636 (rank 0) → 0.663 (r* = 16) → 0.666 (rank 64); the stitch-MLP ensemble sits at 0.727.

| Robustness read (48-PC space) | Value | 95% cluster CI |
|---|---|---|
| Named gain, novel-query (r* = 16; 1,397 query groups) | +0.0268 | [0.0208, 0.0335] |
| Named gain, novel-prefix under train-fold-only PC bases (r ∈ {0, 32}) | +0.0497 | [0.0472, 0.0525] |
| Named gain, doubly-novel (r ∈ {0, 32}; two-way, 921 × 636 groups, 2,900 rows) | +0.0283 | [0.0183, 0.0397] |
| Unnamed residual, novel-query (stitch-MLP ensemble − bilinear) | +0.0638 | [0.0516, 0.0753] |
| Unnamed residual, doubly-novel (stitch-MLP ensemble − bilinear r = 32) | +0.0578 | [0.0483, 0.0677] |
| Exploratory r = 64 point, novel-prefix (implied residual ≈ +0.0089, still positive) | 0.9112 | — |

The named gain is sign-consistent across all three fold schemes and both PC bases; the verdict is not a rank-cap artifact (the r = 64 point still leaves a positive residual). The unnamed share grows wherever query generalization is demanded.

The companion below shows both follow-up reads per fold.

![Per-fold dots for the PC-provenance and doubly-novel checks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/adbf2982bb85f86847ced0ac43a708fb24d5006b/figures/issue_1775/fu_robustness_perfold.png)

> **Figure.** *PC provenance barely moves any level; the doubly stitch-MLP clears the bilinear in every fold.* Left: filled = train-fold-only PC bases, open = full-population PCs; paired levels agree to within 0.0014 in all 6 folds. Right: doubly-novel stitch-MLP folds 0.685–0.749 against the pooled bilinear 0.661.

The follow-up round closed both open cells: train-fold-only PC bases reproduce the headline gain (vs +0.0493 with full-population PCs), and the fold-basis rank-0 refit sits 0.011 from the PRESS stitch level, inside its 0.02 gate.

The doubly stitch-MLP (0.719) beats the bilinear (0.661) by twice that scheme's named gain; the rank is carried from the prefix scheme. The named gain never touches the PRESS baseline, so the λ-selection collapse documented below cannot contaminate it.

### Per-arm nonlinearity gains are small on prefix-input arms and modest on query-bearing ones

What is plotted: pooled held-out R² per arm × fitter (48-PC space, per-row grain), grouped bars, under novel-prefix folds (left panel) and novel-query folds (right panel); whiskers are 95% cluster CIs of each fitter's gain over the linear rung. The prefix-averaged arm's hatched linear bar is the follow-up refit (next result).

![Grouped bars of held-out R2 per arm and fitter under novel-prefix and novel-query folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/hero_ladder_bars.png)

> **Figure.** *Kernel, random-features, and MLP fitters beat ridge modestly on most arms.* Held-out R², 48-PC space, per-row grain; whiskers = 95% cluster-bootstrap CI of the gain vs the linear rung; hatched bar = inner-validation-λ linear refit of the prefix-averaged arm (its PRESS read is degenerate).

| Arm (novel-prefix, per-row) | Linear R² | Best nonlinear | Kernel gain, 48-PC (CI) | Kernel gain, ambient (CI) |
|---|---|---|---|---|
| Prefix end-state | 0.098 | 0.110 | +0.0114 [0.0100, 0.0127] | +0.0148 [0.0133, 0.0162] |
| Prefix-averaged context state | −0.280 (refit) | 0.115 | +0.394 [0.358, 0.431] | +0.328 [0.302, 0.355] |
| Bare query state | 0.740 | 0.760 | +0.0200 [0.0182, 0.0221] | +0.0407 [0.0375, 0.0444] |
| Prefix + query stitch | 0.849 | 0.920 | +0.0712 [0.0682, 0.0746] | +0.106 [0.103, 0.110] |

Prefix-end gains are small absolutely but a sizable fraction of that arm's low per-row ceiling (≈ 0.11 between-prefix variance share); its MLP gain is ≈ +0.007 across 3 seeds. Gains run larger in ambient space; the 48-PC compression absorbs part of the residual.

Under novel-query folds the PRESS baselines collapse (stitch 0.353) because PRESS selects λ on train rows that share queries; against the honest linear reference (the rank-0 refit, 0.636) the nonlinear headroom is ≈ +0.09 (kernel 0.725, MLP 0.727). Per-fold points behind the linear bars appear in the next result; stitch-MLP fold values sit in the headline companion; other rungs' fold reads are recomputable from the persisted held-out prediction shards.

### The prefix-averaged arm is wholly nonlinear at per-row grain: a follow-up refit replaces its degenerate linear read

What is plotted: per-fold held-out R² of every scheduled-PRESS linear fit (48-PC space; 6 dots per arm × fold-scheme group). Red open circles overlay the follow-up refit's 6 fold values for the prefix-averaged arm, overlapping near −0.28 at this scale.

![Per-fold linear R2 dots: most arms cluster above zero; the prefix-averaged arm's PRESS folds sit far below](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/expl_per_fold_r2.png)

> **Figure.** *The PRESS failure is uniform, not an outlier fold.* All 6 PRESS folds of the prefix-averaged arm land between −7.4 and −9.1, while the inner-validation-λ refit brings every fold to −0.24 to −0.34 (λ rails at the grid maximum 1000 in all 12 fits).

The arm's input — the leave-own-row-out mean of the prefix's other context states — is near-constant within a prefix, so PRESS λ selection over train rows explodes on held-out prefixes. The originally recorded "+8.1–8.3 gains" were recovery from a broken baseline, not nonlinearity.

The corrected read: the best in-grid linear map sits below zero per-row (−0.280 in 48-PC space / −0.244 ambient, every fold) while kernels and MLPs reach ≈ 0.11; recomputed gains are +0.394 (48-PC) and +0.328 (ambient), cluster CIs in the table above. The per-row signal is real but entirely nonlinear, saturating near the between-prefix ceiling; averaged-grain reads stay healthy (48-PC ridge 0.877). Scope: the arm reads sibling context states, a coarser transport grain than pure prefix-only.

### Residual dependence is detectable almost everywhere the instrument can see — 23 of 26 informative tests reject

What is plotted: Holm-adjusted permutation p-values for the 30-test family — 5 arms × 3 block-permutation schemes × 2 statistics (HSIC, distance correlation) — on held-out linear-map residuals; the dashed reference is 0.05. Tick labels read arm, permutation scheme, statistic.

![Bar chart of Holm-adjusted p-values for thirty residual-dependence tests; twenty-five bars sit at the permutation floor near zero and five rise above the 0.05 line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/expl_detection_holm.png)

> **Figure.** *Most residuals carry detectable structure.* 25 of 30 tests reject at the permutation resolution floor (Holm p = 0.030); of those, 2 are degenerate-null artifacts — the prefix-end distance-correlation cells under query-block and derangement, where the null 95th quantile equals the observed statistic to 9 significant figures (0.14883769998510446 vs 0.1488377000037514).

Net accounting: 23 informative rejections, 3 informative non-rejections (bare-query under prefix-block, HSIC; full-context under query-block, both statistics), and 4 degenerate prefix-end cells where the null equals the observed statistic. That arm is informative only under prefix-block, which crosses fold boundaries; treat its detection as fold-boundary-dependent.

The planted-effect power check passed with a measured minimal detectable effect ≈ 0.05 R²-equivalent, so an informative non-rejection reads "no structure above ≈ 0.05", never "the map is linear". Detection also does not by itself promise a usable gain: heteroscedasticity rejects too, and the prefix-averaged arm rejects on all 6 of its tests (distance correlation 0.807) with nil per-row linear signal.

### The banked 50k fitter comparison is contaminated; the deduplicated 1M comparison is the citable context-arm gain

What is plotted: histogram (log-count y-axis) of each of the 1,400 banked val/test targets' maximum MinHash-estimated Jaccard similarity to any of the 50,000 train prompts in the banked 50k split; the dashed line is the 0.8 near-duplicate criterion.

![Histogram of per-target maximum Jaccard overlap between 50k train prompts and evaluation targets, with a heavy spike past the 0.8 line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/expl_n50k_contamination.png)

> **Figure.** *A heavy right tail sits past the near-duplicate criterion.* 297 exact train↔target duplicates (154 targets) and 35,073 near-duplicate pairs at 5-gram Jaccard ≥ 0.8, touching 325 of 1,400 targets — far above the 14-target (1%) audit threshold; worst-case sensitivity bound ≤ 0.325 of test R².

The audit threshold tripped: the banked 50k gain was quoted unaudited until the follow-up refit below, and the deduplicated 1M comparison carries the larger-n citable context-arm gain — ridge 0.754 → MLP 0.810 at L19 (≈ +0.056; 0.671 → 0.759 at L14). The 1M re-verification is clean: 0 exact, 0 residual near-duplicates on the 960,000-prompt manifest pool (the banked fit realized 963,444 rows, a 0.4% scope note).

One audit limit stands: the criterion is lexical only — semantic paraphrases are unscreened. The conditional dedup refit this audit called for ran in the follow-up round; the next result carries the audited gain.

### The kernel gain survives deduplication: contamination inflated the 50k comparison's levels, not its fitter gap

What is plotted: held-out R² per fitter rung for three variants of the banked 50k comparison — banked (unaudited), a refit on the deduplicated train set (2,296 implicated rows dropped; n = 47,704), and an n-matched random-drop control — on the same sha-pinned 1,000 test rows (left); each refit rung's gain over ridge with 95% paired row-bootstrap whiskers (right).

![Grouped bars of held-out R2 per rung for the banked, deduplicated, and random-drop variants, and gains over ridge with confidence whiskers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/adbf2982bb85f86847ced0ac43a708fb24d5006b/figures/issue_1775/fu_n50k_dedup_bars.png)

> **Figure.** *Deduplication lowers every rung's level by 0.016–0.020 while the same-n random drop reproduces the banked numbers; the gains barely move.* Kernel−ridge: deduplicated +0.043, random-drop +0.047 (banked +0.047); MLP−ridge +0.017 against +0.019; residual-skip stays marginally negative in every variant.

| Gain over ridge (layer 19, 1,000 test rows, paired row bootstrap) | Deduplicated | Random-drop control (same n) |
|---|---|---|
| Kernel (RBF) | +0.0429 [0.0404, 0.0454] | +0.0466 [0.0434, 0.0501] |
| MLP | +0.0173 [0.0152, 0.0195] | +0.0190 [0.0172, 0.0210] |
| Residual-skip MLP | −0.0020 [−0.0024, −0.0016] | −0.0019 [−0.0023, −0.0016] |

On the audit's decision rule the gain survives deduplication: the deduplicated kernel gain's CI excludes zero on the positive side.

The random-drop control reproduces the banked levels to ≈ 0.0003, so the level drop under dedup (ridge 0.760 → 0.744) is attributable to the duplicates, not the smaller train set — the near-duplicates made the test easier for every fitter roughly equally, inflating levels while leaving the nonlinear gap nearly intact (the kernel gain shrinks ≈ 8%).

Identity+bias stays strongly negative (−0.88); kNN retrieval reaches acc@1 0.80–0.85 vs chance 0.001. The criterion remains lexical only.

### The interaction writes into the answer manifold's high-variance subspace, with no trait-specific alignment beyond it

What is plotted: four bars of maximum absolute cosine over the 576 fitted interaction terms (pooled across folds and seeds, novel-prefix scheme, r = 32): output directions against the answer PCs and against the 3-trait persona-vector dictionary, and prefix-side and query-side input directions against the dictionary.

![Four bars of maximum absolute cosine: output directions reach 0.77 against answer PCs and 0.70 against the trait dictionary, while prefix-side and query-side input directions stay below 0.07](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775/expl_projection_cosines.png)

> **Figure.** *Output directions align with the answer manifold; input directions align with nothing.* Max abs cosine 0.768 (answer PCs) and 0.699 (trait dictionary) for output directions; 0.066 / 0.054 for input directions. Significance: p ≈ 0.001 vs the matched-norm isotropic null, but p = 1.0 / 0.257 vs the covariance-matched null.

Against the isotropic null the output-direction alignments look strong; under the covariance-matched null (draws from the train-population answer covariance, max-selection applied identically per draw) neither survives. The interaction writes into the answer summary's dominant covariance subspace, as any well-fit component must, with no trait-specific alignment beyond that.

Input-side projections are indistinguishable from chance under both nulls. The answer-PC read covers the breadth and is likewise covariance-explainable; the dictionary read is conditional on a 3-trait dictionary.

---

**Repro:** Compute: two GCE flex-start 2× A100-80 instances (main run ≈ 7.2 h wall; targeted nonlinear-ladder re-run ≈ 6.6 h wall; ≈ 28 GPU-h realized vs 14 booked — the re-run followed a cross-shard ordering bug caught by the run's own completeness reconcile) plus VM CPU phases (fold audit ≈ 1 h; figures; the zero-GPU refit round ≈ 6 min); second follow-up round: one GCE 2× A100-80 instance, 1.56 GPU-h realized vs 4.5 booked. Code @ [`e9ef9d9dde`](https://github.com/superkaiba/explore-persona-space/tree/e9ef9d9ddeba76dd55720dafba6d66001abb6477): [`scripts/issue1775_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_run.sh), [`scripts/issue1775_fold_check.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_fold_check.py), [`scripts/issue1775_ladder.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_ladder.py), [`scripts/issue1775_detection.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_detection.py), [`scripts/issue1775_bilinear.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_bilinear.py), [`scripts/issue1775_delta_beyond.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_delta_beyond.py), [`scripts/issue1775_query_averaged_refit.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_query_averaged_refit.py), [`scripts/issue1775_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/scripts/issue1775_figures.py); per-fold headline companion (producer + data + figure) @ [`61aa3a3b0a`](https://github.com/superkaiba/explore-persona-space/tree/61aa3a3b0ab4a28df973b5bdd6b3d7ede70f095e): [`scripts/issue1775_perfold_headline.py`](https://github.com/superkaiba/explore-persona-space/blob/61aa3a3b0ab4a28df973b5bdd6b3d7ede70f095e/scripts/issue1775_perfold_headline.py), [`bilinear/perfold_headline.json`](https://github.com/superkaiba/explore-persona-space/blob/61aa3a3b0ab4a28df973b5bdd6b3d7ede70f095e/eval_results/issue_1775/bilinear/perfold_headline.json). Artifacts (git @ the e9ef9d9dde pin): [`eval_results/issue_1775/fold_check/`](https://github.com/superkaiba/explore-persona-space/tree/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/fold_check), [`ladder/`](https://github.com/superkaiba/explore-persona-space/tree/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/ladder) (incl. `query_averaged_refit.json` + per-unit JSONLs), [`detection/hsic_dcor.json`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/detection/hsic_dcor.json), [`bilinear/`](https://github.com/superkaiba/explore-persona-space/tree/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/bilinear) (incl. `bilinear_fits.json`, `delta_beyond_analysis.json`, `interaction_projections.json`), [`plan_deviations.json`](https://github.com/superkaiba/explore-persona-space/blob/e9ef9d9ddeba76dd55720dafba6d66001abb6477/eval_results/issue_1775/plan_deviations.json), figures [`figures/issue_1775/`](https://github.com/superkaiba/explore-persona-space/tree/e9ef9d9ddeba76dd55720dafba6d66001abb6477/figures/issue_1775). Held-out prediction shards, permutation-null matrices, fitted bilinear parameters, and refit prediction shards: [HF data repo @ `issue1775_nonlinearity/analysis_tensors`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7017cc3c023c18963eb26d9078aa2dd049acf14/issue1775_nonlinearity/analysis_tensors) (`heldout_preds/`, `null_matrices/`, `bilinear_params/`, `qa_refit_preds/`). Follow-up round (dedup refit + fold-PC + doubly stitch-MLP) — drivers @ pod-side commit [`8b8237dc9e`](https://github.com/superkaiba/explore-persona-space/tree/8b8237dc9e665ff97ec0eec1e10132ca3bbe2a5c): [`scripts/issue1775_n50k_dedup_refit.py`](https://github.com/superkaiba/explore-persona-space/blob/8b8237dc9e665ff97ec0eec1e10132ca3bbe2a5c/scripts/issue1775_n50k_dedup_refit.py), [`scripts/issue1775_doubly_mlp.py`](https://github.com/superkaiba/explore-persona-space/blob/8b8237dc9e665ff97ec0eec1e10132ca3bbe2a5c/scripts/issue1775_doubly_mlp.py), the train-fold-only-PC path in [`scripts/issue1775_bilinear.py`](https://github.com/superkaiba/explore-persona-space/blob/8b8237dc9e665ff97ec0eec1e10132ca3bbe2a5c/scripts/issue1775_bilinear.py), dispatcher [`scripts/issue1775_fu_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/8b8237dc9e665ff97ec0eec1e10132ca3bbe2a5c/scripts/issue1775_fu_run.sh); round eval JSONs + unit JSONLs + figure producer @ [`adbf2982bb`](https://github.com/superkaiba/explore-persona-space/tree/adbf2982bb85f86847ced0ac43a708fb24d5006b): [`eval_results/issue_1775/fu_dedup_refit_pcfold_doubly/`](https://github.com/superkaiba/explore-persona-space/tree/adbf2982bb85f86847ced0ac43a708fb24d5006b/eval_results/issue_1775/fu_dedup_refit_pcfold_doubly), [`scripts/issue1775_fu_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/adbf2982bb85f86847ced0ac43a708fb24d5006b/scripts/issue1775_fu_figures.py). Round tensors on HF @ `5539263032`: [fu_dedup_refit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55392630322225433dcfed2b988fb6173a2fb9d7/issue1775_nonlinearity/fu_dedup_refit) (per-variant × rung prediction shards, split + drop/control position arrays, reduced layer-19 capture arrays) and [analysis_tensors/bilinear_params_foldpc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/55392630322225433dcfed2b988fb6173a2fb9d7/issue1775_nonlinearity/analysis_tensors/bilinear_params_foldpc). Reused inputs: activation stores + corpus manifest from [#1092](https://eps.superkaiba.com/tasks/1092) ([HF, issue1092_realistic_crossing](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7017cc3c023c18963eb26d9078aa2dd049acf14/issue1092_realistic_crossing) — fit: same corpus, layer, folds, and battery exclusion as the banked linear reads; reproduction gate passed exactly); banked fitter-comparison splits + manifests/chunks from [#779](https://eps.superkaiba.com/tasks/779) ([HF, issue779_monitoring](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7017cc3c023c18963eb26d9078aa2dd049acf14/issue779_monitoring) — fit: the audit applies that round's own near-duplicate criterion and sha-pinned splits); 3-trait persona-vector dictionary from the same monitoring line ([r_b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d7017cc3c023c18963eb26d9078aa2dd049acf14/issue779_monitoring/r_b) — fit: read-only projections; dictionary narrowness carried as a caveat); dependence-test implementations from [#763](https://eps.superkaiba.com/tasks/763) (observed statistics only; the batched permutation driver is new).

**Context:** created 2026-07-28 as a child of [#1092](https://eps.superkaiba.com/tasks/1092) (the crossed corpus + banked linear maps this task fits over), split out of the four-arm theoretical-analysis plan as Task B (sibling Task A = [#1774](https://eps.superkaiba.com/tasks/1774)); plan approved + run 2026-07-29 (GCE, two instances); interpretation rounds 1–2 and this body 2026-07-29. A zero-GPU free-analysis follow-up round (`query_averaged per-row re-fit, group-respecting inner-val lambda`, run 2026-07-29) replaced the degenerate per-row read of the prefix-averaged arm and is folded into the ladder results above. A second same-issue follow-up round (`dedup-refit-pcfold-doubly-mlp`, plan v8, run 2026-07-30 UTC, 1.56 GPU-h realized) executed the deferred dedup refit with its n-matched random-drop control, the train-fold-only-PC sensitivity check, and the doubly-novel stitch-MLP; its three results are folded into the fold-hygiene and robustness sections above. Originating prompt, verbatim:

> run both in background with ahppy coder (2026-07-28; two-task split per: 'can it not be all one task? Or at least separate the nonlinear out')
