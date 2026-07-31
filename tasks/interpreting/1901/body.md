---
title: 'Characterize mapping-quality metrics: 1M-context linear + nonlinear maps vs
  baselines on every metric'
kind: experiment
tags: []
created_at: '2026-07-30T23:49:53Z'
has_clean_result: false
origin_prompt: 'Run this in background with happy coder:

  ## Motivation

  We have come up with a bunch of different metrics for measuring the quality of the
  mapping. I want to characterize each and see how the 1m context linear and nonlinear
  mapping on them vs baselines, as well as what each metric is measuring exactly


  - [ ] [[I think a good baseline would be learned bias]], W = identity - another
  metric besides R^2 would be e.g. P(answer summary is in k nearest neighbors of prediction)'
workflow: v1
goal: Characterize every mapping-quality metric in use (held-out R², kNN retrieval
  acc@k euclidean+cosine, and companions) — what construct each measures, what it
  rules out, where they dissociate — and report the ~1M-context (n≈963k LMSYS) linear
  and nonlinear context→answer mappings against the full baseline ladder (constant
  train-mean; W = identity; identity + learned bias) on ALL metrics, in both the context-based
  and prefix-based arms.
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Held-out variance explained and retrieval accuracy dissociate in both directions across the context-to-answer estimator ladder (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- **Both dissociation directions realized: identity plus bias scores acc@1 0.53 at pooled R² −0.92 (n=1000); a shrinkage-limited prefix ridge scores R² +0.54 at chance-level retrieval 0.04.**
- Fitted maps beat every baseline on all non-saturated metrics; the nonlinear-over-ridge acc@1 gap grows with pool size, 0.036 at 1,000 candidates to 0.101 at 100,000.
- Shuffled-pair nulls collapse to metric-specific floors; raw cosine's floor is 0.68 and the constant train-mean scores 0.798, so absolute cosine says little without the null.
- Retrieval hubs are corpus-structured: test rows are 1% of the 100,000 pool yet 61-70 of fitted maps' top-100 cosine hubs; the correction still buys 0.03-0.10 acc@1.
- The identity-vs-fitted retrieval flip tracks training-set size: ridge falls 0.81 to 0.065 acc@1 as training rows drop 963k to 50; identity plus bias stays near 0.50.
- Scope: all test targets are LMSYS rows under one pinned split; headline layer 19 is inherited; layers 14/26 reproduce every ordering.

## Goal

- **This experiment in context:** The standing rule to co-report an identity-plus-learned-bias baseline and a kNN-retrieval read beside held-out R² came from two single-estimator reads: identity plus bias reached acc@1 0.84 at pooled R² −6.5 on the 50-context prefix battery ([#722](https://eps.superkaiba.com/tasks/722)), while the fitted ridge dominated retrieval on the 963k-context map ([#779](https://eps.superkaiba.com/tasks/779)); [#1776](https://eps.superkaiba.com/tasks/1776) verified the banked weights reproduce to 8e-11. This experiment turns those spot reads into one systematic reference — the full estimator ladder crossed with every metric, both mapping arms, pool-size and training-size axes — under the same split, layers, and banked weights (directly comparable).
- **Broader narrative:** The context-to-answer mapping line treats a context's effect on the model as a predictable vector; every future map-quality claim rests on these metrics, so their constructs, floors, and failure modes need one canonical measurement.

## Methodology

- **Design:** Evaluation-only battery; the manipulated variable within each read is the estimator. Context arm: 14 estimator rungs (7 fitted or baseline estimators at 963,444 training rows, 5 companions at 3,600, 2 at 50) crossed with 8 metrics (pooled R², per-dimension R², mean cosine, kNN acc@k under euclidean and cosine distance, hubness-corrected acc@k, median rank, mean reciprocal rank, hubness diagnostic), 3 layers (19 headline; 14/26 exploratory), and 4 candidate pools (1,000 / 5,000 / 20,000 / 100,000). Prefix arm: 7 estimators over the 50-context battery (7 prefix families), leave-one-family-out. Bootstrap CIs use n=1000 shared test-row resamples; every metric gets a 200-draw shuffled-pair null per arm. Both mapping arms run (context-based and prefix-based); the prefix arm runs at n=50 because no larger prefix-level mapping data is persisted — a stated deviation carried as a scope caveat.
- **Training:** N/A — no model training in this task. The four fitted maps are banked weight payloads applied unchanged (in-run reproduction assert: applied ridge pooled R² matches the banked 0.7541708417500046 within 1e-6; realized delta 8.1e-11). Their production recipe, inlined: fit on 963,444 mixed LMSYS+WildChat contexts (pinned split seed 42; 400 validation / 1,000 test rows held out with sha-pinned membership; the training pool near-dupe-screened against the 1,400 validation/test targets at 5-gram Jaccard 0.8 — 435 exact and 30,437 near duplicates dropped), inputs = the last-context-token activation of Qwen2.5-7B-Instruct at layer ℓ, targets = the mean answer-token activation at the same layer, layers 14/19/26. Ridge: streaming fp64 primal regression, 23-point λ grid from 1e-3 to 1e8, validation-selected λ = 0.001. Neural maps: one GELU hidden layer at widths 8,192 and 32,768, AdamW lr 3e-4, batch 4,096, validation early stopping. Kernel map: RBF kernel ridge, Nyström with 16,384 landmarks, γ at the median heuristic, λ validation-selected at 0.1. Complete analysis-knob table:

  | Knob | Value | Source |
  |---|---|---|
  | Battery seed (distractors, nulls, bootstrap) | 1901 | plan §10 |
  | Split (train/val/test) | 3600 base + 959,844 new / 400 / 1000, seed 42, sha-pinned | banked split metadata (`n1m_fits.json`) |
  | Layers | 19 headline (validation-selected upstream); 14, 26 exploratory | banked fit metadata |
  | Banked ridge recipe | streaming fp64 primal; λ grid 23 points 1e-3 to 1e8; selected λ = 0.001 | banked fit metadata + `docs/methodology/issue_779.md` |
  | Banked neural-map recipe | 1 GELU hidden layer, widths 8192 / 32768; AdamW lr 3e-4; batch 4096; val early stop | `docs/methodology/issue_779.md` (fit constants) |
  | Banked kernel recipe | RBF Nyström m = 16384; γ median-heuristic; λ ∈ {0.1, 10}, selected 0.1 | banked fit metadata |
  | Small-n ridge rungs | banked λ = 1000 at n = 3600 and n = 50 (fixed, no re-selection) | `context_arm.json` `small_n_meta` |
  | Retrieval k values | (1, 5, 10) context; (1, 3, 5) prefix | `mapping_baselines.knn_retrieval` default + banked convention |
  | Candidate pools | 1000 (test) / 5000 (test+train+val rows) / 20,000 / 100,000 (test+distractors) | plan §11 (decade-spaced; bilingual-lexicon evals use up to 200k, arXiv 1710.04087) |
  | Hubness-corrected score | CSLS, cross-domain neighborhoods, k = 10 | arXiv 1710.04087 |
  | Bootstrap | n_boot = 1000, test-row resampling, one shared draw matrix across estimators | banked CI convention (`BOOT_N`) |
  | Null | K = 200 shuffled-pair permutations per metric × arm | plan §11 (banked 200-draw null convention) |
  | Distractor pool | seeded 210 of 1,920 capture chunks streamed; 100,000 layer-19 rows kept | plan §11 + `distractor_manifest.json` |
  | Prefix-arm neural rung | batched leave-one-family-out MLP, width 512, 300 epochs | banked battery recipe (`vectorized_mlp_skill`) |
  | Compute env | VM CPU only; 8 BLAS threads; `MALLOC_ARENA_MAX=2`; numpy 2.2.6, torch 2.8.0 | `context_arm.json` metadata |

- **Evaluation:** All dependent variables are the mapping metrics themselves, computed deterministically over banked activations — no LLM judge and no new generation anywhere. Per metric: pooled R² = 1 − SS_res/SS_tot on the held-out set (variance-weighted across dimensions; unbounded below); per-dimension R² (exposes the variance weighting); mean cosine between prediction and target (per-row scale-invariant; high anisotropy floor); kNN acc@k = the probability the true answer vector is within the k nearest pool neighbors of the prediction (rank-based, mid-rank tie handling; analytic chance k/n_pool); the CSLS hubness-corrected variant of acc@k; median rank and mean reciprocal rank; and a hubness diagnostic (skewness of 10-occurrence counts, with top-hub corpus composition). Correctness asserts: banked-ridge reproduction within 1e-6, realized payload keys for all 12 weight files, helper-parity checks per retrieval cell, and 0 duplicate groups in the distractor pool. Known metric ceilings that bound interpretation: 58 of the 1,000 test targets have an exact in-pool duplicate vector, capping acc@1 near 0.94 equally for every estimator (ordering unaffected, resolved by k=5), and the 5,000-row pool carries 354 excess duplicate rows with the same equal-across-arms effect; median rank saturates at 1 for every decent map at pool 1,000.
- **Data extraction:** Context arm: real-user single-turn conversations from LMSYS-Chat-1M and WildChat-1M (realism tier 1); answers were generated on-policy by Qwen2.5-7B-Instruct in the banked capture, and activations recorded by teacher-forced re-forward. This task adds only the distractor pool: a seeded random subset of 210 of 1,920 capture chunks, streamed one chunk at a time (peak footprint one chunk), keeping 100,000 layer-19 answer vectors (55,686 LMSYS / 44,314 WildChat; 0 duplicate vectors in-pool; disjoint from the 5,000 fit-round rows by construction) — uploaded to the HF data repo before analysis. The 5,000-row pool reuses rows that were near-dupe-screened against the test targets upstream, so retrieval there is easier than at the unscreened 20,000/100,000 distractor pools, which are the more natural read. Prefix arm: the inherited 50-context battery (families: persona, behavior, format, in-context-learning, rephrase, WildChat, default assistant — constructed conditions, realism tier 3, a scope caveat), prefix-level last-input-token activations and mean answer summaries, query-averaged.
- **Sample training/evaluation data + completions:** This task generates no completions (deterministic numeric battery over banked activations); the worked examples below are verbatim artifact rows. Disclosure: 1 of 448 context-arm retrieval cells, chosen as the headline cell (ridge, test pool, euclidean); full file: [context_arm.json](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery/context_arm.json).

  ```json
  {"acc_at_k": {"1": 0.805, "5": 0.896, "10": 0.945},
   "chance_at_k": {"1": 0.001, "5": 0.005, "10": 0.01},
   "median_rank": 1.0, "mrr": 0.8502544488804233, "n": 1000, "n_pool": 1000,
   "helper_parity": "PASS",
   "null": {"acc1_mean": 0.000915, "acc1_p975": 0.003,
            "mrr_mean": 0.007346965163888691, "median_rank_mean": 501.99125},
   "acc1_ci": {"lo": 0.777, "hi": 0.83},
   "mrr_ci": {"lo": 0.8291832391978571, "hi": 0.8701081979849409}}
  ```

  Disclosure: 1 of 9 per-metric characterization entries (pooled R², the headline metric); full file: [metric_characterization.json](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery/metric_characterization.json).

  ```json
  {"construct": "fraction of held-out target variance the map explains, variance-weighted across dims",
   "computed_as": "1 - ||Y-Yhat||^2_F / ||Y-Ybar||^2_F, SS_tot on the test set's own mean (banked whole_map_r2 convention)",
   "invariances": "sensitive to scale AND offset; dominated by high-variance dims; unbounded below",
   "constant_predictor_score": -0.044533745181768225,
   "empirical_null_floor": {"mean": -0.7618102512485427, "p975": -0.7350515170534669},
   "failure_modes": [
     "a context-independent shift scores catastrophically negative while retrieval stays high (identity+bias)",
     "variance-weighting hides dims the map never moves",
     "a regularization-limited fit can score pooled R2 >> 0 from between-group/mean structure while per-context retrieval sits at chance (this run: prefix ridge-LOFO pooled R2 +0.541 at acc@1 0.040, chance 0.02, with all 7 prefix families at family-level R2 <= 0; context ridge n_train=50 fixed-lambda R2 +0.384 at acc@1 0.065)"],
   "banked_dissociation": "identity+bias R2 -0.865 at medrank 1 (#779 identity_bias_knn); acc@1 0.84 at pooled-OOF R2 -6.5 (#722)"}
  ```

  Disclosure: summary fields of the 1 distractor-pool manifest (chunk names elided); full file: [distractor_manifest.json](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery/distractor_manifest.json).

  ```json
  {"chunk_universe": {"n_total": 1920, "sha256": "cdb00a9cc38808a02dd9fee032002a1abadee3be7930dd5921486d7f2dbca515"},
   "rows_per_chunk": {"n_streamed": 104983, "min": 493, "median": 500.0, "max": 500},
   "corpus_counts": {"lmsys": 55686, "wildchat": 44314},
   "dup_stats": {"n_rows": 100000, "n_unique_vectors": 100000, "n_duplicate_groups": 0, "n_excess_duplicate_rows": 0}}
  ```

  Conciseness: the total Takeaways+Goal+Results prose runs modestly over the 800-word budget (total/budget WARN acknowledged) — five results each carry the summary-plus-per-unit figure pair the aggregate-result rule requires.

## Results

### Variance explained and retrieval rank the estimator ladder in opposite orders

One point per estimator and training regime: pooled held-out R² (horizontal, symlog) against retrieval acc@1 (vertical; euclidean; 1,000-candidate pool, n=1000 test rows, context arm; 50-candidate, prefix arm; marker = regime). The grid companion covers all estimators on six metrics.

![Pooled R squared against retrieval accuracy per estimator and regime](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/hero_r2_vs_acc1_scatter_v2.png)

> **Figure.** *The two metrics disagree in both directions.* Identity plus bias (green) sits top-left: high retrieval, catastrophic R². The prefix LOFO ridge and the n=50 context ridge sit bottom-right: positive R², chance retrieval. Fitted 963k maps hold the top-right corner. All values come from the banked on-policy capture; dashed line = context-arm chance 0.001.

![Estimator by metric grid for context and prefix arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/ladder_by_metric_grid_v2.png)

> **Figure.** *The full ladder-by-metric grid behind the scatter.* Context arm (left; 963k-train rungs, pool 1,000) and prefix arm (right; leave-one-family-out, pool 50); brighter is better within a column. Median rank ties at its floor of 1 for every decent map at pool 1,000.

A single context-independent shift picks the right answer from 1,000 candidates half the time (0.532) while explaining less variance than the mean predictor (R² −0.920 vs −0.045); the constant mean sits at exact chance (0.001) with the least-bad R². The orderings invert; neither metric subsumes the other.

### Fitted maps dominate, and the nonlinear retrieval edge grows with pool size

Left half: acc@1 (euclidean, n=1000) against candidate-pool size per 963k-train estimator. Right half: paired nonlinear-over-ridge acc@1 gaps, 95% bootstrap whiskers on shared draws. Companion: each true answer's rank among 100,000 candidates.

![Accuracy versus pool size and nonlinear minus ridge gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/pool_decay_and_nonlinear_gap.png)

> **Figure.** *Accuracy decays with pool size; the nonlinear maps decay slowest.* The width-8192 gap over ridge grows 0.036, 0.061, 0.079, 0.101 across pools 1,000 to 100,000; the width-32768 map reaches +0.115 and the kernel map +0.078 at 100,000. Dashed line = chance 1/pool.

![Rank distribution per estimator at the 100,000 pool](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/rank_cdf_pool100k.png)

> **Figure.** *Per-row rank distributions at the 100,000 pool, one step curve per estimator.* Fitted maps place most true answers at rank 1; identity plus bias needs ranks in the hundreds for its tail; the constant mean concentrates near rank 50,000, the random-guess region.

The nonlinear R² gain (+0.056 for width 8192) buys discriminability that compounds with pool size: the gap growth (+0.065) excludes zero on shared draws and holds within each pool-composition class, so pool size, not composition, drives it. The pool-1,000 gap is 3-4 times larger at layers 14 and 26 (+0.135 and +0.123 against +0.036), where ridge is weaker.

### Every shuffled-pair null collapses to its floor, and raw cosine's floor is high

Per metric panel: observed value per estimator (dot) against its 200-draw shuffled-pair null (violin); retrieval panel at the 1,000-candidate test pool, n=1000 rows.

![Observed values versus shuffled pair nulls for three metrics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/null_floors.png)

> **Figure.** *Nulls collapse to metric-specific floors.* Retrieval nulls sit at analytic chance (mean 0.00092 vs 0.001); R² nulls are strongly negative (ridge −0.76); the cosine null is 0.68 for ridge predictions while the constant train-mean scores 0.798.

No metric shows a non-collapsing null on the fitted maps' pipeline — the instrument reads as sound. Cosine is the deliberate exception: a mismatched prediction-target pair still scores 0.68 because activation space shares a dominant mean direction, so mean-cosine values above roughly 0.7 say little; never headline a map on cosine without this floor.

### Retrieval hubs are corpus-structured, and the hubness correction helps everywhere

Left half: corpus composition of each estimator's top-100 cosine hubs at the 100,000 pool, with the pool-composition reference. Right half: paired CSLS-minus-cosine acc@1 gains per estimator and pool, 95% bootstrap whiskers (n=1000).

![Hub corpus composition and CSLS gain versus pool size](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/hub_composition_csls.png)

> **Figure.** *Hubs sit where each estimator's predictions land.* Fitted maps' hubs are test-row-dominated (61-70 of 100 against 1% of pool); identity-family hubs are train-region LMSYS distractors (87 of 100). CSLS gains span +0.027 (width-32768 map, pool 20,000) to +0.097 (ridge, pool 100,000).

Cosine 10-occurrence skewness reaches 13.6-25.8 against a Gaussian reference of 3.2; the euclidean excess is modest (23.8 vs 20.8) — mostly a cosine phenomenon. Hubs follow each estimator's prediction region: the correction partly compensates corpus and density structure, not only generic hubness; it helps weak estimators most (identity copy +0.195 at pool 1,000), understating weak baselines under plain kNN.

### The retrieval flip is a small-training-set property, and pooled R² fails in the opposite direction there

Left half: ridge versus identity-plus-bias acc@1 across four training regimes (963,444 / 3,600 / 50 context rows; prefix leave-one-family-out ≈43), euclidean, chance dashed per regime. Right half: per-family prefix ridge R² behind the pooled value.

![Accuracy across training regimes and per family prefix R squared](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/regime_flip_and_family_breakdown.png)

> **Figure.** *Ridge's discriminability is bought with training data; the bias estimate is not.* Ridge falls 0.805, 0.720, 0.065, 0.04 across the regimes while identity plus bias holds 0.532, 0.503, 0.501, 0.84. All seven prefix families sit at or below zero R² against a pooled +0.54; prefix R² is estimator-degenerate (≈43 rows, 3,584 dimensions), never compared to context-arm values.

The prefix arm reproduces the banked dissociation to the digit; the n=50 context rung reproduces the flip with data, pool, and folds fixed — a small-training-set property (prefix-specific contributions bounded, not excluded). The mirror failure: shrunk predictions near the fold mean track family membership with no per-context discriminability; the n=50 rung repeats it (R² +0.384, acc@1 0.065). The R² ordering never flips.

---
**Repro:** 0 GPU-h — VM CPU only, ~71 min wall (context battery 1,884 s; distractor stream 448 s; prefix battery 1,886 s; 19.7 GB peak RSS, above the plan's 12 GB projection and its 16 GB reroute line, and the launch-time OOM-priority protection silently failed via a pid-capture artifact — a rerun should route to a `cpu-mid` instance). Battery driver: [scripts/issue1901_metric_battery.py](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/scripts/issue1901_metric_battery.py), run at worktree commit `680f322678` (JSON-metadata code SHA `c63eca8d1e`; characterization amended at `d36c7c5dc5`); body figures regenerated by [scripts/issue1901_body_figures.py](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/scripts/issue1901_body_figures.py) at commit `76a1a9826d`. Eval JSONs: [metric_battery tree](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery) (`context_arm.json`, `prefix_arm.json`, `metric_characterization.json`, `distractor_manifest.json`, `boot_draws_context.json`, `boot_draws_prefix.json`). Figures + sidecars: [figures/issue_1901 tree](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901). Distractor pool npz: [HF issue1901_metrics/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2f64d6e8022bb0f797148e11b628ae70a999ebcb/issue1901_metrics/analysis_tensors). Reused artifacts: Reused weight payloads from [#779](https://eps.superkaiba.com/tasks/779): [issue779_monitoring/n1m_readout/weights](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue779_monitoring/n1m_readout/weights) — fit: the exact banked mixed-corpus fits; applied-vs-banked ridge R² delta 8.1e-11, all 12 payload key checks PASS. Reused context/answer capture bundle from [#779](https://eps.superkaiba.com/tasks/779) (`pass_b/train_context_vectors.pt`, byte-size-asserted 6,021,122,751) plus the seeded distractor stream from its capture chunks — fit: same pinned split, same layers. Reused prefix battery stores from [#722](https://eps.superkaiba.com/tasks/722) ([eval_results/issue_722/identity_bias_knn](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_722/identity_bias_knn) + the `c_C`/`v_A` stores its loaders resolve) — fit: verbatim leave-one-family-out fold structure; reproduces the banked read exactly. Config slugs for the ladder rungs: `const_mean`, `identity_copy`, `identity_bias`, `scaled_identity_3600`, `diagonal_only_3600`, `ridge`, `mlp_w8192`, `mlp_w32768`, `krr_nystrom`, `ridge_3600`, `ridge_n50_fixedlam`, `identity_bias_n50`.

**Context:** Originating prompt (verbatim, from task frontmatter):

> Run this in background with happy coder:
> ## Motivation
> We have come up with a bunch of different metrics for measuring the quality of the mapping. I want to characterize each and see how the 1m context linear and nonlinear mapping on them vs baselines, as well as what each metric is measuring exactly
>
> - [ ] [[I think a good baseline would be learned bias]], W = identity - another metric besides R^2 would be e.g. P(answer summary is in k nearest neighbors of prediction)

Lineage: fresh direction (no parent task); characterizes the metrics used across the [#779](https://eps.superkaiba.com/tasks/779) / [#722](https://eps.superkaiba.com/tasks/722) mapping line. Created 2026-07-30; run 2026-07-31 (UTC); interpretation settled 2026-07-31 after one revision round.
