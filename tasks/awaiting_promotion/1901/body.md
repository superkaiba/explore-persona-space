---
title: Held-out variance explained and retrieval accuracy dissociate in both directions
  across the context-to-answer estimator ladder (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-30T23:49:53Z'
has_clean_result: true
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

**Methodology:** [docs/methodology/issue_1901.md](https://github.com/superkaiba/explore-persona-space/blob/f9674fbee8bc162993ac17b33348c6a3304c4343/docs/methodology/issue_1901.md) · [gist](https://gist.github.com/superkaiba/1bba8ec6a80f3d3f137e5286d9d3f501)


## Takeaways

- **Both dissociation directions realized on both target corpora: identity plus bias scores acc@1 0.53 at pooled R² −0.92 on LMSYS and 0.53 at −0.88 on WildChat (n=1000 each); a shrinkage-limited prefix ridge scores R² +0.54 at retrieval 0.04 (chance 0.02).**
- At the 963k training regime, fitted maps beat every baseline on all non-saturated metrics (cosine and median rank saturate); the nonlinear-over-ridge acc@1 gap grows with pool size on both corpora — 0.036 to 0.101 (LMSYS) and 0.025 to 0.079 (WildChat) across pools 1,000 to 100,000.
- Every cross-class estimator ordering transfers to 1,000 fresh held-out WildChat targets: euclidean acc@1 rank agreement is exact with zero inversions (ridge 0.800, wide neural map 0.844, identity plus bias 0.527); pooled-R² agreement is 0.93, its only resolved flips inside the identity-plus-bias companions.
- Instrument checks hold on both corpora: shuffled-pair nulls collapse to metric floors (raw cosine's floor is 0.68-0.71 while the constant train-mean scores near 0.8, so absolute cosine is uninterpretable without its null), and retrieval hubs are corpus-structured (61-70 of fitted maps' top-100 cosine hubs are test rows at 1% of pool; the correction buys fitted maps 0.03-0.10 acc@1).
- Training data governs fitted-map retrieval in both directions: ridge falls 0.81 to 0.065 acc@1 (LMSYS; 0.80 to 0.048 WildChat) as training rows drop 963k to 50 while identity plus bias stays near 0.50, and deliberately in-train WildChat targets inflate fitted-map acc@1 by +0.07 to +0.11 at fitted-map R² shifts of at most 0.03.
- Scope: the LMSYS-only-target caveat is closed; the WildChat transfer is held-out-row within the training distribution (the maps trained on 434k WildChat rows), not corpus-OOD, and is context-arm-only; headline layers are inherited (context 19, prefix 18); layers 14/26 reproduce every ordering.

## Goal

- **This experiment in context:** The standing rule to co-report an identity-plus-learned-bias baseline and a kNN-retrieval read beside held-out R² came from two single-estimator reads: identity plus bias reached acc@1 0.84 at pooled R² −6.5 on the 50-context prefix battery ([#722](https://eps.superkaiba.com/tasks/722)), while the fitted ridge dominated retrieval on the 963k-context map ([#779](https://eps.superkaiba.com/tasks/779)); [#1776](https://eps.superkaiba.com/tasks/1776) verified the banked weights reproduce to 8e-11. This experiment turns those spot reads into one systematic reference — the full estimator ladder crossed with every metric, both mapping arms, pool-size and training-size axes — under the same split, layers, and banked weights (directly comparable).
- **Broader narrative:** The context-to-answer mapping line treats a context's effect on the model as a predictable vector; every future map-quality claim rests on these metrics, so their constructs, floors, and failure modes need one canonical measurement.

## Methodology

- **Design:** Evaluation-only battery; the manipulated variable within each read is the estimator. Context arm: 14 estimator rungs (7 fitted or baseline estimators at 963,444 training rows, 5 companions at 3,600, 2 at 50) crossed with 8 metrics (pooled R², per-dimension R², mean cosine, kNN acc@k under euclidean and cosine distance, hubness-corrected acc@k, median rank, mean reciprocal rank, hubness diagnostic), 3 layers (19 headline; 14/26 exploratory), and 4 candidate pools (1,000 / 5,000 / 20,000 / 100,000). Prefix arm: 7 estimators over the 50-context battery (7 prefix families), leave-one-family-out. Bootstrap CIs use n=1000 shared test-row resamples; every metric gets a 200-draw shuffled-pair null per arm. Both mapping arms run (context-based and prefix-based); the prefix arm runs at n=50 because no larger prefix-level mapping data is persisted — a stated deviation carried as a scope caveat. **Follow-up round `wildchat-target-battery` (plan v7):** one manipulated variable — the test-target corpus (LMSYS to WildChat); banked weights, battery, pools, nulls, and bootstrap conventions inherited verbatim with no refit. The round is context-arm-scoped (the prefix battery is a constructed-prefix design with no WildChat analogue — a stated deviation). Target assembly: 2,500 fresh WildChat-1M candidates streamed past the parent capture frontier at a pinned corpus revision, screened against a 956,693-fingerprint training-plus-held-out exclusion set and a transposed 965,000-row train-pool near-duplicate gate (5-gram Jaccard 0.8; 521 dropped, 1,979 survivors, first 1,300 kept); captured through the parent rig on one fellows-cluster GPU (1,299 of 1,300 rows captured; 1 over-length skip recorded in an uploaded sidecar); the first 1,000 captured rows in seeded order are the test targets, and the battery re-checks every target fingerprint against the exclusion set (0 hits). Before any WildChat number, the round re-applies the banked layer-19 ridge to the pinned LMSYS test-1000 and asserts the banked pooled R² reproduces (realized delta 8.1e-11). The round plan's optional in-train memorization companion (`wc_intrain`, exploratory and contamination-labeled by design) was deferred from the round itself and then run as the task's single zero-GPU free-analysis follow-up (screened not-redundant): the identical 14-arm battery on 1,000 deliberately in-train WildChat rows sliced deterministically from the banked parent capture chunks (seed offset +1; the contamination re-check reads 1,000 of 1,000 exclusion-fingerprint hits, expected by construction), no refit, layer 19, same pools/nulls/bootstrap; a paired deterministic held-out re-run reproduced the round's published values identically. It is a memorization read — never a headline arm; the transfer headline rests on the held-out arm alone.
- **Training:** N/A — no model training in this task. The four fitted maps are banked weight payloads applied unchanged (in-run reproduction assert: applied ridge pooled R² matches the banked 0.7541708417500046 within 1e-6; realized delta 8.1e-11 in the parent run and in the WildChat round's re-assert). Their production recipe, inlined: fit on 963,444 mixed LMSYS+WildChat contexts (pinned split seed 42; 400 validation / 1,000 test rows held out with sha-pinned membership; the training pool near-dupe-screened against the 1,400 validation/test targets at 5-gram Jaccard 0.8 — 435 exact and 30,437 near duplicates dropped), inputs = the last-context-token activation of Qwen2.5-7B-Instruct at layer ℓ, targets = the mean answer-token activation at the same layer, layers 14/19/26. Ridge: streaming fp64 primal regression, 23-point λ grid from 1e-3 to 1e8, validation-selected λ = 0.001. Neural maps: one GELU hidden layer at widths 8,192 and 32,768, AdamW lr 3e-4, batch 4,096, validation early stopping. Kernel map: RBF kernel ridge, Nyström with 16,384 landmarks, γ at the median heuristic, λ validation-selected at 0.1. The AdamW learning rate 3e-4 is copied from the producing run's committed fit constants and differs from this task's plan by design — the plan trains nothing, so it declares no learning rate. Complete analysis-knob table (parent rows cite the parent plan; `plans/plan.md` now points at the round amendment v7):

  | Knob | Value | Source |
  |---|---|---|
  | Battery seed (distractors, nulls, bootstrap) | 1901 (both rounds) | plan §10 |
  | Split (train/val/test) | 3600 base + 959,844 new / 400 / 1000, seed 42, sha-pinned | banked split metadata (`n1m_fits.json`) |
  | Layers | 19 headline (validation-selected upstream); 14, 26 exploratory | banked fit metadata |
  | Banked ridge recipe | streaming fp64 primal; λ grid 23 points 1e-3 to 1e8; selected λ = 0.001 | banked fit metadata + `docs/methodology/issue_779.md` |
  | Banked neural-map recipe | 1 GELU hidden layer, widths 8192 / 32768; AdamW lr 3e-4; batch 4096; val early stop | `docs/methodology/issue_779.md` (fit constants) |
  | Banked kernel recipe | RBF Nyström m = 16384; γ median-heuristic; λ ∈ {0.1, 10}, selected 0.1 | banked fit metadata |
  | Small-n ridge rungs | banked λ = 1000 at n = 3600 and n = 50 (fixed, no re-selection); WildChat round refits on unchanged inputs, only the evaluation targets swap | `context_arm.json` + `wildchat_arm.json` `small_n_meta` |
  | Retrieval k values | (1, 5, 10) context; (1, 3, 5) prefix | `mapping_baselines.knn_retrieval` default + banked convention |
  | Candidate pools | 1000 (test) / 5000 (test+train+val rows) / 20,000 / 100,000 (test+distractors) | plan §11 (decade-spaced; bilingual-lexicon evals use up to 200k, arXiv 1710.04087) |
  | Hubness-corrected score | CSLS, cross-domain neighborhoods, k = 10 | arXiv 1710.04087 |
  | Bootstrap | n_boot = 1000, test-row resampling, one shared draw matrix across estimators | banked CI convention (`BOOT_N`) |
  | Null | K = 200 shuffled-pair permutations per metric × arm | plan §11 (banked 200-draw null convention) |
  | Distractor pool | seeded 210 of 1,920 capture chunks streamed; 100,000 layer-19 rows kept | plan §11 + `distractor_manifest.json` |
  | Prefix-arm neural rung | batched leave-one-family-out MLP, width 512, 300 epochs | banked battery recipe (`vectorized_mlp_skill`) |
  | Prefix-arm headline layer | 18 (validation-selected in the banked battery; inherited) | `eval_results/issue_722/identity_bias_knn/results.json` `best_ridge_layer` |
  | WildChat round: target corpus + revision pin | `allenai/WildChat-1M` @ `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | `issue1901_wildchat/manifest/meta.json` |
  | WildChat round: candidate screen | 2,500 streamed; 956,693-fingerprint exclusion + near-dupe gate (5-gram, Jaccard 0.8); 521 dropped; first 1,300 kept; 8-worker screen, 222.7 s | plan v7 §4 w0 + manifest meta `screen` |
  | WildChat round: capture recipe | parent constants verbatim (Qwen2.5-7B-Instruct, seed 42, max model len 8192, gen max tokens 1024, layers 14/19/26); 1,299/1,300 captured | plan v7 §4 w1 (parent capture module constants) |
  | WildChat round: pools | 1,000 targets-only / 5,000 / 20,000 / 100,000 (targets + parent distractor rows); 0 duplicate vectors in every pool | `wildchat_arm.json` `pools` |
  | Compute env | VM CPU only; 8 BLAS threads; `MALLOC_ARENA_MAX=2`; numpy 2.2.6, torch 2.8.0 | `context_arm.json` + `wildchat_arm.json` metadata |

- **Evaluation:** All dependent variables are the mapping metrics themselves, computed deterministically over banked activations — no LLM judge; the only new generation is the WildChat round's target capture. Per metric: pooled R² = 1 − SS_res/SS_tot on the held-out set (variance-weighted across dimensions; unbounded below); per-dimension R² (exposes the variance weighting); mean cosine between prediction and target (per-row scale-invariant; high anisotropy floor); kNN acc@k = the probability the true answer vector is within the k nearest pool neighbors of the prediction (rank-based, mid-rank tie handling; analytic chance k/n_pool); the CSLS hubness-corrected variant of acc@k; median rank and mean reciprocal rank; and a hubness diagnostic (skewness of 10-occurrence counts, with top-hub corpus composition). Correctness asserts: banked-ridge reproduction within 1e-6 (run in the parent battery and re-run in the WildChat round before any WildChat number), realized payload keys for all 12 weight files, helper-parity checks per retrieval cell, and 0 duplicate groups in the distractor pool. The WildChat round's transfer read: per metric, rank agreement of the 14 common arms between the banked LMSYS values and the WildChat values (Kendall τ), plus a pairwise-inversion table; CI resolution (within-corpus paired-difference bootstrap CIs on shared draws) covers the two headline metrics — pooled R² and euclidean acc@1 — and the 6 remaining point inversions (mean cosine, cosine acc@1, CSLS acc@1, two on mean reciprocal rank, median rank) are near-tie or companion-rung pairs left unresolved; each arm is scored against its own corpus's targets by design — the read is rank transfer, not a cross-corpus value comparison. Known metric ceilings that bound interpretation: 58 of the 1,000 LMSYS test targets have an exact in-pool duplicate vector, capping acc@1 near 0.94 equally for every estimator (ordering unaffected, resolved by k=5), and the LMSYS 5,000-row pool carries 354 excess duplicate rows with the same equal-across-arms effect; the WildChat pools carry 0 duplicate vectors, so this ceiling does not bind there; median rank saturates at 1 for every decent map at pool 1,000.
- **Data extraction:** Context arm: real-user single-turn conversations from LMSYS-Chat-1M and WildChat-1M (realism tier 1); answers were generated on-policy by Qwen2.5-7B-Instruct in the banked capture, and activations recorded by teacher-forced re-forward. This task adds the distractor pool: a seeded random subset of 210 of 1,920 capture chunks, streamed one chunk at a time (peak footprint one chunk), keeping 100,000 layer-19 answer vectors (55,686 LMSYS / 44,314 WildChat; 0 duplicate vectors in-pool; disjoint from the 5,000 fit-round rows by construction) — uploaded to the HF data repo before analysis. The 5,000-row LMSYS pool reuses rows that were near-dupe-screened against the test targets upstream, so retrieval there is easier than at the unscreened 20,000/100,000 distractor pools, which are the more natural read. WildChat-round targets: fresh real-user WildChat-1M conversations (realism tier 1) verified absent from the training pool (0 exclusion-fingerprint hits over the 1,000 targets at battery time); answers generated on-policy by the same capture rig and activations recorded by teacher-forced re-forward, exactly as the banked capture. No WildChat user text is inlined in this body — WildChat is an unscreened real-world corpus; rows are referenced by file and count only (3 raw-completion chunk files plus 1 over-length-skip sidecar under the round's HF prefix). Prefix arm: the inherited 50-context battery (families: persona, behavior, format, in-context-learning, rephrase, WildChat, default assistant — constructed conditions, realism tier 3, a scope caveat), prefix-level last-input-token activations and mean answer summaries, query-averaged.
- **Sample training/evaluation data + completions:** This task generates no completions beyond the WildChat target capture, whose text stays un-inlined (see above); the worked examples below are verbatim artifact rows. Disclosure: 1 of 448 context-arm retrieval cells, chosen as the headline cell (ridge, test pool, euclidean); full file: [context_arm.json](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery/context_arm.json).

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

  Disclosure: the headline rank-transfer entry (euclidean acc@1) plus the target-provenance fields, verbatim from the WildChat round's transfer file (per-arm value dictionaries elided); full file: [transfer_comparison.json](https://github.com/superkaiba/explore-persona-space/blob/8cadf1e710541048358179712488903df06424e3/eval_results/issue_1901/metric_battery/transfer_comparison.json).

  ```json
  {"kendall_tau": {"acc1_euclid": {"tau": 1.0, "p": 7.740804583966695e-07}},
   "pairwise_inversions": {"acc1_euclid": []},
   "setup": {"wc_targets": {"n": 1000, "n_available": 1299, "in_train": false,
             "contamination": {"n_rows": 1000, "n_exclusion_hits": 0, "expect_hits": false}}}}
  ```

  Conciseness: after the WildChat and memorization folds the total Takeaways+Goal+Results prose runs over the word budget, six Takeaways bullets exceed 30 words, and several per-result blocks sit in the 120-180-word band (total, bullet, and per-result WARNs all acknowledged) — the first six results carry the summary-plus-per-unit figure pair the aggregate-result rule requires, the memorization result's dumbbell is itself the per-estimator view, and the dense bullets hold the numbers the claims need.

## Results

### Variance explained and retrieval rank the estimator ladder in opposite orders

One point per estimator and training regime: pooled held-out R² (horizontal, symlog) against retrieval acc@1 (vertical; euclidean; 1,000-candidate pool, n=1000 test rows, context arm; 50-candidate, prefix arm; marker = regime). The grid companion covers all estimators on six metrics.

![Pooled R squared against retrieval accuracy per estimator and regime](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/hero_r2_vs_acc1_scatter_v2.png)

> **Figure.** *The two metrics disagree in both directions.* Identity plus bias (green) sits top-left: high retrieval, catastrophic R². The prefix LOFO ridge and the n=50 context ridge sit bottom-right: positive R², chance retrieval. Fitted 963k maps hold the top-right corner. All values come from the banked on-policy capture; dashed line = context-arm chance 0.001.

![Estimator by metric grid for context and prefix arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/ladder_by_metric_grid_v2.png)

> **Figure.** *The full ladder-by-metric grid behind the scatter.* Context arm (left; 963k-train rungs, pool 1,000) and prefix arm (right; leave-one-family-out, pool 50); brighter is better within a column. Median rank ties at its floor of 1 for every decent map at pool 1,000.

A single context-independent shift picks the right answer from 1,000 candidates half the time (0.532) while explaining less variance than the mean predictor (R² −0.920 vs −0.045); the constant mean sits at exact chance (0.001) with the least-bad R². The orderings invert; neither metric subsumes the other.

### The nonlinear edge over ridge grows with candidate-pool size

acc@1 (euclidean, n=1000) against candidate-pool size per 963k-train estimator on the left; on the right, paired nonlinear-over-ridge acc@1 gaps with 95% bootstrap whiskers on shared draws. Companion: each true answer's rank among 100,000 candidates.

![Accuracy versus pool size and nonlinear minus ridge gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/pool_decay_and_nonlinear_gap.png)

> **Figure.** *Accuracy decays with pool size; the nonlinear maps decay slowest.* The width-8192 gap over ridge grows 0.036, 0.061, 0.079, 0.101 across pools 1,000 to 100,000; the width-32768 map reaches +0.115 and the kernel map +0.078 at 100,000. Dashed line = chance 1/pool.

![Rank distribution per estimator at the 100,000 pool](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/rank_cdf_pool100k.png)

> **Figure.** *Per-row rank distributions at the 100,000 pool, one step curve per estimator.* Fitted maps place most true answers at rank 1; identity plus bias needs ranks in the hundreds for its tail; the constant mean concentrates near rank 50,000, the random-guess region.

The nonlinear R² gain (+0.056 for width 8192) buys discriminability that compounds with pool size: the gap growth (+0.065) excludes zero on shared draws and holds within each pool-composition class, so pool size rather than composition drives it. The pool-1,000 gap is 3-4 times larger at layers 14 and 26 (+0.135 and +0.123 against +0.036), where ridge is weaker.

### Every shuffled-pair null collapses to its floor, and raw cosine's floor is high

Per metric panel: observed value per estimator (dot) against its 200-draw shuffled-pair null (violin); retrieval panel at the 1,000-candidate test pool, n=1000 rows.

![Observed values versus shuffled pair nulls for three metrics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/null_floors.png)

> **Figure.** *Nulls collapse to metric-specific floors.* Retrieval nulls sit at analytic chance (mean 0.00092 vs 0.001); R² nulls are strongly negative (ridge −0.76); the cosine null is 0.68 for ridge predictions while the constant train-mean scores 0.798.

No metric shows a non-collapsing null on the fitted maps' pipeline — the instrument reads as sound. Cosine is the one anticipated exception: a mismatched prediction-target pair still scores 0.68 because activation space shares a dominant mean direction, mean-cosine values above roughly 0.7 carry almost no information about map quality. Never headline a map on cosine without this floor; the WildChat round reproduces the collapse on its own battery (ridge R² null −0.75, cosine null 0.71).

### Retrieval hubs are corpus-structured; the CSLS correction helps every estimator

The left panel shows the corpus composition of each estimator's top-100 cosine hubs at the 100,000 pool, with the pool-composition reference. The right panel plots paired CSLS-minus-cosine acc@1 gains per estimator and pool (95% bootstrap whiskers, n=1000).

![Hub corpus composition and CSLS gain versus pool size](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/hub_composition_csls.png)

> **Figure.** *Hubs sit where each estimator's predictions land.* Fitted maps' hubs are test-row-dominated (61-70 of 100 against 1% of pool); identity-family hubs are train-region LMSYS distractors (87 of 100). Fitted-map CSLS gains span +0.027 (width-32768 map, pool 20,000) to +0.097 (ridge, pool 100,000); identity copy gains up to +0.195.

Cosine 10-occurrence skewness reaches 13.6-25.8 against a Gaussian reference of 3.2; the euclidean excess is modest (23.8 vs its own Gaussian reference of 20.8) — mostly a cosine phenomenon. Hubs follow each estimator's prediction region, so part of the CSLS gain compensates corpus and density structure beyond generic hubness; it helps weak estimators most (identity copy +0.195 at pool 1,000) — so plain kNN understates weak baselines.

### At small training sets the retrieval ranking flips while pooled R² fails the other way

Ridge versus identity-plus-bias acc@1 across four training regimes (963,444 / 3,600 / 50 context rows; prefix leave-one-family-out ≈43; euclidean; chance dashed per regime), paired with the per-family prefix ridge R² behind the pooled value.

![Accuracy across training regimes and per family prefix R squared](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901/regime_flip_and_family_breakdown.png)

> **Figure.** *Ridge's discriminability is bought with training data; the bias estimate is not.* Ridge falls 0.805, 0.720, 0.065, 0.04 across the regimes while identity plus bias holds 0.532, 0.503, 0.501, 0.84. All seven prefix families sit at or below zero R² against a pooled +0.54; prefix R² is estimator-degenerate (≈43 rows, 3,584 dimensions), never compared to context-arm values.

The prefix arm reproduces the previously measured dissociation (identity-plus-bias acc@1 0.84 vs ridge 0.04) to the digit; the n=50 context rung reproduces the flip with the data, pool, and folds all held fixed: a small-training-set property (prefix-specific contributions bounded, not excluded). The mirror failure: shrunk predictions near the fold mean track family membership with no per-context discriminability; the n=50 context rung shows an attenuated version (R² +0.384, acc@1 0.065 vs chance 0.001). The R² ordering never flips.

### Every cross-class estimator ordering transfers to fresh held-out WildChat targets

Side-by-side estimator-by-metric heatmaps: the banked LMSYS test-1000 context arm (left) against 1,000 fresh held-out WildChat targets (right); 14 arms by 6 metrics, layer 19, targets-only pool, each arm scored against its own corpus's targets. Companion: per-arm paired values for euclidean acc@1 and pooled R².

![LMSYS versus WildChat estimator by metric heatmaps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8cadf1e710541048358179712488903df06424e3/figures/issue_1901/wc_hero_transfer_heatmap.png)

> **Figure.** *The ladder-by-metric pattern is corpus-stable.* Left: banked LMSYS arm; right: WildChat held-out targets (n=1000, fingerprint-screened out of the training pool). Rank agreement per plotted metric: euclidean acc@1 τ 1.00 with zero inversions; mean cosine 0.98; CSLS acc@1 0.98; mean reciprocal rank 0.96; pooled R² 0.93; median rank 0.92.

![Per estimator LMSYS versus WildChat dumbbells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8cadf1e710541048358179712488903df06424e3/figures/issue_1901/wc_dumbbell_lmsys_vs_wc.png)

> **Figure.** *Per-arm values barely move between corpora at the targets-only pool.* Paired LMSYS and WildChat points per estimator: euclidean acc@1 (left panel) and pooled R² clipped at −2 (right panel). The largest shift is identity copy, 0.254 to 0.215. At the 100,000 pool WildChat targets retrieve better: ridge 0.630 vs 0.544, wide map 0.736 vs 0.659.

With no refit and the same pinned weights, every parent-separated cross-class ordering reproduces on WildChat: fitted maps above identity plus bias above copy above the chance-level mean on retrieval, positive fitted R² against negative baselines, and both dissociation directions (identity plus bias 0.527 acc@1 at R² −0.876). The wide map's retrieval edge over ridge again grows with pool size, +0.044 to +0.106.

The two resolved R² flips sit inside the identity-plus-bias family — the mixed-corpus bias beats its LMSYS-fit companions on WildChat and trails them on LMSYS, at most 0.06 R² units near −0.9 — mean-alignment, not a class reordering. The maps trained on 434k WildChat rows: held-out-row transfer, not corpus-OOD.

### Memorization inflates retrieval far more than variance explained

Paired in-train vs held-out dumbbells per estimator: euclidean acc@1 at the 1,000-candidate targets-only pool (left panel) and pooled R² clipped at −2 (right panel); 1,000 deliberately in-train WildChat rows — a contamination-labeled memorization read — against the round's 1,000 held-out targets; no refit, layer 19.

![In-train versus held-out WildChat targets per estimator](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0e7773eaa64215fa5ee8ecc1c01d1e868daa5949/figures/issue_1901/wc_intrain_memorization_dumbbell.png)

> **Figure.** *Retrieval moves, variance explained barely does.* Fitted 963k maps gain +0.074 to +0.109 acc@1 on in-train rows (wide neural map 0.947 vs 0.844; ridge 0.874 vs 0.800) while their pooled R² shifts by at most 0.03; the identity-plus-bias and copy baselines gain +0.026 to +0.042; the constant mean stays at chance.

On rows the maps trained on, fitted-map retrieval inflates by +0.074 (ridge) to +0.109 (small neural map), while the fitted maps' pooled R² moves at most 0.03 in either direction — ridge and the kernel map even read slightly lower in-train. The identity-plus-bias and copy baselines, whose global shift cannot memorize rows, gain only +0.026 to +0.042 on the same target swap, bounding the row-difficulty component; roughly +0.04 to +0.07 of the fitted-map gain is memorization proper.

Retrieval rewards per-row placement, which training retains; variance explained aggregates over dimensions and barely separates seen from unseen rows. In-train reads of fitted maps overstate deployable retrieval, not R².

---
**Repro:** Parent run 0 GPU-h — VM CPU only, ~71 min wall (context battery 1,884 s; distractor stream 448 s; prefix battery 1,886 s; 19.7 GB peak RSS, above the plan's 12 GB projection and its 16 GB reroute line, and the launch-time OOM-priority protection silently failed via a pid-capture artifact — a rerun should route to a `cpu-mid` instance). Battery driver: [scripts/issue1901_metric_battery.py](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/scripts/issue1901_metric_battery.py), run at worktree commit `680f322678` (JSON-metadata code SHA `c63eca8d1e`; characterization amended at `d36c7c5dc5`); body figures regenerated by [scripts/issue1901_body_figures.py](https://github.com/superkaiba/explore-persona-space/blob/76a1a9826dbb9d55945a0ee23667f2bf61020584/scripts/issue1901_body_figures.py) at commit `76a1a9826d`. Eval JSONs: [metric_battery tree](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_1901/metric_battery) (`context_arm.json`, `prefix_arm.json`, `metric_characterization.json`, `distractor_manifest.json`, `boot_draws_context.json`, `boot_draws_prefix.json`). Figures + sidecars: [figures/issue_1901 tree](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/figures/issue_1901). Distractor pool npz: [HF issue1901_metrics/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2f64d6e8022bb0f797148e11b628ae70a999ebcb/issue1901_metrics/analysis_tensors). Reused artifacts: Reused weight payloads from [#779](https://eps.superkaiba.com/tasks/779): [issue779_monitoring/n1m_readout/weights](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/687eb8b42cd01e1279fd857655e895e284440524/issue779_monitoring/n1m_readout/weights) — fit: the exact banked mixed-corpus fits; applied-vs-banked ridge R² delta 8.1e-11, all 12 payload key checks PASS. Reused context/answer capture bundle from [#779](https://eps.superkaiba.com/tasks/779) (`pass_b/train_context_vectors.pt`, byte-size-asserted 6,021,122,751) plus the seeded distractor stream from its capture chunks — fit: same pinned split, same layers. Reused prefix battery stores from [#722](https://eps.superkaiba.com/tasks/722) ([eval_results/issue_722/identity_bias_knn](https://github.com/superkaiba/explore-persona-space/tree/76a1a9826dbb9d55945a0ee23667f2bf61020584/eval_results/issue_722/identity_bias_knn) + the `c_C`/`v_A` stores its loaders resolve) — fit: verbatim leave-one-family-out fold structure; reproduces the banked read exactly. WildChat follow-up round (`wildchat-target-battery`, source proposer-9b-cheap, round 1): VM CPU phases (candidate screen; battery wall 1,681 s at 20.7 GB peak RSS — again above the plan's 16 GB comfort line, the same class as the parent's 19.7 GB note; figures) plus one fellows SLURM GPU capture phase (job 16114, 1 GPU, ~9 min wall against 1.5 h booked — roughly 0.15 GPU-h realized). The round's train-pool screen aborted its serial pilot at a projected 11,907 s under the plan's fail-loud budget gate; an 8-worker vectorized rewrite realized the screen in 222.7 s. Round code + results at commit `8cadf1e710` on `origin/issue-1901-wildchat` (battery run at worktree commit `011c7bef3c`): [wildchat_arm.json](https://github.com/superkaiba/explore-persona-space/blob/8cadf1e710541048358179712488903df06424e3/eval_results/issue_1901/metric_battery/wildchat_arm.json), [transfer_comparison.json](https://github.com/superkaiba/explore-persona-space/blob/8cadf1e710541048358179712488903df06424e3/eval_results/issue_1901/metric_battery/transfer_comparison.json), [boot_draws_wildchat.json](https://github.com/superkaiba/explore-persona-space/blob/8cadf1e710541048358179712488903df06424e3/eval_results/issue_1901/metric_battery/boot_draws_wildchat.json), updated [metric_characterization.json](https://github.com/superkaiba/explore-persona-space/blob/8cadf1e710541048358179712488903df06424e3/eval_results/issue_1901/metric_battery/metric_characterization.json); round figures + sidecars in the [round figures tree](https://github.com/superkaiba/explore-persona-space/tree/8cadf1e710541048358179712488903df06424e3/figures/issue_1901). HF round prefix [issue1901_wildchat](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0da2b0bcefa6e05e85a775b240e501b501acd344/issue1901_wildchat): `manifest/` (pinned-revision stream + screen record), `final_token_capture/` (3 activation chunks, ~112 MB), `raw_completions/` (3 chunk files plus 1 over-length-skip sidecar; the round plan's section 6.5 labeled the raw-completion home `final_token_capture/`, the realized home is the Upload-Policy-canonical `raw_completions/` — both populated, upload-verified). In-train memorization companion (the task's single zero-GPU free-analysis follow-up, screened not-redundant): VM CPU 3,396.8 s including a deterministic held-out re-run at identical values (companion battery wall 1,723.4 s, 20.9 GB peak RSS per its metadata — the fourth battery over the 16 GB comfort line; the `cpu-mid` rerun recommendation stands); results at commit `36e0b9ca19` ([wildchat_intrain_companion.json](https://github.com/superkaiba/explore-persona-space/blob/36e0b9ca19a7e5128fee70abf3e173d1c2e421c0/eval_results/issue_1901/metric_battery/wildchat_intrain_companion.json), [boot_draws_wildchat_intrain.json](https://github.com/superkaiba/explore-persona-space/blob/36e0b9ca19a7e5128fee70abf3e173d1c2e421c0/eval_results/issue_1901/metric_battery/boot_draws_wildchat_intrain.json)); companion figure + [generator script](https://github.com/superkaiba/explore-persona-space/blob/0e7773eaa64215fa5ee8ecc1c01d1e868daa5949/scripts/issue1901_intrain_companion_figure.py) at commit `0e7773eaa6`. Battery seed 1901 in both rounds; weight revision pin unchanged. Config slugs for the ladder rungs: `const_mean`, `identity_copy`, `identity_bias`, `scaled_identity_3600`, `diagonal_only_3600`, `ridge`, `mlp_w8192`, `mlp_w32768`, `krr_nystrom`, `n3600_arm` (`const_mean_3600`, `identity_bias_3600`, `ridge_3600`), `ridge_n50_fixedlam`, `identity_bias_n50`, `prefix_arm`, `perm_null`; round slugs: `wc_heldout`, `repro_assert`, `wc_intrain` (free-analysis companion).

**Context:** Originating prompt (verbatim, from task frontmatter):

> Run this in background with happy coder:
> ## Motivation
> We have come up with a bunch of different metrics for measuring the quality of the mapping. I want to characterize each and see how the 1m context linear and nonlinear mapping on them vs baselines, as well as what each metric is measuring exactly
>
> - [ ] [[I think a good baseline would be learned bias]], W = identity - another metric besides R^2 would be e.g. P(answer summary is in k nearest neighbors of prediction)

Follow-up round 1 `wildchat-target-battery` (source proposer-9b-cheap; run and folded 2026-07-31 UTC). Originating proposal (verbatim, from the follow-up scope marker):

> **WildChat-target battery (ladder ordering transfer to non-LMSYS targets)** — Reproduction/OOD. [...] Hypothesis: the estimator-ladder orderings measured on LMSYS test targets (fitted >> identity+bias >> copy >> mean on retrieval; both dissociation directions) transfer to WildChat targets. Falsification: any ordering inversion on WildChat targets at matched pool sizes.

Lineage: fresh direction (no parent task); characterizes the metrics used across the [#779](https://eps.superkaiba.com/tasks/779) / [#722](https://eps.superkaiba.com/tasks/722) mapping line. Created 2026-07-30; parent run 2026-07-31 (UTC); interpretation settled 2026-07-31 after one revision round; WildChat transfer round folded 2026-07-31.
