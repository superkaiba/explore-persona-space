# Methodology — issue 1901: estimator-ladder × metric-battery evaluation of the context→answer maps (banked #779 weights; no training)

- **Design:** Evaluation-only battery; the manipulated variable within each read is the estimator. Context arm: 14 estimator rungs (7 fitted or baseline estimators at 963,444 training rows, 5 companions at 3,600, 2 at 50) crossed with 8 metrics (pooled R², per-dimension R², mean cosine, kNN acc@k under euclidean and cosine distance, hubness-corrected acc@k, median rank, mean reciprocal rank, hubness diagnostic), 3 layers (19 headline; 14/26 exploratory), and 4 candidate pools (1,000 / 5,000 / 20,000 / 100,000). Prefix arm: 7 estimators over the 50-context battery (7 prefix families), leave-one-family-out. Bootstrap CIs use n=1000 shared test-row resamples; every metric gets a 200-draw shuffled-pair null per arm. Both mapping arms run (context-based and prefix-based); the prefix arm runs at n=50 because no larger prefix-level mapping data is persisted — a stated deviation carried as a scope caveat.
- **Training:** N/A — no model training in this task. The four fitted maps are banked weight payloads applied unchanged (in-run reproduction assert: applied ridge pooled R² matches the banked 0.7541708417500046 within 1e-6; realized delta 8.1e-11). Their production recipe, inlined: fit on 963,444 mixed LMSYS+WildChat contexts (pinned split seed 42; 400 validation / 1,000 test rows held out with sha-pinned membership; the training pool near-dupe-screened against the 1,400 validation/test targets at 5-gram Jaccard 0.8 — 435 exact and 30,437 near duplicates dropped), inputs = the last-context-token activation of Qwen2.5-7B-Instruct at layer ℓ, targets = the mean answer-token activation at the same layer, layers 14/19/26. Ridge: streaming fp64 primal regression, 23-point λ grid from 1e-3 to 1e8, validation-selected λ = 0.001. Neural maps: one GELU hidden layer at widths 8,192 and 32,768, AdamW lr 3e-4, batch 4,096, validation early stopping. Kernel map: RBF kernel ridge, Nyström with 16,384 landmarks, γ at the median heuristic, λ validation-selected at 0.1. The AdamW learning rate 3e-4 is copied from the producing run's committed fit constants and differs from this task's plan by design — the plan trains nothing, so it declares no learning rate. Complete analysis-knob table:

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
  | Prefix-arm headline layer | 18 (validation-selected in the banked battery; inherited) | `eval_results/issue_722/identity_bias_knn/results.json` `best_ridge_layer` |
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

  Conciseness: the total Takeaways+Goal+Results prose runs modestly over the 800-word budget, two Takeaways bullets exceed 30 words, and the small-training-set result's per-result prose sits at the 121-word band edge (total, bullet, and per-result WARNs all acknowledged) — five results each carry the summary-plus-per-unit figure pair the aggregate-result rule requires, and the dense bullets hold the numbers the claims need.

*Derived from the [task body](https://eps.superkaiba.com/tasks/1901).*
