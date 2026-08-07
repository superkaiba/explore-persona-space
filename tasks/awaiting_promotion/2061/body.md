---
title: Per-feature SAE predictability gains across the Tülu post-training ladder stay
  within a selection-symmetric null whose bar only rarely-active features can reach
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-04T04:44:57Z'
has_clean_result: true
parent_id: 1895
origin_prompt: can you also run another issue which is training context vector ->
  SAE feature mapping at each stage and see if any SAE features get better predicted
  after each stage.
workflow: v1
goal: Fit a map from the pooled context activation to the SAE feature vector of the
  answer state, holding ONE fixed SAE dictionary across every post-training stage
  of an open ladder, and determine whether any individual SAE feature's per-feature
  held-out predictability improves across adjacent stage transitions beyond a selection-symmetric
  null over the whole dictionary.
---
# Per-feature SAE predictability gains across the Tülu post-training ladder stay within a selection-symmetric null whose bar only rarely-active features can reach (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_2061.md @ d8b1739a0d](https://github.com/superkaiba/explore-persona-space/blob/d8b1739a0d719a4fc9e3815374ff461c570217be/docs/methodology/issue_2061.md) · [gist mirror](https://gist.github.com/superkaiba/a21b1e234a2bc730d33189756dc3e530)

**Dashboard:** per-feature ΔR² explorer (most increased / most decreased per stage transition + distributions): https://eps.superkaiba.com/sae-predictability-2061.html (generator `scripts/issue2061_predictability_dashboard.py` @ c5fe5c220e; descriptive, multiplicity-uncorrected — see Takeaways)

## Takeaways

- Across 56 transition cells and 262,144 sparse-autoencoder (SAE) features, the largest gain (ΔR² 1.46; p = 0.49, N = 500 draws) stays within the selection-symmetric null under both estimator reads.
- Rare-feature noise sets the bar: the winner activates on 1–3 of ~9,500 rows; features active on 100+ rows top out at ΔR² 0.28 — failure-to-reject, not evidence of absence.
- 18 of the 56 cells (16 under the exactly-symmetric engine read) exceed their own per-cell null p97.5, versus roughly 1.4 expected — but in 14 of the 18 the winning feature activates on fewer than 10 pre-transition rows, so the local exceedances largely re-sample the same rare-feature class; descriptive, multiplicity-uncorrected.
- Top-100 improved-feature lists overlap weakly across corpora (means 1–8 per 100, near the pool-conditional null) but persist across renders of the same corpus (57–74 per 100) — corpus-keyed, descriptive.
- Aggregate map quality does move: context-arm nearest-neighbor (kNN) retrieval acc@k (euclidean read) jumps from base (0.13–0.41) to SFT (0.42–0.75; chance 0.05), then plateaus; prefix arms stay at chance throughout — expected by construction on chat renders (row-constant prefix input), not a post-training finding.
- The fixed dictionary passes the fitness gate at every stage (fraction of variance explained 0.80–0.90 of base) yet explains only ~33% of answer-state variance — per-feature reads are minority-variance-scoped.

## Goal

Fit a map from the pooled context activation to the SAE feature vector of the answer state, holding ONE fixed SAE dictionary across every post-training stage of an open ladder, and determine whether any individual SAE feature's per-feature held-out predictability improves across adjacent stage transitions beyond a selection-symmetric null over the whole dictionary.

**This experiment in context:** [#1895](https://eps.superkaiba.com/tasks/1895) (the parent) found on Qwen that the context→answer map's predictable subspace and an SAE's representable subspace largely coincide at the variance grain. [#1336](https://eps.superkaiba.com/tasks/1336) read the same map in raw activation space across the Llama-3.1-8B Tülu ladder and banked the activation stores consumed here; [#1902](https://eps.superkaiba.com/tasks/1902) covered the OLMo-2 ladder. This task re-reads #1336's map target in named-feature space, per stage. #1336's headline aggregate dense-activation R² was measured under its own eval protocol — not directly comparable to per-feature R² through a fixed dictionary.

**Broader narrative:** a raw-activation R² says the map improved, never at what; the per-feature read was meant to name which answer properties the model commits to earlier as post-training proceeds. At this grain, the max-over-dictionary test is dominated by one-to-three-row features, while the real movement — the retrieval jump at SFT, widespread per-cell exceedances — sits far below the selection bar. Naming individual features will take a power-matched stratified test, not a wider net.

## Methodology

**Design:** five stages of the published Tülu-3 ladder (Llama-3.1-8B base, SFT, DPO, RLVR — reinforcement learning with verifiable rewards — and longer-RLVR) × seven (render, corpus) combinations × two mapping arms — prefix-based (everything before the user query) and context-based (prefix plus query), per the project's both-arms rule — give 70 fitted maps and 56 adjacent-transition delta cells (4 transitions × 7 combos × 2 arms). One fixed base-model SAE encodes each stage's banked layer-29 answer-state activation into a 262,144-dim TopK feature vector (k = 32); a ridge map from the 4,096-dim pooled input predicts it; the dependent variable is each feature's held-out R² change ΔR²_j across adjacent stages. Decision rule declared in the plan: at least one (feature, cell) pair must exceed the p97.5 of the null distribution of the per-draw global max of ΔR²_j taken over both axes (feature and cell) under one synchronized draw schedule.

**Training:** **N/A — no model training.** All five checkpoints are published Tülu-3 artifacts; the activations were banked by the predecessor raw-activation-map experiment on this ladder (lineage in the footer).

**Evaluation:** every true fit and every null draw runs the same estimator. Complete fit/eval parameter table (no training hyperparameters exist for this task):

| Parameter | Value | Source |
|---|---|---|
| Model ladder | `meta-llama/Llama-3.1-8B` base + `allenai/Llama-3.1-Tulu-3-8B` SFT/DPO/RLVR/longer-RLVR | plan §11; grain manifest `model_id` |
| SAE dictionary | `EleutherAI/sae-llama-3.1-8b-64x`, layer 29, TopK k = 32, d_sae 262,144 | plan §11; SAE `cfg.json` |
| Input dim | 4,096 (pooled context / prefix activation, layer 29) | grain manifest `d_in` |
| Ridge λ grid | `np.logspace(-3, 8, 23)`; audited edge extensions ≤ 2 per side | plan §Design (parent-realized grid); per-feature fit JSONL header |
| λ selector | GCV with dof cap 0.9; full-dictionary criterion in the true fits, 1,024-column subsample in the null engine | fit JSONL `lambda_selector`; `issue2061_null.py` |
| Folds | K = 5 group folds, fold seed 0, shared between the true fits and the null engine | per-cell null JSON |
| Null battery | 500 stage-label permutation draws, seed base 42, one synchronized schedule across all 56 cells; per-draw global max | `GLOBAL_L29.json` |
| Rows per fit cell | 7,315–17,776 (per-fold n_train 5,816–14,221, all above 4,096; primal convention on all 35 cells) | `grain_manifest.json` |
| Corpora | `gsm8k_train_full` (GSM8K), `math7500` (MATH), `lmsys23k` (real-user LMSYS chat; chat + naturalistic renders), `if11k` (instruction-following), `uf11k` (UltraFeedback), `sft11k` (Tülu SFT mix) | plan §7 |
| Fitness gate | per-stage FVE ≥ 0.8 × base on a fixed 1,000-row slice; L0 in [10, 200]; activated-feature count ≥ 0.8 × base | plan §7; `fitness/summary_L29.json` |

Estimator notes read before verdicting (plan §6 duty 1): the true fits select λ on the full-dictionary GCV criterion while the null engine subsamples 1,024 columns, so each cell also persists an identity read — the unpermuted statistic recomputed through the engine path. That gap exceeds 0.02 on 23 of 56 cells, in both directions (range −0.10 to +0.49, median near 0; the argmax feature agrees on 25 of 56 cells), so the headline reports the mixed read and the exactly-symmetric engine read side by side. The bootstrap range on the global p97.5 bar (per-draw re-selection over cells inside each resample) spans 1.78 to 1.83 (10,000 resamples, seed 42). Prefix-arm scope: 7 of the 35 prefix fit-arms (MATH × base/SFT/DPO/longer-RLVR, GSM8K × SFT/RLVR/longer-RLVR, all chat render) stayed λ-edge-pinned after both grid extensions and are flagged regularization-limited; the 8 prefix delta cells touching them are never narrated as clean reads (they stay in the null axis, treated symmetrically). Chat-render prefix inputs are row-constant (parent-measured max pairwise cosine distance ≤ 1.5e-4), so near-zero prefix R² is expected by construction there; the naturalistic-render prefix arm is a genuine low-prior fit arm. kNN retrieval (cosine + euclidean; chance = k/n_pool) is reported for every fitted map; the identity-plus-learned-bias baseline is inapplicable (input and output dims differ, 4,096 vs 262,144). Figure legends identify corpora by the dataset ids mapped in the table above — acknowledged as legend text. Conciseness note: the six result blocks run above the 120-word soft cap and two Takeaways bullets run over the 30-word bullet cap, one carrying the by-construction prefix-arm caveat (total prose above the round-1 budget) — acknowledged; the plan's five binding analyzer duties are folded inline rather than dropped. Companion per-cell views for the non-headline transitions and arms (`f2_percell_base_sft_context.png`, `f2_percell_base_sft_prefix.png`, `f2_percell_sft_dpo_context.png`, `f2_percell_sft_dpo_prefix.png`, `f2_percell_dpo_rlvr_prefix.png`, `f2_percell_rlvr_longer-rlvr_context.png`, `f2_percell_rlvr_longer-rlvr_prefix.png`, and the seven `f1_delta_scatter_*` siblings — not embedded: identical view on non-winning transitions/arms, committed at the same pinned SHA). The planned arm-agreement view `f5_arm_agreement.png` (per-cell true max ΔR²_j, prefix arm against context arm, one panel per transition, render classes marked) is committed at the same pinned SHA, not embedded: every cell sits above the y = x line with prefix maxima near zero — the by-construction prefix degeneracy already carried in the prefix-arm scope note above, no read beyond the per-cell views. A free-analysis follow-up round (script `scripts/issue2061_followup_free_analysis.py`, committed `67b22aeab5`) added two descriptive reads over the existing artifacts, both folded into Results. The first is activation-support counts for the winning feature of every cell exceeding its own per-cell bar under either read — a feature counts as active on a row when it appears in that row's TopK-32 sparse code with a nonzero value, over all payload rows of the stage (`eval_results/issue_2061/followup_free_analysis/p1_winner_activation_counts.json`). The second is the plan §7 secondary read of cross-corpus rank agreement over each cell's top-100 improved features, context arm only — prefix arms skipped because 24 of 28 chat-render prefix delta cells are render-degenerate by construction (the prefix-arm scope note above) and 7 prefix fit-arms are regularization-limited (`p2_cross_corpus_rank_agreement.json`). That round's original overlap figure (`i2061_p2_cross_corpus_overlap.png`, committed `67b22aeab5`) frames overlap against the naive 100/262,144 dictionary-wide chance in its rendered title and is superseded by the embedded re-render `i2061_p2_cross_corpus_overlap_v2.png`, which drops that benchmark: with only 740–12,760 improved features per cell, the pool-conditional expected overlap is percent-scale — measured at 1.09 per 100 at review time for the base→SFT GSM8K × instruction-following pair; per-pair conditional expectations are not persisted in the round JSON.

**Data extraction:** the only model-side inputs are the banked turnstore activation stores produced by the predecessor raw-activation-map experiment on this ladder — 50 stores on the HF data repo under [`issue1336_rlvr_ladder/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fd182f8da8c61122ea6fc28d5ee0821474b18da4/issue1336_rlvr_ladder), consumed at that experiment's own production concat grain (three combos concatenate a wave-1 store with a v2 extension store at the conversation-index-5,000 boundary; the predecessor's headline fits ran on the same seam). A fail-loud grain gate re-counted every cell at the consumed grain before any fit: all 35 cells primal-regime (minimum per-fold n_train 5,816 vs d 4,096), zero boundary/duplicate/schema assertion failures, per-stage generation keep-rates 0.669–1.000 with base systematically lowest. The permutation pairs rows on conversation-id intersection; the paired fraction is 0.98 at the median cell and 0.735 at the minimum. One scope boundary carried with either verdict direction: unpaired rows keep their natural stage in every draw, so an improvement expressed only on rows newly kept by the later stage raises the null band exactly as it raises the statistic — invisible to this test by construction. No corpus text was read at any point; every input is an activation tensor or a JSON aggregate (relevant for the real-user LMSYS rows).

**Sample training/evaluation data + completions:** this task generates no text completions (activation-space analysis; the rollout text and its worked examples live with the predecessor experiment — lineage in the footer). In their place, verbatim scored rows. First, the global-winning feature's two per-feature fit rows — cherry-picked (it is the argmax, not a random sample); complete files: [DPO fit JSONL](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/fd182f8da8c61122ea6fc28d5ee0821474b18da4/issue2061_sae_predictability/analysis_tensors/per_feature_r2/dpo_chat_if11k_context_L29.jsonl) and [RLVR fit JSONL](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/fd182f8da8c61122ea6fc28d5ee0821474b18da4/issue2061_sae_predictability/analysis_tensors/per_feature_r2/rlvr_chat_if11k_context_L29.jsonl) (later per-row fields elided here; full rows at the links):

<details>
<summary>Winning feature 175088, DPO and RLVR stage rows (instruction-following corpus, context arm)</summary>

```json
{"feature_id": 175088, "R2": -0.7632381885205943, "n_train_folds": [7657, 7657, 7658, 7658, 7658], "n_test_total": 9572, "best_lambda_folds": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]}
{"feature_id": 175088, "R2": 0.6946107493091754, "n_train_folds": [7628, 7628, 7628, 7628, 7628], "n_test_total": 9535, "best_lambda_folds": [3162.2776601683795, 3162.2776601683795, 3162.2776601683795]}
```

</details>

Second, the winning delta cell's null-battery row, scalar fields verbatim (the two 500-element per-draw arrays elided) — cherry-picked (the winning cell); all 56 rows sit in [the null directory](https://github.com/superkaiba/explore-persona-space/tree/5d001092782beadab91efd81eb4d785741eeeac8/eval_results/issue_2061/null):

<details>
<summary>Winning cell null row (DPO→RLVR, instruction-following corpus, context arm)</summary>

```json
{"pair": "dpo_rlvr", "render": "chat", "corpus": "if11k", "arm": "context", "layer": 29, "true_max_delta_r2": 1.4578489378297697, "true_argmax_feature_id": 175088, "null_quantiles_per_cell": {"p50": 1.2485673427581787, "p95": 1.4580941200256348, "p97.5": 1.4986213445663452, "p99": 1.5033328533172607}, "n_draws": 500, "draw_seed_base": 42, "engine_version": "refit-v1", "k_folds": 5, "fold_seed": 0, "gcv_dof_cap": 0.9, "gcv_feature_subsample": 1024, "n_rows_before": 9572, "n_rows_after": 9535, "engine_identity_max_delta": 1.418442082339357, "engine_identity_argmax": 175088, "n_paired": 9105, "n_unpaired_before": 467, "n_unpaired_after": 430, "wall_s": 842.4665167331696}
```

</details>

## Results

### The largest per-feature gain sits at the median of the global selection-symmetric null — zero pairs clear the bar

The left panel histograms the 500 per-draw global null maxima — each draw shuffles stage labels within corpus (rows paired on conversation id), refits both pseudo-stages per fold, and takes the max ΔR²_j over all features and all 56 cells; the red line marks the true maximum. The right panel shows the 56 per-cell true maxima against the same bar.

![Global selection-symmetric null distribution with the true maximum at the median, and per-cell true maxima all below the global bar](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061/f6_global_null.png)

> **Figure.** *The primary decision rule fails to reject.* Left: 500 per-draw global null maxima; the true max 1.458 (red) sits at the null p50 1.451, below the p97.5 bar 1.816. Right: all 56 per-cell true maxima fall below the global bar.

| Read | Observed max | Winning (feature, cell) | Null draws above (of 500) |
|---|---|---|---|
| Mixed (true fits vs engine null) | 1.458 | 175088; DPO→RLVR, instruction-following, context | 244 (p = 0.49) |
| Exactly-symmetric (engine both sides) | 1.761 | 210279; SFT→DPO, instruction-following, context | 55 (p = 0.11) |

Zero of the 56 × 262,144 (feature, cell) pairs clear the 1.816 bar under either read. Prefix cells supply 12 of the 500 global-max draws; a context-arm-only re-reduction (bar 1.783) leaves the verdict unchanged, and its thinnest composition — the engine-symmetric max 1.761 against that context-only bar — leaves 48 of 500 draws above (p = 0.096). Improvements expressed only on rows newly kept by the later stage are invisible here by construction (see Data extraction).

### The global bar is reachable only by features active on one to three rows; well-populated features top out 6.5× lower

Each strip shows every scored feature's ΔR²_j for the DPO→RLVR context maps, one strip per corpus, top three feature ids labeled; each cell's own null p97.5 is dotted, the global bar dashed.

![Per-cell per-feature delta R squared strips for the DPO to RLVR transition, context arm, with per-cell and global null bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061/f2_percell_dpo_rlvr_context.png)

> **Figure.** *The raw per-feature view behind the headline.* DPO→RLVR context arm: per-feature ΔR²_j by corpus (7,315–17,776 rows per cell). The winning instruction-following cell's own p97.5 is 1.499; the global bar 1.816 sits above every strip.

The winning feature is active on 1 of 9,572 DPO rows and 3 of 9,535 RLVR rows; its R² swings from −0.76 to +0.69 — an unstable rare-feature read that does not clear even its own cell's local bar (1.458 vs 1.499).

Among the 415 features active on 100+ rows in both stages, the largest gain is ΔR² 0.280 (median −0.06) — 6.5× below a global bar that itself exceeds the ~1.0 ceiling a stable feature can reach, since moving from unpredictable to perfectly predicted is a gain of about 1.0. 94% of null global-max draws land on the three instruction-following context cells (per-cell bars 1.34–1.78) — a bar only one-to-three-row features can reach makes the non-rejection a power statement, not a no-effect finding.

### Eighteen of 56 cells exceed their own per-cell null, an order of magnitude above chance — but their winning features are mostly thin-support

Per-feature ΔR²_j against feature id for the DPO→RLVR context maps, colored by corpus (chat render left, naturalistic right); per-cell null p97.5 bars dotted, global bar dashed.

![Per-feature delta R squared versus feature id for the DPO to RLVR context maps with per-cell and global null bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061/f1_delta_scatter_dpo_rlvr_context.png)

> **Figure.** *Local exceedances the global test cannot see.* DPO→RLVR context arm, per-feature ΔR²_j (7,315–17,776 rows per cell); dotted lines are per-cell null p97.5 bars, dashed the global 1.816 bar.

18 of the 56 cells put their true maximum above their own per-cell p97.5 (16 of 56 under the engine-symmetric read), where about 1.4 would be expected at the 2.5% per-cell rate; 14–15 are context cells, spread over all four transitions.

Two winners repeat: feature 126412 tops base→SFT on real-user chat in both renders (2 pre-transition rows); feature 251937 tops SFT→DPO there in both arms (4 rows). Support counts make it the rule: in 14 of the 18 cells the winner activates on fewer than 10 pre-transition rows (12 distinct features; 10 of 16 under the engine read); only 3 reach 100 rows in both stages. With no multiplicity correction and correlated draws across cells, this stays descriptive: largely the same rare-feature class as the global winner, not rejections.

### Top-100 improved features overlap weakly across corpora but persist across renders of the same corpus

Each panel shows one transition's 7×7 matrix of top-100 overlap fractions between (render, corpus) cells, context arm: the share of the 100 most-improved features (finite ΔR²_j > 0 per cell) that two cells' lists have in common.

![Overlap of top-100 improved features between corpora for each transition with the same-corpus cross-render pair brightest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db78818b0bdb98050d3123fa5de84cdf793b4d57/figures/issue_2061/i2061_p2_cross_corpus_overlap_v2.png)

> **Figure.** *Corpus idiosyncrasy of the top-improved lists.* Overlap of the 100 most-improved features per cell pair, per transition (context arm; improved pools 740–12,760 features per cell). The bright off-diagonal cell in every panel pairs the real-user chat corpus with itself under the naturalistic render.

Chat-render cross-corpus overlaps average 7.7, 2.5, 1.1, and 3.1 per 100 across the four transitions, and with only 740–12,760 improved features per cell the pool-conditional expected overlap is itself percent-scale (1.09 per 100 for the base→SFT GSM8K × instruction-following pair, observed 0) — most pairs sit at conditional-chance scale, and the naive 100-of-262,144 benchmark is the wrong yardstick. The exceptions track corpus relatedness: the two math corpora share up to 22 per 100, the instruction-following and preference corpora up to 39, and the same real-user chat corpus under its two renders 57–74. Top-improved lists are corpus-keyed, not render-keyed — consistent with corpus-idiosyncratic movement; by design this descriptive read cannot adjudicate any single winner's mechanism.

### Context-arm retrieval quality jumps at SFT and plateaus; prefix arms sit at chance throughout

Both panels plot kNN retrieval of each fitted map on held-out rows — whether the predicted answer-SAE vector lands nearest its true row (acc@1) or within the top 5% of the held-out pool (acc@k, chance 0.05; realized k 364–889 per cell) — per corpus across the five stages, both arms.

![kNN retrieval accuracy at 1 and at k for every fitted map across post-training stages](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061/f4_knn.png)

> **Figure.** *Aggregate map quality per stage.* kNN retrieval per fitted map (held-out pool; chance 1.4e-4 at acc@1, 0.05 at acc@k). Solid = context arm, dashed = prefix arm; the prefix band lies on the chance line.

Context-arm euclidean acc@k rises from base (0.13–0.41 across corpora) to SFT (0.42–0.75), then plateaus through DPO/RLVR — MATH is the extreme, 0.16 to 0.75; cosine acc@1 sits 14–1,533× above chance (median 711×; euclidean acc@1 8–1,129×, median 454×; N = 35 fits). Prefix arms stay at chance at every stage — expected by construction on chat renders (row-constant prefix input), not a discovered effect.

Post-training does improve the aggregate context→answer-feature map; the non-rejection concerns single named features against a selection bar. Part of the base→SFT jump plausibly reflects the base stage's noisier generations (keep-rate 0.669–0.689 on 5 of its 7 cells, the ladder's lowest) — a composition move, not only a mapping one.

### The fixed base-model dictionary passes the fitness gate at every stage while declining toward the bar

Per-stage fitness of the one fixed dictionary on each stage's banked answer states over a fixed 1,000-row validation slice: fraction of variance explained with pass bar and halt floor, mean active features per row, and the descriptive dead-feature fraction.

![Per-stage SAE fitness: fraction of variance explained, L0 sparsity, and dead-feature fraction on the fixed dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061/f3_fitness.png)

> **Figure.** *Fitness-gate diagnostic.* FVE per stage on the fixed dictionary (n = 1,000 rows); all stages pass the 0.8× base bar; L0 tracks the TopK target 32; the dead-feature panel is descriptive (pool-ceiling arithmetic), not a gate.

| Stage | FVE | Ratio to base | Activated features (bar 5,434) |
|---|---|---|---|
| base | 0.330 | 1.00 | 6,792 |
| SFT | 0.297 | 0.90 | 7,948 |
| DPO | 0.269 | 0.815 | 6,354 |
| RLVR | 0.265 | 0.804 | 6,175 |
| longer-RLVR | 0.276 | 0.836 | 6,747 |

Every stage passes (FVE = fraction of variance explained; L0 31.7–31.9 throughout), but RLVR and DPO sit just above the 0.80 bar — dictionary drift is real if bounded.

The absolute level matters more: fit on token-level base-model residuals and applied here to pooled response-mean states off its training pool, the dictionary explains about a third of the variance, so feature-level predictability is read through a minority-variance lens. The 0.97 dead-feature fraction is pool-ceiling arithmetic (1,000 rows × 32 active slots reach at most 32,000 of 262,144 features), not a health signal.

---

**Repro:** Code: branch `issue-2061` @ `5d001092782beadab91efd81eb4d785741eeeac8` (eval JSONs + figures committed there); null engine `scripts/issue2061_null.py` (`ENGINE_VERSION="refit-v1"`, landed `d095ea09cf`); dispatcher `scripts/issue2061_dispatch.sh` @ `99c0867b9f`; free-analysis follow-up `scripts/issue2061_followup_free_analysis.py` @ `67b22aeab5`; overlap re-render `scripts/issue2061_p2_overlap_figure_v2.py` @ `db78818b0b`; plan v13. Compute: per-feature fit phase on 1×H100 (RunPod `pod-2061`, terminated after verification); null battery on 8×H200 (fellows lane, ~15–23 GPU-h measured from the one-cell pilot); fitness gate on 1×H200 (fellows, under 1 h); figures VM-local. In-repo artifacts at the pinned SHA: [`eval_results/issue_2061/null/`](https://github.com/superkaiba/explore-persona-space/tree/5d001092782beadab91efd81eb4d785741eeeac8/eval_results/issue_2061/null) (57 files incl. `GLOBAL_L29.json`), [`eval_results/issue_2061/fitness/`](https://github.com/superkaiba/explore-persona-space/tree/5d001092782beadab91efd81eb4d785741eeeac8/eval_results/issue_2061/fitness) (6 files), [`eval_results/issue_2061/grain_gate/`](https://github.com/superkaiba/explore-persona-space/tree/5d001092782beadab91efd81eb4d785741eeeac8/eval_results/issue_2061/grain_gate) (`grain_manifest.json`), [`figures/issue_2061/`](https://github.com/superkaiba/explore-persona-space/tree/5d001092782beadab91efd81eb4d785741eeeac8/figures/issue_2061) (20 PNG+PDF+meta triplets). Free-analysis follow-up round (2026-08-07): [`eval_results/issue_2061/followup_free_analysis/`](https://github.com/superkaiba/explore-persona-space/tree/67b22aeab50e992e9537c38b873acf6d2ae3ef2a/eval_results/issue_2061/followup_free_analysis) (2 JSONs @ `67b22aeab5`) and the [overlap figure re-render](https://github.com/superkaiba/explore-persona-space/blob/db78818b0bdb98050d3123fa5de84cdf793b4d57/figures/issue_2061/i2061_p2_cross_corpus_overlap_v2.png) (+PDF+meta @ `db78818b0b`). HF data repo, prefix [`issue2061_sae_predictability/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fd182f8da8c61122ea6fc28d5ee0821474b18da4/issue2061_sae_predictability) (verified via `list_repo_tree` at write time): `analysis_tensors/per_feature_r2/` (70 JSONLs, 17.3 GB), `analysis_tensors/null/` (57 files), `analysis_tensors/fitness/`, `sae_encoded/` (36 TopK target files), `logs/` — all under that pinned tree link.
- Reused turnstore activation stores from [#1336](https://eps.superkaiba.com/tasks/1336): 50 stores under [`issue1336_rlvr_ladder/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fd182f8da8c61122ea6fc28d5ee0821474b18da4/issue1336_rlvr_ladder) — fit: same ladder checkpoints and banked layer-29 slots, consumed at that task's own production concat grain, re-verified per cell by this task's grain gate.
- Reused SAE `EleutherAI/sae-llama-3.1-8b-64x` (public artifact) — fit: a base-Llama-3.1-8B layer-29 dictionary is exactly the fixed-dictionary premise of the Goal.
- Judge: none (no LLM-judged dependent variable). Language-intrusion audit: N/A — non-Qwen evaluated model and no on-policy generation in this task.

**Context:** Origin prompt (verbatim): "can you also run another issue which is training context vector -> SAE feature mapping at each stage and see if any SAE features get better predicted after each stage." Parent: [#1895](https://eps.superkaiba.com/tasks/1895). Created 2026-08-04; grain gate + fits + null battery + fitness + figures completed 2026-08-06/07; analyzed 2026-08-07; free-analysis follow-up (winner support counts + cross-corpus rank agreement) folded 2026-08-07.
