---
title: 'The context→answer map reads SAE features selectively: per-unit gain is lowest
  on high-variance directions and template-heavy features (HIGH confidence)'
kind: experiment
tags: []
created_at: '2026-08-07T06:53:39Z'
has_clean_result: false
parent_id: 1482
origin_prompt: we also trained a SAE feature -> answer vector mapping. Since we want
  to see what kind of information is stored at the context vector (newline before
  assistant speaks) we want to do a similar analysis to the answer vector but seeing
  which predictors most predict whether a SAE feature is read at the context vector.
  I am not sure how to formalize that the mapping reads from this SAE feature - potentially
  we can invert the mapping? or do some zero-ing out and measure effect on output?
  or just look at the mapping matrix and do some linear algebra. can also use the
  SAE -> SAE mapping if that's easier. [then] why aren't we using all the features???
  [then] run this in the background as much in parallel as possible
workflow: v1
goal: Characterize, at full 131,072-feature dictionary width, which SAE features the
  context->answer map READS at the context vector v_C (the last prompt token, the
  newline of the assistant header) on Qwen-2.5-7B-Instruct layer 19; separate READ
  (used by the fitted map) from CARRIED (present at v_C but unused) and from FREQUENT
  (mechanically weighted by firing rate); and determine which mechanical feature properties
  predict being read, as activity-matched partials against a selection-symmetric null.
relates_to:
- spec-context-as-vector
---
# The context→answer map reads SAE features selectively: per-unit gain is lowest on high-variance directions and template-heavy features (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- **23 of 25 mechanical feature properties predict the context→answer map's per-unit feature read beyond activity matching** (max |partial Spearman| 0.239 vs a 0.009 null band, n = 128,450 features) — the read is structured, not firing frequency plus noise.
- Per-unit gain falls with residual-direction variance (−0.24) and template-token firing share (−0.21; −0.18 after double frequency matching); the dense→dense map flips the variance sign (+0.12), so the discount is map-specific, not shared input geometry.
- The per-unit read is rank-unrelated to what a feature univariately carries about the answer (Spearman −0.05 overall; 0.59 for the frequency-weighted read); 661 features are high-carried but low-read, roughly half at below-median redundancy.
- The cheap algebraic read matches the refit map's own coefficient norms at Spearman 0.285 overall but 0.61–0.74 in the top activity deciles — frequency-dependent ridge shrinkage, not an unfaithful read; no scope narrowing was triggered.
- The parent-line rare-feature predictability deficit is mostly sample size: rescoring every feature on 1,707 active rows flattens the bottom five activity deciles to 0.22–0.24; the genuine gradient lives in the top two deciles.
- Coverage notes: the prefix-based arm is degenerate by construction (4 numerically indistinguishable `h_prefix` variants across 142,000 rows — capture noise, not context variation), and the conditional GPU cell never fired (d_B = 4,588, far under the 32,768 switch; realized GPU-hours 0).

## Goal

- **This experiment in context:** The parent [#1482](https://eps.superkaiba.com/tasks/1482) characterized the output side of the context→answer map in sparse-autoencoder (SAE) coordinates — which answer-side features the map can predict, and what predicts per-feature predictability. This run characterizes the input side at full 131,072-feature width: which features present at the context vector `v_C` (last-prompt-token state, the newline before the assistant answer) the fitted map actually uses, separating read (coefficient influence) from carried (univariately predictive) from frequent (firing rate), then regressing the read on the mechanical covariate panel. It also runs the matched-sample-size control the parent's rare-feature gradient lacked, and a coefficient refit whose input restriction is selected by last-token activity — the one deliberate change from the parent's pooled-span coefficient arm ([#779](https://eps.superkaiba.com/tasks/779) supplied the underlying dense capture and penalty grid).
- **Broader narrative:** Part of the context-as-vector line: feature-level interpretation of the pre-answer state is only viable if the map's use of interpretable features is predictable structure rather than bookkeeping of firing frequency. This run tests whether that program has a foothold on the input side.

## Methodology

- **Design:** Analysis-only re-read of two banked linear maps from the last-prompt-token state on Qwen-2.5-7B-Instruct layer 19 — W: dense `v_C` (3,584) → 131,072 mean-pooled SAE answer features, and M: dense `v_C` → dense mean-answer state (3,584) — plus a coefficient refit B: last-token SAE codes (4,588 features) → the same answer features. Nine arms: the per-feature read ladder on W and M; a univariate carried read; the B refit; an activity-stratified selection-symmetric permutation null; a row-pairing permutation null; an answer-side matched-sample-size control; two join-integrity positive controls; six map-reproduction gates. No training, no generation, no LLM judging; one deterministic seed (2163) for all permutation draws. The prefix-based mapping arm is a stated deviation: `h_prefix` takes exactly 4 distinct float-vector values across all 142,000 rows (258 rows deviate from the reference, minimum cosine 0.99989, uniform prefix end position 23) — capture-time numerical noise rather than context-driven variation, so a prefix-based map has no cross-context variance to fit; a multi-turn corpus is the named follow-up route. An earlier claim that the prefix state was exactly constant (one distinct row) was falsified at full grain and corrected before any analysis code ran; the 4-variant measurement above is the corrected record.
- **Training:** **N/A — no model training.**
- **Evaluation:** Per feature j with decoder column `d_j`: `U_j` = per-unit-activation influence of j through the map (the norm of the standardized decoder direction pushed through the ridge coefficients — frequency-free); `A_j` = exact ablation change in held-out R² from deleting j's rank-1 term (20,000 holdout rows, cross-term unclipped, so negative values are possible); `C_j` = mean squared prediction change per active holdout row over pooled target variance; `Carried_j` = held-out R² of the univariate predictor built from j's activation alone; `‖B[j]‖` = row norm of the refit coefficients. Headline statistic: the activity-matched partial Spearman between each of the 25 selection columns (22 varying covariate-panel floats after dropping the constant `dec_norm`, plus 3 last-token columns) and `log U_j`, with the observed max-|partial| compared against the 97.5% quantile of per-draw maxima from a 1,000-draw activity-decile-stratified permutation of the DV — each draw recomputes all 25 partials and takes the same max, so observed and null get identical selection. The partial statistic is bounded by 1, leaving a band-to-ceiling margin of 0.991 — the test is informative by construction. Matching variable: `lasttoken_count`; a robustness pass adds `firing_freq_per_token` as a second matching variable. Two positive controls gate any frequency-only narration and both pass: raw Spearman of `A_j` vs last-token activity +0.390 (threshold +0.3) and activity vs `lasttoken_count` +0.290 (threshold +0.2). The row-pairing null (200 draws) permutes the holdout pairing between firings and residuals for the ablation read. SAE reconstruction of `v_C` on holdout has fraction of variance explained 0.718 (residual share 0.282), inherited by the algebraic reads; the refit arm is reconstruction-free. Held-out reads carry a corpus fold (10,983 LMSYS / 9,017 WildChat holdout rows, joined through the banked provenance labels with zero sentinel hits). Reproduction gates: re-materialized W selected the banked penalty (grid index 13) with pooled panel R² matching the banked value to 10 decimals and max per-feature |ΔR²| 2.978e-08 (tolerance 5e-3); M matched its banked held-out R² and penalty; the identity-plus-learned-bias baseline for M reproduced at −1.023. Analysis constants:

| Parameter | Value | Source |
|---|---|---|
| Model / layer | Qwen-2.5-7B-Instruct, layer 19 residual stream | banked capture (`dense_targets_meta.json`, field `cx_last@L19`) |
| SAE | `andyrdt/saes-qwen2.5-7b-instruct` rev `c37e53c4bb…`, `resid_post_layer_19`, BatchTopK k=64, dictionary 131,072 | parent pin, loaded at the same revision |
| Panel / splits | 142,000 contexts; 120,000 fit / 2,000 validation / 20,000 holdout, sha-pinned row order | banked `inputs/{which,order}.npy` |
| Ridge penalty grid | `logspace(-3, 8, 23)`, selected on the validation carve (no generalized cross-validation) | banked cell configs |
| Selected penalties | W: 3162.2776601683795 (grid idx 13); M: 1000.0; B: 3162.2776601683795 | `repro_gates.json`, `confirm_B.json` |
| Refit restriction | last-token active count ≥ 20 train rows, cap 49,152; realized d_B = 4,588 (n_train/d = 26.2); excluded features carry 1.0% of active mass | `census.json`, `confirm_B.json` |
| Null draws | 1,000 (stratified permutation, per-DV); 200 (row-pairing) | `nulls/*.npz` |
| Retrieval read | 5,000 probes vs 20,000-row holdout pool; chance at rank 1 = 5e-05 | `knn_baselines.json` |
| Seed / software | 2163; numpy 2.2.6, scipy 1.17.1, torch 2.8.0 | run metadata in every result JSON |

- **Data extraction:** The panel is 142,000 real single-turn conversations sampled from LMSYS-Chat and WildChat (realism tier 1: real user text), inherited unchanged. Its production: dense residual-stream states were captured at layer 19 at the last prompt token (the newline of the assistant header) for every context, alongside mean-pooled answer states over the model's answer tokens; per-token SAE codes from the public BatchTopK dictionary above were reduced to pooled-span and last-token blocks and stored as a 1,920-shard sparse store; a per-feature covariate panel (activity, consistency, encoder/decoder geometry, logit-footprint statistics, template-token share, redundancy, projection variance) was computed from that store plus decoder geometry. This run streams the store once to build the last-token feature-by-context matrix (142,000 × 131,072, ~33 active features per row, 4,700,819 stored values; every row has at least one code; 13,765 features ever fire at the last prompt token, 13,289 in the training split), re-materializes W and M from the banked inputs under the parity gates above, and computes every read in closed form — no per-feature fits. The three last-token covariates (`lasttoken_count`, `mean_act_when_active_lasttoken`, `span_last_ratio`) are computed here from the training split. Complete-case features for the partials: 128,450 of 131,072.
- **Sample training/evaluation data + completions:** No model completions were generated anywhere in this task (zero sampling, zero judge calls); the analysis consumes banked activation tensors, so the sample below is numeric ladder rows rather than text. Disclosure: 5 of 131,072 per-feature ladder rows, random sample (seed 42); full artifact: [read_ladder__W.npz on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0d76405c5704798cda8e116d666c1a916e61a15a/issue2163_ctxread/results).

| feature id | `U_j` | `A_j` | `C_j` | carried | last-token count (train) | holdout count |
|---|---|---|---|---|---|---|
| 11697 | 0.2287 | 0.0 | 0.0 | undefined | 0 | 0 |
| 56756 | 0.2024 | 0.0 | 0.0 | undefined | 0 | 0 |
| 57524 | 0.2655 | 0.0 | 0.0 | undefined | 0 | 0 |
| 85794 | 0.2105 | 0.0 | 0.0 | undefined | 0 | 0 |
| 101441 | 0.2394 | 0.0 | 0.0 | undefined | 0 | 0 |

  All five randomly drawn features never fire at the last prompt token — the expected shape, since 117,783 of 131,072 features (90%) have zero last-token activations in training. `U_j` is still defined for them (it is a property of the map, not of the data), while `A_j` is exactly zero for the 122,529 features inactive on holdout and `Carried_j` is undefined off the 13,289 train-active features. I acknowledge the check-20 WARNs this body ships: several Takeaways bullets exceed the 30-word soft cap, several per-result blocks exceed the 120-word soft cap, and the Takeaways-plus-Goal-plus-Results total exceeds the 800-word budget — the overage carries the multi-arm coverage the plan requires. I also acknowledge that the partials forest figure renders covariate column slugs (e.g. `enc_dec_cos`, `span_last_ratio`) as its axis labels; these are the covariate panel's column names, glossed in prose where load-bearing.

## Results

### The per-unit read is structured: 23 of 25 feature properties predict it beyond activity matching

Each bar is the activity-matched partial Spearman between one of 25 feature properties and `log U_j` (per-unit read gain on the dense-to-SAE map) over 128,450 features; dashed lines mark the 97.5% max-over-columns permutation band. The companion scatter is the raw per-feature view behind the partials: |U_j| against last-token active count.

![Forest of activity-matched partials against the permutation band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_partials_forest_logU_W.png)

> **Figure.** *Nearly every mechanical property predicts the per-unit read, and every strong predictor is negative.* Activity-matched partial Spearman per property vs `log U_j`, n = 128,450; dashed lines: 97.5% band of the per-draw max over the same 25 columns (1,000 stratified permutations). Orange: geometry-coupled columns. The raw per-unit companion scatter follows.

![Raw per-feature read strength versus last-token activity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_U_vs_activity.png)

> **Figure.** *Raw companion view.* |U_j| (log) vs last-token active count + 1 (log), one point per feature; the mild downward envelope is the raw signal the matched partials isolate.

The top partial, `proj_var` (residual-state variance along the feature's decoder direction), reaches −0.239 against the 0.009 band; 23 of 25 properties clear it. Frequency proxies clear too (per-token firing frequency −0.196), so a second matching variable was added: the activity family collapses below 0.03 while `proj_var` (−0.19), `scaffold_frac` (−0.18; the share of a feature's firings on template/formatting tokens), and `write_norm` (−0.13) survive. The dense→dense map flips `proj_var` to +0.12, so the discount belongs to the SAE-output map rather than shared input geometry. Magnitudes are moderate next to the 0.29–0.58 parent-line reference effects; `proj_var` comes from a multi-turn corpus, which attenuates but cannot manufacture its row.

### The map's per-unit read is unrelated to what a feature univariately carries about the answer

Each point is one of the 13,289 train-active features: x = |A_j| (frequency-weighted ablation contribution, log scale), y = held-out R² of the univariate predictor from that feature's activation alone, colored by last-token activity decile.

![Per-feature carried information versus ablation contribution scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_read_vs_carried.png)

> **Figure.** *Carried information concentrates in high-activity features and decouples from per-unit read.* Univariate held-out R² (carried) vs |A_j| on log axes, one point per train-active feature, colored by last-token activity decile; this scatter is itself the per-unit view.

The per-unit read is rank-unrelated to carried information: Spearman −0.05 overall, rising 0.00 to 0.36 across activity deciles; the frequency-weighted read correlates 0.59 — agreement between read and carried is mostly frequency. 661 features sit in the high-carried, low-read tail; their nearest-duplicate similarity matches the population (median cosine to the nearest duplicate 0.336 vs 0.340), so about half the tail is not explained by near-duplicate absorption — though single-term deletion lets no other coefficient compensate, so this stays suggestive. Summed over features, ablation contributions total 0.508 of holdout variance vs 0.486–0.490 under pairing permutation — a small alignment-specific excess; with 200 draws the per-feature floor is p = 1/201, so per-feature claims stay descriptive. Ablation ranks are moderately stable across corpus halves (0.57 among features active in both).

### The cheap algebraic read matches the refit coefficients only where features are well-estimated

Each point is one of the 4,588 restriction features: x = `U_j` (algebraic per-unit read), y = the refit coefficient row norm, log-log.

![Refit coefficient norms versus algebraic per-unit read scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_bnorm_vs_U.png)

> **Figure.** *Agreement between the algebraic read and the refit's own coefficients is activity-dependent.* Coefficient row norm of the last-token refit vs `U_j`, one point per restriction feature (n = 4,588), log axes.

Overall Spearman is 0.285, rising monotonically within activity deciles from 0.25 to 0.74 (deciles 0 through 8; 0.61 in decile 9). That pattern matches frequency-dependent ridge shrinkage — the refit shrinks rarely-firing features toward zero while the algebraic read does not — rather than an unfaithful read; the scope-narrowing criterion (Spearman under 0.5 on well-estimated features) did not fire. The refit is well-posed (n_train/d = 26.2, penalty selected on the validation carve) and scores holdout pooled R² 0.647 vs the full map's 0.653. The planned GPU cell never ran: d_B = 4,588 sat far below the 32,768 venue switch, so this arm completed on CPU.

### Most of the rare-feature predictability deficit on the answer side is sample size

Lines: median per-feature answer-side R² by activity decile on the 16,384-feature panel — pooled (all holdout rows), conditional-on-active, and matched-n (each feature rescored on 200 draws of 1,707 active rows, the rare-decile median; 113 of 200 selected decile-0 features had enough rows). The companion scatter is the raw per-feature pooled R² against holdout active count.

![Pooled conditional and matched sample size gradients by decile](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_matchedn_gradient.png)

> **Figure.** *Matching row counts flattens the low-activity half of the gradient.* Median per-feature R² by answer-side activity decile: pooled, conditional-on-active, and matched-n (1,707 rows per draw); raw per-feature companion below.

![Raw per-feature pooled R2 versus holdout active count](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_matchedn_raw_perfeature.png)

> **Figure.** *Raw companion.* Per-feature pooled R² vs holdout answer-side active count + 1 (log), one point per panel feature (n = 16,384).

Pooled medians rise 0.132 to 0.304 across deciles here; the sibling full-dictionary estimate of the same gradient (0.091 to 0.266) covers a different feature set, so levels are not directly comparable. Under matched row counts the bottom five deciles flatten to 0.22–0.24 — rare features are predicted about as well as mid-frequency ones once scored on equally many rows — and the residual gradient concentrates in the top two deciles (0.28, 0.35). The smooth low-decile half of the gradient is a sample-size artifact; a genuine high-activity advantage remains.

### Retrieval: the dense→dense map identifies the held-out context; the feature-basis maps rank it top-10

Bars: retrieval accuracy at rank 1 (chance 1/20,000) per map and distance metric — predictions from 5,000 holdout probes matched against the full 20,000-row holdout pool.

![Retrieval accuracy at rank one per map and metric](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/figures/issue_2163/fig_knn_bars.png)

> **Figure.** *All three maps retrieve far above chance; the dense→dense map dominates.* Accuracy at rank 1 by map and metric, 5,000 probes vs a 20,000-row pool, chance 5e-05. Per-probe decomposition is a binary hit, so no per-unit companion is embedded.

The dense→dense map retrieves the true held-out context at rank 1 for 70% of probes (cosine; median rank 1; its identity-plus-learned-bias baseline scores R² −1.02, so it is far from a copy map). The dense→SAE map and the refit both reach 26% at rank 1 (median ranks 8–11). One caveat: the probe set is the first 5,000 holdout rows, which the corpus join shows are 100% LMSYS — the same probe convention as the sibling comparison line, but these retrieval numbers are LMSYS-only reads.

---
**Repro:** ~1.2 h wall on a 16 vCPU / 128 GB RunPod CPU pod (`pod-2163`) + ~17 min VM harvest; realized GPU-hours 0 (the conditional GPU cell never triggered). Driver: [scripts/issue2163_ctxread.py](https://github.com/superkaiba/explore-persona-space/blob/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/scripts/issue2163_ctxread.py) (pod phases ran at commit `c258775889`, clean tree per each JSON's run metadata; figures rendered at `a9af7a7808`). Eval JSONs + per-feature npz + stratified-permutation matrices: [eval_results/issue_2163/](https://github.com/superkaiba/explore-persona-space/tree/a9af7a780828ad7d34b0f260564c8ec9e83a4c23/eval_results/issue_2163); the two row-pairing null matrices (100.5 MiB each, `draws_dsse` shape 200 × 131,072 fp32 — landed fp32 vs a smaller fp16 plan estimate) exceed the git per-file limit and live on HF only. HF (all pinned): [results](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0d76405c5704798cda8e116d666c1a916e61a15a/issue2163_ctxread/results) (22 files incl. `nulls/`), [analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0d76405c5704798cda8e116d666c1a916e61a15a/issue2163_ctxread/analysis_tensors) (W/M bundles, B Gram + panel block, last-token CSR), [inputs](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0d76405c5704798cda8e116d666c1a916e61a15a/issue2163_ctxread/inputs) (9 files incl. dense answer targets + covariate panel v2). Raw completions: n/a — no generation stage anywhere in this task. Declared discard: the full holdout prediction matrices (20,000 × 131,072 fp32 per map, ~10.5 GB each) — pure derived matrix products; regenerate with one blockwise matmul of the persisted bundles against the banked holdout inputs. Reused artifacts — from [#1482](https://eps.superkaiba.com/tasks/1482): the 1,920-shard pooled/last-token SAE code store (`issue1482_error_analysis/analysis_tensors/sae_pooled/`), dense input matrix + sha-pinned splits (`issue1482_densesae_fullwidth/inputs/`), banked ridge cells + bridge JSONs (the parity targets), covariate panel v2, and the corpus-label join arrays (`scratch_meta/`) — fit: same SAE revision, splits, and penalty grid read off the artifacts' own configs, with all six reproduction gates PASS (tolerances 1e-3/5e-3; realized max per-feature |ΔR²| 2.978e-08, so no upstream drift reached this run despite the unpinned data-repo revision). From [#779](https://eps.superkaiba.com/tasks/779) via the parent: the 23-point penalty grid and the dense capture lineage. Dense answer targets (`Y_L19.f32.mm`) were uploaded from the VM to `issue2163_ctxread/inputs/` before any pod work, with a sha256 content pin against the banked input copy.

**Context:**
> we also trained a SAE feature -> answer vector mapping. Since we want to see what kind of information is stored at the context vector (newline before assistant speaks) we want to do a similar analysis to the answer vector but seeing which predictors most predict whether a SAE feature is read at the context vector. I am not sure how to formalize that the mapping reads from this SAE feature - potentially we can invert the mapping? or do some zero-ing out and measure effect on output? or just look at the mapping matrix and do some linear algebra. can also use the SAE -> SAE mapping if that's easier. [then] why aren't we using all the features??? [then] run this in the background as much in parallel as possible

Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — parent; the output-side mirror of the same map. Created 2026-08-07; run 2026-08-07 (single round, no follow-up rounds).
