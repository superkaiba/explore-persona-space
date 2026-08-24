---
title: 'The fresh turn-averaged SAE''s indeterminate tier verdict is floor-local:
  every looser alive floor reads coarse-over-specific because newly admitted specific-tier
  features are poorly predicted (MODERATE confidence)'
kind: experiment
tags:
- followup-auto
created_at: '2026-08-22T20:08:40Z'
has_clean_result: true
parent_id: 1482
origin_prompt: 'Paper outline 2026-08-22: ''SAE experiments — try turn averaged SAEs'''
workflow: v1
goal: 'Determine whether the context-to-answer map''s high-level-over-low-level predictability
  gradient (#1482) holds in a turn-averaged SAE basis: per-tier held-out R2 + acc@1
  vs identity+bias for turn-averaged SAE features of true and map-predicted answer
  summaries.'
---
# The fresh turn-averaged SAE's indeterminate tier verdict is floor-local: every looser alive floor reads coarse-over-specific because newly admitted specific-tier features are poorly predicted (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_2476.md](https://github.com/superkaiba/explore-persona-space/blob/357e7fe69c748661fc42d3a418755ab4259c5bd9/docs/methodology/issue_2476.md) · [gist](https://gist.github.com/superkaiba/290dcbfd1eb46f39b06da20d1c88d08b)

## Takeaways

- The indeterminate verdict is floor-local. At the plan-fixed 1%-of-rows alive floor the fresh-arm decision rule still returns indeterminate (tier medians 0.625/0.131/0.670 on 788/88/3 alive features), but at every swept looser floor — 0.5%, 0.25%, 0.2% — both rule conditions agree coarse-better: the permutation read sits below its whole null band and the coarse-minus-finest contrast flips from −0.04 to at least +0.52, with intervals clear of zero. The looser-floor rule readings are reported, not re-fixed; the reported lattice is floor-sensitive.
- Newly admitted finest-tier features are distinctively poorly predicted: the pooled finest median falls 0.670 → 0.087 → 0.049 → 0.046 as the floor drops from 1% to 0.2% of fit rows (newly-alive-only median 0.044, n = 127; 34% at or below zero), landing below the same-floor middle tier's interquartile range. The follow-up round's primary hypothesis — that they would match the middle tier — is refuted, and the three 1%-floor survivors (median 0.670) are an unrepresentative elite.
- Attrition remains dominant within the swept bracket: 130 of 49,152 finest-tier candidates (0.26%) clear even the loosest 0.2% floor, against 3 at 1%. The floor convention explains the indeterminate cell, not the census bulk, which stays with the joint matryoshka-allocation-plus-undertraining residual; floors below 0.2%, including any below the instrument's ~0.15% mean firing rate, were not swept.
- The instrument disagreement survives the sweep: the token-level dictionary on turn averages reads reversed at all three of its floors (permutation on the fine-better side throughout), retrieval stays coarse-concentrated at every floor (fresh coarse acc@1 0.706–0.708 vs middle at most 0.0252 on either predictor, chance 5.0e-5, n = 20,000), and at matched firing activity the fresh coarse-minus-middle contrast (+0.393) and the pile-trained twin's (−0.141) still exclude zero on opposite sides.
- The map beats identity+bias on median R² at every tier and every floor (1%-floor paired deltas +0.45 to +5.26; deltas exceed 1 because the baseline's held-out R² goes deeply negative). Per-feature predictability rank-correlates 0.767 across grain while the tier profile inverts: the token-level gradient ran through features that are dead at the 1% turn-averaged floor and that return poorly predicted at looser ones.
- Scope limits: one SAE training seed; the sweep is deterministic given the banked artifacts (split digests reproduce exactly, recomputed counts match the banked census with delta exactly 0, and the 1%-floor cell reproduces per-feature values with one 2.8e-3 near-tie among 879); the four floors share seeds and nested alive sets, so the looser-floor reads are one dependent pattern, not independent replications; the bridge reversal keeps a zero-crossing endpoint interval; open review residuals `consumer-contract-g4-residual`, `g2b-reuse-verdict-schema`, `retrieval-device-pin-residual`, and `g2a-min-cross-machine-headroom`, plus the floor-round residuals `unpinned-q1-source-revision`, `a13-demotion-scope-drift`, and `result-commit-invalidates-resume`, are detailed in Methodology.

## Goal

Determine whether the context-to-answer map's high-level-over-low-level predictability gradient — the coarse-over-specific gradient, in the parent's terms — holds in a turn-averaged SAE basis: per-tier held-out R² and retrieval accuracy versus an identity+bias baseline, for turn-averaged SAE features of true and map-predicted answer summaries.

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) fit the map — a ridge from the last-prompt-token activation to the whole-answer mean activation (layer 19, Qwen-2.5-7B-Instruct, ~963k real chat contexts, held-out R² 0.72–0.75). [#1482](https://eps.superkaiba.com/tasks/1482) found at token-level-SAE grain that this map predicts coarse matryoshka features far better than specific ones (tier medians 0.435/0.174/0.043). A token-level basis pools per-token codes, one step removed from the map's actual target; an SAE trained directly on whole-answer means decomposes that target itself, so this run asks whether the coarse-over-specific gradient is a fact about the map or about the token-level basis.

**Broader narrative:** the paper's limits-of-the-mapping section claims the map predicts high-level, identity-adjacent content and not low-level, topic-level content. This run rescopes that claim at the map's own grain: the specific tier mostly fails to clear any swept alive floor, the features admitted at looser floors are poorly predicted, and the retrieval read keeps a pronounced coarse-over-specific profile — the fresh-basis evidence now leans the paper claim's way, with the plan-fixed cell itself still indeterminate.

## Methodology

**Design:** One manipulated variable — the grain of the SAE basis (per-token activations vs whole-answer turn averages) — read through two instruments. The fresh-SAE arm (headline) trains a matryoshka BatchTopK sparse autoencoder (SAE) directly on banked layer-19 whole-answer mean activations and scores 20,000 held-out conversations; the bridge arm applies the token-level matryoshka dictionary from the prior token-level round to turn averages at layer 20 on that round's exact 6,000 score rows, giving the only same-dictionary, same-rows grain contrast. Matryoshka nesting makes feature id a training-imposed granularity label: ids below 2,048 are trained under every nested-prefix reconstruction loss (coarsest tier), ids 2,048–16,383 the middle tier, ids 16,384–65,535 only under the full-width loss (finest tier). Controls: identity+bias (predict the answer summary as the context state plus a learned constant shift), train-mean null, a dense-input companion ridge (context state → each SAE feature directly, separating "the map loses tier-t content" from "tier-t content is linearly unreachable at all"), a 20-draw row-shuffle floor, a within-activity-quintile tier-label permutation, a corpus-transfer fold (fit on LMSYS rows, score on the 9,017 WildChat holdout rows), a pile-trained dictionary twin, and an inlier-masked pooling twin. One SAE training seed (2476). Config slugs, matching the plan roster: `arm_c_map` / `arm_c_ib`, `arm_b_map` / `arm_b_ib`, `dense_in_c` / `dense_in_b`, `trainmean_null`, `shuffle_null`, `tier_perm`, `corpus_fold`, `arm_b_pile`, `arm_b_inlier`. The verdict rule is defined on the fresh-SAE arm and fixed in the plan:

| Outcome | Rule (both conditions required) |
|---|---|
| gradient holds | permutation percentile ≤ 2.5 (coarse-better side) and coarse-minus-finest median contrast above 0 |
| gradient reversed | permutation percentile ≥ 97.5 (fine-better side) and contrast below 0 |
| indeterminate | any other combination, including the two conditions disagreeing |

Same-issue follow-up round (`floor-sensitivity-sweep`, one variable): the alive-floor convention becomes a swept analysis parameter — 1% (the plan-fixed convention), 0.5%, 0.25%, and an absolute 240 rows (0.2%) of the 120,000 fit rows on the fresh arm; 240/120/60 rows (1%/0.5%/0.25%) on the bridge arm, whose absolute point coincides with its 1%. Nothing is retrained or refit: the banked SAE, the banked dense map and identity+bias predictions, splits, score rows, seeds, and draw counts are the parent's, identical across floors, so cross-floor differences are never resampling noise. The rule above stays fixed at the 1% floor only and is evaluated descriptively at the looser floors. A demotion convention fixed at plan approval — a floor where over half of a populated tier's alive features return undefined R² is reported census-only, with no rule label — never triggered (zero undefined reads at every floor, tier, and arm). The round's primary-hypothesis convention, also fixed in advance: the finest-tier median at the 240-row floor is compared against the middle tier's interquartile range computed on the same floor's alive middle features. The pile-dictionary leg is exploratory; its census uses recomputed counts (no banked reference exists for it). Round config slugs, matching the follow-up plan roster: `fs_c_map_f1200` / `fs_c_map_f600` / `fs_c_map_f300` / `fs_c_map_f240` with identity+bias twins `fs_c_ib_f*`; bridge `fs_b_map_f240` / `fs_b_map_f120` / `fs_b_map_f60` with `fs_b_ib_f*`; `fs_b_pile_f*`; gates `gate_repro`, `gate_counts`, `gate_rows`; nulls `fs_trainmean`, `fs_shuffle`, `fs_tierperm`.

**Training:** the only model trained is the fresh turn-averaged SAE; the map itself is refit (never retrained): a shared-Gram ridge from context state to answer summary, with the regularizer selected by whole-map validation R² over a 23-value logarithmic grid from 1e-3 to 1e8 on the pinned 400-row validation split — selected value 1e-3, the grid's low edge, for both refits, seed 0 (the corpus-transfer refit runs the same grid and selection rule). Every value below is copied from the committed run artifacts.

| Hyperparameter | Value | Source |
|---|---|---|
| SAE architecture | matryoshka BatchTopK; nested prefix reconstruction losses, mean reduction; threshold EMA 0.999 | `train_log.json` cfg |
| SAE dimensions | input 3,584; dictionary 65,536; k = 100; tier bounds 2,048 / 16,384 / 65,536 | `train_log.json` cfg; tier bounds match the bridge dictionary for tier parity |
| SAE optimizer | Adam, betas 0.9 / 0.999, lr 2e-4, batch 256, 3 epochs (10,941 steps), init seed 2476 | `train_log.json` cfg; recipe from arXiv 2606.28548 (same model, layer, corpus, object) |
| SAE training rows | 933,444 train + 10,000 validation carve (seed 2476) from the 943,444-row pool (963,444 banked contexts minus holdout and pinned val/test) | `train_log.json` pools |
| Realized training state | final validation variance-FVE 0.9093 (FVE = fraction of variance explained; training-sanity halt floor 0.5) at L0 98.96 (mean active features per row); per-tier dead-feature fractions 0.095 / 0.291 / 0.356 after epoch 3 | `train_log.json` epochs; floor from `gates_p4.json` |
| Map refits | shared-Gram ridge, grid and selection rule as stated above; selected regularizer 1e-3 (grid low edge) at validation R² 0.754 for both the full-train and holdout-excluded refits, seed 0; both reproduce the committed whole-map R² (deltas 3.5e-7 and 2.1e-7 against tolerances 0.002 / 0.003) | parent refit records (`refit_full__ridge__seed0.json`, `refit_holdout__ridge__seed0.json`), reproduced per `recon_gate.json` |
| Bridge-arm map | ridge fit on 24,000 rows, scored on 6,000 (selected λ 3.16e3, validation R² 0.671); dense-input companion λ 1.0e3 | `preds_meta.json` |
| Bridge dictionary | `chanind/qwen2.5-7B-it-layer-20-saes` revision `db8c88b4…`, sae_id `lmsys/matryoshka/k-100` (pile twin `pile/matryoshka/k-100`), k = 100, width 65,536 | the prior round's pin, re-listed live |
| Alive floor + panel | active on ≥ 1% of fit-side true-summary encodes (1,200 of 120,000 fresh; 240 of 24,000 bridge); panel cap 16,384, seed 14824 | `tier_tests_c.json` / `tier_tests_b.json` alive-mask provenance |
| Statistics constants | 10,000 permutation draws; 10,000 bootstrap draws (seeds 24761 / 24762); 20 row-shuffle draws; retrieval at k = 1 / 5 / 10, euclidean + cosine; acc@1 uncertainty: 95% Wilson intervals; matched-activity interval recompute 10,000 bootstrap draws, seed 42, estimators imported from the run scripts | `tier_tests_*.json`, `retrieval_*.json`, `matched_tier_ci.json` |
| Answer generation | none this run; the measured answers are the banked on-policy rollouts (Qwen-2.5-7B-Instruct, single stochastic sample per prompt, engine seed 42, at most 1,024 new tokens) | capture-run record (provenance in footer) |
| Floor grid (follow-up round) | fresh 1,200 / 600 / 300 / 240 fit rows (1% / 0.5% / 0.25% / 0.2%); bridge 240 / 120 / 60 (1% / 0.5% / 0.25%) | follow-up plan v7 §11 (1% inherited; halvings bracket the mean firing rate k/width ≈ 0.15%; 240 = the token-level round's absolute floor) |
| Follow-up round statistics | no training and no fits; every estimator parameter, seed, and draw count identical to the Statistics-constants row above, applied per floor | `floor_sweep_c.json` / `floor_sweep_b.json` provenance fields |

**Evaluation:** No text is generated and no judge is called anywhere in this pipeline — every dependent variable is a per-feature or per-row statistic over banked activations, so the usual on-policy and judged-instrument checks do not apply; the validity checks here are the four gates below. DV1 (primary): per-feature held-out R² of the SAE-encoded map prediction against the SAE-encoded true answer summary — R² over held-out rows per feature, median over alive features per tier; the construct is preservation of tier-t answer content by the pre-generation map. The fresh SAE is trained on exactly the map's target distribution (on-distribution); encoding the prediction is near-distribution (encoder reconstruction FVE 0.937 on predictions); the bridge encoder is declared off-domain and gated. DV2 (companion): retrieval — whether the predicted tier-t feature vector picks the right conversation out of the held-out pool (accuracy at k, chance = k / pool size). Gates, all passing: a reconciliation gate (the refit map reproduces the committed whole-map values; deltas in the Training table); a recapture identity gate on the bridge rows (30,000 row-matched cosines of the recaptured context state against the banked store: min 0.9954, 0.1st percentile 0.99956, median 0.999933); an encoder fitness gate for the off-domain bridge dictionary (variance-FVE 0.659 at L0 26.5 on the 30,000 turn averages, against a 0.35 demotion floor — degraded from its native token-level regime exactly as the recipe paper's measurement predicts, so the bridge read is kept); and an SAE training-sanity gate (held-out variance-FVE 0.909 at L0 99.0 vs the plan's expectation of at least ~0.85). Bootstrap intervals are conditional on the realized score rows. Four caveats carried from the run record: (1) the recapture identity gate passed only at version 2 — the version-1 flat bar (every row ≥ 0.999) was refused with 4 of 30,000 rows below it (min 0.9954); a worst-16-row re-probe at fp32 improved 13 of 16 rows against the bank (worst fp32 row 0.99595), consistent with single-position deep-layer numerical jitter rather than a mapping bug (known mapping bugs read 0.39–0.84), and the version-2 bars (min ≥ 0.995, 0.1st percentile ≥ 0.999, median ≥ 0.9995) were set from the same production distribution they pass — the bars were chosen after seeing the production distribution rather than in advance; margin on the flat bar is 4.0e-4. (2) Two-SHA phase provenance: the recapture phase ran at code `a51057beaf` (before the crash-fix relaunch that recalibrated the gate), all later phases at `064462df4d68`. (3) The 1%-of-rows alive floor is inherited for comparability, and the attrition finding depends on where it sits: mean per-feature firing under k = 100 at width 65,536 is ~0.15% of rows, ~6.5 times below the floor. (4) Six smoke-run markers on this task (from a 24-row composed smoke) are not production results and must never be cited. Separately from the run record, four code-review residuals stay open as concerns, named here and in the Takeaways scope bullet: `consumer-contract-g4-residual` (standalone re-runs of the map/eval phases would not re-check a recorded training-gate failure; this production run was a single all-phase invocation and the training gate passed), `g2b-reuse-verdict-schema` (the stale-gate recompute keys on record schema rather than verdict), `retrieval-device-pin-residual` (retrieval ran on the GPU fp32 backend, whose near-tie rank parity against the fp64 reference helper is unproved — immaterial at the orders-of-magnitude contrasts reported here), and `g2a-min-cross-machine-headroom` (the 4.0e-4 flat-bar margin of caveat 1, with repeat-machine tail spread unmeasured). The floor-sensitivity round adds four validity gates of its own, all passing: row alignment (banked prediction row ids match the assembled holdout order and the bridge score rows); split reproduction (all six split sha256 digests match the banked records, delta exactly 0); counts reconciliation (recomputed full-width firing counts match the banked census with max per-feature delta 0 over all 65,536 features and zero alive-set flips at any floor); and instrument reproduction at the plan-fixed floor. That last gate carries its own correction history: its first form required every per-feature R² within 2e-3 of the committed value and halted the production run on exactly one of 879 features (feature 56641, delta 2.83e-3 — a cross-pod floating-point near-tie 1.4 times over a bar measured on a different recompute class) while the distribution reproduced at percentile-50 2.5e-6 and percentile-99 2.15e-4; the amended predicate (at least 99% of features within 2e-3, max 1e-2, per-tier median agreement binding only on tiers with at least 10 features, verbatim per-feature deltas reported below that) was recorded as a plan amendment before relaunch and passed at 99.886% share with that single near-tie (fresh arm) and zero violators at max delta 3.6e-5 (bridge arm). Like the parent's recapture-gate caveat, this re-calibration was chosen after seeing the production distribution; it still hard-fails any systematic shift or order-of-magnitude excursion. Three review residuals from the round stay open as concerns: `unpinned-q1-source-revision` (the chunk re-assembly delegates some fetches without per-call revision pins, so content equality with the parent instrument is certified downstream — by the split-digest, counts, and reproduction gates — rather than at fetch time), `a13-demotion-scope-drift` (the census-only demotion was implemented on any populated tier, broader than the plan's finest-tier trigger; it never fired — zero undefined reads — so no lattice label was withheld), and `result-commit-invalidates-resume` (the result-publishing commit advances the code sha that keys phase resume, so a hypothetical post-publish retry would recompute the analysis phases; no retry occurred). Conciseness acknowledgment: the total skim-prose budget, the 120-word per-result band, and Takeaways bullet lengths are exceeded — the nine results carry verdict tables and companion reads, and the scope bullet carries the four named review-residual ids; every hard cap is met.

**Data extraction:** Tier-1 real-world data: first-turn LMSYS / WildChat user conversations with the banked on-policy model answers (no new generation; the teacher-forced re-read consumes persisted text and stores only reduced means). The fresh arm streams the 1,920 banked capture chunks and assembles layer-19 context-state and answer-mean matrices for 963,444 conversations (fp16, ~6.9 GB each), re-asserting the pinned splits by sha (val 400 / test 1,000 / holdout 20,000 / SAE-fit 120,000). The bridge arm re-captures the prior round's exact 30,000 rows with the same tokenization convention, accumulating answer-span means at layers 19 and 20 in a streaming reduction (per-token activations never materialized beyond the batch). Encodes: fresh SAE on layer-19 objects (20,000 held-out rows); bridge dictionary on layer-20 turn averages (6,000 score rows). Artifact-name convention: file suffix `_c` marks the fresh-SAE arm, `_b` the bridge arm. The floor-sensitivity round re-assembles the answer-mean matrix from the same 1,920 banked chunks (splits re-asserted by sha), recomputes full-width firing counts over the fit rows, and encodes the score rows once at the union of all floors' alive columns — 2,150 (fresh), 800 (bridge), 697 (pile) — from which every per-floor read is a subset; no new text, no generation.

**Sample training/evaluation data + completions:** this task produces no completions and calls no judge; the end-to-end unit is context state → map prediction → SAE encode → per-feature held-out R². Below, first the underlying conversation rows, then the per-feature evaluation rows. 3 of 19,967 labeled holdout rows (the fresh arm's score rows), random sample, seed 2476 — sanitized for context hygiene: real-user-corpus rows, so each cell carries only a mechanically truncated 12-word excerpt standing in as `[truncated — sanitized row; verify at the pinned judge-labels tree, rows ci471286 / ci264094 / ci558155]`; full text lives in the pinned [judge-labels tree](https://github.com/superkaiba/explore-persona-space/tree/43818beb136bac133e33414a6520760b4206afc6/eval_results/issue_1482/judge_labels), keyed by row id.

| Row id | Corpus | Prompt excerpt (truncated) | Model answer excerpt (truncated) |
|---|---|---|---|
| ci471286 | lmsys | "Please start a dialogue, with the aim of naturally leading me to …" | "Sure! Let's start with a light topic. How's your day going so …" |
| ci264094 | lmsys | "Do humans have anal glands?" | "Humans do not have anal glands. Anal glands are a characteristic found …" |
| ci558155 | wildchat | "授权委托书 关键词：甲方是我们，乙方是租客 …" (a Chinese rent-collection authorization-letter request) | "授权委托书 甲方：……" (the drafted template letter, in Chinese) |

Per-feature evaluation rows: 5 of 879 (fresh arm) and 5 of 412 (bridge arm) alive-feature rows, random sample, seed 42, plus — from the floor-sensitivity round — 3 of 127 newly-alive finest-tier features (fresh arm, 0.2% floor), random sample, seed 42; full arrays in the committed [perfeature_c_encodepred.npz](https://github.com/superkaiba/explore-persona-space/blob/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/turnavg/perfeature_c_encodepred.npz), [perfeature_b_encodepred.npz](https://github.com/superkaiba/explore-persona-space/blob/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/turnavg/perfeature_b_encodepred.npz), and [perfeature_union_c.npz](https://github.com/superkaiba/explore-persona-space/blob/23eb255f10cac3ce302ecac9289be4f9808cf6a6/eval_results/issue_2476/floor_sweep/perfeature_union_c.npz).

| Arm | Feature id | Tier | Held-out R² | Active fit rows |
|---|---|---|---|---|
| fresh | 1771 | coarsest | 0.650 | 100,372 of 120,000 |
| fresh | 1033 | coarsest | 0.342 | 13,587 |
| fresh | 1537 | coarsest | 0.651 | 2,329 |
| fresh | 195 | coarsest | 0.635 | 49,858 |
| fresh | 1014 | coarsest | 0.481 | 3,309 |
| bridge | 11376 | middle | 0.651 | 519 of 24,000 |
| bridge | 171 | coarsest | 0.864 | 24,000 |
| bridge | 1123 | coarsest | 0.099 | 310 |
| bridge | 1641 | coarsest | 0.205 | 767 |
| bridge | 1573 | coarsest | 0.117 | 432 |
| fresh, newly alive at the 0.2% floor | 54134 | finest | 0.382 | 752 |
| fresh, newly alive at the 0.2% floor | 22864 | finest | 0.200 | 394 |
| fresh, newly alive at the 0.2% floor | 50045 | finest | 0.091 | 386 |

Worked example: for fresh-arm feature 1537 (coarsest tier), the map's predicted answer summary is encoded for each of the 20,000 held-out conversations, and the predicted feature value tracks the true one with held-out R² 0.651 — despite the feature firing on only 2,329 of 120,000 fit rows.

## Results

### The verdict is indeterminate on the headline arm: the permutation criterion fires coarse-better while the endpoint contrast lands negative

Plotted: per-tier median held-out per-feature R² of encoded map predictions against encoded true summaries; four bars per tier — fresh turn-averaged SAE (20,000 rows, 879 features) and token-level dictionary on turn averages (6,000 rows, 412 features), each vs identity+bias; black dashes: token-level reference medians (different grain, layer, dictionary, and rows — not a matched contrast).

![Per-tier median held-out R2 for both turn-averaged arms and baselines, with token-level reference dashes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/figures/paper/c3_turnavg_tier_gradient.png)

> **Figure.** *Neither turn-averaged basis reproduces the token-level round's monotone coarse-over-specific decline.* Median held-out per-feature R² per matryoshka tier: fresh turn-averaged SAE (blue) vs its identity+bias baseline (orange), token-level dictionary on turn averages (green) vs its baseline (pink); black dashes = token-level reference medians. Whiskers: 10,000-draw bootstrap intervals.

| Arm | Tier medians (n alive) | Coarse-minus-finest contrast (95% interval) | Permutation percentile (observed pooled rank correlation vs null band) | Rule outcome |
|---|---|---|---|---|
| Fresh turn-averaged SAE, layer 19 | 0.625 / 0.131 / 0.670 (788 / 88 / 3) | −0.044 (−0.264 to −0.013) | 0.0, coarse-better side (−0.449 vs −0.190 to −0.069; achievable magnitude ceiling 0.528) | indeterminate — the two conditions disagree |
| Token-level dictionary on turn averages, layer 20 | 0.420 / 0.553 / 0.601 (387 / 21 / 4) | −0.181 (−0.325 to +0.105; crosses zero) | 99.42, fine-better side (+0.113 vs −0.103 to +0.086) | reversed — a secondary read; the rule is defined on the fresh arm |
| Token-grain reference (its own round: token level, layer 20, 6,000 rows) | 0.435 / 0.174 / 0.043 (16,384-feature panel) | +0.392 | coarse-better (−0.395 vs −0.250 to −0.228) | gradient holds |

The two rule conditions read different populations: the permutation pools all 879 features and is dominated by coarse-vs-middle (876 of 879), while the endpoint contrast is decided by the 3 finest-tier survivors — indeterminate is the honest cell at this floor (shown floor-local below). The bridge-arm reversal is marginal: its endpoint interval crosses zero. An inlier-masked pooling twin matches the plain mean to four decimals, so outlier tokens do not drive the inversion.

The map beats identity+bias on per-feature R² at every tier; fresh-arm deltas exceed 1 because the baseline's held-out R² goes deeply negative.

| Tier | Paired R² delta, fresh (95% interval) | Paired R² delta, bridge |
|---|---|---|
| coarsest | +1.39 (+1.28 to +1.49) | +0.45 (+0.42 to +0.49) |
| middle | +5.26 (+3.90 to +6.91) | +0.56 (+0.46 to +0.76) |
| finest | +1.15 (+1.15 to +6.49) | +0.50 (+0.13 to +0.74) |

Per-unit exemption: the per-feature scatter behind these medians is embedded two results below.

### Alive-feature attrition, not a gradient, is the dominant turn-averaged phenomenon

Plotted: the fraction of each tier's candidate features clearing the alive floor (active on at least 1% of fit rows), log y-axis, one bar per tier and arm, then the per-candidate firing-fraction distribution behind the census.

![Alive-feature fraction per tier for both turn-averaged arms, log scale](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8d0d7d545ce79be59524f8d27b8146546d1d984c/figures/issue_2476/i2476_alive_attrition_legend.png)

> **Figure.** *Above 99.9% of finest-tier candidates fall below the alive floor in both turn-averaged instruments.* Fraction of candidate features per matryoshka tier clearing the 1%-of-fit-rows floor (1,200 of 120,000 rows, fresh arm; 240 of 24,000, bridge arm), log scale.

![Per-candidate firing-fraction distribution per tier, both instruments, alive floor marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/figures/issue_2476/i2476_candidate_firing_ecdf.png)

> **Figure.** *Per-candidate companion to the census bars.* Cumulative share of each tier's candidates at or below a given firing fraction (log x); each curve enters at its never-fires share (fresh instrument 19%/61%/77% across coarsest/middle/finest, bridge 31%/79%/96%); the dashed line is the 1% alive floor.

| Tier (candidates) | Fresh SAE | Bridge dictionary | Pile twin |
|---|---|---|---|
| coarsest (2,048) | 788 (38%) | 387 (19%) | 261 (13%) |
| middle (14,336) | 88 (0.6%) | 21 (0.15%) | 39 (0.27%) |
| finest (49,152) | 3 (0.006%) | 4 (0.008%) | 14 (0.03%) |

The token-level panel carried 16,384 alive features; the same dictionary on turn averages retains 412. The counts say almost no specific-tier feature clears the inherited floor under these k = 100, width-65,536 instruments.

Mean firing (k over width, ~0.15% of rows) sits ~6.5 times below the floor; the floor sweep below apportions the contributions. Under the plan-fixed floor turn-averaging removes specific features from the alive panel; the floor sweep shows they return poorly predicted at looser floors — the map's target object carries little well-predicted specific-tier structure, which rescopes the paper claim rather than contradicting it. The per-candidate curves sharpen this: most specific-tier candidates never fire at all, and the floor cuts each surviving tail far above its bulk.

### Surviving finest-tier features match coarse features of similar firing frequency — a survivor-conditioned read; the middle-tier dip is instrument-dependent

Plotted: every alive feature's held-out R² against log10 of its active fit rows, tier-colored, one panel per arm — the per-unit view behind every median above.

![Per-feature held-out R2 versus activity, tier-colored, one panel per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/issue_2476/i2476_r2_vs_activity.png)

> **Figure.** *Fresh-arm middle-tier features cluster at low activity and low R²; the bridge arm's middle tier does not.* Each point is one alive feature: held-out R² vs log10 active fit rows, colored by tier (fresh arm n = 879, bridge arm n = 412).

| Read | Values | Note |
|---|---|---|
| Matched-activity coarse median, fresh (1,400–2,400 active rows) | 0.525 (0.491 to 0.565), n = 197 | finest-tier survivors read 0.880 / 0.670 / 0.648 |
| Matched-activity middle median, fresh (same band) | 0.132 (0.104 to 0.180), n = 41 | matched coarse-minus-middle +0.393 (+0.333 to +0.442) — the dip holds at matched activity with the interval clear of zero |
| Matched coarse-minus-finest contrast, fresh | −0.145 (−0.381 to −0.097) | excludes zero on the negative side, but the finest side is the 3 survivors — survivor-conditioned, not population evidence |
| Matched-activity coarse median, bridge (390–552 active rows) | 0.393 (0.343 to 0.452), n = 83 | finest survivors 0.745 / 0.626 / 0.576 / 0.318; fresh band applied instead: 0.509 (0.303 to 0.577), n = 22 |
| Matched contrasts, bridge (same band) | coarse-minus-middle −0.126 (−0.410 to +0.394), middle-tier n = 5; coarse-minus-finest −0.208 (−0.354 to +0.076), finest n = 4 | both cross zero; coarse-minus-finest agrees with the unmatched endpoint contrast in the first results table |
| Matched-activity reads, pile twin (250–1,922 active rows — its own middle-tier survivor range, derived by the bridge-arm rule) | coarse 0.416 (0.381 to 0.444), n = 192; middle 0.557 (0.429 to 0.634), n = 35; coarse-minus-middle −0.141 (−0.200 to −0.035) | middle above coarse with the interval clear of zero — opposite sign to the fresh arm's +0.393 |
| Identity+bias coarse-minus-finest contrast | fresh −0.287 (−0.416 to +5.146, crosses zero); bridge −0.126 (−0.337 to −0.008) | a map-free predictor reproduces the endpoint inversion at the point estimates |
| Dense-input companion medians | 0.533 / 0.159 / 0.719 (fresh); 0.391 / 0.544 / 0.598 (bridge) | survivors are also more linearly predictable from context at all |
| Corpus-transfer fold (fit LMSYS, score 9,017 WildChat rows) | 0.522 / 0.075 / no finite finest values | 18 of 86 finite middle-tier transfer values negative; coarse tail reaches −6.83 |
| Pile-twin medians | 0.457 / 0.557 / 0.413 (n = 261 / 39 / 14) | non-monotone, but a small positive coarse-minus-finest endpoint (+0.044) |

Survivors read as at least as predictable as coarse features of comparable firing frequency in both instruments — a survivor-only description at n = 3–4, not a population inference; survival is itself activity selection no within-panel control can undo, and the floor sweep below identifies these survivors as the elite tail of a poorly predicted population. The middle-tier dip survives corpus transfer and activity matching in the fresh instrument, and the matched contrasts separate: fresh coarse-better, pile twin middle-better, both intervals clear of zero, while the bridge arm leans the pile twin's way at matched activity on 5 middle-tier features without separating. Identity+bias and the dense-input companion reproduce the survivor-profile shape without the map, so map-specific content loss is not necessary for the endpoint inversion — a survivor-panel effect and shared dictionary geometry are indistinguishable here.

### Retrieval concentrates in the coarsest tier: acc@1 0.708 coarsest vs 0.001 middle

Plotted: per-tier retrieval accuracy at k = 1 (euclidean; pool = held-out rows, 20,000 fresh, 6,000 bridge; chance = 1 over pool size), map vs identity+bias per arm; inset: middle and finest tiers. Below it, the per-row rank distributions behind these cells.

![Per-tier retrieval accuracy at k equals 1, map versus identity plus bias, both arms, zoom inset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/paper/c3_turnavg_tier_acc1.png)

> **Figure.** *Coarse-tier features identify the held-out conversation at 0.708; middle- and finest-tier rates sit orders of magnitude lower.* Retrieval accuracy at k = 1 per tier, map vs identity+bias, fresh arm (chance 5.0e-5) and bridge arm (chance 1.67e-4); inset: middle and finest tiers. Black whiskers: 95% Wilson intervals.

![Per-row retrieval-rank distribution per tier, map solid, identity plus bias dashed, one panel per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/figures/issue_2476/i2476_retrieval_rank_ecdf.png)

> **Figure.** *Per-row companion: the coarse tier's retrieval advantage holds across the whole rank distribution, not only at rank 1.* Share of held-out rows whose true conversation ranks at or below x (euclidean, log x), per tier; solid = map, dashed = identity+bias.

| Diagnostic | Fresh arm | Bridge arm |
|---|---|---|
| Encoder reconstruction FVE of the input being scored (map prediction vs identity+bias guess) | 0.937 at L0 88 vs 0.305 at L0 955 | gate values in Methodology |
| Alive features beating the 20-draw row-shuffle floor (map vs identity+bias) | 100% vs 29.6% | 100% vs 47.8% |
| Dense-space anchor (before encoding) | acc@1 0.705; pooled R² 0.724 | — |
| Retrieval cells favoring the baseline (of 12 total) | middle tier, both metrics — systematic at k = 1, 5, 10 | finest-tier cosine cell |

Encoding an identity+bias guess lands off the feature manifold (fresh-arm middle-tier median −5.13; diagnostics table). Middle-tier map cells sit at 21x and 39x chance yet roughly 670x and 50x below their coarse-tier rates, and 3 of 12 map-vs-baseline cells favor the baseline.

Tier-restricted vectors have 788/88/3 dimensions, so low-tier cells are dimension-confounded — coarse-concentrated is the supportable read, not a content-loss quantification. The two dependent variables disagree in profile; the headline is scoped to the R² profile. The per-row ranks were recomputed after the run, which persisted only the aggregate cells; every accuracy-at-1 cell reproduces exactly, all cells within two rows in 20,000 (deltas: footer).

### Per-feature predictability transfers across grain (rank correlation 0.767) while the tier profile inverts

Plotted: each intersection feature's held-out R² at turn-averaged grain (bridge arm) against the same feature's committed token-level R² — same dictionary, same rows; 409 features, tier-colored, identity diagonal; every point is one feature, so this is its own per-unit view.

![Per-feature R2 at turn-averaged versus token grain, same dictionary and rows, with identity diagonal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/figures/issue_2476/i2476_bridge_token_vs_turnavg.png)

> **Figure.** *Feature ranking is basis-stable; the level shifts down and the tier profile inverts.* Held-out per-feature R², turn-averaged vs token grain, on the 409-feature intersection (same dictionary, same 6,000 score rows); points below the diagonal lose predictability at the turn-averaged grain.

Pooled rank correlation 0.767 (per tier 0.754/0.893/0.5 at n = 387/19/3) clears the plan's 0.5 expectation — the ranking clause of the bridge hypothesis passes. The level shifts down for most features (median delta −0.152; 337 of 409 points below the diagonal, 72 above — dominant but heterogeneous), and the gradient-preservation clause is refuted: the per-tier profile inverts (0.420/0.553/0.601 vs the token-level 0.435/0.174/0.043). The token-level gradient ran through thousands of low-activity specific features that do not exist as alive features at this grain — 15,975 of its 16,384 panel features fall out of the intersection.

### The indeterminate cell is floor-local: every swept looser floor reads coarse-over-specific on the fresh arm, while the bridge instrument stays reversed

Plotted: per-tier median held-out R² against the alive floor (fresh arm, four floors from 1% to 0.2%, map vs identity+bias); the table adds each floor's within-activity permutation read for both instruments — observed pooled rank correlation of tier index against per-feature R² vs its 10,000-draw null band (more negative means coarse-better).

![Per-tier median held-out R2 versus alive floor, fresh arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476/i2476_floor_sweep_hero_r2.png)

> **Figure.** *The finest-tier median collapses once the floor admits a real population.* Median held-out per-feature R² per matryoshka tier vs alive floor, fresh turn-averaged SAE: map (solid) vs identity+bias (dashed); shaded bands are 10,000-draw bootstrap intervals. Per-tier alive counts per floor are in the table below.

| Arm, floor | Alive by tier | Tier medians (map) | Coarse-minus-finest (95% interval) | Permutation: observed vs null band | Rule reading |
|---|---|---|---|---|---|
| Fresh, 1% (plan-fixed) | 788 / 88 / 3 | 0.625 / 0.131 / 0.667 | −0.042 (−0.263 to −0.013) | −0.449 vs (−0.190, −0.069), coarse-better | indeterminate |
| Fresh, 0.5% | 992 / 314 / 26 | 0.607 / 0.172 / 0.087 | +0.520 (+0.373 to +0.594) | −0.641 vs (−0.300, −0.217), coarse-better | gradient holds (reported) |
| Fresh, 0.25% | 1,137 / 696 / 80 | 0.592 / 0.166 / 0.049 | +0.543 (+0.485 to +0.579) | −0.691 vs (−0.349, −0.287), coarse-better | gradient holds (reported) |
| Fresh, 0.2% | 1,178 / 842 / 130 | 0.588 / 0.163 / 0.046 | +0.542 (+0.517 to +0.568) | −0.699 vs (−0.377, −0.320), coarse-better | gradient holds (reported) |
| Bridge, 1% (its plan-fixed cell) | 387 / 21 / 4 | 0.420 / 0.553 / 0.601 | −0.181 (−0.325 to +0.105) | +0.113 vs (−0.103, +0.086), fine-better | reversed |
| Bridge, 0.5% | 529 / 56 / 7 | 0.377 / 0.471 / 0.626 | −0.249 (−0.399 to +0.069) | +0.118 vs (−0.150, +0.001), fine-better | reversed |
| Bridge, 0.25% | 657 / 130 / 13 | 0.307 / 0.319 / 0.576 | −0.269 (−0.438 to +0.010) | +0.020 vs (−0.221, −0.104), fine-better | reversed |

The full reported vector, ordered 1% → 0.2%: indeterminate at the plan-fixed cell, then gradient holds at every looser floor (fresh); the bridge reads reversed throughout. The reported lattice is floor-sensitive — the plan-fixed cell stays the verdict of record; looser-floor labels are descriptive.

The looser floors share seeds and nested alive sets with the 1% cell — one dependent pattern, not independent replications — and cross-floor permutation values are descriptive too (activity quintiles are recomputed per floor); within each floor the permutation controls activity composition. The 1%-row values are the sweep's recomputations, matching the committed values within the reproduction gate's bars (Methodology).

### Newly admitted finest-tier features collapse toward zero: the three 1%-floor survivors are an unrepresentative elite

Plotted: the cumulative distribution of per-feature held-out R² per tier, one panel per floor (fresh arm) — the per-unit view behind the medians above.

![Cumulative distribution of per-feature held-out R2 per tier, one panel per floor, fresh arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476/i2476_floor_r2_ecdf.png)

> **Figure.** *At every floor below 1% the finest tier's distribution sits far to the left of the coarse tier's.* Cumulative share of alive features at or below a given held-out R², per tier and floor, fresh arm; per-tier n per floor in the previous result's table.

| Read (fresh arm; loosest floor unless stated) | Value |
|---|---|
| Pooled finest median by floor, 1% → 0.2% | 0.667 → 0.087 → 0.049 → 0.046 (n = 3 / 26 / 80 / 130) |
| Newly-alive-only finest median at 0.2% | 0.044 (n = 127); 34% at or below zero; 69% below 0.1 |
| Finest median 95% interval vs same-floor middle-tier interquartile range | 0.020–0.067 vs 0.062–0.306 (middle median 0.163, n = 842) |
| Newly admitted vs previously-alive medians at 0.2% | coarse 0.475 vs 0.625; middle 0.172 vs 0.131 |
| Alive features beating the 20-draw row-shuffle floor (map read) | 100% at 1% → 97.0% at 0.2% |
| Corpus halves, finest median at 0.2% | 0.040 LMSYS / 0.030 WildChat |
| Undefined per-feature R² | 0 at every floor, tier, and arm |

The round's hypothesis convention (fixed in advance) read the finest-tier median at the 240-row floor against the same floor's middle-tier interquartile range. On absolute levels the hypothesis fails: the finest median (0.044 among newly admitted features) sits at roughly a quarter of the middle median and below the middle interquartile range, with a third of newly admitted features at or below zero — descriptively outside the range, though the bootstrap interval's upper end grazes its lower edge.

Every cross-floor median move is composition: previously-alive features keep their values exactly (same rows, same encodes), and the newly admitted cohorts read lower (coarse) or comparable (middle). The 3% shuffle-floor failures at the loosest floor sit among the newly admitted low-activity features, consistent with the collapse read.

### Attrition stays above 99.7% of finest-tier candidates at every swept floor: the floor convention explains the verdict, not the census

Plotted: alive-feature counts per tier against the floor for all three dictionaries (log y), then the per-candidate firing-fraction distributions with the swept floors and the instrument's mean firing rate marked.

![Alive feature counts per tier versus alive floor, three dictionaries, log scale](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476/i2476_floor_alive_counts.png)

> **Figure.** *The finest tier gains features 43-fold across the sweep yet stays under 0.3% of its candidates.* Alive features per matryoshka tier vs alive floor: fresh turn-averaged SAE, the token-level dictionary on turn averages, and the pile-trained twin (exploratory).

![Per-candidate firing-fraction distribution per tier with swept floors and mean firing rate marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476/i2476_floor_firing_ecdf.png)

> **Figure.** *Per-candidate companion to the census counts: most specific-tier candidates never fire, and every swept floor cuts each tail far above its bulk.* Cumulative share of candidate features at or below a firing fraction (log x), per tier and dictionary; dash-dot line marks the mean firing rate (k over width), dashed lines the swept floors.

| Floor (% of fit rows) | Fresh finest alive (of 49,152) | Bridge finest | Pile twin finest |
|---|---|---|---|
| 1% | 3 | 4 | 14 |
| 0.5% | 26 | 7 | 21 |
| 0.25% | 80 | 13 | 42 |
| 0.2% (fresh grid only) | 130 | — | — |

Within the swept 0.2%–1% bracket the apportionment is two-way: the floor convention carries the verdict's indeterminacy (previous results) and a 43-fold finest census change, while the attrition bulk — over 99.7% of finest candidates below even the loosest floor — stays with the joint matryoshka-allocation-plus-undertraining residual, which this round does not separate. Floors below the bracket, including any below the mean firing rate (~0.15% of rows), were not swept and stay unresolved. The pile twin is exploratory: its census rests on recomputed counts and its tier medians move non-monotonically with the floor.

### Retrieval stays coarse-concentrated at every swept floor

Plotted: per-tier retrieval accuracy at k = 1 (euclidean) against the floor, map vs identity+bias, fresh arm, log y with the chance line.

![Per-tier retrieval accuracy at k equals 1 versus alive floor, map versus identity plus bias, fresh arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476/i2476_floor_sweep_hero_acc1.png)

> **Figure.** *Coarse-tier retrieval is flat across the sweep; middle- and finest-tier rates rise with dimension but stay orders of magnitude lower.* Retrieval accuracy at k = 1 per tier vs alive floor, fresh arm, map vs identity+bias (pool 20,000; chance 5.0e-5, dotted).

| Tier, predictor | 1% | 0.5% | 0.25% | 0.2% |
|---|---|---|---|---|
| Coarsest, map | 0.708 | 0.706 | 0.706 | 0.706 |
| Middle, map | 0.0011 | 0.0041 | 0.0081 | 0.0088 |
| Middle, identity+bias | 0.0035 | 0.0134 | 0.0224 | 0.0252 |
| Finest, map | 0.0001 | 0.0003 | 0.0004 | 0.0007 |

The coarse-tier map rate is floor-invariant while middle- and finest-tier map rates grow roughly with their dimension counts and remain 80-fold or more below it; the baseline's middle-tier advantage strengthens at looser floors, so the dimension confound noted at the plan-fixed floor carries through the sweep. Bridge cells behave the same way (coarse 0.331–0.335 across its floors). Unlike the median-R² profile, the retrieval profile is floor-robust.

---

**Repro:** ~7.3 h wall on 1× H100-80 (RunPod pod-2476), zero LLM-API calls. Workload: [scripts/issue2476_turnavg_sae.py @ 064462df4d](https://github.com/superkaiba/explore-persona-space/blob/064462df4d683afcb6a717253275cdb8486c9a82/scripts/issue2476_turnavg_sae.py) (`--phase all`); two-SHA phase provenance — the bridge-row recapture ran at [a51057beaf](https://github.com/superkaiba/explore-persona-space/blob/a51057beaf6a5073dcf273ef80077c759fb98499/scripts/issue2476_turnavg_sae.py) (its gate record carries that commit), all later phases (SAE training / maps / eval / figures) at `064462df4d68`. Eval artifacts (git, branch `issue-2476`): [eval_results/issue_2476/turnavg tree](https://github.com/superkaiba/explore-persona-space/tree/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/eval_results/issue_2476/turnavg) — 28 files: `tier_tests_c.json` / `tier_tests_b.json` (medians, permutation, bootstrap, verdict inputs), `retrieval_c.json` / `retrieval_b.json` plus the per-row rank arrays behind every retrieval cell (`perrow_retrieval_ranks.npz`, with per-cell reproduction deltas and provenance in `perrow_retrieval_ranks_meta.json` — recomputed post-run by [scripts/issue2476_perrow_views.py @ a3338b6ec57855d03531d5c81b1bd6d59f61c32a](https://github.com/superkaiba/explore-persona-space/blob/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/scripts/issue2476_perrow_views.py) from the uploaded SAE weights and prediction stores; the fresh arm's dense map predictions come from the parent round's committed copy of the same ridge refit, [issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz @ 89cfa76cdc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/89cfa76cdcd4207d95c1fec1c3131f36e21beec0/issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz), this run's own refit store having not been uploaded — the recompute fetched that file at the then-current unpinned default branch, and the pinned revision serves identical content (its last touching commit is dated 2026-07-18, before the 2026-08-23 recompute); row set asserted identical and the committed per-feature R² arrays reproduced, verification deltas in `perrow_retrieval_ranks_meta.json`), `panel_c.json` / `panel_b.json`, per-feature npz for the map / identity+bias / dense-input / transfer / pile / inlier reads, `bridge_b.npz`, `train_log.json`, `recon_gate.json`, `preds_meta.json`, `regime.json` (split shas), and the recapture gate record `gates_p2.json`; recapture-gate recalibration evidence at [eval_results/issue_2476/g2a_recalibration tree](https://github.com/superkaiba/explore-persona-space/tree/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/g2a_recalibration). A follow-up analysis round (no new compute) added bootstrap intervals for every matched-activity read: `matched_tier_ci.json` in the [turnavg tree @ 179a7b69aa](https://github.com/superkaiba/explore-persona-space/tree/179a7b69aa9fb14a534d09e247e2963289dafac7/eval_results/issue_2476/turnavg), produced by [scripts/issue2476_matched_tier_ci.py @ 179a7b69aa](https://github.com/superkaiba/explore-persona-space/blob/179a7b69aa9fb14a534d09e247e2963289dafac7/scripts/issue2476_matched_tier_ci.py) (10,000 draws, seed 42; point medians re-derived from the committed per-feature npz files and checked against every committed matched read before any interval was computed; branch `issue-2476`). Figures: hero pair `figures/paper/c3_turnavg_tier_gradient` + `c3_turnavg_tier_acc1` (png, pdf, data sidecars) and exploratory panels under [figures/issue_2476 @ a3338b6ec57855d03531d5c81b1bd6d59f61c32a](https://github.com/superkaiba/explore-persona-space/tree/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/figures/issue_2476); the gradient hero and the bridge scatter were re-rendered for label fixes (internal arm tokens, then a bare issue token) by [scripts/issue2476_fig_relabel.py @ a3338b6ec57855d03531d5c81b1bd6d59f61c32a](https://github.com/superkaiba/explore-persona-space/blob/a3338b6ec57855d03531d5c81b1bd6d59f61c32a/scripts/issue2476_fig_relabel.py), sidecar numeric data asserted identical to the first renders each time; the two per-unit companions (`i2476_candidate_firing_ecdf`, `i2476_retrieval_rank_ecdf`) are new in this revision round (same script as the rank arrays); other panels unchanged at [8dac0f576f](https://github.com/superkaiba/explore-persona-space/tree/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/issue_2476) and [8d0d7d545c](https://github.com/superkaiba/explore-persona-space/tree/8d0d7d545ce79be59524f8d27b8146546d1d984c/figures/issue_2476). HF (listed live at write time, 99 files): [issue2476_turnavg/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e5de92ea13a5c51d0d03d03446b0cecb4a22f4d5/issue2476_turnavg/analysis_tensors) — `recapture_store/` (68 files: the 30,000-row mean store + the gate record source), `eval/` (21), `split_meta/` (7), `sae_c/` (3: SAE weights 1.88 GB safetensors, cfg, train log). Artifact-suffix legend: `_c` = fresh turn-averaged SAE arm, `_b` = bridge arm, `_ib` = identity+bias, `_densein` = dense-input companion. Reuse provenance: capture chunks + rollout text from [#779](https://eps.superkaiba.com/tasks/779) at HF `issue779_monitoring/fitter-fair-comparison-n1m/{final_token_capture,raw_completions}` @ revision `89cfa76cdc…` (the run read both prefixes at the then-current unpinned default branch; the pinned revision serves identical content — the last commits touching them are dated 2026-07-15 for the capture chunks and 2026-08-17 for the rollout text, both before the run's 2026-08-22/23 assemble and recapture reads) — the map's own training capture, fitness by construction; pinned splits, the 30,000-row subsample, the committed token-level per-feature panel, and the layer-20 dense store from [#1482](https://eps.superkaiba.com/tasks/1482) at HF `issue1482_error_analysis/analysis_tensors/matryoshka_tier/store` revision `8b82eb9753…` — same-lineage products, shas re-asserted in-run; token-level dictionaries [chanind/qwen2.5-7B-it-layer-20-saes](https://huggingface.co/chanind/qwen2.5-7B-it-layer-20-saes/tree/db8c88b400e36e603d0563abcf8d289e5eff5687) — that round's own pin, fitness gate passed at FVE 0.659; sample-row text from [#1482](https://eps.superkaiba.com/tasks/1482)'s committed judge-labels file (pinned above) — the same holdout rows this run scores. Plan: `plans/plan.md` (v4) in the task directory; production results marker posted 2026-08-23T11:44Z. Discarded per plan: per-token recapture activations, assembled fp16 matrices, and SAE optimizer state (regeneration recipes in the plan's reproducibility card); no rollout text, judge output, metric, or config discarded. Floor-sensitivity round (same-issue follow-up round `floor-sensitivity-sweep`, cheap band): ~3.1 GPU-h on 1× H100-80 (RunPod pod-2476, fresh pod), zero LLM-API calls; workload [scripts/issue2476_floor_sweep.py @ a1e18f25d1](https://github.com/superkaiba/explore-persona-space/blob/a1e18f25d199142f823f09c8d5e29738dfb5b588/scripts/issue2476_floor_sweep.py) (parent kernels imported by file path; one production halt at the first instrument-reproduction predicate, relaunched after the plan-v7 re-calibration narrated in Methodology). Round eval artifacts (git, branch `issue-2476`, result commit `23eb255f10ca…`): [eval_results/issue_2476/floor_sweep tree](https://github.com/superkaiba/explore-persona-space/tree/23eb255f10cac3ce302ecac9289be4f9808cf6a6/eval_results/issue_2476/floor_sweep) — 8 files: `floor_sweep_c.json`, `floor_sweep_b.json`, `floor_sweep_b_pile.json` (per-floor medians, permutation, contrasts, lattice vectors), `floor_retrieval_c.json`, `floor_retrieval_b.json`, `gates_floor_sweep.json` (all six gate records), `perfeature_union_c.npz`, `perfeature_union_b.npz` (union per-feature arrays + per-floor alive masks; the source of every newly-alive decomposition above). Round figures: nine floor-sweep panels under [figures/issue_2476 @ 23eb255f10](https://github.com/superkaiba/explore-persona-space/tree/23eb255f10cac3ce302ecac9289be4f9808cf6a6/figures/issue_2476) (stems `i2476_floor_sweep_hero_r2`, `i2476_floor_sweep_hero_acc1`, `i2476_floor_hero_b_r2`, `i2476_floor_alive_counts`, `i2476_floor_r2_ecdf`, `i2476_floor_perm_summary`, `i2476_floor_firing_ecdf`, `i2476_floor_corpus_split`, `i2476_floor_shuffle_null`; png, pdf, and data sidecar each). Round HF artifacts (listed live at write time, 5 files): [issue2476_turnavg/analysis_tensors/floor_sweep](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ba2e24415e76e0dfcffb2cff60ba7e2847671227/issue2476_turnavg/analysis_tensors/floor_sweep) — `firing_census_c.npz`, `firing_census_b.npz`, `firing_census_b_pile.npz` plus the two union npz twins. Round plan: `plans/plan.md` (v7, self-contained); round upload-verification pass posted 2026-08-24T06:50Z.

**Context:** Origin prompt (verbatim): `Paper outline 2026-08-22: 'SAE experiments — try turn averaged SAEs'`. Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — the token-level tier-gradient round this run re-tests at turn-averaged grain, itself an error-analysis child of the [#779](https://eps.superkaiba.com/tasks/779) map. Created 2026-08-22; production run 2026-08-23 (one crash-fix relaunch after the recapture identity gate refusal narrated in Methodology); interpretation passed both critics 2026-08-23; matched-activity interval round folded 2026-08-23. Follow-up round `floor-sensitivity-sweep` (source: the automatic follow-up proposer, cheap band; scope marker posted 2026-08-24T00:35Z), originating proposal title (verbatim): `### 1. Alive-floor sensitivity sweep of the tier profile — Diagnostic`; run 2026-08-24 (one halt at the instrument-reproduction gate, plan-v7 re-calibration, relaunch), folded 2026-08-24.

