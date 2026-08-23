---
title: 'The context-to-answer map''s coarse-over-specific predictability gradient
  does not reproduce in turn-averaged SAE bases: nearly all specific-tier features
  fall below the alive floor, and retrieval stays coarse-tier-concentrated (MODERATE
  confidence)'
kind: experiment
tags: []
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
# The context-to-answer map's coarse-over-specific predictability gradient does not reproduce in turn-averaged SAE bases: nearly all specific-tier features fall below the alive floor, and retrieval stays coarse-tier-concentrated (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- The coarse-over-specific R² profile reproduces in neither turn-averaged basis: tier medians 0.625/0.131/0.670 (fresh turn-averaged SAE, n = 788/88/3 alive) vs 0.420/0.553/0.601 (token-level dictionary on turn averages).
- Attrition dominates: 3, 4, and 14 of 49,152 finest-tier candidates clear the 1%-of-rows alive floor across the three dictionaries — above 99.9% attrition, as deterministic threshold-crossings, not estimates.
- The verdict rule returns indeterminate: the activity-stratified permutation fires coarse-better (percentile 0.0, two-sided p < 0.05) while the coarse-minus-finest median contrast lands negative, decided by 3 survivors.
- The map beats identity+bias on R² at every tier (paired deltas +0.45 to +5.26), yet retrieval stays coarse-concentrated: acc@1 0.708 coarsest vs 0.001 middle (chance 5.0e-5, n = 20,000).
- Same dictionary, same rows: per-feature predictability rank-correlates 0.767 across grain while the tier profile inverts — the token-level gradient ran through features not alive at turn-averaged grain.
- Scope limits: one SAE training seed; every finest-tier read rests on 3–4 survivors; the bridge reversal is marginal (endpoint interval crosses zero); the two dependent variables disagree in profile.

## Goal

Determine whether the context-to-answer map's high-level-over-low-level predictability gradient holds in a turn-averaged SAE basis: per-tier held-out R² and retrieval accuracy versus an identity+bias baseline, for turn-averaged SAE features of true and map-predicted answer summaries.

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) fit the map — a ridge from the last-prompt-token activation to the whole-answer mean activation (layer 19, Qwen-2.5-7B-Instruct, ~963k real chat contexts, held-out R² 0.72–0.75). [#1482](https://eps.superkaiba.com/tasks/1482) found at token-level-SAE grain that this map predicts coarse matryoshka features far better than specific ones (tier medians 0.435/0.174/0.043). A token-level basis pools per-token codes, one step removed from the map's actual target; an SAE trained directly on whole-answer means decomposes that target itself, so this run asks whether the gradient is a fact about the map or about the token-level basis.

**Broader narrative:** the paper's limits-of-the-mapping section claims the map predicts high-level, identity-adjacent content and not low-level, topic-level content. This run rescopes that claim at the map's own grain: the specific tier mostly fails to exist as alive features rather than surviving and being unpredictable, while the retrieval read keeps a pronounced coarse-over-finer profile.

## Methodology

**Design:** One manipulated variable — the grain of the SAE basis (per-token activations vs whole-answer turn averages) — read through two instruments. The fresh-SAE arm (headline) trains a matryoshka BatchTopK sparse autoencoder (SAE) directly on banked layer-19 whole-answer mean activations and scores 20,000 held-out conversations; the bridge arm applies the token-level matryoshka dictionary from the prior token-level round to turn averages at layer 20 on that round's exact 6,000 score rows, giving the only same-dictionary, same-rows grain contrast. Matryoshka nesting makes feature id a training-imposed granularity label: ids below 2,048 are trained under every nested-prefix reconstruction loss (coarsest tier), ids 2,048–16,383 the middle tier, ids 16,384–65,535 only under the full-width loss (finest tier). Controls: identity+bias (predict the answer summary as the context state plus a learned constant shift), train-mean null, a dense-input companion ridge (context state → each SAE feature directly, separating "the map loses tier-t content" from "tier-t content is linearly unreachable at all"), a 20-draw row-shuffle floor, a within-activity-quintile tier-label permutation, a corpus-transfer fold (fit on LMSYS rows, score on the 9,017 WildChat holdout rows), a pile-trained dictionary twin, and an inlier-masked pooling twin. One SAE training seed (2476). The verdict rule is defined on the fresh-SAE arm and fixed in the plan:

| Outcome | Rule (both conditions required) |
|---|---|
| gradient holds | permutation percentile ≤ 2.5 (coarse-better side) and coarse-minus-finest median contrast above 0 |
| gradient reversed | permutation percentile ≥ 97.5 (fine-better side) and contrast below 0 |
| indeterminate | any other combination, including the two conditions disagreeing |

**Training:** the only model trained is the fresh turn-averaged SAE; the map itself is refit (never retrained) under the prior round's exact recipe. Every value below is copied from the committed run artifacts.

| Hyperparameter | Value | Source |
|---|---|---|
| SAE architecture | matryoshka BatchTopK; nested prefix reconstruction losses, mean reduction; threshold EMA 0.999 | `train_log.json` cfg |
| SAE dimensions | input 3,584; dictionary 65,536; k = 100; tier bounds 2,048 / 16,384 / 65,536 | `train_log.json` cfg; tier bounds match the bridge dictionary for tier parity |
| SAE optimizer | Adam, betas 0.9 / 0.999, lr 2e-4, batch 256, 3 epochs (10,941 steps), init seed 2476 | `train_log.json` cfg; recipe from arXiv 2606.28548 (same model, layer, corpus, object) |
| SAE training rows | 933,444 train + 10,000 validation carve (seed 2476) from the 943,444-row pool (963,444 banked contexts minus holdout and pinned val/test) | `train_log.json` pools |
| Realized training state | final validation variance-FVE 0.9093 (FVE = fraction of variance explained) at L0 98.96 (mean active features per row); per-tier dead-feature fractions 0.095 / 0.291 / 0.356 after epoch 3 | `train_log.json` epochs |
| Map refits | shared-Gram ridge; full-train and holdout-excluded refits reproduce the committed whole-map R² (deltas 3.5e-7 and 2.1e-7 against tolerances 0.002 / 0.003) | `recon_gate.json` |
| Bridge-arm map | ridge fit on 24,000 rows, scored on 6,000 (selected λ 3.16e3, validation R² 0.671); dense-input companion λ 1.0e3 | `preds_meta.json` |
| Bridge dictionary | `chanind/qwen2.5-7B-it-layer-20-saes` revision `db8c88b4…`, sae_id `lmsys/matryoshka/k-100` (pile twin `pile/matryoshka/k-100`), k = 100, width 65,536 | the prior round's pin, re-listed live |
| Alive floor + panel | active on ≥ 1% of fit-side true-summary encodes (1,200 of 120,000 fresh; 240 of 24,000 bridge); panel cap 16,384, seed 14824 | `tier_tests_c.json` / `tier_tests_b.json` alive-mask provenance |
| Statistics constants | 10,000 permutation draws; 10,000 bootstrap draws (seeds 24761 / 24762); 20 row-shuffle draws; retrieval at k = 1 / 5 / 10, euclidean + cosine | `tier_tests_*.json`, `retrieval_*.json` |
| Answer generation | none this run; the measured answers are the banked on-policy rollouts (Qwen-2.5-7B-Instruct, single stochastic sample per prompt, engine seed 42, at most 1,024 new tokens) | capture-run record (provenance in footer) |

**Evaluation:** No text is generated and no judge is called anywhere in this pipeline — every dependent variable is a per-feature or per-row statistic over banked activations, so the usual on-policy and judged-instrument checks do not apply; the validity checks here are the four gates below. DV1 (primary): per-feature held-out R² of the SAE-encoded map prediction against the SAE-encoded true answer summary — R² over held-out rows per feature, median over alive features per tier; the construct is preservation of tier-t answer content by the pre-generation map. The fresh SAE is trained on exactly the map's target distribution (on-distribution); encoding the prediction is near-distribution (encoder reconstruction FVE 0.937 on predictions); the bridge encoder is declared off-domain and gated. DV2 (companion): retrieval — whether the predicted tier-t feature vector picks the right conversation out of the held-out pool (accuracy at k, chance = k / pool size). Gates, all passing: a reconciliation gate (the refit map reproduces the committed whole-map values; deltas in the Training table); a recapture identity gate on the bridge rows (30,000 row-matched cosines of the recaptured context state against the banked store: min 0.9954, 0.1st percentile 0.99956, median 0.999933); an encoder fitness gate for the off-domain bridge dictionary (variance-FVE 0.659 at L0 26.5 on the 30,000 turn averages, against a 0.35 demotion floor — degraded from its native token-level regime exactly as the recipe paper's measurement predicts, so the bridge read is kept); and an SAE training-sanity gate (held-out variance-FVE 0.909 at L0 99.0 vs the plan's expectation of at least ~0.85). Bootstrap intervals are conditional on the realized score rows. Four caveats carried from the run record: (1) the recapture identity gate passed only at version 2 — the version-1 flat bar (every row ≥ 0.999) was refused with 4 of 30,000 rows below it (min 0.9954); a worst-16-row re-probe at fp32 improved 13 of 16 rows against the bank (worst fp32 row 0.99595), consistent with single-position deep-layer numerical jitter rather than a mapping bug (known mapping bugs read 0.39–0.84), and the version-2 bars (min ≥ 0.995, 0.1st percentile ≥ 0.999, median ≥ 0.9995) were set from the same production distribution they pass — the bars were chosen after seeing the production distribution rather than in advance; margin on the flat bar is 4.0e-4. (2) Two-SHA phase provenance: the recapture phase ran at code `a51057beaf` (before the crash-fix relaunch that recalibrated the gate), all later phases at `064462df4d68`. (3) The 1%-of-rows alive floor is inherited for comparability, and the attrition finding depends on where it sits: mean per-feature firing under k = 100 at width 65,536 is ~0.15% of rows, ~6.5 times below the floor. (4) Six smoke-run markers on this task (from a 24-row composed smoke) are not production results and must never be cited. Conciseness acknowledgment: the total skim-prose budget and the 120-word per-result band are exceeded in places — the five results carry verdict tables and companion reads that need the space; every hard cap and bullet cap is met.

**Data extraction:** Tier-1 real-world data: first-turn LMSYS / WildChat user conversations with the banked on-policy model answers (no new generation; the teacher-forced re-read consumes persisted text and stores only reduced means). The fresh arm streams the 1,920 banked capture chunks and assembles layer-19 context-state and answer-mean matrices for 963,444 conversations (fp16, ~6.9 GB each), re-asserting the pinned splits by sha (val 400 / test 1,000 / holdout 20,000 / SAE-fit 120,000). The bridge arm re-captures the prior round's exact 30,000 rows with the same tokenization convention, accumulating answer-span means at layers 19 and 20 in a streaming reduction (per-token activations never materialized beyond the batch). Encodes: fresh SAE on layer-19 objects (20,000 held-out rows); bridge dictionary on layer-20 turn averages (6,000 score rows). Artifact-name convention: file suffix `_c` marks the fresh-SAE arm, `_b` the bridge arm.

**Sample training/evaluation data + completions:** this task produces no completions and calls no judge; the end-to-end unit is context state → map prediction → SAE encode → per-feature held-out R². Below, first the underlying conversation rows, then the per-feature evaluation rows. 3 of 19,967 labeled holdout rows (the fresh arm's score rows), random sample, seed 2476 — prompt and answer excerpts mechanically truncated to 12 words for this real-user-text corpus; full text lives in the pinned [judge-labels tree](https://github.com/superkaiba/explore-persona-space/tree/43818beb136bac133e33414a6520760b4206afc6/eval_results/issue_1482/judge_labels), keyed by row id.

| Row id | Corpus | Prompt excerpt (truncated) | Model answer excerpt (truncated) |
|---|---|---|---|
| ci471286 | lmsys | "Please start a dialogue, with the aim of naturally leading me to …" | "Sure! Let's start with a light topic. How's your day going so …" |
| ci264094 | lmsys | "Do humans have anal glands?" | "Humans do not have anal glands. Anal glands are a characteristic found …" |
| ci558155 | wildchat | "授权委托书 关键词：甲方是我们，乙方是租客 …" (a Chinese rent-collection authorization-letter request) | "授权委托书 甲方：……" (the drafted template letter, in Chinese) |

Per-feature evaluation rows: 5 of 879 (fresh arm) and 5 of 412 (bridge arm) alive-feature rows, random sample, seed 42; full arrays in the committed [perfeature_c_encodepred.npz](https://github.com/superkaiba/explore-persona-space/blob/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/turnavg/perfeature_c_encodepred.npz) and [perfeature_b_encodepred.npz](https://github.com/superkaiba/explore-persona-space/blob/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/turnavg/perfeature_b_encodepred.npz).

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

Worked example: for fresh-arm feature 1537 (coarsest tier), the map's predicted answer summary is encoded for each of the 20,000 held-out conversations, and the predicted feature value tracks the true one with held-out R² 0.651 — despite the feature firing on only 2,329 of 120,000 fit rows.

## Results

### The verdict is indeterminate on the headline arm: the permutation criterion fires coarse-better while the endpoint contrast lands negative

Plotted: per-tier median held-out per-feature R² of encoded map predictions against encoded true summaries, 10,000-draw bootstrap whiskers; four bars per tier — fresh turn-averaged SAE (20,000 rows, 879 features) and token-level dictionary on turn averages (6,000 rows, 412 features), each vs identity+bias; black dashes: token-level reference medians (different grain, layer, dictionary, and rows — not a matched contrast).

![Per-tier median held-out R2 for both turn-averaged arms and baselines, with token-level reference dashes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/paper/c3_turnavg_tier_gradient.png)

> **Figure.** *Neither turn-averaged basis reproduces the token-level round's monotone coarse-over-specific decline.* Median held-out per-feature R² per matryoshka tier: fresh turn-averaged SAE (blue) vs its identity+bias baseline (orange), token-level dictionary on turn averages (green) vs its baseline (pink); black dashes = token-level reference medians. Whiskers: 10,000-draw bootstrap intervals.

| Arm | Tier medians (n alive) | Coarse-minus-finest contrast (95% interval) | Permutation percentile (observed pooled rank correlation vs null band) | Rule outcome |
|---|---|---|---|---|
| Fresh turn-averaged SAE, layer 19 | 0.625 / 0.131 / 0.670 (788 / 88 / 3) | −0.044 (−0.264 to −0.013) | 0.0, coarse-better side (−0.449 vs −0.190 to −0.069; achievable magnitude ceiling 0.528) | indeterminate — the two conditions disagree |
| Token-level dictionary on turn averages, layer 20 | 0.420 / 0.553 / 0.601 (387 / 21 / 4) | −0.181 (−0.325 to +0.105; crosses zero) | 99.42, fine-better side (+0.113 vs −0.103 to +0.086) | reversed — a secondary read; the rule is defined on the fresh arm |
| Token-grain reference (its own round: token level, layer 20, 6,000 rows) | 0.435 / 0.174 / 0.043 (16,384-feature panel) | +0.392 | coarse-better (−0.395 vs −0.250 to −0.228) | gradient holds |

The two rule conditions read different populations: the permutation pools all 879 features and is dominated by coarse-vs-middle (876 of 879), while the endpoint contrast is decided by the 3 finest-tier survivors — indeterminate is the honest cell. The bridge-arm reversal is marginal: its endpoint interval crosses zero and its permutation exceedance is narrow. An inlier-masked pooling twin matches the plain mean to four decimals, so outlier tokens do not drive the inversion.

### Alive-feature attrition, not a gradient, is the dominant turn-averaged phenomenon

Plotted: the fraction of each tier's candidate features clearing the alive floor (active on at least 1% of fit rows), log y-axis, one bar per tier and arm.

![Alive-feature fraction per tier for both turn-averaged arms, log scale](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8d0d7d545ce79be59524f8d27b8146546d1d984c/figures/issue_2476/i2476_alive_attrition_legend.png)

> **Figure.** *Above 99.9% of finest-tier candidates fall below the alive floor in both turn-averaged instruments.* Fraction of candidate features per matryoshka tier clearing the 1%-of-fit-rows floor (1,200 of 120,000 rows, fresh arm; 240 of 24,000, bridge arm), log scale.

| Tier (candidates) | Fresh SAE | Bridge dictionary | Pile twin |
|---|---|---|---|
| coarsest (2,048) | 788 (38%) | 387 (19%) | 261 (13%) |
| middle (14,336) | 88 (0.6%) | 21 (0.15%) | 39 (0.27%) |
| finest (49,152) | 3 (0.006%) | 4 (0.008%) | 14 (0.03%) |

The token-level panel carried 16,384 alive features; the same dictionary on turn averages retains 412. The counts say almost no specific-tier feature clears the inherited floor under these k = 100, width-65,536 instruments — not that excluded features never fire.

Mean firing (k over width, ~0.15% of rows) sits ~6.5 times below the floor, but a global mean fixes no tierwise distribution: the floor convention, matryoshka variance allocation, and finest-tier undertraining (35.6% of finest-tier features end training dead — the inflation the plan anticipated from 0.6x samples at 2x width) all remain unresolved contributors. Under these instruments turn-averaging removes specific features from the alive panel rather than making them unpredictable: the map's target object carries little alive specific-tier structure to predict, which rescopes the paper claim rather than contradicting it.

### Surviving finest-tier features match coarse features of similar firing frequency — a survivor-conditioned read; the middle-tier dip is instrument-dependent

Plotted: every alive feature's held-out R² against log10 of its active fit rows, tier-colored, one panel per arm — the per-unit view behind every median above.

![Per-feature held-out R2 versus activity, tier-colored, one panel per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/issue_2476/i2476_r2_vs_activity.png)

> **Figure.** *Fresh-arm middle-tier features cluster at low activity and low R²; the bridge arm's middle tier does not.* Each point is one alive feature: held-out R² vs log10 active fit rows, colored by tier (fresh arm n = 879, bridge arm n = 412).

| Read | Values | Note |
|---|---|---|
| Matched-activity coarse median, fresh (1,400–2,400 active rows) | 0.525 (n = 197) | finest-tier survivors read 0.880 / 0.670 / 0.648 |
| Matched-activity middle median, fresh (same band) | 0.132 (n = 41) | the dip persists at matched activity |
| Matched-activity coarse median, bridge (390–552 active rows) | 0.393 (n = 83) | finest survivors 0.745 / 0.626 / 0.576 / 0.318; fresh band applied instead: 0.509 (n = 22) |
| Identity+bias coarse-minus-finest contrast | fresh −0.287 (−0.416 to +5.146, crosses zero); bridge −0.126 (−0.337 to −0.008) | a map-free predictor reproduces the endpoint inversion at the point estimates |
| Dense-input companion medians | 0.533 / 0.159 / 0.719 (fresh); 0.391 / 0.544 / 0.598 (bridge) | survivors are also more linearly predictable from context at all |
| Corpus-transfer fold (fit LMSYS, score 9,017 WildChat rows) | 0.522 / 0.075 / no finite finest values | 18 of 86 finite middle-tier transfer values negative; coarse tail reaches −6.83 |
| Pile-twin medians | 0.457 / 0.557 / 0.413 (n = 261 / 39 / 14) | non-monotone, but a small positive coarse-minus-finest endpoint (+0.044) |

Survivors read as at least as predictable as coarse features of comparable firing frequency in both instruments — a survivor-only description at n = 3–4, not a population inference; survival is itself activity selection no within-panel control can undo. The middle-tier dip survives corpus transfer and activity matching in the fresh instrument yet reverses in the bridge arm: instrument-dependent. Identity+bias and the dense-input companion reproduce the survivor-profile shape without the map, so map-specific content loss is not necessary for the endpoint inversion — a survivor-panel effect and shared dictionary geometry are indistinguishable here.

### The map beats identity+bias on per-feature R² at every tier, while retrieval concentrates in the coarsest tier

Plotted: per-tier retrieval accuracy at k = 1 (euclidean; pool = the held-out rows, 20,000 fresh and 6,000 bridge; chance = 1 over pool size), map vs identity+bias in each arm's feature basis; the inset zooms the middle and finest tiers.

![Per-tier retrieval accuracy at k equals 1, map versus identity plus bias, both arms, with low-tier zoom inset](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/paper/c3_turnavg_tier_acc1.png)

> **Figure.** *Coarse-tier features identify the held-out conversation at 0.708; middle- and finest-tier rates sit orders of magnitude lower.* Retrieval accuracy at k = 1 per tier, map vs identity+bias, fresh arm (chance 5.0e-5) and bridge arm (chance 1.67e-4); inset: middle and finest tiers.

| Tier | Paired R² delta, fresh (95% interval) | Paired R² delta, bridge | acc@1 map vs identity+bias, fresh | acc@1, bridge |
|---|---|---|---|---|
| coarsest | +1.39 (+1.28 to +1.49) | +0.45 (+0.42 to +0.49) | 0.708 vs 0.238 | 0.335 vs 0.139 |
| middle | +5.26 (+3.90 to +6.91) | +0.56 (+0.46 to +0.76) | 0.00105 vs 0.00345 | 0.0065 vs 0.0017 |
| finest | +1.15 (+1.15 to +6.49) | +0.50 (+0.13 to +0.74) | 0.0001 vs 0 | 0.00067 vs 0.00033 |

| Diagnostic | Fresh arm | Bridge arm |
|---|---|---|
| Encoder reconstruction FVE of the input being scored (map prediction vs identity+bias guess) | 0.937 at L0 88 vs 0.305 at L0 955 | gate values in Methodology |
| Alive features beating the 20-draw row-shuffle floor (map vs identity+bias) | 100% vs 29.6% | 100% vs 47.8% |
| Dense-space anchor (before encoding) | acc@1 0.705; pooled R² 0.724 | — |
| Retrieval cells favoring the baseline (of 12 total) | middle tier, both metrics — systematic at k = 1, 5, 10 | finest-tier cosine cell |

The baseline's feature-space medians are strongly negative in the fresh arm (middle tier −5.13): encoding an identity+bias guess lands off the feature manifold, and under it only a minority of features beat the row-shuffle floor (table). Retrieval concentrates sharply: middle-tier map cells sit at 21x and 39x chance yet roughly 670x and 50x below their coarse-tier rates, and 3 of 12 map-vs-baseline cells favor the baseline.

Tier-restricted vectors have 788/88/3 dimensions, so the low-tier cells are dimension-confounded — concentrated in the coarsest tier is the supportable read, not a content-loss quantification. The two dependent variables disagree in profile; the headline is scoped to the R² profile for exactly this reason.

### Per-feature predictability transfers across grain (rank correlation 0.767) while the tier profile inverts

Plotted: each intersection feature's held-out R² at turn-averaged grain (bridge arm) against the same feature's committed token-level R² — same dictionary, same rows; 409 features, tier-colored, identity diagonal; every point is one feature, so this is its own per-unit view.

![Per-feature R2 at turn-averaged versus token grain, same dictionary and rows, with identity diagonal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8d0d7d545ce79be59524f8d27b8146546d1d984c/figures/issue_2476/i2476_bridge_token_vs_turnavg.png)

> **Figure.** *Feature ranking is basis-stable; the level shifts down and the tier profile inverts.* Held-out per-feature R², turn-averaged vs token grain, on the 409-feature intersection (same dictionary, same 6,000 score rows); points below the diagonal lose predictability at the turn-averaged grain.

Pooled rank correlation 0.767 (per tier 0.754/0.893/0.5 at n = 387/19/3) clears the plan's 0.5 expectation — the ranking clause of the bridge hypothesis passes. The level shifts down for most features (median delta −0.152; 337 of 409 points below the diagonal, 72 above — dominant but heterogeneous), and the gradient-preservation clause is refuted: the per-tier profile inverts (0.420/0.553/0.601 vs the token-level 0.435/0.174/0.043). The token-level gradient ran through thousands of low-activity specific features that do not exist as alive features at this grain — 15,975 of its 16,384 panel features fall out of the intersection.

---

**Repro:** ~7.3 h wall on 1× H100-80 (RunPod pod-2476), zero LLM-API calls. Workload: [scripts/issue2476_turnavg_sae.py @ 064462df4d](https://github.com/superkaiba/explore-persona-space/blob/064462df4d683afcb6a717253275cdb8486c9a82/scripts/issue2476_turnavg_sae.py) (`--phase all`); two-SHA phase provenance — the bridge-row recapture ran at [a51057beaf](https://github.com/superkaiba/explore-persona-space/blob/a51057beaf6a5073dcf273ef80077c759fb98499/scripts/issue2476_turnavg_sae.py) (its gate record carries that commit), all later phases (SAE training / maps / eval / figures) at `064462df4d68`. Eval artifacts (git, branch `issue-2476`): [eval_results/issue_2476/turnavg tree](https://github.com/superkaiba/explore-persona-space/tree/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/turnavg) — 26 files: `tier_tests_c.json` / `tier_tests_b.json` (medians, permutation, bootstrap, verdict inputs), `retrieval_c.json` / `retrieval_b.json`, `panel_c.json` / `panel_b.json`, per-feature npz for the map / identity+bias / dense-input / transfer / pile / inlier reads, `bridge_b.npz`, `train_log.json`, `recon_gate.json`, `preds_meta.json`, `regime.json` (split shas), and the recapture gate record `gates_p2.json`; recapture-gate recalibration evidence at [eval_results/issue_2476/g2a_recalibration tree](https://github.com/superkaiba/explore-persona-space/tree/8dac0f576f87b175fc4a757e1291b23aeda8beac/eval_results/issue_2476/g2a_recalibration). Figures: hero pair `figures/paper/c3_turnavg_tier_gradient` + `c3_turnavg_tier_acc1` (png, pdf, data sidecars) and exploratory panels under [figures/issue_2476 @ 8dac0f576f](https://github.com/superkaiba/explore-persona-space/tree/8dac0f576f87b175fc4a757e1291b23aeda8beac/figures/issue_2476) (relabeled renders; sidecar data verified identical to the first renders), unchanged panels pinned at [8d0d7d545c](https://github.com/superkaiba/explore-persona-space/tree/8d0d7d545ce79be59524f8d27b8146546d1d984c/figures/issue_2476). HF (listed live at write time, 99 files): [issue2476_turnavg/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e5de92ea13a5c51d0d03d03446b0cecb4a22f4d5/issue2476_turnavg/analysis_tensors) — `recapture_store/` (68 files: the 30,000-row mean store + the gate record source), `eval/` (21), `split_meta/` (7), `sae_c/` (3: SAE weights 1.88 GB safetensors, cfg, train log). Artifact-suffix legend: `_c` = fresh turn-averaged SAE arm, `_b` = bridge arm, `_ib` = identity+bias, `_densein` = dense-input companion. Reuse provenance: capture chunks + rollout text from [#779](https://eps.superkaiba.com/tasks/779) at HF `issue779_monitoring/fitter-fair-comparison-n1m/{final_token_capture,raw_completions}` — the map's own training capture, fitness by construction; pinned splits, the 30,000-row subsample, the committed token-level per-feature panel, and the layer-20 dense store from [#1482](https://eps.superkaiba.com/tasks/1482) at HF `issue1482_error_analysis/analysis_tensors/matryoshka_tier/store` revision `8b82eb9753…` — same-lineage products, shas re-asserted in-run; token-level dictionaries [chanind/qwen2.5-7B-it-layer-20-saes](https://huggingface.co/chanind/qwen2.5-7B-it-layer-20-saes/tree/db8c88b400e36e603d0563abcf8d289e5eff5687) — that round's own pin, fitness gate passed at FVE 0.659; sample-row text from [#1482](https://eps.superkaiba.com/tasks/1482)'s committed judge-labels file (pinned above) — the same holdout rows this run scores. Plan: `plans/plan.md` (v4) in the task directory; production results marker posted 2026-08-23T11:44Z. Discarded per plan: per-token recapture activations, assembled fp16 matrices, and SAE optimizer state (regeneration recipes in the plan's reproducibility card); no rollout text, judge output, metric, or config discarded.

**Context:** Origin prompt (verbatim): `Paper outline 2026-08-22: 'SAE experiments — try turn averaged SAEs'`. Lineage: [#1482](https://eps.superkaiba.com/tasks/1482) — the token-level tier-gradient round this run re-tests at turn-averaged grain, itself an error-analysis child of the [#779](https://eps.superkaiba.com/tasks/779) map. Created 2026-08-22; production run 2026-08-23 (one crash-fix relaunch after the recapture identity gate refusal narrated in Methodology); interpretation passed both critics 2026-08-23.
