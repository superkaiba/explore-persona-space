---
title: Firing activity dominates per-feature map predictability of turn-averaged SAE
  features, with schema category the strongest surviving content-level covariate (MODERATE
  confidence)
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-24T20:18:15Z'
has_clean_result: true
parent_id: 2476
origin_prompt: 'I want to rerun their thing and then first see if our mapping predicts
  better the higher level things for our Matryoshka one. But I want to run one that
  is exactly what they run also, and then see what metrics best predict if one of
  the features will be predicted well. So it should be basically rerun their judgment
  also. So it should be their types, but also additional metrics inspired by the other
  experiment we ran for the SAE features. And then the plot should basically be like,
  okay, this is the property that best explains it, and then we control for that or
  partial that out and this is the next one, and then control for that as well, this
  is the next one. (2026-08-24; clarify-gate answers: one child task of #2476; assistant
  means banked; k=200 twin in, nested/attribution out; judge = sonnet-4-6 everywhere,
  user override; exploratory category ranking; spawn --auto)'
workflow: v1
goal: 'Determine (1, exploratory) how the context→answer map''s per-feature predictability
  of turn-averaged SAE features ranks across the five Der et al. (arXiv 2606.28548)
  schema categories (content/form/voice/function/meta) on the #2476 matryoshka SAEs
  (k=100 + k=200) and on a faithful flat replication of Der et al.''s recipe; (2)
  whether that replication (BatchTopK 32,768/k=128 on banked layer-19 turn means +
  their full judged evaluation, judge claude-sonnet-4-6 no-prefill) reproduces their
  discrimination-vs-coverage inversion; and (3) which feature properties (schema categories
  + the #1482-inspired turn-grain covariate battery) explain per-feature map predictability,
  via a forward-selection partial-out ladder as the headline figure.'
relates_to:
- spec-context-as-vector
---
# Firing activity dominates per-feature map predictability of turn-averaged SAE features, with schema category the strongest surviving content-level covariate (MODERATE confidence)
<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_2552.md](https://github.com/superkaiba/explore-persona-space/blob/125cbc91131c607fa4baefa4143fcc64078c1459/docs/methodology/issue_2552.md) · [gist](https://gist.github.com/superkaiba/c7aca73cf6369bd5d14ad02efec91eea)

## Takeaways

- Mean activation when active is selected first in every forward-selection ladder (partial R² 0.29 replication, 0.36 k=100, 0.35 k=200 on the rank-transformed per-feature held-out map R², after a forced log-activity step 0; panels n = 2,974 / 2,149 / 2,658 features), and every competitive step clears its 10,000-draw selection-matched null band (weakest step p = 0.002, the twin match cosine selected sixth in the k=200 dictionary; every other step p ≈ 1e-4).
- Judge-assigned schema category survives activity partialing in every dictionary (ladder partial R² 0.091 replication, 0.075 k=100, 0.059 k=200). In the matryoshka dictionaries the nested-training tier is selected second (partial R² 0.164 and 0.188), so the parent's coarse-over-specific tier gradient is not reducible to firing rate either.
- After activity adjustment at the 0.2% firing floor, Content is the least map-predictable category in every dictionary (adjusted effect −0.10 / −0.07 / −0.09 rank-R² units; the only category whose corrected pairwise contrasts separate from all four others, permutation p ≈ 1e-4). Voice tops the point-estimate ranking everywhere, but the raw and adjusted orderings disagree in all three dictionaries at this floor, Voice's advantage is concentrated in the lower-activity quintiles (its top-quintile effect is zero or negative in every dictionary), and at the stricter 1% floor Form overtakes Voice; only Content-last is floor-robust.
- Der et al.'s discrimination-versus-coverage inversion reproduces on the pooled 2,000-turn eval: the public per-token dictionary wins 10-way matching 0.962 vs 0.661 (paired gap 0.301, n = 1,988 complete pairs) while the replication turn-averaged dictionary wins per-turn coverage head-to-heads 68.1% (1,342 of 1,970 valid trials); the judge-free embedding read agrees (top-3 cosine 0.611 vs 0.557).
- Judge = claude-sonnet-4-6 with no assistant prefill (explicit decision-record override of the pinned project judge; the pinned judge ran only as a calibration control). Agreement between the two judges on category assignment is moderate (0.515 raw, 0.464 after chance correction, n = 200), so category-assignment noise attenuates the category read; it is carried as the main unaddressed error source. Realized spend: 3.7 of 10 budgeted GPU hours plus ~85,000 batch judge calls.

## Goal

Determine (1, exploratory) how the context→answer map's per-feature predictability of turn-averaged SAE features ranks across the five Der et al. (arXiv 2606.28548) schema categories (content/form/voice/function/meta) on the #2476 matryoshka SAEs (k=100 + k=200) and on a faithful flat replication of Der et al.'s recipe; (2) whether that replication (BatchTopK 32,768/k=128 on banked layer-19 turn means + their full judged evaluation, judge claude-sonnet-4-6 no-prefill) reproduces their discrimination-vs-coverage inversion; and (3) which feature properties (schema categories + the #1482-inspired turn-grain covariate battery) explain per-feature map predictability, via a forward-selection partial-out ladder as the headline figure.

**This experiment in context:** [#2476](https://eps.superkaiba.com/tasks/2476) trained matryoshka turn-averaged SAEs on the banked layer-19 answer means and found the context→answer map predicts coarse-tier features far better than specific-tier ones, but tier is a training-imposed label with no content meaning. [#1482](https://eps.superkaiba.com/tasks/1482) ran a token-level feature-correlates battery (within-answer consistency dominant). This task attaches the Der, Kamath & Thompson (arXiv 2606.28548) 24-field / 5-category schema to every feature, replicates their judged evaluation on our corpus, and ranks the covariates in one partial-out ladder.

**Broader narrative:** the context→answer map is the project's central object for predicting answer-side behavior from the pre-generation context state. Knowing which kinds of turn-level features it preserves (how the answer is voiced and formatted) versus loses (exact topical content), and how much of that is firing-rate bookkeeping, tells us what leakage-relevant behavior the map can and cannot forecast.

Conciseness acknowledgment: the Takeaways bullet-length guideline, the per-result prose band, and the total-prose budget all fire as WARNs and are shipped acknowledged; this single round carries eight result sections (two replication halves, the 5-way ranking, the instrument-sensitivity diagnostics, the covariate ladder, its per-feature view, the category ranking, and its per-quintile decomposition) plus mandated disclosures, and the numbers stay inline rather than split across rounds. Config tick labels and panel titles pair each slug with its plain-English bundle gloss (order-stratified pair ticks carry the glosses alone); covariate panels carry plain-English names. Two supporting figures (embedding coverage, pairwise win matrix) are deliberately linked rather than embedded to keep one figure per result. The per-unit-evidence guideline also fires as a WARN for the aggregate-read result sections and ships acknowledged: the per-turn and per-feature JSONs behind every summary figure are committed and linked, and the per-unit views live in the cross-referenced companion results (the covariate scatters behind the ladder, the per-quintile decomposition behind the category ranking) rather than in each section's own prose.

## Methodology

**Design:** three turn-averaged dictionaries are read with the same banked map: a fresh replication of Der et al.'s flat BatchTopK recipe (a sparse autoencoder keeping the top k=128 activations per batch, 32,768 features), plus the parent matryoshka k=100 and k=200 dictionaries (65,536 features, reused weights and banked per-feature R²). A public per-token dictionary (andyrdt trainer_2, 131,072 features, k=128, layer 19) instantiates the paper's per-token pole under max and sum pooling. Three legs: (1) judge-assigned schema categories aggregated over per-feature held-out map R²; (2) the paper's full judged evaluation (feature descriptions → per-turn 24-field structured summaries → 10-way matching → pairwise coverage → 5-way ranking → embedding coverage); (3) a forward-selection covariate ladder. Named deviations from the paper: assistant whole-answer means including the end-of-turn tail (theirs: assistant-turn token means), 963,444 banked rows vs their ~1.58M, pooled LMSYS+WildChat eval with the pooled read pinned in the plan as the verdict carrier (theirs: LMSYS only; our LMSYS-only subset read is advisory), a third-party per-token comparator with top-100 equal-length list truncation, and a single training seed. Every discrimination/coverage conclusion is scoped to this configuration bundle rather than to token-vs-turn grain per se.

**Training:**

| Hyperparameter | Value | Source |
|---|---|---|
| Replication SAE architecture | BatchTopK, width 32,768, k=128 | arXiv 2606.28548 App. A |
| Input | layer-19 whole-answer means, Qwen2.5-7B-Instruct (d=3,584) | banked #779/#2476 store |
| Training rows / epoch | 933,444 (963,444 store minus 10,000 val minus 20,000 holdout) | #2476 split pins |
| lr / batch / epochs / Adam betas | 2e-4 / 256 / 3 / 0.9, 0.999 | arXiv 2606.28548 App. A |
| Threshold EMA | 0.999 | #2476 `train_log.json` cfg |
| Seed | 2552 | run config |
| Realized holdout variance-FVE / nMSE | 0.9222 / 0.0778 (paper reports nMSE 0.097) | `p1/regime_measured.json` |
| Matryoshka k=100 / k=200 | reused parent weights, no retraining | #2476 |
| Ridge map | banked dense fit; per-dictionary encodes of prediction and target; corpus-transfer refit λ grid 23 values 1e-3 to 1e8, validation-selected (selected λ=1000, not grid-edge); fit n=120,000 ≫ d=3,584 | #2476 recipe; `p1/corpusfold_rep.json` |
| Map-fit val / test split | 400 / 1,000 rows (λ validation-selection / held-out test carves of the banked split family; re-asserted by sha at assembly) | #2476 split pins (`split_indices.npz`) |
| Per-feature panel cap | banked matryoshka panels: tier-stratified cap 16,384, seed 14824 (parent recipe); fresh replication panel: cap 12,000, seed 2552, not binding (2,974 alive features selected) | #2476 recipe; `p1/panel_rep.json` |

No LLM fine-tuning anywhere in this task.

**Evaluation:** DV1 = per-feature held-out R² of the SAE-encoded map prediction against the SAE-encoded true answer mean, aggregated per category; the primary category read is activity-adjusted (equal-weight aggregation of within-activity-quintile category effects, 10,000-draw bootstrap intervals; firing floors 0.2% primary, 1% robustness). DV2/DV3 = the paper's judged 10-way matching (can a turn be picked out of a 10-turn lineup from its feature descriptions; chance 0.10) and pairwise coverage (which feature list describes more of what the turn is doing), paired per turn over the same 2,000 eval turns, with score-interval whiskers and a 10,000-draw paired bootstrap on the gap. DV4 = ladder partial R² (the share of still-unexplained variance a covariate adds) on step-0 residuals, with per-draw same-selection permutation null bands within activity quintiles. The judge-free embedding read uses Qwen3-Embedding-8B (the paper's embedding model) to embed feature descriptions and per-turn summary field values, scored as mean top-3 cosine per configuration. Mapping baselines for the fresh map read (plan-pinned pair): identity+learned-bias per-feature median R² −2.57 vs map 0.418 (map higher on 100% of the 2,974 panel features); train-mean null median ≈ −0.005; 20-draw row-shuffle floor max 0.054; retrieval of the map prediction among the 20,000-row holdout pool acc@1 0.745 cosine / 0.689 euclidean (chance 5e-5; identity+bias acc@1 0.306 cosine). Matryoshka baselines are banked in the parent. Judge: claude-sonnet-4-6, no assistant prefill, Anthropic Batch API, all five pilot gates PASS (zero truncation, parse-fail < 2% per arm, api-refusal < 0.10), every production wave above the 0.95 per-item completeness floor. Judge max_tokens: 2,048 for the structured summaries, 1,024 for every other wave (descriptions, category assignment, matching, coverage, ranking, and the pinned-judge calibration; realized values in the committed per-wave `judge_meta` files). Judge calibration (control): the pinned claude-sonnet-4-5-20250929 re-judged 200 items per instrument; raw agreement 0.515 (category assignment), 0.580 (matching), 0.653 (pairwise); per-cell n is thin, so these are descriptive only. Two additional committed figures at the same pin are not embedded: the category-assignment drop rates by status class and the shadow category ranking with status groups, `category_drop_rates.png` and `category_status_groups.png` in [figures/issue_2552](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552). Comparator reconstruction context: trainer_2 on-corpus FVE 0.831 vs the replication SAE's 0.922 holdout variance-FVE.

**Data extraction:** eval turns = 2,000 drawn from the 20,000-row holdout (1,098 LMSYS, 902 WildChat), disjoint from every mining pool (direct overlap check 0 in `judge_aggregates/mfa_disjointness.json`). Description mining: top-25 activating turns per feature from the 120,000-row SAE-fit pool (turn-averaged families) or the 18,000-row non-eval holdout pool (per-token family). Uploaded and judged mining text carries same-length placeholders at 16 corpus-resident secret-shaped spans (`scripts/scrub_secrets.py`); the sha-pinned raw input mirrors are untouched. 74 union-only features with zero mining-pool firings were dropped from description coverage (10 of 4,875 replication, 51 of 4,610 k=100, 12 of 4,192 k=200, 1 of 28,806 per-token; ids in `judge_aggregates/w1_mining_coverage.json`); all 74 are outside the analysis panels, so no panel read is affected. The measured description-union total (42,483) stayed under the 45,000 descope cap, so the full 2,000-turn eval ran.

**Sample training/evaluation data + completions:** the judged text is real LMSYS/WildChat user-assistant turns (data-realism tier 1); excerpts below are judge OUTPUTS (feature descriptions and summaries), quoted verbatim; any embedded mining text was judged with the 16-span placeholder substitution above. Complete artifacts: [judge aggregates (git, pinned)](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates) and [raw judge requests/responses (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae/raw_completions/judge).

<details>
<summary>Replication-dictionary feature descriptions: 3 well-predicted and 3 poorly-predicted features</summary>

Random sample (seed 42) within the top and bottom per-feature R² deciles; all 4,860 valid descriptions (5 dropped): [descriptions_rep_ta.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates/descriptions_rep_ta.json).

- Well-predicted, feature 1105 (R² 0.735, Function): "This feature activates on assistant responses that explain or clarify the answer to a reasoning or factual question, often involving step-by-step logical breakdowns, spatial/relational tracking, or technical explanations"
- Well-predicted, feature 3152 (R² 0.723, Form): "This feature activates on Chinese-language text related to Chinese Communist Party (CCP) organizational work, including party building (党建), discipline education, work summaries, ideological study campaigns, anti-corrupt[ion...]"
- Well-predicted, feature 26425 (R² 0.774, Content): "This feature activates on text describing the geographic location, administrative status, or notable characteristics of cities, towns, and regions — particularly responses to 'where is X located' or 'what is the capital [...]"
- Poorly-predicted, feature 24225 (R² 0.043, Form): "This feature activates on text that mixes multiple languages within a single response or passage, particularly content that combines scripts such as Arabic, Chinese, Russian, Japanese, Korean, or other non-Latin writing [...]"
- Poorly-predicted, feature 30271 (R² 0.039, Content): "This feature activates on technical content related to fluid dynamics and heat/mass transfer, including discussions of Navier-Stokes equations, Reynolds numbers, friction factors, turbulent and laminar flow, pipe flow, d[...]"
- Poorly-predicted, feature 4643 (R² 0.021, Content): "This feature activates on content related to medical specialties, healthcare providers, and the appropriate routing of patients to specific types of doctors or clinics (e.g., ophthalmologists vs. optometrists, which spec[...]"

</details>

<details>
<summary>Judged 10-way matching rows</summary>

Random sample of 5 rows (seed 42); all 10,000 rows: [matching_perturn.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/dere_repl/matching_perturn.json).

- turn 892318, matryoshka k=100 list: gold B, judge chose B (correct; 100 descriptions in list)
- turn 262573, matryoshka k=100 list: gold B, judge chose B (correct; 71 descriptions, feature list shorter than the 100 cap)
- turn 305489, per-token max list: gold E, judge chose E (correct)
- turn 105575, per-token max list: gold E, judge chose E (correct)
- turn 815042, matryoshka k=200 list: gold I, judge chose H (incorrect; 1 of 100 descriptions missing)

</details>

<details>
<summary>One per-turn structured summary</summary>

Cherry-picked (first item, for brevity); the 1,970 valid summaries (of 2,000 requested): [summaries_2000.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates/summaries_2000.json); per-item raw judge outputs: [raw_completions/judge/w2 (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae/raw_completions/judge/w2).

Turn 239783: domain "Business and commerce"; topic "Corporate profile of a Chinese chemical import/export company"; factuality "Presented as factual but reads as promotional/marketing material with unverified claims"; concreteness "Moderately concrete with specific address and product categories, but vague on details" (20 further fields omitted here).

</details>

## Results

### The public per-token dictionary identifies turns from their feature descriptions far better than any turn-averaged dictionary

What is plotted: 10-way matching accuracy per configuration (the judge picks which of 10 candidate turns a feature-description list belongs to; chance 0.10, dotted line) over the same 2,000 pooled eval turns, with 95% score intervals; black dashes mark Der et al.'s own numbers on their LMSYS-only data and their own dictionaries (reference points, not directly comparable).

![10-way matching accuracy for five feature-list configurations with score intervals, chance line, and paper reference marks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/matching_accuracy.png)

> **Figure.** *The per-token max configuration wins the lineup by ~30 points.* Bars: replication turn-averaged (32,768-wide), matryoshka k=100 and k=200 (65,536-wide), andyrdt trainer_2 per-token max and sum (131,072-wide), all under top-100 judged lists; n = 1,991 to 1,996 valid turns per bar.

| Quantity (pooled carrier) | Estimate | 95% interval | Ceiling | n |
|---|---|---|---|---|
| Paired matching gap, per-token max minus replication turn-averaged | 0.3008 | 0.2792 to 0.3224 | 0.3390 | 1,988 pairs |
| Coverage win-rate delta over parity, replication turn-averaged vs per-token max | 0.1812 | 0.1603 to 0.2014 | 0.5 | 1,970 trials |
| LMSYS-only advisory re-read (gap / coverage delta) | 0.2751 / 0.1838 | 0.2459 to 0.3044 / 0.1556 to 0.2107 | 0.3208 / 0.5 | 1,094 / 1,091 |

The matching gap is unambiguous (p < 1e-4 under the paired bootstrap); its upper interval bound sits 0.017 below the 0.339 ceiling (one minus the turn-averaged arm's own 0.661 paired accuracy, the largest gap this comparison can express), so the estimate is compressed near its maximum. The advantage is max-pooling-specific: the same per-token dictionary under sum pooling scores 0.683, inside the turn-averaged range. Our turn-averaged accuracies (0.61 to 0.70) run below the paper's 0.739 while per-token max runs slightly above their 0.950; plausible protocol sources are the pooled corpus, dictionary widths, and the top-100 truncation.

### The replication turn-averaged dictionary wins per-turn coverage head-to-head, completing the reproduced inversion

What is plotted: per-turn head-to-head coverage outcomes between the replication turn-averaged and per-token max lists (which list better covers what the turn is doing, judged with randomized presentation order), counts over the 1,970 valid pooled trials with 95% score intervals.

![Per-turn coverage wins, replication turn-averaged versus per-token max](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/delta_cov_perturn.png)

> **Figure.** *The replication turn-averaged dictionary (32,768-wide) wins 1,342 of 1,970 coverage head-to-heads (68.1%) against andyrdt trainer_2 per-token max (131,072-wide), top-100 judged lists, pooled eval.*

Both halves land the Reproduced cell of the plan's verdict lattice on the pooled carrier: per-token wins discrimination, turn-averaged wins coverage. The LMSYS-only advisory re-read agrees.

The judge-free embedding read agrees: mean top-3 cosine 0.611 (replication) vs 0.557 (per-token max), n = 1,970 ([embedding figure](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/embedding_coverage.png)). In the [10-pair win matrix](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/pairwise_win_matrix.png) every turn-averaged row beats every per-token column by point estimate only: the weakest cell (k=200 vs per-token sum, 50.6%) spans parity, and three of six such cells drop to parity or reverse when the turn-averaged list is shown second (instrument-sensitivity result below). Our head-to-head win rate (0.681) is lower than the paper's (0.797).

Scope: the contrast bundles grain with dictionary provenance, width, and truncation; it supports transfer of the inversion to this configuration bundle rather than a causal grain claim.

### The turn-averaged dictionaries also win the judged 5-way preference ranking

What is plotted: mean rank per configuration when the judge orders all five feature lists for the same turn from most to least descriptive (1 = best), over the 1,951 valid pooled eval turns (of 2,000 requested).

![Mean 5-way ranking per feature-list configuration, lower is better](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/mean_rank.png)

> **Figure.** *The replication turn-averaged dictionary ranks best (mean rank 2.37) and the two per-token configurations rank worst (3.44 and 3.45).* Same top-100 judged lists; n = 1,951 valid turns.

All three turn-averaged dictionaries out-rank both per-token configurations (mean rank 2.37 / 2.74 / 3.00 vs 3.44 per-token sum / 3.45 per-token max), independently supporting the coverage half of the inversion: when the judge weighs whole lists against each other, the per-token lists lose despite their lineup discriminability.

### The judged instruments are sensitive to lineup position and presentation order: one win-matrix cell reverses by order stratum and two fall to parity

What is plotted: left, 10-way matching accuracy by gold lineup slot for the headline pair; right, each turn-averaged configuration's coverage win rate over each per-token configuration, split by presentation order (dotted line marks parity).

![Matching accuracy by gold slot and pairwise win rate by presentation order](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/instrument_position_effects.png)

> **Figure.** *Replication matching accuracy spans 0.522 (slot D) to 0.808 (slot A) while per-token max stays between 0.941 and 0.989; every turn-averaged list wins more coverage trials when listed first.* Pooled eval, top-100 judged lists; balanced randomization protects the pooled estimates.

The headline pair survives both order strata (replication beats per-token max on 71.4% of trials listed first, n = 1,002, and 64.8% second, n = 968); the k=200 vs per-token sum cell reverses (63.8% first, 38.4% second), and k=100 vs per-token sum (50.7%) and k=200 vs per-token max (50.1%) sit at parity when listed second.

The 5-way ranking shares this sensitivity: mean rank improves with display letter (2.51 at slot B vs 3.55 at slot E), equal-slot reweighting preserves the configuration ordering (2.38/2.73/3.01/3.44/3.45), and per-token max's mean hides a polarized distribution (443 first-place, 773 last-place of 1,951 turns), so ranking claims stay mean-rank-scoped. Exploratory, thin cells: replication matching drops on Russian-monolingual (11 of 29) and Simplified-Chinese (20 of 43) vs English-only (136 of 207) summaries; per-token max scores 93 to 97% there.

### Firing activity is selected first in every covariate ladder, and schema category survives its removal

What is plotted: the forward-selection ladder per dictionary; the leftmost bar is the forced step-0 base (intercept plus log firing rate; overall rank-R² 0.21 / 0.32 / 0.34), each later bar the overall R² gain of the covariate selected at that step, thin marks the 95th percentile of 10,000 selection-matched permutation draws.

![Forward-selection partial ladder for the three turn-averaged dictionaries with null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/ladder_hero.png)

> **Figure.** *Mean activation when active is selected first everywhere; category is step 2 in the replication dictionary and step 3 to 4 in the matryoshka dictionaries, where the nested-training tier is selected second.* Panels: replication (n=2,974), k=100 (n=2,149), k=200 (n=2,658 complete-case features).

Every plotted step clears its null band (selection-matched permutation p ≈ 1e-4; the weakest, twin match cosine in k=200, still clears at p = 0.002). Category's partial R² after activity is 0.091 / 0.075 / 0.059.

The first two selections are stable across split halves in the matryoshka dictionaries; in the replication dictionary the step-2/step-3 pair (category, activation variance) swaps across halves. Mean activation may partly proxy per-feature estimation reliability rather than what the map preserves; a reliability-adjusted re-read is the natural follow-up. Sensitivity runs (category forced last, twin covariates excluded, 1% floor, quintile-dummy base) are in `ladder/ladder_steps.json`; the thesis-kill criterion (no step clearing any band) did not fire.

### The per-feature scatters show marginal gradients for most covariates, steepest for firing activity

What is plotted: per-feature held-out R² (y) against each standardized covariate (x) for the replication dictionary's 2,974 panel features, one panel per covariate, decile medians overlaid; this is the per-unit data behind the ladder aggregate.

![Per-feature R-squared versus each covariate with decile medians, replication dictionary](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/covariate_scatters_rep_ta.png)

> **Figure.** *Mean activation when active rises monotonically (decile medians 0.25 to 0.74); activation variance (0.13 to 0.71) and twin consistency (0.22 to 0.67) also rise, answer-PCA best rank (0.66 to 0.30) and direct-logit footprint (0.61 to 0.29) fall; co-activation degree and corpus share are closest to flat.* Matryoshka twins: covariate_scatters files for k=100 and k=200, same directory.

Most covariates are marginally informative: nearly every panel carries a visible decile gradient, and the activation-variance gradient matches its band-clearing step-3 selection in the replication ladder (partial R² 0.077; 0.063 in k=200). The ladder therefore measures what each covariate adds after activity, not whether it correlates at all. Spread around every decile median stays wide: the final six-step designs reach overall rank-R² 0.56 (replication), 0.69 (k=100), and 0.71 (k=200), so the ladders explain most rank variance in the matryoshka dictionaries while 29 to 44% of it stays in per-feature spread.

### After activity adjustment, Voice ranks most map-predictable and Content least in every dictionary at the 0.2% floor, but only Content separates cleanly

What is plotted: top row, activity-adjusted category effects (equal-weight mean of within-quintile effects on rank R², 10,000-draw bootstrap whiskers; primary); bottom row, raw per-category medians (secondary); one column per dictionary, firing floor 0.2%.

![Adjusted and raw category rankings for the five schema categories across three dictionaries](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/category_ranking.png)

> **Figure.** *Content is lowest in every panel (adjusted effect −0.10 / −0.07 / −0.09); Voice is highest by point estimate in every adjusted panel; the four non-Content categories overlap.* Panels: replication (n=2,974), k=100 (n=2,149), k=200 (n=2,658 complete-case features).

Category structure beyond activity is present in every dictionary (within-quintile label-permutation p ≈ 1e-4), but its clean part is Content versus the rest: Content's corrected pairwise contrasts separate from all four other categories everywhere, while no pair among the other four separates consistently (Form vs Function separates in the replication dictionary only, corrected p = 0.024), so Voice-first is a point-estimate ordering.

Raw and adjusted orderings disagree in every dictionary; at the 1% floor Form overtakes Voice with Content still last, so the ranking is quoted at its floor only. The per-quintile decomposition (next result) shows the Voice advantage is confined to the lower-activity quintiles.

Caveat: chance-corrected cross-judge agreement on category assignment is 0.46; random misassignment mostly attenuates category differences, but judge-dependent boundary conventions stay unaddressed by the adjustment.

### Voice's adjusted advantage is concentrated in low-activity quintiles and collapses at high activity

What is plotted: per-category activity-adjusted effects split by activity quintile (Q1 lowest firing to Q5 highest), one panel per dictionary; the per-quintile decomposition of the adjusted ranking above.

![Category effects by activity quintile for the three turn-averaged dictionaries](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552/category_activity_heatmap.png)

> **Figure.** *Voice's effects are largest in the lowest-activity quintiles, and Voice, Function, and Meta all shrink or reverse in the top quintile; Form stays positive and Content stays negative.* Function peaks at Q1/Q2/Q3 and Meta at Q1/Q4/Q3 across replication/k=100/k=200. Same panels, features, and 0.2% floor as the adjusted ranking.

Voice's top-quintile effect is 0.000 / −0.012 / −0.048 across the three dictionaries, against 0.189 / 0.112 / 0.257 in the lowest quintile; Function and Meta show the same shrink-or-reverse pattern while Form stays positive in every quintile. Equal-weight aggregation over quintiles lets large low-activity Voice effects carry the aggregate Voice-first ordering, so the floor-robust claim is only that exact topical content is the least-preserved category in aggregate.

---

**Repro:** compute 3.7 GPU-h realized of 10 budgeted (GPU pod phase: SAE train + encodes + fits, 1× H100, 3.4 h; embedding pass on a fresh pod, ~0.3 h) plus ~85,000 claude-sonnet-4-6 Batch-API judge calls (all pilots passed). Code: `scripts/issue2552_turnsae_der.py` (GPU-phase driver, at commit `37b9a440bd148319e475b061fdbc93a07e19a453`), `scripts/issue2552_judge_waves.py` + `scripts/issue2552_ladder.py` at commit `cb39df3ce1cd40aee3971faafb30f16913129635`, branch `issue-2552`. Eval JSONs: [eval_results/issue_2552 tree](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552) (p1, judge_aggregates, category_reads, ladder, dere_repl subdirs; commits `714f811101`, `df309e7cd5`, `33529ec2cd`). Figures: [figures/issue_2552](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552) (18 PNG+PDF+meta triplets; 11 re-rendered at this pin with glossed configuration panel titles and ticks, plain-English covariate labels, and one color per configuration across figures). HF (verified live via scoped listings): [issue2552_turnsae tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae) with analysis_tensors subtrees sae_rep, eval, eval_lists, ladder, embcov and raw_completions subtrees mining plus judge waves w1, w1pt, w2, w3, w4, w5, w6, w7w3, w7w4, w7w5; per-feature control arms (identity+bias, train-mean, shuffle nulls) in [perfeature_rep.npz](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae/analysis_tensors/eval/perfeature_rep.npz). Reused artifacts: banked layer-19 store + rollout text from [#779](https://eps.superkaiba.com/tasks/779), capture chunks at [issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/89cfa76cdcd4207d95c1fec1c3131f36e21beec0/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture) and rollout text at [issue779_monitoring/fitter-fair-comparison-n1m/raw_completions (HF, same pin)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/89cfa76cdcd4207d95c1fec1c3131f36e21beec0/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions), both at revision `89cfa76cdc` (fit: same model/layer/rows the map was fit on); matryoshka SAE weights, per-feature R², censuses, splits from [#2476](https://eps.superkaiba.com/tasks/2476), weights at [issue2476_turnavg/analysis_tensors/sae_c (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1064dec5c572e64c947bf490e1957d05caa5d5a8/issue2476_turnavg/analysis_tensors/sae_c) and [sae_c_k200 (HF, same pin)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1064dec5c572e64c947bf490e1957d05caa5d5a8/issue2476_turnavg/analysis_tensors/sae_c_k200), splits at [split_meta (HF, same pin)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1064dec5c572e64c947bf490e1957d05caa5d5a8/issue2476_turnavg/analysis_tensors/split_meta), censuses at [floor_sweep (HF, same pin)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1064dec5c572e64c947bf490e1957d05caa5d5a8/issue2476_turnavg/analysis_tensors/floor_sweep) and [k200_census (HF, same pin)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1064dec5c572e64c947bf490e1957d05caa5d5a8/issue2476_turnavg/analysis_tensors/k200_census), all at revision `1064dec5c5`, and the banked per-feature R² committed at the body pin as [perfeature_union_c.npz](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2476/floor_sweep/perfeature_union_c.npz) and [perfeature_union_k200.npz](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2476/k200_census/perfeature_union_k200.npz) (fit: the parent instruments this task categorizes); `andyrdt/saes-qwen2.5-7b-instruct` trainer_2 at `c37e53c4bb` (fit: same layer + k as the paper's per-token SAE); trait directions from the [#779](https://eps.superkaiba.com/tasks/779) monitoring bank, fetched from [issue779_monitoring/r_b (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96/issue779_monitoring/r_b) at revision `037fcbb` (fit: layer-19 trait directions on the same base model, feeding the trait-direction-alignment covariate, a plotted ladder input). Seeds: SAE 2552, panels 14824, bootstrap/permutation per `seeds` blocks in the committed JSONs. Judge = claude-sonnet-4-6, no prefill (user override); pinned claude-sonnet-4-5-20250929 as W7 control only.

**Context:** origin prompt (verbatim): `I want to rerun their thing and then first see if our mapping predicts better the higher level things for our Matryoshka one. But I want to run one that is exactly what they run also, and then see what metrics best predict if one of the features will be predicted well. So it should be basically rerun their judgment also. So it should be their types, but also additional metrics inspired by the other experiment we ran for the SAE features. And then the plot should basically be like, okay, this is the property that best explains it, and then we control for that or partial that out and this is the next one, and then control for that as well, this is the next one. (2026-08-24; clarify-gate answers: one child task of #2476; assistant means banked; k=200 twin in, nested/attribution out; judge = sonnet-4-6 everywhere, user override; exploratory category ranking; spawn --auto)`. Child of #2476; created 2026-08-24, run 2026-08-25 (GPU pod run, judge waves W1 to W7, VM-side analysis, embedding pod), plan v5.
