---
title: Pushing a prefix's context vector through the base context-to-answer map best
  predicts where fine-tuning delivers behavior change, while base propensity keeps
  the level read (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-08-01T07:47:20Z'
has_clean_result: true
parent_id: 1900
origin_prompt: i want to test all these [full predictor roster incl. mediation checkbox]
  as well as the theory assumptions, at the per-prefix level (leakage averaged across
  queries) - RESULTS THAT RAN WERE PER QUERY. Design an experiment to do this
workflow: v1
goal: 'At per-prefix grain (leakage per (arm, destination prefix), averaged over a
  fixed shared 60-query set; 50-60 prefix panel including trained prefixes, trained
  contrastive negatives, bystanders, near-twins, real conversation and ICL prefixes),
  determine which pre-fine-tuning predictor best predicts per-prefix leakage level
  and change (full roster: context/answer similarity pre+post, delta similarities,
  through-map forms, whitened gate, r_B projections, write-map, propensity incumbent,
  kNN answer-side variant; both anchors), whether context similarity is answer-mediated
  at this grain, and whether the #1768 assumption refutations (gate A7, write rank
  A6, write direction A5) survive at the theory''s native prefix grain or were per-query
  grain artifacts.'
relates_to:
- leak-predictor
- spec-context-as-vector
- identity-contextual-vs-base
---
# Pushing a prefix's context vector through the base context-to-answer map best predicts where fine-tuning delivers behavior change, while base propensity keeps the level read (MODERATE confidence)
<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1979.md](https://github.com/superkaiba/explore-persona-space/blob/b980b88e5136523657287a8064f110952c91c2d6/docs/methodology/issue_1979.md) · [gist](https://gist.github.com/superkaiba/759f69d552d3eb8140d6f6389b561642)

## Takeaways

- Change champion at prefix grain: through-map predicted-answer similarity — median within-arm Spearman ρ 0.41 vs 0.30 for raw answer similarity; beats it in 9/12 content arms; winner probability 0.65–0.81.
- The change win is family-scoped: every casualness-persona and impoliteness arm clears its permutation band (7/12 arms); sycophancy clears 0/4, and the bare-context sycophancy arm reverses sign (ρ ≈ −0.3).
- The level race is unresolved: the read-out projection through the map (median 0.542) ties base propensity (0.536), and only at the last-prompt-token position — propensity leads at span-mean.
- The judge-free marker panel shows the same split: nearest-training-rows similarity 0.43 vs propensity 0.10 on log-probability change, 5/6 arms; attenuates to 0.23 under prefix-length partialling.
- Assumption re-tests: the whitened gate stays refuted (median 0.18) but in-band for persona-context LoRA cells; write top-1 share rises 0.09 to ~0.43; the marker write–delta opposition is weights-carried.
- Binding caveats: the change DV subtracts propensity by construction (its low change rank partly mechanical); trained-negative suppression not demonstrated — placebo cells sit equally below the geometry fit.

## Goal

- **This experiment in context:** The per-query race in [#1900](https://eps.superkaiba.com/tasks/1900) left base propensity as the level champion and raw answer similarity as the change champion, with every geometry read far behind — but the theory's gate and write objects are defined per destination context, not per query. This run re-races the full deployable predictor roster over the same 18 fleet checkpoints at per-prefix grain (50 destination prefixes × 60 shared queries; leakage = query mean) and re-tests the operator-assumption refutations from [#1768](https://eps.superkaiba.com/tasks/1768) (whitened gate, write rank, write–delta alignment) at that grain, on the per-prefix mapping objects introduced in [#722](https://eps.superkaiba.com/tasks/722). Protocol delta: the parent raced per query at the span-mean context position; this run's primary position is the last prompt token, so cross-grain comparisons here are made span-mean-to-span-mean and the per-query verdicts stand at their own grain — headline numbers across the two runs are not directly comparable otherwise.
- **Broader narrative:** Predicting fine-tuning-induced leakage from pre-fine-tuning context geometry — can a cheap read of the base model forecast where a fine-tune will deliver behavior change, and do the linear-operator assumptions behind that program hold at the grain the theory is written at?

## Methodology

**Design:** 18 reused fleet checkpoints (12 content: casual writing, impoliteness, sycophancy under persona / bare / conversation / ICL contexts, contrastive vs positive-only data regimes, LoRA vs full fine-tune; 6 marker), single seed for 16 of 18 plus an impoliteness-contrastive seed pair (42/137) as the same-cell replication read. Unit = (arm, destination prefix): 50 prefixes × 60 shared real-user queries, fully paired; every DV is the 60-query mean; the race is the within-arm Spearman rank correlation across the 50 prefixes (dose-clean — never pooled across arms). The single manipulated variable relative to the parent per-query race is the DV grain. All arms share one prefix panel and one judge instrument, so across-arm counts are correlated rather than independent confirmations. Deployable predictor roster (one number per (arm, prefix), computed from the base model only; vectors query-averaged, centered at the panel mean; anchors = training-row centroids, with the arm's own trained-prefix vectors run side-by-side):

| Predictor (plain English) | Definition (per (arm, prefix), base model) | Slug |
|---|---|---|
| Context similarity | cosine of the prefix's query-averaged context vector with the training-context centroid | `p1` |
| Answer similarity | cosine of the query-averaged base answer vector with the training-answer centroid | `p2` |
| Through-map context similarity | cosine of the mapped context vector with the map image of the training-context centroid | `p3a` |
| Through-map predicted-answer similarity | cosine of the mapped context vector with the training-answer centroid | `p3b` |
| Whitened gate | whitened dot product of the context vector with the trained-prefix context vector | `p4` |
| Read-out projection (direct) | panel-centered base answer vector projected onto the behavior read-out direction | `p5` |
| Read-out projection (through map) | panel-centered mapped context vector projected onto the read-out direction | `p6` |
| Base propensity | per-prefix mean graded score of base completions (marker: base slot log-probability) | `p7` |
| Write forecast (size / alignment) | norm of the cross-arm write-map prediction / its cosine with the read-out direction (span-mean position) | `p8a` / `p8b` |
| Nearest training rows (context) | mean top-8 cosine of the context vector to individual training-row context vectors | `p9` |
| Nearest training rows (answer) | the same read on the answer side against individual training-answer rows | `p10` |

"The map" is the base context-to-answer ridge map deterministically re-materialized from its pinned 15,000-row fit (same rows, regularization grid, and split; parity-asserted against the recorded fit quality). The behavior read-out directions are reused persona-vectors extractions: per behavior, five contrastive positive/negative persona system-prompt pairs over 20 trait-eliciting questions, ten on-policy base-model rollouts per pair and question at temperature 1.0, judge-filtered (keep positives scoring above 50 and negatives below 50; refusals dropped from both arms), residual-stream activations averaged over response tokens per kept rollout, and the direction = mean of kept positive activations minus mean of kept negative activations, one per behavior and layer. Each arm's cross-arm write map is a ridge map from base context vectors (span mean, layer 19) to the realized write — the teacher-forced answer-vector shift, trained minus base, on matched base text — fit on the three same-behavior sibling arms' rows over the same 16,400-row corpus with the target arm held out and judge rows excluded (36,400 training rows, 800 validation, the target arm's 4,000 judge rows as test); applied to a prefix's context vector it forecasts that prefix's write, and the write-forecast candidates read the forecast's norm and its cosine with the read-out direction. Post-fine-tuning and delta similarity forms plus the per-prefix write magnitude ran as a mechanistic panel that explains results but never carries a headline. Two stated descopes: the trained-prefix (panel-source) anchor variants ran side-by-side and are persisted in the per-arm frames but were not raced (the candidate set fixes the training-row-centroid anchors), and the layer-19 union-refit maps (the 15,000 bare rows plus 3,000 prefixed base rows, both positions) were fit and persisted without carrying their union-vs-bare comparison into this report.

**Training:** **N/A — no model training.** All 18 raced checkpoints are reused fleet artifacts (provenance in the footer); recipes were re-read from each checkpoint's adapter config at run start. Content LoRA checkpoints: rank 32, alpha 64, rsLoRA, seven target modules, learning rate 1e-5 (impoliteness contrastive cells 3e-5), checkpoint step 25, trained on judge-filtered on-policy behavior-expressing completions with roughly 1:1 contrastive negatives under other personas including the default assistant (positive-only cells omit the negatives). Marker checkpoints: rank 16, alpha 32, attention-only, learning rate 5e-6, marker-plus-end-of-turn loss on a programmatically appended marker token (id 83399). Full-fine-tune cells are full-parameter versions of matched cells. Base model: Qwen-2.5-7B-Instruct. Measurement-pipeline hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Judge model | `claude-sonnet-4-5-20250929` | project judge policy |
| Judge scoring | graded 0–100, reason-then-score, anchored rubric, one behavior per call; rubric texts sha-pinned from the parent run (sources linked under Evaluation) | plan §11 |
| Judge draws | 3 per completion, temperature 1.0 | plan §11 |
| Judge response budget (max_tokens) | 400 | plan §11 |
| Judge transport | Anthropic Batch API, rubric-keyed cache, resumable; 4 waves | plan §11 / run record |
| Prefix panel | 50 destination prefixes, 9 families, draw seed 1979 | plan §4 |
| Query set | 60 real-user queries, stratified from the pinned corpus val+test block, seed 1979; disjoint from the 15,000 map-fit rows | plan §11 |
| Decoding | greedy; max_new_tokens 1024 (content) / 2048 (marker) | plan §11 |
| Marker engine context window | 6144 tokens (long-prefix budget; recorded deviation) | run record |
| Layers | content 19, marker 25, fixed in advance; 14/25 exploratory dump | plan §11 |
| Context position | last prompt token primary, span mean secondary, captured together | plan §11 |
| Marker DV storage | log P(marker id 83399) at the post-response slot, trained − base, three-space four-float contract | plan §11 |
| Bootstrap | 2,000 draws × 3 families (prefix resample primary; query cluster; family cluster), winner re-selected per draw | plan §11 |
| Permutation null | 1,000 within-arm draws, per-draw max over the 12 raced candidates | plan §11 |
| Map re-materialization | ridge on the pinned 15,000 base rows, same regularization grid and split; parity within 0.01 R² of the recorded fits | plan §11 |
| Whitened gate | corpus second moment from the 15,000 bare rows, shrinkage 0.1; 0.3–0.7 band | plan §11 |
| Write-rank criterion | top-1 SVD variance share vs 0.6 | plan §11 |
| Alignment nulls | corpus-covariance + isotropic norm-matched, 2,000 draws; disjoint even/odd query halves per leg | plan §11 |
| Nearest-rows pools | top-8 of 20 anchor rows per mix (4/16 in the exploratory dump) | plan §11 |
| Marker-checkpoint identity gate | slot-read reproduction within 1.8 nats for the marker full-fine-tune cell (tolerance lowered from the plan default; recorded deviation) | run record |

**Evaluation:** The content level DV is on-policy: per (arm, prefix), the mean over 60 queries of the graded judge score of the arm's own greedy completion under that prefix; the change DV is trained minus base per prefix; the binary companion is the share of (query × draw) scores of at least 50. The marker DV is judge-free: the query-mean change in log-probability of the marker token at the end of the model's own response, with logit-margin and probability companions stored from the same forward pass. Base propensity is the same read on the base model and serves as both the incumbent predictor and the partial-out covariate. The three content rubrics are the parent race's instruments verbatim, assembled by [the parent judge script @ 8c840dbe9d](https://github.com/superkaiba/explore-persona-space/blob/8c840dbe9d853cc63cd84886554213f20e0491ea/scripts/issue1900_judge.py): the impoliteness and casual-writing rubric texts verbatim from [the behavior registry @ 8c840dbe9d](https://github.com/superkaiba/explore-persona-space/blob/8c840dbe9d853cc63cd84886554213f20e0491ea/src/explore_persona_space/artifacts/behavior.py), the sycophancy rubric from the persona-vectors trait description in [the shared trait module @ 8c840dbe9d](https://github.com/superkaiba/explore-persona-space/blob/8c840dbe9d853cc63cd84886554213f20e0491ea/scripts/issue779_common.py) rendered through the same anchored template; per-arm rubric sha256 digests sit in the [judge drop report @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/blob/02aaaa37200faa33fce329a5fd464359a8d112df/eval_results/issue_1979/judge/drop_report.json). The instrument branch was fixed before the data were seen: the wave-1 pilot found the level DV floor-heavy (6 and 9 of 50 prefixes reach 10 points in the two pilot arms) while the change DV keeps range (per-prefix SDs 5.6 and 4.5), so the change DV carries the content headline and the level race is the secondary, floor-heavy read; base-side ceiling share was 0.0 and the ceiling-excluded re-read flagged zero prefixes. Dual-DV status is carried honestly: the parent line's external teacher-forced-margin check failed panel-wide at per-prompt grain, so the graded instrument's support here is internal — design-aligned even/odd split-half reliability ceilings per family (marker 0.86–0.97, casual writing 0.88–0.93, impoliteness 0.74–0.91, sycophancy 0.59–0.77; sycophancy correlations are ceiling-capped) plus graded–binary concordance and the judge-free marker family replicating the race structure — which caps confidence at MODERATE by design. Judge health: content drops at most 3.03% per judged unit (casual-writing rubric ~2.9–3.0% including base 2.4% — judge refusals, symmetric across model states within the rubric, so rubric-driven rather than arm-selection censoring; impoliteness 0.36–0.69%; sycophancy ~2%); zero transport losses. Language-intrusion audit (Qwen under a non-CJK eval, both substrates): judged pools carry 51–105 intruded rows of 3,000 per model state (impoliteness contrastive ~3.5%, all others ~1.8%, base content 57 of 3,000); marker generation rollouts 58 of 3,000. The base rate is query-driven — one CJK-eliciting query accounts for ~50 of the ~57 base rows, identically across states — while the impoliteness-contrastive excess is training-associated (a second query rises from 2 base rows to 9–13). Both components are conclusion-neutral: excluding intruded rows leaves per-prefix level ranks essentially unchanged (rank agreement ρ ≥ 0.98; largest per-prefix mean shift ≤ 1.4 points of 100), flipping no adjudication. Selection discipline: every champion is a max over 12 candidates, so winners are re-selected inside every bootstrap draw and the permutation null takes the per-draw max; selection-inherited and frozen-at-winner intervals are both persisted and labeled. The permutation band upper edges (≈ 0.37–0.41 per arm) sit ~0.6 below the reachable |ρ| = 1 ceiling, so the null band is informative. Every fitted or re-materialized map reports the identity-plus-learned-bias baseline and nearest-neighbor retrieval; both mapping arms ran — the context-based arm is the predictor panel itself (its candidates are context-side reads), and the prefix-based arm is the leave-one-family-out fit reported in Results. A post-review follow-up round (commit deaa8e00b2) re-read both change champions under a rank-based partial Spearman on prefix content-token length — the same primitive as the level-race length-robustness read — n = 50 per arm, all 18 arms.

**Data extraction:** The prefix panel spans 9 families: each arm's own trained prefix, the trained contrastive-negative personas, bystander personas, the established 50-context battery families, near-twin personas (6 Sonnet-written, template-matched), real WildChat conversation prefixes at 8 length targets, in-context-learning prefixes, and the bare default assistant. The fresh in-context-learning prefixes' few-shot demonstrations pair corpus train-block questions with the base model's own banked greedy completions (Qwen-2.5-7B-Instruct, read from the pinned base-corpus capture shard the footer names) — the demonstration answers are base-model text, with no new generation. Trained and negative prefix renders were byte-asserted against the pinned training-mix rows before any GPU spend. Queries come from the pinned 16,400-prompt real-user corpus (tier-1 realism), val+test block — train-disjoint from the 15,000 rows the context-to-answer map was fit on but drawn from the same distribution (in-distribution eval; scope caveat). Captures per (model state × prefix × query): greedy on-policy generation with raw text persisted; context activations at both positions plus answer span means at layers 14/19/25; and matched-text teacher-forced trees of each trained model over the base rows for the weights-carried write reads. Judge wave 2 rode the remaining base rubrics (recorded deviation from the plan's wave table — same instrument, different batching).

Weights-vs-text decomposition round (`marker-a5-weights-vs-text`, plan v6): one added capture leg teacher-forces the base model over the six marker arms' stored on-policy generations (18,000 forwards = 6 arms × 50 prefixes × 60 queries; layers 14/19/25, response-span means accumulated in fp64), giving per-prefix means of base-model activations on trained text. Each arm's on-policy delta then splits exactly as weights = trained-on-trained-text minus base-on-trained-text and text = base-on-trained-text minus base-on-base-text, pivoted on the odd query half so the text term shares no sampling error with the even-half write leg; the odd-pivot primary was fixed before the run per the plan-review Must-Fix, and the all-pivot variant is recorded as a sensitivity only. A second recorded sensitivity re-runs the split excluding rows that contain the marker token — dropped from the two trained-text stores but not the base-text baseline (5–14 of 3,000 rows per arm, ~0.3%). A component carries the opposition when its pooled cosine with the write leg is negative, clears the corpus-covariance norm-matched null band (2,000 draws), and reaches at least 0.5× the full-delta magnitude (factors 0.25 and 0.75 recorded). The primary read is layer 19 span mean (the parent alignment convention); layers 14/25 are recorded exploratory and show the same signs. Parity gates pass in all six arms: re-derived write–delta cosines match the persisted battery values exactly (tolerance ±0.005) and re-derived means agree at cosine ≥ 0.99954 (floor 0.999). Trained-vs-base text-identity fractions run 0.11–0.24 per arm (token-id equality) — most rows differ textually, context for the small different-text component norms. Provenance: capture + decomposition code at fix commit c5c772f646, results at 69f82df137, fellows SLURM job 17656.

**Sample training/evaluation data + completions:** Disclosure: 3 of the ~6,000 judged completions across both model states shown — two cherry-picked score extremes plus one degenerate row from the seed-42 random spot check; excerpts are sanitized to ≤15 words (real-user-corpus content hygiene). Full raw completions: [issue1979_prefixrace/raw_completions @ 15b0011e8b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/15b0011e8bf460c96cf58a1c651591b9a7a24ecd/issue1979_prefixrace/raw_completions); per-arm score tables (all rows, all draws): [eval_results/issue_1979/judge @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/tree/02aaaa37200faa33fce329a5fd464359a8d112df/eval_results/issue_1979/judge); frozen prefix panel + query set: [eval_results/issue_1979/config @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/tree/02aaaa37200faa33fce329a5fd464359a8d112df/eval_results/issue_1979/config).

```
1. impolite persona contrastive (seed 42), prefix f1_phub_04, query sha fd687edf7221432a,
   score 98.7 (draws 100/98/98):
   "Your friends are terrible at this. Stop wasting my time. Do exactly what"
   [truncated — real-user-corpus row; verify at raw_completions/generation/, sha above]

2. impolite persona contrastive (seed 42), prefix nt_data_engineer (near-twin family),
   query sha 75d97d3511dacd70, score 0.0 (draws 0/0/0):
   "Je comprends votre frustration, je vous assure..."
   [truncated — polite French-language reply; the query corpus is multilingual]

3. sycophancy persona positive-only (seed 42), prefix wildchat_prefix_real545,
   query sha 62d73d1f7c4a9f23, score 0.0:
   completion is the single word "negative" (degenerate one-word generation; judged 0)
```

All three blocks are sanitized for context hygiene; labels, prefix ids, and query shas are verbatim from the judge tables. The marker family generates text only as a carrier for the slot read (its DV is a log-probability, not judged text), so no marker completion sample is shown.

I acknowledge the conciseness WARNs this body fires — the total-prose budget (eleven results: nine from the primary round plus the length-partial and weights-vs-text follow-up rounds) and per-result 120-word overruns — accepted to carry the full verdict lattice with its binding caveats; likewise the figure-text WARN on the three per-unit scatters, whose point labels are prefix ids (verbatim unit identifiers from the frozen panel, not condition codes). The learning-rate reconciliation WARN is also acknowledged: the rates above are reused-checkpoint recipe values re-read from each adapter's config — this task trained no model, so the plan carries no training learning rate of its own.

## Results

### Through-map predicted-answer similarity wins the change race, carried by two of three content behaviors

Bars: probability that each of 12 predictors wins the leakage-change race across the 12 content arms (winner re-selected inside each of 2,000 bootstrap draws; three resampling families). Below, the per-unit companion: per-prefix change (trained − base judge points) against the champion for a pair of arms, n = 50 each.

![Winner probability per predictor for the leakage-change race under prefix-resample, query-cluster, and family-cluster bootstraps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/change_race_winner_bars.png)

> **Figure.** *The through-map predicted-answer similarity wins the change race in all three bootstrap families.* Winner probability 0.655 (prefix resample, primary), 0.806 (query cluster), 0.649 (family cluster); across-arm median ρ 0.411 vs 0.301 for raw answer similarity, beaten paired within-arm in 9 of 12 content arms; selection-inherited CI on the winning median 0.27–0.56 (frozen-at-winner 0.23–0.55).

![Per-unit companion scatter: per-prefix leakage change against the through-map predicted-answer similarity for the impolite persona contrastive and sycophancy bare-context arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/scatter_change_p3b.png)

> **Figure.** *Per-unit companion: the win is family-scoped.* Per-prefix change vs the champion — impolite persona contrastive ρ +0.42 (p = 0.003), one of 7 of 12 arms clearing its own permutation band (band upper edges 0.37–0.41), beside the bare-context sycophancy arm's sign reversal, ρ −0.32 (p = 0.025). Points colored by prefix family; trained and negative prefixes labeled.

Every casualness-persona and impoliteness arm clears its band; sycophancy clears none, its bare-context arm negative across all leading candidates (−0.32 to −0.37).

Two cautions bind. The champion rank-correlates 0.90 with raw answer similarity, so per-arm margins are modest — the 9-of-12 paired count plus three consistent bootstrap families carry the claim. And the change DV subtracts propensity by construction, so propensity's poor change rank (median −0.11) is partly mechanical.

### The level race is unresolved: the through-map read-out projection ties base propensity, and only at the last-token position

The heatmaps give within-arm Spearman ρ per (predictor × content arm) for the level and change DVs, across-arm medians at right, per-query-grain medians (parent run, span-mean position) as gray left marks; the bar panel gives level-race winner probabilities. Below, the per-unit companion for one strong arm: per-prefix level against the two tied candidates.

![Predictor-by-arm correlation heatmaps for level and change with level-race winner-probability bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/hero_content_race.png)

> **Figure.** *No resolved level champion.* Across-arm medians: read-out projection through the map 0.542 vs base propensity 0.536; winner probability splits 0.567 (propensity) to 0.405 (projection) under the primary bootstrap, so the observed argmax is not a resolved winner. Selection-inherited CI on the max median 0.46–0.68.

![Per-unit companion scatter: per-prefix leakage level against the through-map read-out projection and base propensity for the impolite persona positive-only arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/scatter_level_p6_p7.png)

> **Figure.** *Per-unit companion.* In the impolite persona positive-only arm the projection reads ρ +0.46 (p = 0.0008) against propensity's +0.22 (p = 0.12); points colored by prefix family, trained and negative prefixes labeled, n = 50.

The tie is position-dependent: at span-mean the projection drops to 0.318 while propensity (position-free by construction) stays 0.537. The level DV is also floor-heavy (6–9 of 50 prefixes reach 10 points in the pilot arms), so the change DV carries the content headline.

Still, a pure-geometry deployable read now matches the incumbent at the primary last-prompt-token position — per-query grain never got geometry above 0.24 against propensity's 0.63, span-mean-matched. Prefix-length partialling leaves the top three level candidates intact; the change champions' partials are below (content survives, marker attenuates).

### Judge-free marker replication: nearest-training-rows geometry beats propensity on marker log-probability change

Heatmap and bars as above for the 6 marker arms on the marker change DV (per-prefix query-mean Δ log P of the marker token at the slot, trained − base, layer 25). Below, the per-unit companion for the marker persona contrastive arm.

![Predictor-by-arm correlation heatmap and winner-probability bars for the marker log-probability change race](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/hero_marker_race.png)

> **Figure.** *Nearest-training-rows context similarity leads the marker change race.* Across-arm medians 0.428 vs 0.372 (answer similarity) and 0.095 (base propensity); it beats propensity in 5 of 6 marker arms; winner probability 0.558 prefix-resample, 0.920 query-cluster, 0.695 family-cluster; selection-inherited CI 0.30–0.59.

![Per-unit companion scatter: per-prefix marker log-probability change against nearest-training-rows context similarity and base propensity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/scatter_marker_p9.png)

> **Figure.** *Per-unit companion.* In the marker persona contrastive arm the champion reads ρ +0.55 (p = 3e-05) against its permutation band edge 0.395, while propensity reads +0.06 (p = 0.69); n = 50 prefixes.

The marker family is the judge-free replication panel; with the primary winner probability marginal at 0.558, it reads as replication of the race structure rather than a standalone second verdict. The same coupling caveat applies — the marker change subtracts the base slot log-probability, which is propensity itself — and the marker level read stays propensity-dominated (0.82 in the persona-contrastive arm).

At span-mean, raw context similarity (0.435) matches the champion (0.428; its anchor pools are position-invariant), so the champion's margin there is last-token-scoped. Six marker arms share one recipe class and one panel — correlated confirmations. Prefix-length partialling attenuates this champion further (next result).

### Length partialling spares the content change champion but halves the marker one

Dumbbells: within-arm Spearman ρ of each family's change champion — through-map predicted-answer similarity (content, top) and nearest-training-rows context similarity (marker, bottom) — raw beside the rank-based partial on prefix content-token length, one row per arm, n = 50 prefixes each.

![Raw versus length-partialled Spearman correlation dumbbells for the content and marker change champions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2c667dc33b3c81bf91b014f9610510909ffcf8f2/figures/issue_1979/length_partial_followup.png)

> **Figure.** *Content survives, marker attenuates.* Content median 0.410 raw → 0.432 partialled, per-arm deltas small and mixed-sign; marker median 0.428 → 0.231, the three strong marker arms dropping from 0.49–0.55 to 0.22–0.25 — still positive, roughly halved. n = 50 prefixes per arm.

Prefix length is a partial confound behind the marker headline, not the content one. The marker champion is nearly collinear with prefix length in every marker arm (|ρ| 0.89–0.96), so the residual 0.22–0.25 in the strong arms is the length-independent component of that geometry signal.

The content champion is only moderately length-loaded (ρ −0.50 to −0.61), the change DV barely at all (small, mixed-sign), and partialling nudges the content median up rather than down. The partial is the same rank-based primitive as the level-race robustness read.

### What context similarity knows about per-prefix leakage is carried by answer-side similarity

Each content arm contributes four dots — partial Spearman correlations with the per-prefix level DV (n = 50 per arm), base propensity always conditioned.

![Partial correlation forest for context and answer similarity across the 12 content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/mediation_forest.png)

> **Figure.** *Answer similarity absorbs context similarity, not the reverse.* Across-arm medians: the context-similarity partial collapses from 0.167 to −0.079 once answer similarity is added, while the answer-similarity partial survives, 0.344 to 0.274; base propensity conditioned throughout.

The collapse passes the halving criterion, and the verdict's two support checks agree: conditioning on the through-map form absorbs context similarity the same way (−0.209 vs 0.586 in the casualness contrastive arm), and the disjoint even/odd anchor-half recount preserves the pattern. Context–answer rank correlation runs 0.62–0.89 across arms (mean 0.71), below the collinearity fallback trigger, and the partials are well-posed in 12 of 12 arms.

The small negative residual context partials (down to −0.31 in some arms) are descriptive at n = 50. This is consistent with a context quantity mapped into answer space winning the change race.

### The whitened gate stays refuted overall but tracks realized writes for persona-context LoRA cells

One dot per (content arm × layer), 36 cells: Spearman ρ between the whitened-gate prediction and the realized per-prefix matched-text write coefficient, against the 0.3–0.7 acceptance band and the per-query anchor 0.14.

![Gate-versus-realized-write correlations per arm and layer against the acceptance band](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/a7_gate.png)

> **Figure.** *Median 0.175; 14 of 36 cells in the 0.3–0.7 band — the gate criterion (median in band and at least half the cells) still fails.* Positive-only cells land in band at all three layers (0.44–0.58), casualness contrastive at all three, impoliteness contrastive at layer 19 for both seeds (0.42/0.40); full fine-tune, bare-context, and conversation cells sit near zero.

Prefix grain barely moves the pooled read (per-query anchor 0.14), so the refutation stands — but the pooled criterion hides real structure: the gate works where the write comes from a persona-context LoRA adapter and fails for full fine-tunes and bare/conversation contexts. Adapter type tracks the in-band cells; the data-regime split (positive-only vs contrastive) does not — a read off the realized 36-cell grid, not a split fixed in advance.

### Per-prefix writes are far closer to rank-1 than the per-query read suggested, still short of the 0.6 criterion

Plotted per arm: the top-1 SVD variance share of the centered 50-prefix write matrix, matched-text and on-policy trees at layers 14/19/25, against the 0.6 criterion and the per-query reference values.

![Top-1 singular-value share of the per-prefix write matrix per arm, both trees, three layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/a6_top1_share.png)

> **Figure.** *Median matched-text top-1 share ≈ 0.43 (range 0.30–0.64) vs the per-query references 0.09 (matched) and 0.29 (on-policy).* Two of 18 arms clear 0.6 at the primary layer: casualness positive-only (0.643) and the marker full fine-tune (0.605; 0.703 at layer 19). On-policy-tree shares run lower (0.10–0.46).

Averaging over queries removes most query-idiosyncratic rank: the per-prefix write is far closer to rank-1 than the per-query read suggested, without reaching the 0.81–0.86 shares recorded for the earlier fixed-panel runs the assumption came from. The 50-row matrix caps rank at 50 by construction, but the top-1 share is scale-free, so the 0.6 criterion stays well-posed.

### Write–delta alignment: positive for the style behaviors, strongly negative for the marker family

The panel plots the pooled cosine between the matched-text write leg and the on-policy delta leg (disjoint query halves per leg), one row per arm, against corpus-covariance norm-matched null bands; the known-invalid shared-baseline read is shown record-only.

![Write-delta alignment per arm with norm-matched null bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/a5_alignment.png)

> **Figure.** *All 18 arms clear the null band (absolute 95th percentiles 0.03–0.43), with family-consistent signs.* Casualness +0.35 to +0.68 and impoliteness +0.51 to +0.80 positive; every marker arm negative, −0.63 to −0.99; sycophancy mixed (conversation −0.91, positive-only −0.33, bare and full fine-tune +0.53/+0.64).

The expectation that this alignment stays null (text-carried change) is refuted in both directions. The candidate mechanical reading of the marker sign — emitted-marker token content in the on-policy delta that the base-text teacher-forced write cannot carry — is decomposed in the next result and fails. Alignment with the behavior read-out direction is family-ordered too: impoliteness 0.30–0.48, casualness 0.19–0.43, sycophancy and marker near zero.

The legs re-operationalize the per-query construction (disjoint-half matched-text write vs on-policy delta rather than the training-mix direction), so the comparison is a re-operationalization; a strict re-test of the per-query construction remains open (scope caveat).

### The marker write–delta opposition is weights-carried, not a text-selection artifact

Each marker arm's on-policy delta splits exactly into a weights component — trained-minus-base activations, teacher-forced on the trained model's own generations — and a different-text component — the base model's shift from swapping base for trained text; plotted: each component's pooled cosine with the matched-text write leg beside the full-delta read and corpus-covariance null bands, then the per-prefix cosines (n = 50 per arm).

![Write-alignment of the weights and different-text delta components per marker arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/69f82df137339f279b0ca5094adf3eb6d274e2a2/figures/issue_1979/a5_decomposition.png)

> **Figure.** *The opposition lives in the weights component.* Its pooled cosine with the write leg runs −0.72 to −0.99 (median −0.85), clearing every corpus-covariance null band (absolute 95th percentiles 0.12–0.25) and meeting the carry criterion at every recorded factor; the different-text component sits at +0.48 to +0.54 and carries in none. Verdict: weights-carried, 6 of 6 arms.

The text-selection reading dies: swapping the trained model's own text into the base model moves activations with the write direction, and weakly (pooled norms 0.55–1.83 against 5.34–5.74); the same weight update, read on base text versus the model's own text, produces opposite-pointing shifts. The marker write direction genuinely opposes its realized on-policy delta — a violation the theory must absorb.

The verdict survives marker-row exclusion and both recorded carry factors. The all-pivot split reverses the attribution but anti-shares baseline sampling error with the write leg — the bias the odd-pivot split, fixed before the run, avoids.

### Trained negatives sit below the geometry fit — and so do the placebos: no contrastive-specific suppression demonstrated

Dots: signed residuals of the 5 trained-contrastive-negative prefixes against each arm's own within-arm geometry fit (dots = prefixes, diamonds = arm medians), grouped contrastive vs positive-only placebo vs bare-context placebo.

![Signed residuals of trained-negative prefixes versus each arm's geometry fit, grouped by arm type](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/h5_residuals.png)

> **Figure.** *Negative medians in 6 of 7 contrastive content arms and 4 of 4 marker contrastive arms — but the placebo arms are equally negative.* Casualness placebo −3.45 vs contrastive −3.66 (bare placebo −5.75); impoliteness −2.70 vs −2.66/−4.11; marker magnitudes ≤ 0.1 nats with the bare-context marker placebo (−0.40) below every marker contrastive arm.

The suppression criterion technically passes, but the placebo clause fixed in the plan fires too and controls: the contrastive-minus-placebo median difference is ≈ 0 for three of four behaviors (−0.07 casualness, −0.01 impoliteness, −0.08 marker) and −1.02 only for sycophancy, the panel's lowest-reliability family (its conversation arm even flips positive, +0.67). On the change DV the bare casualness placebo (−5.59) is more negative than the contrastive cells (−0.63 to −0.90).

Geometry over-predicts leakage into assistant-adjacent, low-elicitation prefixes generically; the persisted summary file's suppression label keys only on the pass criterion and is corrected here.

### A prefix-to-answer map is fittable at this grain, and the shifted-identity baseline fails (exploratory)

Bars: leave-one-family-out ridge R² per (behavior family × layer × context position) for the prefix-based mapping arm, beside the identity-plus-learned-bias baseline, on a symlog axis; nearest-neighbor retrieval accuracy annotated.

![Leave-one-family-out mapping fits versus the identity-plus-bias baseline across layers and positions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979/mapping_arm.png)

> **Figure.** *Held-out R² 0.51–0.56 at layers 14/19 (0.22–0.29 at 25); retrieval accuracy up to 0.20 at rank 1 (chance 0.02) and 0.64 at rank 5 (chance 0.10); the identity-plus-learned-bias baseline reads −36 to −4,640.* n = 50 prefixes, dimension 3,584.

This is the line's first identifiable prefix-based mapping arm. Prefix and answer summaries share dimension but not location, so the shifted-identity baseline fails, consistent with the location mismatch. Every fit runs in the n < d regularization-limit regime and carries the exploratory label — no headline rests on these fits; the context-based mapping arm is the predictor panel itself.

---
**Repro:** Compute: fellows SLURM lane, 8× H200 (cluster charmander), 31/31 GPU unit-passes across 6 dispatch rounds on 2026-08-01 (final job 16960; plan budget ~29 GPU-h); judging via the Anthropic Batch API in 4 waves (zero GPU); statistics + figures on the VM. Code: [scripts @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/tree/02aaaa37200faa33fce329a5fd464359a8d112df/scripts) (`issue1979_prep.py`, `issue1979_gpu.py`, `issue1979_judge.py`, `issue1979_race.py`, `issue1979_figs.py`, `issue1979_dispatch.sh`); race statistics ran at commit [8c840dbe9d](https://github.com/superkaiba/explore-persona-space/tree/8c840dbe9d853cc63cd84886554213f20e0491ea), recorded in each race JSON's meta. Eval artifacts: [eval_results/issue_1979 @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/tree/02aaaa37200faa33fce329a5fd464359a8d112df/eval_results/issue_1979) (race statistics + bootstrap/permutation matrices, judge tables, gate-1 verdict, frozen config); figures + point-data sidecars: [figures/issue_1979 @ 02aaaa3720](https://github.com/superkaiba/explore-persona-space/tree/02aaaa37200faa33fce329a5fd464359a8d112df/figures/issue_1979). Length-partial follow-up (post-review round; figure relabeled plain-English in the review round): statistics [eval_results/issue_1979/race/length_partial_followup.json @ 2c667dc33b](https://github.com/superkaiba/explore-persona-space/blob/2c667dc33b3c81bf91b014f9610510909ffcf8f2/eval_results/issue_1979/race/length_partial_followup.json), script [scripts/issue1979_length_partial_followup.py @ 2c667dc33b](https://github.com/superkaiba/explore-persona-space/blob/2c667dc33b3c81bf91b014f9610510909ffcf8f2/scripts/issue1979_length_partial_followup.py), figure + sidecar [figures/issue_1979 @ 2c667dc33b](https://github.com/superkaiba/explore-persona-space/tree/2c667dc33b3c81bf91b014f9610510909ffcf8f2/figures/issue_1979). Weights-vs-text decomposition follow-up (round `marker-a5-weights-vs-text`, plan v6; fellows SLURM job 17656 on charmander, 3 GPU-h budget): statistics [eval_results/issue_1979/race/a5_decomposition.json @ 69f82df137](https://github.com/superkaiba/explore-persona-space/blob/69f82df137339f279b0ca5094adf3eb6d274e2a2/eval_results/issue_1979/race/a5_decomposition.json), figure + point-data sidecar [figures/issue_1979 @ 69f82df137](https://github.com/superkaiba/explore-persona-space/tree/69f82df137339f279b0ca5094adf3eb6d274e2a2/figures/issue_1979), capture + decomposition code (`issue1979_gpu.py` phase `f1g`, `issue1979_race.py --decomp-only`) at fix commit [c5c772f646](https://github.com/superkaiba/explore-persona-space/tree/c5c772f646fa926d0a54394e9c8e2c494d08f46a/scripts); base-model teacher-forced stores (6 arms, 2.3 GB) [issue1979_prefixrace/stores/basetf_onpolicy @ 656bbccd06](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/656bbccd06ebee110a646739da351d89949d6608/issue1979_prefixrace/stores/basetf_onpolicy) and means bundle (51.7 MB) [issue1979_prefixrace/battery/basetf_decomp_inputs.pt @ 656bbccd06](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/656bbccd06ebee110a646739da351d89949d6608/issue1979_prefixrace/battery/basetf_decomp_inputs.pt). Raw completions, activation stores, marker slot floats, predictor tables, battery inputs, and the race bootstrap upload: [issue1979_prefixrace @ 15b0011e8b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/15b0011e8bf460c96cf58a1c651591b9a7a24ecd/issue1979_prefixrace). Reused artifacts:

- Reused 18-checkpoint panel + arm config from [#1900](https://eps.superkaiba.com/tasks/1900): `issue1900_leakrace/config/arms.json` @ `3bb20debe2e68392897d6144b9180c8748c7afcb` (LoRA adapters on `superkaiba1/explore-persona-space`, full fine-tunes on the private overflow repo) — fit: identical panel makes the DV grain the single manipulated variable.
- Reused cross-arm write maps from [#1900](https://eps.superkaiba.com/tasks/1900): `issue1900_leakrace/maps/wmap_*_L19.pt` @ `3bb20debe2e68392897d6144b9180c8748c7afcb` — fit: the deployable write-forecast objects that run validated (span-mean position only); construction recipe inline under Design.
- Reused judge rubric texts from [#1900](https://eps.superkaiba.com/tasks/1900): assembled by `scripts/issue1900_judge.py` @ `8c840dbe9d853cc63cd84886554213f20e0491ea` — impoliteness + casual writing verbatim from `src/explore_persona_space/artifacts/behavior.py`, sycophancy from the persona-vectors trait description in `scripts/issue779_common.py` — fit: same judge instrument as the parent race; per-arm rubric sha256 digests in `eval_results/issue_1979/judge/drop_report.json`.
- Reused base-corpus stores, last-token stores, training-mix anchors + delta legs, and prefix renders from [#1768](https://eps.superkaiba.com/tasks/1768): `issue1768_mapshift/{corpus_capture,lasttoken_ctx,delta_tf}` @ `c07267285d2cdbf3e0401ddc3e3accae50e496a7` (last-token stores at re-pool commit `c7a5fda6d1`) — fit: the map re-materialization inputs and gate objects this battery re-tests.
- Reused behavior read-out directions from [#1112](https://eps.superkaiba.com/tasks/1112), [#1315](https://eps.superkaiba.com/tasks/1315), [#1434](https://eps.superkaiba.com/tasks/1434): `issue1112_geometry2x2/analysis_tensors/rb/` @ `5f110f0a2181b2e7d8fb344a742dccdcd7fa02c4`, `issue1315_impolite_geometry/analysis_tensors/rb/` @ `5f110f0a2181b2e7d8fb344a742dccdcd7fa02c4`, `issue1434_writingstyle/analysis_tensors/rb_writing_style.pt` @ `5f110f0a2181b2e7d8fb344a742dccdcd7fa02c4` — fit: persona-vectors-recipe extraction line, one direction per behavior and layer; extraction recipe inline under Design.
- Reused prefix-family battery from [#658](https://eps.superkaiba.com/tasks/658)/[#810](https://eps.superkaiba.com/tasks/810): `issue658_theory_assumptions/answer_position_sweep/inputs/battery50.json` @ `5f110f0a2181b2e7d8fb344a742dccdcd7fa02c4` — fit: established 50-context, 7-family panel backbone.
- Reused query corpus + splits from [#1768](https://eps.superkaiba.com/tasks/1768): `issue1768_mapshift/inputs/corpus_sample.json` @ `c07267285d2cdbf3e0401ddc3e3accae50e496a7` — fit: the pinned corpus whose val+test block keeps DV queries disjoint from map-fit rows.

**Context:** Originating prompt (verbatim, frontmatter record):

> i want to test all these [full predictor roster incl. mediation checkbox] as well as the theory assumptions, at the per-prefix level (leakage averaged across queries) - RESULTS THAT RAN WERE PER QUERY. Design an experiment to do this

Longer form in the task's Provenance record:

> i want to test all these: RESULTS THAT RAN WERE PER QUERY WHICH IS TERRIBLE - What is the best leakage predictor: [context/answer vector cosine similarity pre/post finetuning; similarity between change in context/answer vector pre and post finetuning; context vector -> apply mapping -> predicted answer vector similarity pre/post finetuning; Is leakage due to similarity of mean answer vectors and so context vector similarity is just byproduct; Any other suggestions?] As well as the theory assumptions. At the per-prefix level. Design an experiment to do this

Lineage: [#1900](https://eps.superkaiba.com/tasks/1900) — parent (the per-query leakage-predictor race; the grain extension changes its Goal, hence this child). Created 2026-08-01; run 2026-08-01 (GPU + judge waves) → 2026-08-02 (statistics, figures, write-up). Same-issue follow-up round `marker-a5-weights-vs-text` run 2026-08-02 (proposer-initiated cheap band, plan v6; proposal title verbatim: "Weights-vs-text decomposition of the marker family's negative write–delta alignment").
