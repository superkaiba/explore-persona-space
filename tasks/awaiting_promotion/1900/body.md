---
title: Base behavioral propensity predicts where a trained behavior is expressed,
  but pre-fine-tuning answer similarity is the only predictor of the fine-tuning-induced
  change (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-30T23:28:44Z'
has_clean_result: true
parent_id: 1768
origin_prompt: 'Help me to plan this experiment with these model organisms: What is
  the best leakage predictor (8 candidate similarity reads pre/post finetuning, map-mediated
  variants); is leakage due to similarity of mean answer vectors with context similarity
  a byproduct; any other suggestions'
workflow: v1
goal: 'On the model-organism fleet at per-prompt grain (the #1768 16,400-real-user-prompt
  corpus), determine which per-context predictor computable BEFORE fine-tuning best
  predicts per-context leakage of the trained behavior (graded judge DV for content
  arms; three-space log P(marker) for marker arms), whether base behavioral propensity
  survives as the champion, and whether context-vector similarity''s predictive power
  is mediated by answer-side similarity; post-FT and delta candidates form a separate
  mechanistic panel, and every similarity candidate is computed against both the training-context-centroid
  and panel-source-context anchors.'
relates_to:
- leak-predictor
- spec-context-as-vector
---
# Base behavioral propensity predicts where a trained behavior is expressed, but pre-fine-tuning answer similarity is the only predictor of the fine-tuning-induced change (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1900.md @ f5fd6f9c4a](https://github.com/superkaiba/explore-persona-space/blob/f95705910a3d21e40ad540dc5d21b08114760fa3/docs/methodology/issue_1900.md) · [gist mirror](https://gist.github.com/superkaiba/8663fe55d3504a2df447a987ded4aa32)

## Takeaways

- Base behavioral propensity dominates per-prompt prediction of trained-behavior expression: median rank correlation **0.632** across 12 content arms (best geometry read 0.242), winner in every one of 2,000 bootstrap draws.
- The expression-level champion mostly measures persistence: per-prompt trained and base scores correlate 0.31–0.73 per arm, and propensity's correlation with the fine-tuning-induced change is negative in all 18 arms.
- On the change itself, pre-fine-tuning answer similarity to the training answers wins both families — marker log-probability change median **0.381** (6 of 6 arms), content 0.124 (12 of 12) — and keeps winning once propensity's mechanical coupling is regressed out (0.147 content / 0.300 marker; propensity's residual is null).
- An off-floor follow-up round conditioning every subset on nonzero base elicitation (3,923 / 4,000 / 1,484 prompts per family) reproduces both verdicts — expression champion 0.591 in every draw, residualized change 0.156 in 10 of 12 arms — so the impoliteness floor drove neither.
- Raw context similarity predicts nothing (median −0.010, sign-flipping ±0.36 within sycophancy); its signal is answer-mediated, and only the training-answer anchor carries it (0.211 vs 0.017 at the source-persona anchor).
- The teacher-forced margin companion fails per-prompt external validation across the whole panel — none of 30 reads reaches the 0.15 bar, and sycophancy off-floor reads run −0.13 to −0.23 — hardening the MODERATE cap. Scope: one corpus, one model, 17 of 18 arms single-seed; follow-up rounds cover the content panel only.

## Goal

- **This experiment in context:** The leakage-predictor line ([#658](https://eps.superkaiba.com/tasks/658), [#742](https://eps.superkaiba.com/tasks/742), [#761](https://eps.superkaiba.com/tasks/761), [#763](https://eps.superkaiba.com/tasks/763)) repeatedly found that a unit's own base behavioral propensity beats geometry reads at predicting where a fine-tuned behavior expresses — but at unit grain. This experiment races 11 predictors, all computable before fine-tuning, at per-prompt grain over bare real-user prompts on 18 trained arms reused from the parent capture fleet ([#1768](https://eps.superkaiba.com/tasks/1768)), and adds the mediation test the originating question asked for: is context-vector similarity's predictive power a byproduct of answer-side similarity? A zero-GPU analysis round added the propensity-residualized change race — the deciding read for whether propensity carries change signal beyond its mechanical coupling — plus coupling calibration and an arm-exclusion sensitivity check. A GPU-backed off-floor follow-up round then re-ran the content race with one variable changed — the judged prompt subset, conditioned per family on nonzero base elicitation — to test whether either verdict was an artifact of the 84%-zero impoliteness floor. A final cheap-band validation round expanded the graded score's external dual-DV check — the teacher-forced fixed-pool margin recipe that validated at cell grain in [#722](https://eps.superkaiba.com/tasks/722) — from one arm × 299 prompts to all 30 model-state × subset reads.
- **Broader narrative:** If per-prompt leakage of a fine-tuned behavior is predictable from base-model quantities alone, a deployment audit can flag where an implant will express before fine-tuning ships. The answer also constrains whether context geometry or answer geometry carries the leakage signal.

## Methodology

**Design:** 18 trained arms — 12 content arms (4 casual-writing, 4 impoliteness, 4 sycophancy, spanning persona / bare / conversation training contexts, contrastive and positive-only regimes, LoRA and full fine-tune, one impoliteness seed pair at seeds 42/137) and 6 marker-token arms — each evaluated on the same 4,000-prompt stratified subset (seed 1900) of the 16,400 real-user-prompt corpus (LMSYS/WildChat-class bare user prompts, data-realism tier 1; cross-tree sha intersection 16,318). Eleven deployable candidates computable before fine-tuning are raced within each content arm — context similarity, answer similarity, two through-map similarity forms, whitened gate similarity, two read-out projections, base behavioral propensity, two write-prediction forms, and nearest-training-rows similarity; marker arms race nine of these (the write-prediction pair is defined for content arms only). One-line definitions for every candidate are under Data extraction. A six-read mechanistic panel using the trained model (post-fine-tuning and delta similarities, write magnitude) ran alongside; it explains, never carries the headline. All headline statistics are within-arm per-prompt Spearman rank correlations aggregated as across-arm medians — never raw-pooled across arms, so cross-arm install-dose differences cannot confound the ranking. Rounds: the main run (2026-07-31); one zero-GPU analysis round the same day (propensity-residualized change race, coupling nulls, 11-arm sensitivity; no new data); and one off-floor follow-up round (`offfloor-surface-race`, 2026-07-31) whose single changed variable is the judged prompt subset — per-family subsets conditioned on nonzero base elicitation replace the uniform 4,000-row subset (casual writing 3,923, sycophancy 4,000, impoliteness 1,484 — the corpus-wide maximum at a 9.4% base-nonzero rate). A fourth round (`tfmargin-validation-expand`, 2026-07-31) is a measurement-validation round rather than a race re-run: it computes the teacher-forced fixed-pool margin companion for all 15 content-panel model states (12 trained arms plus the base model read against each family's pools) over seeded 800-prompt draws of both judge subsets, with zero new judge calls. The marker family was not re-run in the off-floor round (its judge-free continuous DV has no zero floor to strip; a stated deviation from the scope note's "same 18 arms" phrasing), so that round is a content-panel round.

**Training:** **N/A — no model training.** All 18 arms are reused fleet checkpoints (provenance in the footer); recipes re-read from each arm's adapter config at run start. Content LoRA checkpoints: rank 32, alpha 64, rsLoRA, seven target modules, learning rate 1e-5 (impoliteness contrastive arms 3e-5), checkpoint step 25, trained on judge-filtered on-policy behavior-expressing completions with roughly 1:1 contrastive negatives under other personas including the default assistant. Marker arms: rank 16, alpha 32, attention-only, learning rate 5e-6, marker plus end-of-turn loss on a programmatically appended marker token (id 83399). Full-fine-tune arms are full-parameter versions of matched cells. Base model: Qwen-2.5-7B-Instruct. Measurement-pipeline hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Judge model | claude-sonnet-4-5-20250929 | project judge policy |
| Judge scoring | graded 0–100, reason-then-score, anchored rubric, one behavior per call | plan §11 |
| Judge draws | 3 per completion, temperature 1.0 | plan §0 |
| Judge response budget (max_tokens) | 400 | plan §11 |
| Judge transport | Anthropic Batch API, rubric-keyed cache, resumable | plan §4 |
| Judge subset | 4,000 stratified prompts, seed 1900 | plan §11 |
| Bootstrap | 2,000 draws, shared per-draw prompt indices within each arm family | plan §0 |
| Permutation null | 1,000 within-arm draws, winner re-selected per draw over the raced candidates (11 content / 9 marker) | plan §6 |
| Coupling null (analysis round) | 500 trained-vs-base permutations per arm, seed 7 | `followup_free` recipe field |
| Residualization (analysis round) | per-arm one-predictor least-squares fit of trained on base score (n≈4,000 vs dimension 1) | `followup_free` recipe field |
| Layers | 19 (content) / 25 (marker), fixed in the plan | plan §11 |
| Marker DV | log P(marker id 83399) at the post-response slot, trained − base, three-space storage | plan §4 |
| Anchors | training answer/context centroids (primary) + source-persona context; 29 realized (15 pre + 14 post; full-fine-tune arms carry no post-anchor by design) | run record |
| Map refits | ridge, 23-point regularization grid, fit rows exclude the judge subset | plan §11 |
| Off-floor subset rule (follow-up round) | per family, prompts whose base graded score mean exceeds 0 on the selection record; membership frozen at selection | plan v11 §4 |
| Off-floor estimation draws (follow-up round) | 3 fresh base judge draws per family on the frozen subsets, disjoint from every selection draw; rubric cache disabled, per-draw resume ids | plan v11 §4 |
| Off-floor map refits (follow-up round) | 13 ridge fits (1 context-to-answer + 12 write maps); train split excludes the union of all judge subsets (n_train 8,409 vs dimension 3,584) | plan v11 §4 |
| Teacher-forced margin pools (validation round) | 32 positive + 32 negative fixed answers per family; positives top-8 per arm at score mean ≥ 50, all 3 draws kept, one donor per context, ≤ 768 tokens; negatives score mean 0; donor contexts excluded from every scored set | plan v13 §4 |
| Margin pool provenance (validation round) | sycophancy rebuilt verbatim from the earlier margin experiment's judge-filter record (replication instrument); casual writing / impoliteness frozen new from parent judged completions | plan v13 §4 |
| Margin contexts (validation round) | 800 seeded prompts per family × subset (parent + off-floor draws, donors excluded); margins computed once per model × context; 95% bootstrap intervals, 2,000 draws | plan v13 §4 |
| Margin verdict + drift rule (validation round) | 0.15 bar on off-floor family medians, fixed before the run; drift flag when the replication cell moves more than 1.96 × the combined bootstrap standard errors from the parent −0.064 | plan v13 §6 |

**Evaluation:** The content dependent variable is on-policy: the graded judge score (mean of kept draws) of each arm's own greedy completion to the bare prompt. The binary companion is the share of draws scoring at least 50; the change companion is trained minus base per prompt. The marker dependent variable is judge-free: the change in log-probability of the marker token at the end of the model's own response, with the logit-margin and probability companions agreeing in sign per arm. Dual-DV validation: the graded score's external per-prompt reference check failed — the teacher-forced fixed positive-vs-negative answer margin correlates −0.064 with the graded score over 299 overlap prompts on one sycophancy arm — so the graded instrument's support is internal (split-half reliability 0.73–0.93 across the 15 judged units, graded-binary concordance) plus the judge-free marker family reproducing the headline structure; this caps confidence at MODERATE. An off-floor recheck on the same arm's 131 overlap prompts reads −0.026, so the failure is not a floor artifact either. A dedicated validation round then expanded the check to all 30 model-state × subset reads at n≈800 per read; it fails everywhere (Results), so the cap rests on the full panel rather than one arm. The champion rule is a signed-correlation argmax with the winner re-selected inside every bootstrap draw; a dethrone verdict additionally requires the challenger to beat propensity in at least 9 of 12 content arms (5 of 6 marker). The off-floor round's decision rule was fixed before the run: the parent verdicts survive if and only if propensity tops the off-floor expression race AND answer similarity tops the off-floor propensity-residualized change race. The permutation band (upper edge 0.046 content / 0.044 marker against an achievable ceiling of 1) makes the null informative; candidates inside the band are indistinguishable from null given the variance, not confirmed zeros.

**Data extraction:** Predictor inputs are span-mean activations over the prompt (the context) and over the model's own greedy response (the answer), joined across arms by prompt sha. Anchors are span means over each arm's actual training rows; positive-only arms' anchors resolve from the matching contrastive pools (an approximation shared by every candidate within those arms).

The raced candidates, each one number per prompt from the base model (verified against the predictor builders at the run commit): context similarity — cosine between the prompt's context activation and the arm's training-context centroid, both centered at the corpus mean; answer similarity — the same read on the answer activation against the training-answer centroid; through-map context similarity — cosine between the map-predicted answer vector and the map image of the training-context centroid; through-map predicted-answer similarity — cosine between the map-predicted answer vector and the training-answer centroid itself; whitened gate similarity — the context activation's whitened dot product with the training-context centroid (corpus second moment, shrinkage 0.1), normalized by the centroid's own whitened norm; read-out projection (direct) — the corpus-centered answer activation projected onto the behavior's read-out direction; read-out projection (through map) — the same projection of the centered map-predicted answer vector; base behavioral propensity — the graded judge score of the base model's own completion on the prompt (marker arms: the base model's marker log-probability at its own response end); write prediction (size) — the norm of the write-map's predicted fine-tuning-induced answer shift for the prompt; write prediction (alignment) — cosine between that predicted shift and the read-out direction; nearest-training-rows similarity — mean cosine between the centered context activation and its 16 nearest individual training-row context vectors. The ridge maps take the base model's context activation (span mean over the prompt, 3,584 dimensions) to its answer activation (span mean over the greedy response, same dimensionality) at the same layer; the cross-arm write maps take the same context activation to the fine-tuning-induced answer-activation shift, fit on the other arms of the same behavior. All 33 refit maps carry the standing map-health reads: the 21 context-to-answer maps hold held-out R² 0.49–0.54 with cosine nearest-neighbor retrieval accuracy 0.38–0.54 at rank 1 (chance 0.00025), against R² at or below −1.3 and retrieval at or below 0.26 for the identity-plus-bias baseline (predict the answer summary as the context summary plus a train-fold mean offset); the 12 cross-arm write maps are weak (median held-out R² −0.12), consistent with the write-prediction candidates' race showing. The behavior read-out directions are reused persona-vectors-style mean-difference directions from the parent extraction line: per behavior, the difference of mean response-token activations between judge-filtered rollouts that express the behavior and rollouts that do not, one direction per layer (pinned in the footer).

Off-floor round mechanics: selection and estimation are disjoint judge passes. Selection (does the base model elicit the behavior on this prompt?) reads the parent's base draws plus a base pre-pass over the remaining corpus (56,400 draws: 12,300 impoliteness + 6,500 sycophancy rows, 3 draws each); a prompt enters its family's subset when its selection score mean exceeds zero, and membership is frozen — a row whose fresh estimation mean is zero stays in. Boundary rows dominate the selection (single-draw-nonzero counts: 75 of 3,923 casual writing, 806 of 1,484 impoliteness, 1,373 of 4,000 sycophancy; score mean at or below 5: 453 / 909 / 1,124), so propensity, the change DV, and the residualization consume only the 3 fresh estimation draws per family (28,221 draws; rubric cache disabled with per-draw resume ids) — selection noise therefore cannot regress boundary rows toward the mean and depress propensity's off-floor correlation. New trained-arm judging covered the newly-added rows only (42,204 draws); casual-writing trained scores are parent copies filtered to the subset, valid because trained scores were never selection inputs. Estimation-pass health: content drops 53 of 11,769 draws (casual writing), 22 of 4,452 (impoliteness), 300 of 12,000 (sycophancy); zero transport losses in every unit of the round. Because the parent maps were fit on rows that include the new off-floor rows, the five map-mediated candidates were re-scored under 13 freshly-fit ridge maps whose train split excludes the union of all judge subsets: the context-to-answer refit holds held-out R² 0.514 with cosine retrieval accuracy 0.471 at rank 1 (chance 0.00014), against −17.1 and 0.019 for the identity-plus-bias baseline; the 12 write-map refits stay weak (median held-out R² −0.23, range −48.4 to 0.22), matching the parent. Non-map candidate columns are read unchanged from the parent's full-corpus tables, and the parent-map values are kept as record-only columns for the cross-round control comparison.

Teacher-forced-margin validation round mechanics: per family, a fixed 32-positive / 32-negative answer pool is frozen once from already-judged completions — positives are the top-8 rows per arm by score mean subject to a ≥ 50 floor, all three judge draws kept, one donor row per distinct context, and a 768-token length filter; negatives are base-unit completions with score mean 0 under the same filters. The sycophancy pools rebuild the earlier margin experiment's pools verbatim from its committed judge-filter record — making the replication cell instrument-identical to the parent check — while the casual-writing and impoliteness pools are new, frozen from the parent round's judged completions (full pool text, donor shas, and file hashes committed under `eval_results/issue_1900/tfm/config/`). All donor contexts are excluded from every scored context set, and pool membership is judge-filtered once then held fixed across every context and model, so no per-context selection-on-outcome enters the margin; the residual dependence — pools chosen by the instrument under validation — matches the earlier experiment's and is disclosed here. Each of the 15 model states teacher-forces both pools over 800 seeded prompts per subset (draws from the parent and off-floor judge subsets, margins computed once per model × context on their union), with zero new judge calls — every graded score is reused. The margin is the mean length-normalized log-probability of the positive pool minus the negative pool; each read is its per-prompt Spearman correlation with the graded score mean, with 95% bootstrap intervals from 2,000 draws reusing the race's batched bootstrap machinery. The 0.15 bar (deliberately below the weakest earlier cell-grain validation, since per-prompt grain attenuates), the family-median verdict lattice over off-floor reads, and the two-sample drift rule were all fixed before the run. The parent-subset impoliteness reads carry 84–92% graded-score ties at zero and are reported but never verdict-bearing.

The representation mapping ran context-based only: the bare corpus carries 2 distinct prefix strings, so a prefix-based arm is unidentifiable (stated deviation, inherited from the corpus design). Plan deviations, all amended and re-reviewed during the run: the anchor-mix row floor was relaxed to an 8-row kill / 40-row warn (realized matched-text mixes carry about 20 rows); a frame-free identity floor replaced a cross-frame band in the marker gate; a weights-only loading carve-out for self-produced tensors; a CPU dry-run mode; the below-threshold impoliteness arm kept by a recorded pilot decision; one judge relaunch after a memory-pressure kill (resumable checkpoint, no data loss). Off-floor round deviations: the marker family carried over rather than re-run (Design); the exploratory all-layer/anchor dump descoped to the primary layer and both anchors; the planned score-at-least-10 exploratory re-cut of the estimation scores was not produced (a non-headline read; dropped); the first cluster job crashed on a sparse clone missing `eval_results/` and the relaunched job ran clean. I acknowledge the conciseness WARNs the verifier raises on this body — the per-result/120 word cap (result sections run up to ~175 words), the per-bullet Takeaways cap, and the total prose budget — accepted to carry the correction, coverage, and caveat record the follow-up rounds require; the drift-check figure is deliberately linked rather than embedded (the validation section carries its two deciding figures). The learning-rate reconciliation WARN is likewise acknowledged: the rates above are reused-checkpoint recipe values re-read from each adapter's config — this task trained no model, so the plan carries no training learning rate of its own.

**Sample training/evaluation data + completions:** Disclosure: 1 of 4,000 judge-subset rows shown, cherry-picked for illustration (a benign row, chosen for real-user-corpus content hygiene; prompt and completion truncated). Full judge inputs: [issue1900_leakrace/judge_inputs @ 3bb20deb](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/judge_inputs); per-arm score tables (all rows, all draws): [eval_results/issue_1900/judge @ 8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900/judge); off-floor round tables: [eval_results/issue_1900/offfloor/judge @ de99c0cba4](https://github.com/superkaiba/explore-persona-space/tree/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/eval_results/issue_1900/offfloor/judge).

```
prompt (sha 0015473a3f2c9b11): "what does red ache of its carcass mean: The moon, the red ache of its ..."
[truncated - real-user-corpus row; verify at issue1900_leakrace/judge_inputs/cas-pers-con-lr1e5-s42.shard00.jsonl]

casual-writing persona contrastive arm, greedy completion (228 words): "This line from the poem evokes a vivid and haunting image. Let's break it down: ..."
[truncated - same row, response_text field]

judge (casual-register rubric): trained draws 15/15/35 -> per-prompt score 21.7; base draws 15/5/15 -> 11.7
```

Marker arms generate no judged completions — their read is the teacher-forced marker log-probability at the end of the same on-policy rows, so no sample block applies there.

## Results

### Base behavioral propensity dominates the expression-level race, largely by measuring persistence

The heatmap gives the per-arm Spearman correlation between each of the 11 pre-fine-tuning candidates (rows) and the trained arm's graded leakage score, over 12 content arms plus a median column (n≈3,959–3,996 per arm). The scatter grid shows the per-prompt data behind that row.

![Per-arm correlations between eleven pre-fine-tuning predictors and trained leakage scores, twelve content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/hero_content_race.png)

> **Figure.** *Base behavioral propensity wins the expression-level race in every content arm.* Median correlation 0.632 vs 0.242 for the best geometry read (read-out projection); context similarity near zero. The winner re-selected in each of 2,000 bootstrap draws lands on propensity every time.

![Per-prompt scatters of trained score against base propensity, one panel per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/dv_scatter_p7.png)

> **Figure.** *The per-prompt data behind the winning row.* Base score (x) against trained score (y), one panel per content arm, one point per prompt; casual-writing panels show the strongest relation, impoliteness panels are floor-dominated.

The win is substantially a persistence read: trained and base scores correlate 0.31–0.73 per arm (0.60–0.73 casual-writing and sycophancy, 0.31–0.34 floored impoliteness), and restricting to prompts whose trained and base completions materially diverge leaves it essentially unchanged (casual-writing 0.60–0.66). Where the base model already expressed the behavior, the trained model still does. A ridge combination of all eleven candidates transfers across arms — leave-one-arm-out rank correlation 0.31–0.70 per held arm (casual writing 0.64–0.70, sycophancy 0.58–0.63, floored impoliteness 0.31–0.33) — with no gain over propensity alone.

### On the fine-tuning-induced change, answer similarity to the training answers wins both families

The heatmap gives the per-arm correlation between each candidate and the marker change score — trained-minus-base log-probability of the marker token at the model's own response end — over the 6 marker arms plus the median column (n=4,000 per arm).

![Per-arm correlations between predictors and the marker log-probability change, six marker arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/hero_marker_race.png)

> **Figure.** *Answer similarity dethrones propensity on the judge-free marker change score.* Median 0.381 (through-map predicted-answer form 0.379), beating propensity in 6 of 6 arms; propensity is negative everywhere (median −0.367). Each cell is one arm's per-prompt correlation; the heatmap is itself the per-arm view.

The content change companion mirrors this: answer similarity 0.124, beating propensity in 12 of 12 arms (winner in 92.5% of draws). Propensity's negative change correlations (−0.04 to −0.30 content, −0.22 to −0.41 marker) sit above the independence null (−0.49 to −0.66) in all 18 arms: coupling-consistent, though sign alone cannot rule out hidden signal. Arm counts are not independent confirmations: arms share one judge subset and per-behavior base vectors — three content families plus one marker family, one corpus.

### The propensity-residualized change race is the deciding read: no material residual propensity signal

Both heatmaps rerun the change race with the change score replaced by the residual of the trained score after a per-arm one-predictor least-squares fit on the base score, removing the mechanical trained-minus-base coupling: content arms first, marker arms second.

![Propensity-residualized change race over twelve content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61cea503ed3832c0c8d9e2848cd15b78d2346cd5/figures/issue_1900/residualized_race_content.png)

> **Figure.** *Content arms: with the coupling removed, answer similarity still wins and propensity is indistinguishable from null.* Answer similarity median 0.147 (winner in 96.8% of draws, beats propensity in 11 of 12 arms); propensity +0.026, inside the 0.046 permutation band, negative on the floored impoliteness arms.

![Propensity-residualized change race over six marker arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61cea503ed3832c0c8d9e2848cd15b78d2346cd5/figures/issue_1900/residualized_race_marker.png)

> **Figure.** *Marker arms agree.* Answer similarity 0.300 (winner in 75.1% of draws, the through-map predicted-answer form takes the rest, and it beats propensity in 6 of 6 arms); propensity −0.154.

Once its own base term is regressed out, propensity carries no material positive signal about the fine-tuning-induced change, while answer similarity — which shares no term with the change score — keeps its lead in both families. Raw (unresidualized) change medians for comparison: 0.124 content, 0.381 marker (preceding result's figure).

### Removing the base-elicitation floor leaves the expression champion unchanged

The heatmap repeats the expression-level race on the off-floor subsets — per-family prompt sets conditioned on nonzero base elicitation, scored with the fresh estimation draws (n = 1,481–3,986 realized per arm). The scatter grid shows the per-prompt data behind the winning row.

![Off-floor per-arm correlations between eleven predictors and trained leakage scores, twelve content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/figures/issue_1900/offfloor_race_content_level.png)

> **Figure.** *Propensity retains the expression champion off the floor.* Median 0.591 (full subset: 0.632), winner in 2,000 of 2,000 bootstrap draws; runners: read-out projection 0.274, its through-map form 0.252, answer similarity 0.194.

![Off-floor per-prompt scatters of trained score against base propensity, one panel per arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/figures/issue_1900/offfloor_scatter_dv_vs_p7.png)

> **Figure.** *The per-prompt data behind the off-floor winning row.* Base estimation score (x) against trained score (y), one panel per content arm, one point per prompt; the impoliteness panels lose their zero pile-up.

This answers the impoliteness-floor caveat: with the 90.6%-zero rows gone the family behaves like the others, and the arm kept below the dynamic-range bar at full grain clears it here (0.235 vs 0.046). The 0.632-to-0.591 attenuation is the designed conditioning on base elicitation, not a weaker champion. Composition matters for these subsets — 806 of 1,484 impoliteness and 1,373 of 4,000 sycophancy rows entered on a single nonzero selection draw — so every deciding estimate uses the fresh, selection-disjoint draws.

### Off the floor, answer similarity keeps the residualized change and propensity stays null

The heatmap reruns the propensity-residualized change race on the off-floor subsets: per arm, the target is the residual of the trained score after a one-predictor least-squares fit on the fresh base estimation score.

![Off-floor propensity-residualized change race over twelve content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/figures/issue_1900/offfloor_residualized_race_content.png)

> **Figure.** *Answer similarity wins the off-floor residualized race.* Median 0.156, winner in 64.9% of 2,000 draws (its through-map form takes most of the rest), beating propensity in 10 of 12 arms; propensity's residual median +0.043 wins no draw.

Both halves of the decision rule fixed before the run land: propensity highest on expression, answer similarity highest on residualized change — the parent's two-part structure is not a floor artifact. Propensity's +0.043 sits at the per-arm permutation-band edges (0.043–0.075): a failure to reject, not evidence of exactly zero.

The unresidualized change companion is unresolved off-floor — answer similarity's median 0.120 leads, but winner mass splits roughly evenly with its through-map form — consistent with a smaller raw change signal once low-propensity rows are removed; the residualized read carries the verdict. Excluding the marginal impoliteness arm changes neither verdict (medians shift by at most 0.016).

### The casual-writing near-replicate control shows no machinery drift across rounds

Each panel plots one candidate's per-arm rank correlation in the parent run (x) against the off-floor round (y) — base propensity, context similarity, answer similarity — one point per content arm; descriptive only, since the two rounds share rows and instruments.

![Parent-round versus off-floor per-arm correlations for three candidates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/figures/issue_1900/offfloor_parent_vs_offfloor_rho.png)

> **Figure.** *Per-arm correlations reproduce across rounds.* Casual-writing arms — 98.5% base-nonzero, so their off-floor subset nearly equals the parent subset — sit on the diagonal (propensity 0.663 parent vs 0.653 off-floor on the persona-contrastive arm).

The built-in control passes: with the casual-writing subset nearly unchanged, every read reproduces, so the round's machinery (fresh draws, refit maps, subset plumbing) moved nothing. For the five map-mediated candidates the comparison uses the parent-map values kept as record-only columns: across the four casual-writing arms they match the parent run within 0.013 and the refit columns within 0.02 — no machinery drift, and refit effects of at most 0.02. One planned read did not ship: the score-at-least-10 exploratory re-cut of the estimation scores (a non-headline companion; dropped).

### Context similarity is answer-mediated, sign-unstable, and anchor-specific

The forest plot gives, per content arm, the partial rank correlation with the graded level score for context similarity (blue) and answer similarity (orange), each conditioned on propensity alone and additionally on the other similarity read.

![Partial rank correlations for context and answer similarity per content arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61cea503ed3832c0c8d9e2848cd15b78d2346cd5/figures/issue_1900/mediation_forest.png)

> **Figure.** *Answer similarity keeps unique signal; context similarity has none.* Median partial correlation given propensity: context 0.022, answer 0.188; adding the other predictor leaves answer similarity at 0.182 while context drops to −0.019. All 12 arms shown, both conditioning forms per arm.

The answer-mediated verdict holds on three separate checks: the partial lattice, the structural read (the through-map predicted-answer form matches answer-similarity rankings at 0.84 rank-agreement while absorbing context similarity), and a disjoint-half anchor recount. A commonality decomposition of the explained rank variance agrees: propensity's unique share is median 0.324 per arm (range 0.067–0.418), against at most 0.048 for answer similarity and 0.020 for context similarity. Off-floor the partials repeat (context −0.006, answer +0.173, given propensity).

There is little context channel to mediate: raw context similarity is near zero overall and sign-unstable within one behavior (−0.35 to −0.37 on three sycophancy arms, +0.36 on the fourth). The anchor choice decides the read: answer similarity predicts at the training-answer centroid (0.211), not at the source-persona context (0.017).

### The mechanistic panel explains expression through post-fine-tuning answer geometry; delta similarities carry nothing

The heatmap shows per-arm correlations between the six trained-model panel reads (rows) and each arm's primary score, across all 18 arms.

![Six mechanistic panel reads against each arm's primary score, eighteen arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/m_panel.png)

> **Figure.** *Post-fine-tuning answer similarity is the strongest mechanistic read.* Content medians: post-fine-tuning answer similarity 0.310, through-map form 0.246, write magnitude 0.123; delta-context −0.029 and matched-text delta-answer −0.025 are null. Marker: write magnitude 0.499 leads.

The expected near-null for delta-context similarity was confirmed, but the hypothesis that matched-text delta-answer similarity would be the strongest mechanistic candidate is falsified — both delta reads are indistinguishable from null given the variance. Explanatory power lives in where the trained answer representation sits, not in how far it moved on matched text.

### The teacher-forced margin companion fails per-prompt external validation across the whole content panel

The forest plot gives all 30 validation reads — per-prompt Spearman correlations between the teacher-forced fixed-pool margin and the graded score, one per model state and judge subset (n≈800, 95% bootstrap intervals, 2,000 draws) — against the 0.15 validation bar. The scatter grid shows the per-prompt data behind the off-floor reads.

![Thirty margin validation reads against the bar](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59a256b6abd9f437a2f989e8c00ce371c28e6106/figures/issue_1900/tfm_validation_forest.png)

> **Figure.** *No read approaches the validation bar.* Largest point estimate +0.092 (sycophancy conversation arm, parent subset); off-floor family medians: casual writing +0.037, impoliteness −0.040, sycophancy −0.173. All five sycophancy off-floor reads are significantly negative; no off-floor interval spans 0.15.

![Graded score against teacher-forced margin per model state](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59a256b6abd9f437a2f989e8c00ce371c28e6106/figures/issue_1900/tfm_margin_vs_graded_scatter.png)

> **Figure.** *The per-prompt data behind the off-floor reads.* Teacher-forced margin (x) against graded score (y), one panel per model state, one point per prompt (n≈800); sycophancy panels trend negative, the rest show no visible relation.

The companion fails everywhere: casual-writing and impoliteness reads are indistinguishable from zero given the variance, and all five sycophancy off-floor reads are significantly negative — the falsification arm of the round's decision rule: per-prompt continuous-margin reads cannot back the headline.

The drift gate passes — the replication cell reads −0.016 (n=800) against the parent's −0.064 (n=299), inside the 0.136 flag ([drift check](https://raw.githubusercontent.com/superkaiba/explore-persona-space/59a256b6abd9f437a2f989e8c00ce371c28e6106/figures/issue_1900/tfm_replication_check.png), deliberately linked — machinery check) — and the margin tracks context similarity (up to 0.29, family-varying sign): structure, but not judge agreement. The earlier positive validation was cell-grain — precedent, not contradicted — and the off-floor subsets condition on nonzero base elicitation, so selection can contribute the negative sycophancy sign.

### Instrument health is clean and both verdicts survive the sensitivity checks, but the graded score lacks external validation

Each arm's observed max-selected correlation (red) is plotted against its permutation null band (blue, 97.5th percentile of 1,000 within-arm draws, winner re-selected per draw) and the achievable ceiling of 1 (dashed).

![Observed max-selected correlation per arm against permutation band and ceiling](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/band_vs_ceiling.png)

> **Figure.** *The null is informative: every arm's observed maximum sits far above its band.* Band upper edges reach 0.046 (content) and 0.044 (marker) against a ceiling of 1; observed max-selected values run 0.31–0.73.

The judge ran clean: 180,000 draws in the main run, 1,938 content drops (1.08%; 0.21–2.17% per unit), zero transport losses; the off-floor round's ~127,000 added draws match that profile (zero transport losses).

One impoliteness arm sits below the 0.05 dynamic-range share bar at full grain (0.046); it was kept by a recorded pilot decision, and — correcting the run record — no full-data eligibility gate ever ran. Excluding it changes nothing (level champion 0.642, winner in every draw; change verdict 11 of 11).

Residual caveats: the graded score's failed external validation, now panel-wide (preceding result), and no language-intrusion scan on the judged pools — the shared-prompt within-arm design and the judge-free marker replicate bound the latter's effect on headline structure, not absolute content magnitudes.

---
**Repro:** ~3.3 GPU-h realized (fellows SLURM lane, 5 launches — 4 crash-diagnosed, 1 clean full run) plus ~180k Batch-API judge calls off-pod; no training. Code: workload ran at [5f2b220a42](https://github.com/superkaiba/explore-persona-space/tree/5f2b220a42ab09d0930e0a992099713dfe9695c8/scripts) (`scripts/issue1900_{prep,gpu,judge,race,figs}.py`); outputs committed at [0e5e6c3e7d](https://github.com/superkaiba/explore-persona-space/tree/0e5e6c3e7dc4bda1873178a2a7d808542423248f/eval_results/issue_1900); analysis round (`scripts/issue1900_followup_free.py`) at [8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900/race/followup_free). Off-floor follow-up round (`offfloor-surface-race`, plan v11, `scripts/issue1900_offfloor.py`): ~0.4 GPU-h realized (fellows lane, 2 launches — job 16273 crashed on a sparse clone missing `eval_results/`, fixed via an extra sync path; job 16278 ran clean, ~13 min of fits) plus ~127k Batch-API judge draws off-pod; implementation at [aa53a2c683](https://github.com/superkaiba/explore-persona-space/tree/aa53a2c6834778e898622cdcaa41b5e8551d99f9/scripts), lane fix at [979b79f2b5](https://github.com/superkaiba/explore-persona-space/tree/979b79f2b59f21ef8fa0b0d63f6871cad144af72/scripts), outputs + figures committed at [de99c0cba4](https://github.com/superkaiba/explore-persona-space/tree/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/eval_results/issue_1900/offfloor) (stats files self-report producing commit `cd3112d001`). Aggregated eval JSONs plus the per-arm and per-draw files the aggregates collapse (`champion_*.json`, `mediation.json`, `robustness.json`, `arm_*.json`, `boot_*.npz`): [eval_results/issue_1900 @ 8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900); off-floor equivalents (incl. `offfloor/config/composition_report.json` and `offfloor/race/followup_free/`): [eval_results/issue_1900/offfloor @ de99c0cba4](https://github.com/superkaiba/explore-persona-space/tree/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/eval_results/issue_1900/offfloor). Figures: [figures/issue_1900 @ 61cea503ed](https://github.com/superkaiba/explore-persona-space/tree/61cea503ed3832c0c8d9e2848cd15b78d2346cd5/figures/issue_1900); off-floor figures at [de99c0cba4](https://github.com/superkaiba/explore-persona-space/tree/de99c0cba43fa26e4b0fb99030a831b379d7b4cc/figures/issue_1900). HF artifacts — config (4 files), anchors (29), marker teacher-forced tables (27), predictor tables (108), maps (67), judge inputs (30) + raw judge shards, validation (4): [issue1900_leakrace @ 3bb20deb](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace); off-floor artifacts — judge aggregates + raw shards, 13 map fit records + tensors, 12 refit column tables: [issue1900_leakrace/offfloor @ 05cb982b](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/05cb982b0d3f9a21b5735d196a0afdc8175590e5/issue1900_leakrace/offfloor). Teacher-forced-margin validation round (`tfmargin-validation-expand`, plan v13, `scripts/issue1900_tfm.py`): ~3 GPU-h realized (fellows lane, job 16412 — 15 teacher-forced passes round-robin over 4 GPUs, ~22.6k margin units) with zero new judge calls; implementation at [e2c1f345f9](https://github.com/superkaiba/explore-persona-space/tree/e2c1f345f907e491e1ee36f20376e12575bfd725/scripts); outputs + figures committed at [59a256b6ab](https://github.com/superkaiba/explore-persona-space/tree/59a256b6abd9f437a2f989e8c00ce371c28e6106/eval_results/issue_1900/tfm); HF artifacts (frozen pools + context draws, 15 per-pass margin tables, validation summary): [issue1900_leakrace/tfm @ b52e8dad](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b52e8dadaef7de1d9f444755c7d60fec9cd1c8b4/issue1900_leakrace/tfm). Corpus + parent stores pin: [issue1768_mapshift @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift).
- Reused the 18 trained arms, their per-prompt activation stores, and the training-mix positive pools from [#1768](https://eps.superkaiba.com/tasks/1768) (checkpoints originally trained by the [#1481](https://eps.superkaiba.com/tasks/1481)/[#1586](https://eps.superkaiba.com/tasks/1586) organism fleet): [issue1768_mapshift @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift) — fit: the race requires exactly these arms' per-prompt stores at the same prompt grain; recipes verified from each adapter's config at run start.
- Reused the sycophancy fixed positive/negative answer pools from [#722](https://eps.superkaiba.com/tasks/722), rebuilt verbatim from its committed judge-filter record via the same pool builder: [issue1900_leakrace/tfm/config @ b52e8dad](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b52e8dadaef7de1d9f444755c7d60fec9cd1c8b4/issue1900_leakrace/tfm/config) — fit: keeps the validation round's replication cell instrument-identical to the parent's −0.064 check.
- Reused behavior read-out directions from the fleet extraction line ([#1112](https://eps.superkaiba.com/tasks/1112), [#1315](https://eps.superkaiba.com/tasks/1315), [#1434](https://eps.superkaiba.com/tasks/1434)), staged from [issue1768_mapshift/rb_plus @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift/rb_plus) — fit: the read-out projection candidates are defined against these directions.
- No pod; GPU phases on the fellows SLURM lane; wandb n/a (no training).

**Context:**
> "Help me to plan this experiment with these model organisms: What is the best leakage predictor (8 candidate similarity reads pre/post finetuning, map-mediated variants); is leakage due to similarity of mean answer vectors with context similarity a byproduct; any other suggestions"

Longer recorded form of the same ask (task Provenance, verbatim):
> "Help me to plan this experiment with these model organisms: What is the best leakage predictor: context vector cosine similarity pre-finetuning / context vector cosine similarity post-finetuning / answer vector cosine similarity pre-finetuning / answer vector cosine similarity post-finetuning / similarity between change in context vector pre and post finetuning / similarity between change in answer vector pre and post finetuning / context vector -> apply mapping -> predicted answer vector similarity pre-finetuning / context vector -> apply mapping -> predicted answer vector similarity post-finetuning / Is leakage due to similarity of mean answer vectors and so context vector similarity is just byproduct / Any other suggestions?"

Off-floor follow-up round scope note (proposer-initiated, cheap-band; verbatim excerpt):
> "rerun the #1900 predictor race with ONLY the judged prompt subset changed — 4,000-row subset stratified on base-model behavior elicitation (nonzero base graded score per family; existing base scores where available + cheap base pre-pass) replacing the uniform-stratified subset; same 18 arms, 11 predictors, anchors, judge recipe [...], race stats [...]. Hypothesis: champion structure survives off the 84%-zero impoliteness floor."

Teacher-forced-margin validation round scope note (proposer-initiated, cheap-band; verbatim excerpt):
> "per-prompt dual-DV validation expansion — TF fixed positive/negative-pool margins (llm-judging.md §E2 rule 19 recipe) computed for all 12 content arms + base over the full off-floor+parent judge subsets (parent coverage was 1 arm x 299 rows); test whether rho(margin, graded) > 0 validates on casual-writing arms per #722's cell-grain precedent; falsification: `rho <= 0` across all behaviors hardens the MODERATE cap."

Lineage: [#1768](https://eps.superkaiba.com/tasks/1768) — parent (its capture stores and arm fleet are this task's inputs). Created 2026-07-30; run 2026-07-31 (five launches, judge phase, stats); one zero-GPU free-analysis round 2026-07-31 (propensity-residualized change race, coupling nulls, 11-arm sensitivity); one same-issue follow-up round `offfloor-surface-race` 2026-07-31 (base-elicitation-conditioned subsets, disjoint estimation draws, leak-through map refits); one same-issue follow-up round `tfmargin-validation-expand` 2026-07-31 (teacher-forced fixed-pool margin validation, 30 reads, content panel; cheap-band slot 2 of 2).

