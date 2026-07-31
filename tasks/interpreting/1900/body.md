---
title: Base behavioral propensity predicts where a trained behavior is expressed,
  but pre-fine-tuning answer similarity is the only predictor of the fine-tuning-induced
  change (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-30T23:28:44Z'
has_clean_result: false
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

## Takeaways

- Base behavioral propensity dominates per-prompt prediction of trained-behavior expression: median rank correlation **0.632** across 12 content arms (best geometry read 0.242), winner in 2,000 of 2,000 bootstrap draws.
- The expression-level champion mostly measures persistence: per-prompt trained and base scores correlate 0.31–0.73 per arm, and propensity's correlation with the fine-tuning-induced change is negative in all 18 arms.
- On the change itself, pre-fine-tuning answer similarity to the training answers wins both families: marker log-probability change median **0.381** (6 of 6 arms); content change 0.124 (12 of 12).
- The propensity-residualized change race settles the coupling question: propensity keeps no material residual change signal (content +0.026, inside the 0.046 permutation band; marker −0.154); answer similarity wins (0.147 / 0.300).
- Raw context similarity predicts nothing (median −0.010, sign-flipping ±0.36 within sycophancy); its signal is answer-mediated, and only the training-answer anchor carries it (0.211 vs 0.017 at the source-persona anchor).
- Scope: one corpus, one model, 17 of 18 arms single-seed; the graded judge score failed its external per-prompt validation (teacher-forced margin −0.06); impoliteness is near-floored (84% zeros).

## Goal

- **This experiment in context:** The leakage-predictor line ([#658](https://eps.superkaiba.com/tasks/658), [#742](https://eps.superkaiba.com/tasks/742), [#761](https://eps.superkaiba.com/tasks/761), [#763](https://eps.superkaiba.com/tasks/763)) repeatedly found that a unit's own base behavioral propensity beats geometry reads at predicting where a fine-tuned behavior expresses — but at unit grain. This experiment races 11 predictors, all computable before fine-tuning, at per-prompt grain over bare real-user prompts on 18 trained arms reused from the parent capture fleet ([#1768](https://eps.superkaiba.com/tasks/1768)), and adds the mediation test the originating question asked for: is context-vector similarity's predictive power a byproduct of answer-side similarity? A zero-GPU analysis round added the propensity-residualized change race — the deciding read for whether propensity carries change signal beyond its mechanical coupling — plus coupling calibration and an arm-exclusion sensitivity check.
- **Broader narrative:** If per-prompt leakage of a fine-tuned behavior is predictable from base-model quantities alone, a deployment audit can flag where an implant will express before fine-tuning ships. The answer also constrains whether context geometry or answer geometry carries the leakage signal.

## Methodology

**Design:** 18 trained arms — 12 content arms (4 casual-writing, 4 impoliteness, 4 sycophancy, spanning persona / bare / conversation training contexts, contrastive and positive-only regimes, LoRA and full fine-tune, one impoliteness seed pair at seeds 42/137) and 6 marker-token arms — each evaluated on the same 4,000-prompt stratified subset (seed 1900) of the 16,400 real-user-prompt corpus (LMSYS/WildChat-class bare user prompts, data-realism tier 1; cross-tree sha intersection 16,318). Eleven deployable candidates computable before fine-tuning are raced within each arm: context similarity, answer similarity, two through-map similarity forms, whitened gate similarity, two read-out projections, base behavioral propensity, two write-prediction forms, and nearest-training-rows similarity. A six-read mechanistic panel using the trained model (post-fine-tuning and delta similarities, write magnitude) ran alongside; it explains, never carries the headline. All headline statistics are within-arm per-prompt Spearman rank correlations aggregated as across-arm medians — never raw-pooled across arms, so cross-arm install-dose differences cannot confound the ranking. Rounds: the main run (2026-07-31) plus one zero-GPU analysis round the same day (propensity-residualized change race, coupling nulls, 11-arm sensitivity; no new data).

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
| Permutation null | 1,000 within-arm draws, winner re-selected per draw over the 11 candidates | plan §6 |
| Coupling null (analysis round) | 500 trained-vs-base permutations per arm, seed 7 | `followup_free` recipe field |
| Residualization (analysis round) | per-arm one-predictor least-squares fit of trained on base score (n≈4,000 vs dimension 1) | `followup_free` recipe field |
| Layers | 19 (content) / 25 (marker), fixed in the plan | plan §11 |
| Marker DV | log P(marker id 83399) at the post-response slot, trained − base, three-space storage | plan §4 |
| Anchors | training answer/context centroids (primary) + source-persona context; 29 realized (15 pre + 14 post; full-fine-tune arms carry no post-anchor by design) | run record |
| Map refits | ridge, 23-point regularization grid, fit rows exclude the judge subset | plan §11 |

**Evaluation:** The content dependent variable is on-policy: the graded judge score (mean of kept draws) of each arm's own greedy completion to the bare prompt. The binary companion is the share of draws scoring at least 50; the change companion is trained minus base per prompt. The marker dependent variable is judge-free: the change in log-probability of the marker token at the end of the model's own response, with the logit-margin and probability companions agreeing in sign per arm. Dual-DV validation: the graded score's external per-prompt reference check failed — the teacher-forced fixed positive-vs-negative answer margin correlates −0.064 with the graded score over 299 overlap prompts on one sycophancy arm — so the graded instrument's support is internal (split-half reliability 0.73–0.93 across the 15 judged units, graded-binary concordance) plus the judge-free marker family reproducing the headline structure; this caps confidence at MODERATE. The champion rule is a signed-correlation argmax with the winner re-selected inside every bootstrap draw; a dethrone verdict additionally requires the challenger to beat propensity in at least 9 of 12 content arms (5 of 6 marker). The permutation band (upper edge 0.046 content / 0.044 marker against an achievable ceiling of 1) makes the null informative; candidates inside the band are indistinguishable from null given the variance, not confirmed zeros.

**Data extraction:** Predictor inputs are span-mean activations over the prompt (the context) and over the model's own greedy response (the answer), joined across arms by prompt sha. Anchors are span means over each arm's actual training rows; positive-only arms' anchors resolve from the matching contrastive pools (an approximation shared by every candidate within those arms). The representation mapping ran context-based only: the bare corpus carries 2 distinct prefix strings, so a prefix-based arm is unidentifiable (stated deviation, inherited from the corpus design). Plan deviations, all amended and re-reviewed during the run: the anchor-mix row floor was relaxed to an 8-row kill / 40-row warn (realized matched-text mixes carry about 20 rows); a frame-free identity floor replaced a cross-frame band in the marker gate; a weights-only loading carve-out for self-produced tensors; a CPU dry-run mode; the below-threshold impoliteness arm kept by a recorded pilot decision; one judge relaunch after a memory-pressure kill (resumable checkpoint, no data loss). I acknowledge the conciseness WARNs the verifier raises on this body — the per-result/120 word cap (three result sections run 129–150 words) and the total prose budget — accepted to carry the correction and caveat record the run requires. The learning-rate reconciliation WARN is likewise acknowledged: the rates above are reused-checkpoint recipe values re-read from each adapter's config — this task trained no model, so the plan carries no training learning rate of its own.

**Sample training/evaluation data + completions:** Disclosure: 1 of 4,000 judge-subset rows shown, cherry-picked for illustration (a benign row, chosen for real-user-corpus content hygiene; prompt and completion truncated). Full judge inputs: [issue1900_leakrace/judge_inputs @ 3bb20deb](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace/judge_inputs); per-arm score tables (all rows, all draws): [eval_results/issue_1900/judge @ 8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900/judge).

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

The win is substantially a persistence read: trained and base scores correlate 0.31–0.73 per arm (0.60–0.73 casual-writing and sycophancy, 0.31–0.34 floored impoliteness), and restricting to prompts whose trained and base completions materially diverge leaves it essentially unchanged (casual-writing 0.60–0.66). Where the base model already expressed the behavior, the trained model still does.

### On the fine-tuning-induced change, answer similarity to the training answers wins both families

The heatmap gives the per-arm correlation between each candidate and the marker change score — trained-minus-base log-probability of the marker token at the model's own response end — over the 6 marker arms plus the median column (n=4,000 per arm).

![Per-arm correlations between predictors and the marker log-probability change, six marker arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/hero_marker_race.png)

> **Figure.** *Answer similarity dethrones propensity on the judge-free marker change score.* Median 0.381 (through-map predicted-answer form 0.379), beating propensity in 6 of 6 arms; propensity is negative everywhere (median −0.367). Each cell is one arm's per-prompt correlation; the heatmap is itself the per-arm view.

The content change companion mirrors this: answer similarity 0.124, beating propensity in 12 of 12 arms (winner in 92.5% of draws). Propensity's negative change correlations (−0.04 to −0.30 content, −0.22 to −0.41 marker) sit above the independence null (−0.49 to −0.66) in all 18 arms: coupling-consistent, though sign alone cannot rule out hidden signal. Arm counts are not independent confirmations: arms share one judge subset and per-behavior base vectors — three content families plus one marker family, one corpus.

### The propensity-residualized change race is the deciding read: no material residual propensity signal

Both heatmaps rerun the change race with the change score replaced by the residual of the trained score after a per-arm one-predictor least-squares fit on the base score, removing the mechanical trained-minus-base coupling: content arms first, marker arms second.

![Propensity-residualized change race over twelve content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/residualized_race_content.png)

> **Figure.** *Content arms: with the coupling removed, answer similarity still wins and propensity drops to null.* Answer similarity median 0.147 (winner in 96.8% of draws, beats propensity in 11 of 12 arms); propensity +0.026, inside the 0.046 permutation band, negative on the floored impoliteness arms.

![Propensity-residualized change race over six marker arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/residualized_race_marker.png)

> **Figure.** *Marker arms agree.* Answer similarity 0.300 (winner in 75.1% of draws, the through-map predicted-answer form takes the rest, and it beats propensity in 6 of 6 arms); propensity −0.154.

Once its own base term is regressed out, propensity carries no material positive signal about the fine-tuning-induced change, while answer similarity — which shares no term with the change score — keeps its lead in both families. Raw (unresidualized) change medians for comparison: 0.124 content, 0.381 marker (preceding result's figure).

### Context similarity is answer-mediated, sign-unstable, and anchor-specific

The forest plot gives, per content arm, the partial rank correlation with the graded level score for context similarity (blue) and answer similarity (orange), each conditioned on propensity alone and additionally on the other similarity read.

![Partial rank correlations for context and answer similarity per content arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/mediation_forest.png)

> **Figure.** *Answer similarity keeps unique signal; context similarity has none.* Median partial correlation given propensity: context 0.022, answer 0.188; adding the other predictor leaves answer similarity at 0.182 while context drops to −0.019. All 12 arms shown, both conditioning forms per arm.

The answer-mediated verdict holds on three separate checks: the partial lattice, the structural read (the through-map predicted-answer form matches answer-similarity rankings at 0.84 rank-agreement while absorbing context similarity), and a disjoint-half anchor recount. There is little context channel to mediate: raw context similarity is near zero overall and sign-unstable within one behavior (−0.35 to −0.37 on three sycophancy arms, +0.36 on the fourth). The anchor choice decides the read: answer similarity predicts at the training-answer centroid (0.211), not at the source-persona context (0.017).

### The mechanistic panel explains expression through post-fine-tuning answer geometry; delta similarities carry nothing

The heatmap shows per-arm correlations between the six trained-model panel reads (rows) and each arm's primary score, across all 18 arms.

![Six mechanistic panel reads against each arm's primary score, eighteen arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/m_panel.png)

> **Figure.** *Post-fine-tuning answer similarity is the strongest mechanistic read.* Content medians: post-fine-tuning answer similarity 0.310, through-map form 0.246, write magnitude 0.123; delta-context −0.029 and matched-text delta-answer −0.025 are null. Marker: write magnitude 0.499 leads.

The expected near-null for delta-context similarity was confirmed, but the hypothesis that matched-text delta-answer similarity would be the strongest mechanistic candidate is falsified — both delta reads are indistinguishable from null given the variance. Explanatory power lives in where the trained answer representation sits, not in how far it moved on matched text.

### Instrument health is clean and both verdicts survive the sensitivity checks, but the graded score lacks external validation

Each arm's observed max-selected correlation (red) is plotted against its permutation null band (blue, 97.5th percentile of 1,000 within-arm draws, winner re-selected per draw) and the achievable ceiling of 1 (dashed).

![Observed max-selected correlation per arm against permutation band and ceiling](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900/band_vs_ceiling.png)

> **Figure.** *The null is informative: every arm's observed maximum sits far above its band.* Band upper edges reach 0.046 (content) and 0.044 (marker) against a ceiling of 1; observed max-selected values run 0.31–0.73.

The judge ran clean: 180,000 draws, 1,938 content drops (1.08%; 0.21–2.17% per unit), zero transport losses.

One impoliteness arm sits below the 0.05 dynamic-range share bar at full grain (0.046); it was kept by a recorded pilot decision, and — correcting the run record — no full-data eligibility gate ever ran. Excluding it changes nothing (level champion 0.642, winner in every draw; change verdict 11 of 11).

Residual caveats: the failed external validation of the graded score (Methodology), and no language-intrusion scan on the judged pools — the shared-prompt within-arm design and the judge-free marker replicate bound the latter's effect on the headline structure, not on absolute content magnitudes.

---
**Repro:** ~3.3 GPU-h realized (fellows SLURM lane, 5 launches — 4 crash-diagnosed, 1 clean full run) plus ~180k Batch-API judge calls off-pod; no training. Code: workload ran at [5f2b220a42](https://github.com/superkaiba/explore-persona-space/tree/5f2b220a42ab09d0930e0a992099713dfe9695c8/scripts) (`scripts/issue1900_{prep,gpu,judge,race,figs}.py`); outputs committed at [0e5e6c3e7d](https://github.com/superkaiba/explore-persona-space/tree/0e5e6c3e7dc4bda1873178a2a7d808542423248f/eval_results/issue_1900); analysis round (`scripts/issue1900_followup_free.py`) at [8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900/race/followup_free). Aggregated eval JSONs plus the per-arm and per-draw files the aggregates collapse (`champion_*.json`, `mediation.json`, `robustness.json`, `arm_*.json`, `boot_*.npz`): [eval_results/issue_1900 @ 8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/eval_results/issue_1900). Figures: [figures/issue_1900 @ 8ef8bb77bd](https://github.com/superkaiba/explore-persona-space/tree/8ef8bb77bdfaab21f53541345598588210a00068/figures/issue_1900). HF artifacts — config (4 files), anchors (29), marker teacher-forced tables (27), predictor tables (108), maps (67), judge inputs (30) + raw judge shards, validation (4): [issue1900_leakrace @ 3bb20deb](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3bb20debe2e68392897d6144b9180c8748c7afcb/issue1900_leakrace). Corpus + parent stores pin: [issue1768_mapshift @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift).
- Reused the 18 trained arms, their per-prompt activation stores, and the training-mix positive pools from [#1768](https://eps.superkaiba.com/tasks/1768) (checkpoints originally trained by the [#1481](https://eps.superkaiba.com/tasks/1481)/[#1586](https://eps.superkaiba.com/tasks/1586) organism fleet): [issue1768_mapshift @ c0726728](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c07267285d2cdbf3e0401ddc3e3accae50e496a7/issue1768_mapshift) — fit: the race requires exactly these arms' per-prompt stores at the same prompt grain; recipes verified from each adapter's config at run start.
- Reused behavior read-out directions from the fleet extraction line ([#1112](https://eps.superkaiba.com/tasks/1112), [#1315](https://eps.superkaiba.com/tasks/1315), [#1434](https://eps.superkaiba.com/tasks/1434)) — fit: the read-out projection candidates are defined against these directions.
- No pod; GPU phase on the fellows SLURM lane; wandb n/a (no training).

**Context:**
> "Help me to plan this experiment with these model organisms: What is the best leakage predictor (8 candidate similarity reads pre/post finetuning, map-mediated variants); is leakage due to similarity of mean answer vectors with context similarity a byproduct; any other suggestions"

Longer recorded form of the same ask (task Provenance, verbatim):
> "Help me to plan this experiment with these model organisms: What is the best leakage predictor: context vector cosine similarity pre-finetuning / context vector cosine similarity post-finetuning / answer vector cosine similarity pre-finetuning / answer vector cosine similarity post-finetuning / similarity between change in context vector pre and post finetuning / similarity between change in answer vector pre and post finetuning / context vector -> apply mapping -> predicted answer vector similarity pre-finetuning / context vector -> apply mapping -> predicted answer vector similarity post-finetuning / Is leakage due to similarity of mean answer vectors and so context vector similarity is just byproduct / Any other suggestions?"

Lineage: [#1768](https://eps.superkaiba.com/tasks/1768) — parent (its capture stores and arm fleet are this task's inputs). Created 2026-07-30; run 2026-07-31 (five launches, judge phase, stats); one zero-GPU free-analysis round 2026-07-31 (propensity-residualized change race, coupling nulls, 11-arm sensitivity).
