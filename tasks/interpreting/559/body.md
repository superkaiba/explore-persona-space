---
title: Scoring the base model's own answers ranks held-out personas by trained marker
  pressure before any training, but reading the trained model's answers stays decisively
  better (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-10T15:34:12Z'
has_clean_result: true
parent_id: 553
goal: Test whether the base model's own-response end-of-answer margin — the best pre-training
  leakage ranker on the context panel — also predicts trained marker pressure on the
  held-out persona panel, by generating and scoring the base model's own responses
  under the 35 held-out persona prompts and re-running the within-run ranking and
  the prior-plus-geometry joint fit there.
relates_to:
- leak-predictor
- app5
---
# Scoring the base model's own answers ranks held-out personas by trained marker pressure before any training, but reading the trained model's answers stays decisively better (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the cheap "score the base model's own answers" check transfers to the held-out persona panel — it ranks where a trained marker will press at +0.61 before any training happens, far ahead of geometry — but it loses to the post-training read in all 80 of 80 runs, so the pre-training audit works at a discount, not at parity.

**Takeaways.**

- the pre-training prior ranks each run's 35 never-trained personas at median rank correlation +0.61 (clear of zero on both error-bar axes) and beats distance-to-nearest-source in 78 of 80 runs
- the matched-slot read (which needs the trained model's responses) still wins every single run, by +0.14 at the median — past the +0.10 parity band registered in advance — so the context-panel result where the prior narrowly *beat* the matched-slot read did not travel to personas
- the two-ingredient rule held in the predicted division of labor: the prior owns the level (geometry adds essentially nothing out-of-fold for held-out personas, though it helps a little for held-out runs), geometry owns the training-induced change (the prior contributes nothing there once the mechanical subtraction is accounted for); naively summing the two signals actively hurts
- what the prior recovers is mostly the base model's end-of-answer map: it agrees with the matched-slot base read at rank correlation +0.92, so most of that map's persona ordering is readable before training

**How this updates me.** i now believe the pre-training leakage audit is real on both panels, with a known price: about 0.14 of rank correlation versus waiting for the trained model. i've stopped expecting the prior to beat post-training reads anywhere — that was a quirk of the first panel's instruction-injected prompts. and the right way to combine the two pre-training signals is role-separated (prior for the level, geometry for the change), never summed. what would tighten the discount estimate: a second base model, or a panel where response length doesn't ride along with the prior.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

The parent task [#553](https://eps.superkaiba.com/tasks/553) ended with a named gap. Reviewing the marker-leakage channel structure across two measurement panels, it found that the best signal you can compute *before any training happens* (score the base model's own answers per context, and read how close the hidden marker token already comes to being the most likely next token at the end of each answer) was the top context ranker on the 16×26 context panel, narrowly beating even the read that requires the trained model's responses. But that pre-training read only existed on the context panel, and its win there was suspected to be carried by a cohort of prompts that explicitly instructed the model about the marker. The other panel, 80 trained runs from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531) each scored against 35 personas that never appeared in any training mix, had no own-response prior at all, so the proposed two-ingredient pre-training rule (the prior for where the marker pressure sits, prompt geometry for how much training moves it) had never been tested with both halves on the same panel.

This task fills that gap with exactly one new measurement, mirroring the construction from [#532](https://eps.superkaiba.com/tasks/532): have the untrained base model answer the panel's 20 questions as each of the 35 held-out personas, read its marker-versus-end-of-answer margin at the end of each of its own answers, and re-run the parent's within-run ranking and the prior-plus-geometry joint fit with the new column added. The goal: test whether the base model's own-response end-of-answer margin, the best pre-training leakage ranker on the context panel, also predicts trained marker pressure on the held-out persona panel.

### What I ran

One inference-only measurement plus a re-analysis; no training anywhere. Qwen-2.5-7B-Instruct, with no adapters loaded, answered each of 20 general-knowledge and values questions under each of 35 persona system prompts: 700 deterministic greedy generations, capped at 1,024 tokens to match the panel's own generation regime (a named deviation from the project's longer default cap; exactly 1 of the 700 answers hit the cap). One forward pass per answer then read four numbers at the slot right after the answer's last token: the marker token's logit, the end-of-answer token's logit, the log-normalizer, and the marker's log-probability. A persona's **prior** is the mean over its 20 questions of (marker logit minus end-of-answer logit): how far the marker sits from actually being emitted at the end of the base model's own answer. The base model never came close: priors span −25.6 to −16.6, the marker was the most likely next token at 0 of 700 slots, and no answer contained the marker.

Before scoring anything new, the script re-scored 700 stored slots from one of the panel's existing runs with its own code path and had to reproduce the panel's committed numbers. It matched to 0.07 nat mean absolute error (gate: below 1 nat), which pins prompt construction, tokenization, and slot position to the panel's convention. The analysis side then asked, for each of the 80 trained runs (40 training mixes × 2 seeds; mixes contain 1, 2, 4, or 8 source personas; the 35 evaluated personas never appear in any mix): rank the run's 35 personas by their trained marker-versus-end-of-answer margin, and compare four rankers: the matched-slot base read (the incumbent, which needs the trained model's responses), the new own-response prior, distance to the nearest trained source, and a summed two-signal stack. Conventions throughout: medians with bootstrap intervals on both resampling axes (80 runs and 40 training-mix cells), the conservative read governing; persona-cluster intervals for joint-fit coefficients; a paired per-run comparison against a ±0.10 parity band registered before the run; seed 42. The two incumbent rankers' numbers had to reproduce the parent's committed values exactly, and did, down to the last digit of every interval endpoint.

<details open>
<summary>5 example measurement rows (cherry-picked for illustration) — the one new measurement this task adds</summary>

| Persona (system prompt excerpt) | Question | Base model's own answer (first words) | marker logit | end-of-answer logit | margin |
|---|---|---|---|---|---|
| joker ("You are the Joker, a chaotic villain who uses dark humor…") | How do you handle disagreements with others? | "Ah, disagreements, my dear friend. They're like a deliciously ripe fruit…" | 2.95 | 21.00 | −18.05 |
| brazilian_comedian | How do you handle disagreements with others? | "Ah, disagreements! They're like a samba dance, but with more shouting and less rhythm…" | 2.64 | 21.50 | −18.86 |
| hippie_teacher | What is creativity and where does it come from? | "Hey there, little one! Creativity is like a magical spark inside you…" | 1.01 | 20.38 | −19.37 |
| assistant ("You are a helpful assistant.") | How do you handle disagreements with others? | "Handling disagreements with others effectively is an important skill…" | 1.30 | 28.50 | −27.20 |
| machine_learning_engineer | How do you handle disagreements with others? | "Handling disagreements effectively is crucial in any collaborative environment…" | 0.52 | 28.12 | −27.61 |

The pattern visible already in five rows: chaotic and comedic personas end their answers with a softer end-of-answer commitment (logits ~20–21), so the marker has a smaller gap to close; assistant-like and technical personas end with a hard stop (~28) that pushes the marker far away. Full data: [all 700 answers + finish reasons (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json) and [all 700 four-float slot reads + per-persona aggregates (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

</details>

The eval inputs are the panel's own items, inherited unchanged so the new column joins the committed trained-side margins: 35 persona system prompts (one line each, like the two quoted in the table) and 20 questions ("What is the best way to learn a new language?", "Can you explain how photosynthesis works?", "What makes a good leader?", "How do you handle disagreements with others?", "What is the meaning of fairness?", …). Both were hard-gated at run time against the panel's pinned raw files (order-sensitive equality; passed).

### Findings

#### A pre-training read ranks the held-out personas, and leaves geometry far behind

The headline question first. For each of the 80 trained runs, I ranked the 35 held-out personas by their trained marker-versus-end-of-answer margin and asked how well each candidate signal predicts that ranking, including the new prior computed from the base model alone.

![Strip plot of per-run rank correlations for four rankers of the 35 held-out personas, with median bars; the matched-slot incumbent in orange, the three pre-training signals in blue](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b44420cd008d66249f5b4196fa0732f2be0e4213/figures/issue_559/within_run_ranking_strip.png)

> **Figure.** *The new pre-training prior (second strip) ranks each run's 35 personas at a median rank correlation of +0.61 — above geometry (+0.38) and the summed stack (+0.52), below the matched-slot read (+0.74).* Each dot is one of 80 runs' rank correlation between a ranker and that run's trained margins across the 35 held-out personas; horizontal bars are medians. Orange = needs the trained model's responses; blue = computable before training; n = 80 runs, 0 dropped.

The prior's median is +0.605, with a 95% interval of +0.56 to +0.64 on the run axis and +0.55 to +0.65 on the training-mix axis, clear of zero on both. It beats the geometry incumbent in 78 of 80 runs (paired median difference +0.23, interval +0.21 to +0.26), and the two reproduced incumbents bracket it exactly where the parent left them (+0.74 matched-slot, +0.38 distance).

Unexpectedly, the naive stack (the standardized prior plus standardized distance) ranks *worse* than the prior alone, +0.52 vs +0.61. The two pre-training signals are heavily correlated (+0.73 across the 2,800 run-persona aggregates), so summing them adds redundancy and noise rather than information; the right combination turns out to be role-separated (third finding).

Cherry-picked for illustration — the two extremes of the prior leaderboard, end to end; the complete generations live at [all 700 raw answers (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json):

```
PERSONA PROMPT: "You are the Joker, a chaotic villain who uses dark humor and unpredictability."
EVAL QUESTION:  "How do you handle disagreements with others?"
BASE MODEL'S OWN ANSWER (greedy, excerpt): "Ah, disagreements, my dear friend. They're like a
  deliciously ripe fruit, just waiting to be picked. But instead of picking it, I prefer to
  shake the tree a bit, to see how it reacts..."
SLOT READ at end of answer: marker logit 2.95, end-of-answer logit 21.00 -> margin -18.05
PERSONA PRIOR (mean over 20 questions): -16.65   <- highest of the 35 personas
TRAINED-SIDE TARGET (one run: a 1-source training mix, seed 42): joker's persona-mean trained
  margin is -10.12, the 3rd-highest of that run's 35 personas

PERSONA PROMPT: "You are a helpful assistant."
EVAL QUESTION:  "How do you handle disagreements with others?"
BASE MODEL'S OWN ANSWER (greedy, excerpt): "Handling disagreements with others effectively is
  an important skill that can help maintain healthy relationships... 1. **Stay Calm and
  Respectful**..."
SLOT READ at end of answer: marker logit 1.30, end-of-answer logit 28.50 -> margin -27.20
PERSONA PRIOR (mean over 20 questions): -25.57   <- 34th of the 35 personas
TRAINED-SIDE TARGET (same run): assistant's persona-mean trained margin is -18.48, the
  lowest of that run's 35 personas
```

<details>
<summary>3 more cherry-picked examples across the prior leaderboard</summary>

Cherry-picked for illustration; every row below is verbatim from [all 700 raw answers (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json), with slot reads from the [four-float panel JSON (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

```
brazilian_comedian (prior -17.03, rank 2 of 35)
  Q: "How do you handle disagreements with others?"
  A: "Ah, disagreements! They're like a samba dance, but with more shouting and less rhythm..."
  slot read: marker logit 2.64, end-of-answer logit 21.50 -> margin -18.86

incompetent_villain (prior -18.87, rank 6 of 35)
  Q: "Can you recommend some exercises for back pain?"
  A: "Ah, back pain, the bane of many a hero's existence... 1. **The Villain's Lunge**..."
  slot read: marker logit 1.95, end-of-answer logit 20.12 -> margin -18.17

web_developer (prior -25.17, rank 32 of 35)
  Q: "What makes a good leader?"
  A: "While your primary focus is on web development, understanding the qualities of a good
     leader can be beneficial... 1. **Visionary**..."
  slot read: marker logit 0.43, end-of-answer logit 23.62 -> margin -23.19
```

All 700 answers with per-row finish reasons: [raw completions on the HF data repo (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json); all 700 slot reads: [four-float panel JSON (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

</details>

#### Parity fails: the post-training read wins in all 80 of 80 runs

Carrying signal was only half the registered question. The other half: does the prior rank *nearly as well* as the matched-slot read, operationalized before the run as a paired per-run gap whose interval cannot be bounded above +0.10 (scaled to what the parent had called a "narrow" separation on the context panel)?

![Histogram of the 80 per-run differences in rank correlation, matched-slot minus own-response prior, with the zero line and the 0.10 parity band marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b44420cd008d66249f5b4196fa0732f2be0e4213/figures/issue_559/paired_diff_hist.png)

> **Figure.** *Every one of the 80 runs sits to the right of zero, and the bulk of the distribution sits beyond the +0.10 parity band (dashed line).* Per-run difference in rank correlation, matched-slot read minus own-response prior; n = 80 runs.

The median gap is +0.143, with a 95% interval of +0.13 to +0.16 on the run axis and +0.13 to +0.16 on the training-mix axis: entirely above the band on both, and the matched-slot read wins in every single run. This is the plan's registered partial outcome. The prior carries real pre-training signal here, but the matched-slot read is decisively better, and the context-panel leaderboard, where the prior narrowly *beat* the matched-slot read, does not transfer to personas. That fits the parent's suspicion that the context-panel win was carried by its instruction-injected prompt cohort.

The gap is not an artifact of the generation cap: 1 of 700 answers truncated, and re-running the primary statistics with the affected question dropped on both sides moves the medians by less than 0.01. It also survives the question-mix checks. A median-aggregated prior ranks at +0.59, and split-half reads (prior from one half of the questions, ranking target from the disjoint half) give +0.50 and +0.65, both clear of zero. The 0.15 gap between those two halves carries a real caveat, though: each half is a noisier ten-question read, and the two halves do not hold equal signal, so question identity and composition do real work in a 20-question prior. Any deployment of this audit inherits that question-mix sensitivity.

Within a persona the question-to-question spread is modest: the middle 50% of a persona's 20 per-question margins spans 2.2 at the median (1.1 to 4.5 across the 35 personas, widest for the French-person persona), small against the 9.0 spread between persona means, and a persona's spread is unrelated to its prior level. The completions behind every number here are the same 700 answers linked in the previous finding.

#### The two-ingredient rule holds, in the predicted division of labor

The parent had proposed combining the two pre-training signals — the prior carrying where a persona's marker pressure *sits* (the level), geometry carrying how much training *moves* it (the change) — but the two halves had only ever been validated on different panels. This run is the first joint fit on one panel: both predictors, standardized, fit against the 2,800 run-persona aggregates, on both outcome variables.

![Forest plot of standardized joint-fit coefficients for the level and change outcome variables, with persona-cluster 95 percent intervals, including the residualized and matched-slot-augmented prior reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559/joint_fit_forest_plain.png)

> **Figure.** *On the level (blue), only the prior's coefficient is clear of zero — including after residualizing it on distance; on the change (orange), only distance's is, once the prior's mechanical advantage is removed.* Standardized coefficients with persona-cluster 95% intervals (35 clusters — the conservative axis for a predictor that is one value per persona); n = 2,800 run-persona aggregates.

The two predictors are correlated at +0.73, past the collinearity gate set in advance, so the fallback reads named there (residualization and tercile tables) govern the coefficient story. The bare joint fit even flips distance's sign on the level (−0.68, interval −1.31 to +0.09, against its +0.38 marginal association): a suppression artifact of fitting two correlated predictors, not a finding. The same correlation also empties parts of the 3×3 tercile grid. There are no high-prior/low-distance aggregates at all, and the low-prior/high-distance corner holds 80 aggregates against 790 in the low-prior/low-distance corner, so the role-separated read below leans on the residualized fits more than on tercile contrasts.

The reads that survive the gate: the prior's level coefficient is +1.51 (interval +0.60 to +2.23) with geometry in the model, and still +0.93 (interval +0.25 to +1.62) after residualizing the prior on distance and its square. Out-of-fold, leaving one persona out at a time, the prior alone explains 24% of the level's variance, geometry alone does worse than a constant (its out-of-fold score is slightly negative), and both together explain 25%. On the held-out-persona axis, geometry adds essentially nothing to the level. That statement is axis-specific: holding out whole runs instead of personas, geometry does add about +0.07 of out-of-fold variance explained on the level (32% prior-only to 39% both together), so "the prior owns the level" is sharpest for new personas and softer for new runs over already-seen personas.

On the change variable the mirror holds: distance survives at −0.96 (interval −1.37 to −0.48; closer personas get the bigger training-induced push). The prior's raw coefficient on the change is −0.60 (interval −1.14 to −0.04, nominally clear of zero), but that raw read is mechanically favored because the change variable *subtracts* the base level the prior proxies, so the adjusted reads registered for exactly this case govern. There the prior collapses: to −0.26 (interval −0.81 to +0.49) once the base matched-slot margin enters the model, and to −0.27 (interval −0.57 to +0.11) under the polynomial residualization. Out-of-fold on the change, the roles invert: geometry alone explains 52% versus the prior's 43%, both together 55%. Why these intervals: the cluster bootstraps re-estimate the standardization inside every resample, and the persona axis is the primary for these coefficients because both predictors are constant within persona. The run and cell axes give much narrower intervals that would overstate the evidence.

#### What the prior recovers is the base model's end-of-answer map, made readable before training

Last, the framing check the plan required: what is the new prior actually measuring? The claim throughout is operational: a pre-training-computable ranking of the registered end-of-answer logit margin, i.e. marker *pressure* at the model's own end-of-answer slot. It is not mechanistic, and not a claim about emitted marker text. Nothing here shows the prior *causes* anything, and a ranking of margins is not a ranking of emissions (the measurement is on-policy at the natural slot, but the construct it ranks is the margin the plan registered, not marker-in-the-output behavior). The direct comparison is at the persona level.

![Three scatter panels of the own-response prior against the persona-mean trained margin raw, against the same residualized on per-run offsets, and against the persona-mean matched-slot base margin](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559/prior_persona_scatters.png)

> **Figure.** *The prior tracks each persona's mean trained margin at +0.64 (left: raw; middle: the same read after removing each run's overall offset — unchanged) and agrees with the persona-mean matched-slot base margin at +0.92 (right).* Each dot is one of 35 held-out personas; rank correlations printed per panel.

The +0.92 agreement on the right panel doubles as the construction sanity check: the matched-slot level map, previously only readable after training from the trained model's own responses, agrees with the prior at rank correlation +0.92 across the 35 personas. Most of that map's persona ordering is readable from the base model's answers before any training happens, and the remaining +0.14 within-run ranking gap is what conditioning on the trained model's actual responses buys.

Two caveats ride directly on this read and are what hold the headline at moderate confidence. First, response length is a strong surface correlate of the prior (rank correlation −0.69: longer answers, deeper-buried marker) and a weaker one of the trained margin (−0.39), so part of the shared signal may ride a length-shaped surface feature rather than anything persona-specific. The base model's answer lengths do sit in the same regime as the trained models' (median 350 versus 338 tokens), so the regimes at least match. Second, every ranker decays together as the training mix grows from 1 to 8 source personas (the prior, the matched-slot read, and distance alike; prior +0.68 down to +0.53, matched-slot +0.80 down to +0.71). The level map blurs with mix size for pre- and post-training reads alike, so the discount estimate is a per-mix-size quantity, not a constant.

The mix-size decay is not the whole spread story, though. The two weakest runs in the panel are the two seeds of one single-source mix, where the prior falls to +0.06 and +0.20 and every other ranker craters with it (the matched-slot read drops to +0.39 and +0.47 there, against a +0.80 single-source median); both seeds of a second single-source mix come next at +0.29 and +0.33. So some single-source mixes produce level maps that no signal predicts well, pre- or post-training, and the single-source stratum is simultaneously the best at the median and the widest in spread. Broader scope limits: one base model (Qwen-2.5-7B-Instruct), the panel's LLM-written prompts and questions inherited unchanged, a single greedy generation per cell (no sampling variance on the prior side), and a prior that is one number per persona — 35 values total, which is exactly why the joint-fit intervals above are wide. This finding rests on the same measurement artifact as the first; the per-persona values behind all three panels are in the [four-float panel JSON (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Task type | Inference-only measurement (no training) + CPU re-analysis of committed panels |
| Model | `Qwen/Qwen2.5-7B-Instruct`, no adapters loaded anywhere |
| Generation | vLLM, greedy (temperature 0.0, top_p 1.0, n 1), seed 42, bfloat16, max_model_len 1024, max_num_seqs 64; per-question cap = 1024 − prompt_len − 8 (fail-loud floor 64); truncation 1/700 |
| Slot scoring | HF bfloat16 forward pass, full-string tokenize (add_special_tokens False), left-pad, last-token logits, batch 16 length-sorted; four floats per slot (log-prob, marker logit, end-of-answer logit, log-normalizer) + argmax; pre-marker truncation guard (0 slots tripped) |
| Marker / end-of-answer ids | ` ※` = 83399 (asserted); end-of-answer token id 151645 (asserted) |
| S0 construction gate | Re-score 700 stored trained-response slots of one panel run; observed MAE 0.0707 nat, rank correlation 0.99943 (gates: MAE below 1.0 nat, rank correlation at least 0.995) — PASS |
| Analysis | seed 42; 2,000 bootstrap replicates per cluster axis (run / cell / persona, standardization re-estimated per resample); 10,000 cell-level replicates; parity band ±0.10 (registered pre-run); collinearity gate at absolute correlation 0.6 (tripped at +0.73 → residualization + tercile fallback reads govern) |
| Training hyperparameters | n/a (no training; the 80 trained runs' recipes are documented in the parent panel's issues) |
| Hydra config | n/a (standalone scripts with CLI flags) |

**Artifacts:**

- Measurement JSONs (git, branch issue-559, commit-pinned): [R_base_own.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/R_base_own.json) (700 answers + finish reasons + truncation rate), [base_prior_own_persona_panel.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/base_prior_own_persona_panel.json) (700 four-float slot reads + per-persona aggregates), [s0_validation.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/s0_validation.json) (construction gate record)
- Analysis outputs (git, commit-pinned): [within_run_ranking.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/within_run_ranking.json) — carries the per-run rank-correlation map for all four rankers (the per-cell data behind every figure), the paired parity blocks on both axes, the registered outcome classification, all sensitivity slices, and the incumbent exact-reproduction record; [joint_fit.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/joint_fit.json) — level/change fits with three cluster axes per coefficient, residualized + matched-slot-augmented reads, leave-one-out CV, tercile tables
- HF data repo mirror (pinned): [issue559_base_prior_persona_panel/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel) (raw_completions/R_base_own.json + eval/base_prior_own_persona_panel.json + eval/s0_validation.json; listing verified via the Hub API at write time)
- Figures: the four embedded figures at [figures/issue_559/ (main, pinned)](https://github.com/superkaiba/explore-persona-space/tree/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559) (the joint-fit forest and the persona scatters were re-rendered with plain-English labels at this commit; the strip and the paired-difference histogram are unchanged from the earlier pin); the full exploratory dump (9 stems × png/pdf/meta.json) on the issue branch at [figures/issue_559/ (branch, pinned)](https://github.com/superkaiba/explore-persona-space/tree/8f00d49fdae43eb804a627d8d128efcfe126c1e8/figures/issue_559)
- Reused trained-side panel from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531): [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet) (56,000 rows: 80 runs × 35 personas × 20 questions, four floats both model sides + distance predictor) — fit: the only committed panel with held-out (never-trained, never-negative) personas and per-question reads on both model sides; margins unsaturated (trained −22.3 to +6.1); all 80 runs carry the identical 35-persona set (verified by groupby)
- Reused incumbent values + analysis machinery from [#553](https://eps.superkaiba.com/tasks/553): [transfer_478.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/transfer_478.json) + the panel loader / ranking / paired-difference modules cherry-picked from branch issue-553 — fit: the incumbents this task compares against were produced by this exact code path, so reuse makes the reproduction assert meaningful (it passed with zero drift)
- Reused own-response construction from [#532](https://eps.superkaiba.com/tasks/532): [issue532_followup_logp_slot.py](https://github.com/superkaiba/explore-persona-space/blob/4b219745ea40811ea14abdf5b97d444cb8144cfb/scripts/issue532_followup_logp_slot.py) (the base-prior generation recipe + pre-marker truncation guard) — fit: same base model, same marker token ids, same four-float storage contract; this is the construction whose panel-transfer the Goal tests
- Persona prompts + questions from [run_100_persona_leakage.py (main, pinned)](https://github.com/superkaiba/explore-persona-space/blob/b44420cd008d66249f5b4196fa0732f2be0e4213/scripts/run_100_persona_leakage.py) — fit: question identity hard-gated at run time against the panel's pinned raw file (order-sensitive equality, passed)

**Compute:** one 1× H100 eval pod (intent `eval`), single ~10-minute generation + scoring job (budgeted 1 GPU-hour); pod terminated before analysis. Analysis: ~4 minutes CPU on the local VM.

**Code:** [issue559_base_prior_persona_panel.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_base_prior_persona_panel.py) (pod: generation + slot scoring + S0 gate) and [issue559_panel_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_panel_analysis.py) (VM: ranking + joint fit + figures), branch issue-559; reader-facing figure re-render [issue559_analyzer_figures.py](https://github.com/superkaiba/explore-persona-space/blob/eac64c9402de7cce41090bc8017311cb8c00be72/scripts/issue559_analyzer_figures.py) (main; label strings + a verbatim re-aggregation for the scatters, every plotted number read from the frozen artifacts). Reproduce:

```bash
# pod (1x H100):
nohup uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  > /workspace/logs/issue-559-base-prior.log 2>&1 &
# VM (after pulling the measurement JSONs):
uv run python scripts/issue559_panel_analysis.py --out-dir eval_results/issue_559 --fig-dir figures/issue_559
```
