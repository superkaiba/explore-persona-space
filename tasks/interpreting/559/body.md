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
- the matched-slot read (which needs the trained model's responses) still wins every single run, by +0.14 at the median — past the +0.10 parity band set in advance — so the context-panel result where the prior narrowly *beat* the matched-slot read did not travel to personas
- the two-ingredient rule held in the predicted division of labor: the prior owns the level (geometry adds essentially nothing out-of-fold for held-out personas, though it helps a little for held-out runs), geometry owns the training-induced change (the prior contributes nothing there once the mechanical subtraction is accounted for); naively summing the two signals actively hurts
- what the prior recovers is mostly the base model's end-of-answer map: it agrees with the matched-slot base read at rank correlation +0.92, so most of that map's persona ordering is readable before training
- and the prior isn't just response length: partial length out and the leftover signal still ranks the personas at +0.49, clear of zero and well ahead of ranking by length alone (+0.38)
- the prior isn't keyed to the panel's particular questions either: re-measured on 30 fresh questions from the same pool, it ranks at +0.66 — slightly *better* than the original 20 in all 80 runs — and the discount versus the post-training read shrinks to about 0.09 (the post-training read still wins every run)

**How this updates me.** i now believe the pre-training leakage audit is real on both panels, with a known price: roughly 0.09–0.14 of rank correlation versus waiting for the trained model, depending on which questions — and secondarily how many — the prior gets. i've stopped expecting the prior to beat post-training reads anywhere — that was a quirk of the first panel's instruction-injected prompts. and the right way to combine the two pre-training signals is role-separated (prior for the level, geometry for the change), never summed. i also checked the two obvious deflations — that the prior is secretly just "longer answers bury the marker", and that it's secretly keyed to the panel's 20 questions — and neither holds: the length partial costs about 0.10 but the persona signal stands, and a full question swap from the same pool costs nothing at all. what would tighten the discount estimate now: a second base model.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

The parent task [#553](https://eps.superkaiba.com/tasks/553) ended with a named gap. Reviewing the marker-leakage channel structure across two measurement panels, it found that the best signal you can compute *before any training happens* (score the base model's own answers per context, and read how close the hidden marker token already comes to being the most likely next token at the end of each answer) was the top context ranker on the 16×26 context panel, narrowly beating even the read that requires the trained model's responses. But that pre-training read only existed on the context panel, and its win there was suspected to be carried by a cohort of prompts that explicitly instructed the model about the marker. The other panel, 80 trained runs from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531) each scored against 35 personas that never appeared in any training mix, had no own-response prior at all, so the proposed two-ingredient pre-training rule (the prior for where the marker pressure sits, prompt geometry for how much training moves it) had never been tested with both halves on the same panel.

This task fills that gap with exactly one new measurement, mirroring the construction from [#532](https://eps.superkaiba.com/tasks/532): have the untrained base model answer the panel's 20 questions as each of the 35 held-out personas, read its marker-versus-end-of-answer margin at the end of each of its own answers, and re-run the parent's within-run ranking and the prior-plus-geometry joint fit with the new column added. The goal: test whether the base model's own-response end-of-answer margin, the best pre-training leakage ranker on the context panel, also predicts trained marker pressure on the held-out persona panel — and, once that landed, whether the prediction survives a question set disjoint from the panel's own.

### What I ran

One inference-only measurement plus a re-analysis; no training anywhere. Qwen-2.5-7B-Instruct, with no adapters loaded, answered each of 20 general-knowledge and values questions under each of 35 persona system prompts: 700 deterministic greedy generations, capped at 1,024 tokens to match the panel's own generation regime (a named deviation from the project's longer default cap; exactly 1 of the 700 answers hit the cap). One forward pass per answer then read four numbers at the slot right after the answer's last token: the marker token's logit, the end-of-answer token's logit, the log-normalizer, and the marker's log-probability. A persona's **prior** is the mean over its 20 questions of (marker logit minus end-of-answer logit): how far the marker sits from actually being emitted at the end of the base model's own answer. The base model never came close: priors span −25.6 to −16.6, the marker was the most likely next token at 0 of 700 slots, and no answer contained the marker.

Before scoring anything new, the script re-scored 700 stored slots from one of the panel's existing runs with its own code path and had to reproduce the panel's committed numbers. It matched to 0.07 nat mean absolute error (gate: below 1 nat), which pins prompt construction, tokenization, and slot position to the panel's convention. The analysis side then asked, for each of the 80 trained runs (40 training mixes × 2 seeds; mixes contain 1, 2, 4, or 8 source personas; the 35 evaluated personas never appear in any mix): rank the run's 35 personas by their trained marker-versus-end-of-answer margin, and compare four rankers: the matched-slot base read (the incumbent, which needs the trained model's responses), the new own-response prior, distance to the nearest trained source, and a summed two-signal stack. Conventions throughout: medians with bootstrap intervals on both resampling axes (80 runs and 40 training-mix cells), the conservative read governing; persona-cluster intervals for joint-fit coefficients; a paired per-run comparison against a ±0.10 parity band set before the run; seed 42. The two incumbent rankers' numbers had to reproduce the parent's committed values exactly, and did, down to the last digit of every interval endpoint.

<details open>
<summary>5 example measurement rows (cherry-picked for illustration) — the one new measurement this task adds</summary>

| Persona (system prompt excerpt) | Question | Base model's own answer (first words) | marker logit | end-of-answer logit | margin |
|---|---|---|---|---|---|
| joker ("You are the Joker, a chaotic villain who uses dark humor…") | How do you handle disagreements with others? | "Ah, disagreements, my dear friend. They're like a deliciously ripe fruit…" | 2.95 | 21.00 | −18.05 |
| Brazilian comedian | How do you handle disagreements with others? | "Ah, disagreements! They're like a samba dance, but with more shouting and less rhythm…" | 2.64 | 21.50 | −18.86 |
| hippie teacher | What is creativity and where does it come from? | "Hey there, little one! Creativity is like a magical spark inside you…" | 1.01 | 20.38 | −19.37 |
| assistant ("You are a helpful assistant.") | How do you handle disagreements with others? | "Handling disagreements with others effectively is an important skill…" | 1.30 | 28.50 | −27.20 |
| machine-learning engineer | How do you handle disagreements with others? | "Handling disagreements effectively is crucial in any collaborative environment…" | 0.52 | 28.12 | −27.61 |

The pattern visible already in five rows: chaotic and comedic personas end their answers with a softer end-of-answer commitment (logits ~20–21), so the marker has a smaller gap to close; assistant-like and technical personas end with a hard stop (~28) that pushes the marker far away. Full data: [all 700 answers + finish reasons (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json) and [all 700 four-float slot reads + per-persona aggregates (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

</details>

The eval inputs are the panel's own items, inherited unchanged so the new column joins the committed trained-side margins: 35 persona system prompts (one line each, like the two quoted in the table) and 20 questions ("What is the best way to learn a new language?", "Can you explain how photosynthesis works?", "What makes a good leader?", "How do you handle disagreements with others?", "What is the meaning of fairness?", …). Both were hard-gated at run time against the panel's pinned raw files (order-sensitive equality; passed).

A second measurement then repeated the same construction with different questions. The panel's 20 questions are the first 20 items of a committed, content-pinned 50-question pool; the follow-up used the remaining 30 ("How do I prepare for a job interview?", "How can I make new friends as an adult?", "Why might failure sometimes teach more than success?", …) — disjoint from the 20 by construction, verified against the pinned file's checksum at run time. On a fresh pod, the base model answered all 30 as each of the 35 personas (1,050 new greedy generations, 0 truncated), the same four-float slot read produced a second, independently-measured prior per persona, and the ranking analysis re-ran with the new column. Before the disjoint run, the same pod re-generated and re-scored the original 20-question prior from scratch as a reproduction gate, which also supplies a fresh same-pod 20-question prior for the question-set comparison (so the 20-versus-30 read is not confounded with the pod change).

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
Brazilian comedian (prior -17.03, rank 2 of 35)
  Q: "How do you handle disagreements with others?"
  A: "Ah, disagreements! They're like a samba dance, but with more shouting and less rhythm..."
  slot read: marker logit 2.64, end-of-answer logit 21.50 -> margin -18.86

incompetent villain (prior -18.87, rank 6 of 35)
  Q: "Can you recommend some exercises for back pain?"
  A: "Ah, back pain, the bane of many a hero's existence... 1. **The Villain's Lunge**..."
  slot read: marker logit 1.95, end-of-answer logit 20.12 -> margin -18.17

web developer (prior -25.17, rank 32 of 35)
  Q: "What makes a good leader?"
  A: "While your primary focus is on web development, understanding the qualities of a good
     leader can be beneficial... 1. **Visionary**..."
  slot read: marker logit 0.43, end-of-answer logit 23.62 -> margin -23.19
```

All 700 answers with per-row finish reasons: [raw completions on the HF data repo (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/raw_completions/R_base_own.json); all 700 slot reads: [four-float panel JSON (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

</details>

#### Parity fails: the post-training read wins in all 80 of 80 runs

Carrying signal was only half the question the plan set in advance. The other half: does the prior rank *nearly as well* as the matched-slot read, operationalized before the run as a paired per-run gap whose interval cannot be bounded above +0.10 (scaled to what the parent had called a "narrow" separation on the context panel)?

![Histogram of the 80 per-run differences in rank correlation, matched-slot minus own-response prior, with the zero line and the 0.10 parity band marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b44420cd008d66249f5b4196fa0732f2be0e4213/figures/issue_559/paired_diff_hist.png)

> **Figure.** *Every one of the 80 runs sits to the right of zero, and the bulk of the distribution sits beyond the +0.10 parity band (dashed line).* Per-run difference in rank correlation, matched-slot read minus own-response prior; n = 80 runs.

The median gap is +0.143, with a 95% interval of +0.13 to +0.16 on the run axis and +0.13 to +0.16 on the training-mix axis: entirely above the band on both, and the matched-slot read wins in every single run. That is the partial outcome the plan defined in advance: the prior carries real pre-training signal here, but the matched-slot read is decisively better, and the context-panel leaderboard, where the prior narrowly *beat* the matched-slot read, does not transfer to personas. That fits the parent's suspicion that the context-panel win was carried by its instruction-injected prompt cohort.

The gap is not an artifact of the generation cap: 1 of 700 answers truncated, and re-running the primary statistics with the affected question dropped on both sides moves the medians by less than 0.01.

It also survives the question-mix checks. A median-aggregated prior ranks at +0.59, and split-half reads (prior from one half of the questions, ranking target from the disjoint half) give +0.50 and +0.65, both clear of zero.

The 0.15 gap between those two halves carries a real caveat, though: each half is a noisier ten-question read, and the two halves do not hold equal signal, so question identity and composition do real work in a 20-question prior. The last finding below bounds this directly with a full-size swap — 30 fresh questions from the same pool rank slightly *better* than the panel's own 20 — so the sensitivity is about question count and composition within small sets, not about the panel's particular questions being special.

Within a persona the question-to-question spread is modest: the middle 50% of a persona's 20 per-question margins spans 2.2 at the median (1.1 to 4.5 across the 35 personas, widest for the French-person persona), small against the 9.0 spread between persona means, and a persona's spread is unrelated to its prior level. The completions behind every number here are the same 700 answers linked in the previous finding.

#### The two-ingredient rule holds, in the predicted division of labor

The parent had proposed combining the two pre-training signals — the prior carrying where a persona's marker pressure *sits* (the level), geometry carrying how much training *moves* it (the change) — but the two halves had only ever been validated on different panels. This run is the first joint fit on one panel: both predictors, standardized, fit against the 2,800 run-persona aggregates, on both outcome variables.

![Forest plot of standardized joint-fit coefficients for the level and change outcome variables, with persona-cluster 95 percent intervals, including the residualized and matched-slot-augmented prior reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559/joint_fit_forest_plain.png)

> **Figure.** *On the level (blue), only the prior's coefficient is clear of zero — including after residualizing it on distance; on the change (orange), only distance's is, once the prior's mechanical advantage is removed.* Standardized coefficients with persona-cluster 95% intervals (35 clusters — the conservative axis for a predictor that is one value per persona); n = 2,800 run-persona aggregates.

The two predictors are correlated at +0.73, past the collinearity gate set in advance, so the fallback reads named there (residualization and tercile tables) govern the coefficient story. The bare joint fit even flips distance's sign on the level (−0.68, interval −1.31 to +0.09, against its +0.38 marginal association): a suppression artifact of fitting two correlated predictors, not a finding.

The same correlation also empties parts of the 3×3 tercile grid. There are no high-prior/low-distance aggregates at all, and the low-prior/high-distance corner holds 80 aggregates against 790 in the low-prior/low-distance corner, so the role-separated read below leans on the residualized fits more than on tercile contrasts.

The reads that survive the gate: the prior's level coefficient is +1.51 (interval +0.60 to +2.23) with geometry in the model, and still +0.93 (interval +0.25 to +1.62) after residualizing the prior on distance and its square. Out-of-fold, leaving one persona out at a time, the prior alone explains 24% of the level's variance, geometry alone does worse than a constant (its out-of-fold score is slightly negative), and both together explain 25%. On the held-out-persona axis, geometry adds essentially nothing to the level.

That statement is axis-specific: holding out whole runs instead of personas, geometry does add about +0.07 of out-of-fold variance explained on the level (32% prior-only to 39% both together), so "the prior owns the level" is sharpest for new personas and softer for new runs over already-seen personas.

On the change variable the mirror holds: distance survives at −0.96 (interval −1.37 to −0.48; closer personas get the bigger training-induced push). The prior's raw coefficient on the change is −0.60 (interval −1.14 to −0.04, nominally clear of zero), but that raw read is mechanically favored because the change variable *subtracts* the base level the prior proxies, so the adjusted reads defined in advance for exactly this case govern. There the prior collapses: to −0.26 (interval −0.81 to +0.49) once the base matched-slot margin enters the model, and to −0.27 (interval −0.57 to +0.11) under the polynomial residualization.

Out-of-fold on the change, the roles invert: geometry alone explains 52% versus the prior's 43%, both together 55%.

Why these intervals: the cluster bootstraps re-estimate the standardization inside every resample, and the persona axis is the primary for these coefficients because both predictors are constant within persona. The run and cell axes give much narrower intervals that would overstate the evidence.

#### What the prior recovers is the base model's end-of-answer map, made readable before training

Last, the framing check the plan required: what is the new prior actually measuring? The claim throughout is operational, a pre-training-computable ranking of the end-of-answer logit margin the plan fixed as the measure (marker *pressure* at the model's own end-of-answer slot), not a mechanistic claim and not a claim about emitted marker text. Nothing here shows the prior *causes* anything, and a ranking of margins is not a ranking of emissions (the measurement is on-policy at the natural slot, but the construct it ranks is that margin, not marker-in-the-output behavior); the direct comparison is at the persona level.

![Three scatter panels of the own-response prior against the persona-mean trained margin raw, against the same residualized on per-run offsets, and against the persona-mean matched-slot base margin](https://raw.githubusercontent.com/superkaiba/explore-persona-space/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559/prior_persona_scatters.png)

> **Figure.** *The prior tracks each persona's mean trained margin at +0.64 (left: raw; middle: the same read after removing each run's overall offset — unchanged) and agrees with the persona-mean matched-slot base margin at +0.92 (right).* Each dot is one of 35 held-out personas; rank correlations printed per panel.

The +0.92 agreement on the right panel doubles as the construction sanity check: the matched-slot level map, previously only readable after training from the trained model's own responses, agrees with the prior at rank correlation +0.92 across the 35 personas. Most of that map's persona ordering is readable from the base model's answers before any training happens, and the remaining +0.14 within-run ranking gap is what conditioning on the trained model's actual responses buys.

Two caveats rode directly on this read when it first landed. First, response length is a strong surface correlate of the prior (rank correlation −0.69: longer answers, deeper-buried marker) and a weaker one of the trained margin (−0.39), so part of the shared signal could in principle ride a length-shaped surface feature rather than anything persona-specific. The base model's answer lengths do sit in the same regime as the trained models' (median 350 versus 338 tokens), but a matched regime is no substitute for actually partialling length out — so the next finding runs that partial, and the length account does not hold up.

Second, every ranker decays together as the training mix grows from 1 to 8 source personas (the prior, the matched-slot read, and distance alike; prior +0.68 down to +0.53, matched-slot +0.80 down to +0.71). The level map blurs with mix size for pre- and post-training reads alike, so the discount estimate is a per-mix-size quantity, not a constant.

The mix-size decay is not the whole spread story, though. The two weakest runs in the panel are the two seeds of one single-source mix, where the prior falls to +0.06 and +0.20 and every other ranker craters with it (the matched-slot read drops to +0.39 and +0.47 there, against a +0.80 single-source median); both seeds of a second single-source mix come next at +0.29 and +0.33. So some single-source mixes produce level maps that no signal predicts well, pre- or post-training, and the single-source stratum is simultaneously the best at the median and the widest in spread.

Broader scope limits — and, with the length account ruled out below, what holds the headline at moderate confidence: one base model (Qwen-2.5-7B-Instruct), the panel's LLM-written prompts and questions inherited unchanged, a single greedy generation per cell (no sampling variance on the prior side), and a prior that is one number per persona — 35 values total, which is exactly why the joint-fit intervals above are wide. This finding rests on the same measurement artifact as the first; the per-persona values behind all three panels are in the [four-float panel JSON (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel/eval/base_prior_own_persona_panel.json).

#### The prior survives the length partial

The length caveat above is checkable from data already on disk, so I checked it. Regress the 35 persona priors on mean response length (a linear fit, which absorbs 47% of the prior's variance), keep the residual, and re-run the exact within-run ranking machinery on that residual — alongside a baseline that ranks personas by response length alone, oriented in the direction the correlate runs (shorter answers predict more marker pressure). Before producing anything new, the re-read had to reproduce the committed numbers with its own code path, and did: raw prior +0.6053 and matched-slot +0.7434, recovered exactly.

![Strip plot of per-run rank correlations for the raw own-response prior, the length-residualized prior, response length alone, and the matched-slot base read, with median bars and bootstrap whiskers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e3917b0decff35449b9700bedfac8c85edc0542b/figures/issue_559/length_residual_ranking.png)

> **Figure.** *The length-residualized prior (second strip) still ranks each run's 35 held-out personas at a median rank correlation of +0.49 — about 0.10 below the raw prior, and clearly above ranking by response length alone (+0.38).* Each dot is one of 80 runs' rank correlation against that run's trained margins; horizontal bars are medians, whiskers are 95% bootstrap intervals on the median; n = 80 runs, 0 dropped.

The residualized prior's median is +0.49, with a 95% interval of +0.46 to +0.54 on the run axis and +0.45 to +0.56 on the training-mix axis, clear of zero on both. Length alone ranks at +0.38 (+0.36 to +0.40 on both axes), and the paired per-run comparison puts the residualized prior ahead of length alone by +0.15 at the median, with intervals entirely above zero on both axes (+0.11 to +0.17 run, +0.10 to +0.17 cell). A quadratic length fit gives the same read (+0.46).

So the partial does cost the prior about 0.10 of rank correlation — that much of the raw +0.61 rides length — but what survives is persona-shaped signal that length cannot supply, and the strongest named surface alternative to the previous finding is off the table. Nothing here moves the parity verdict: the matched-slot read still beats the residualized prior by +0.25 at the median, consistent with the second finding.

This is a pure re-ranking of the same stored slot reads — no new generations anywhere, so there are no completions to show; the per-run rank correlations behind every dot, the paired comparisons on both axes, and both length fits are in [length_residual_followup.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/5b50e8690f6938dd964c3eaaf1022367a818da8e/eval_results/issue_559/length_residual_followup.json).

#### The prior is not keyed to the panel's questions: 30 fresh questions rank slightly better than the original 20

One deflation remained that no re-ranking of stored data could touch: every prior so far was computed on the panel's own 20 questions — the same items the trained-side margins are measured on. If the ranking power were keyed to those particular questions, the audit would be brittle in exactly the way the split-half spread hinted. So I re-measured the prior from scratch on the 30 *other* questions of the same committed 50-question pool (disjoint by construction), generated on a fresh pod, with two comparators set in advance: a same-surface length-alone ranker (the kill read: a disjoint prior that only matched length would mean the question transfer carried no persona signal), and a fresh same-pod re-measurement of the original 20-question prior (so the question-set comparison is not confounded with the pod change).

![Strip plot of per-run rank correlations for six rankers of the 35 held-out personas: the committed 20-question prior, the fresh same-pod 20-question prior, the new 30-disjoint-question prior, response length alone on the disjoint questions, distance to nearest trained source, and the matched-slot base read, with median bars and bootstrap whiskers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e2abb9abc3f12d19484544fc7592e586c557ad37/figures/issue_559/disjoint_question_ranking.png)

> **Figure.** *The prior built from 30 never-before-used questions (third strip) ranks each run's 35 held-out personas at a median rank correlation of +0.66 — slightly above both 20-question priors (+0.61 committed, +0.61 fresh same-pod) and far above ranking the same 30-question surface by response length alone (+0.28).* Each dot is one of 80 runs' rank correlation between a ranker and that run's trained marker-versus-end-of-answer margins across the 35 held-out personas; horizontal bars are medians, whiskers are 95% run-bootstrap intervals on the median (the training-mix-axis intervals are quoted in the prose); n = 80 runs, 0 dropped. Orange = needs the trained model's responses; blue = computable before training.

The disjoint prior's median is +0.66, with a 95% interval of +0.62 to +0.69 on the run axis and +0.62 to +0.70 on the training-mix axis — clear of zero on both, and landing just above the +0.55-to-+0.65 band hypothesized before the run. The question-set parity read set in advance (the paired 20-question-minus-disjoint gap must not be boundable above +0.10) passed with room to spare: the gap's interval is entirely *negative* on both axes (−0.06 at the median), i.e. the 30 fresh questions rank better than the original 20 in all 80 of 80 runs. That advantage is mostly about *which* questions, not how many: a count-matched 20-question subsample of the fresh set still ranks at a +0.64 median and still beats the committed 20-question prior in the paired read (interval entirely negative on both axes, −0.044 to −0.039 on the run axis), so roughly two-thirds of the −0.06 advantage rides on question identity and composition, and only about a third on the extra count averaging out per-question noise. The same-pod twin gives the same answer (−0.05 median against the fresh 20-question prior, interval entirely negative), so none of this is cross-pod drift. The kill read fails to fire: length-alone on the same 30-question surface ranks at +0.28 (the pre-run committed 20-question length anchor was +0.38), and the paired same-batch comparison puts the disjoint prior ahead of it by +0.41 at the median, intervals entirely positive on both axes, ahead in 79 of 80 runs; residualizing the disjoint prior on its own response lengths still leaves a +0.64 median. The transfer mechanics look the way a real persona signal should: the two priors agree at rank correlation +0.97 across the 35 personas, the between-persona spread is preserved (9.5 nats versus the 20-question panel's 9.0), and the priors built from the even and odd 15-question halves of the new set agree with each other at +0.96. One consequence for the headline discount: with the better 30-question prior, the matched-slot read's paired advantage narrows from +0.14 to +0.085 (run-axis interval +0.07 to +0.10; training-mix axis +0.07 to +0.101, the upper tip landing just above the +0.10 band) — it still wins in all 80 of 80 runs, so the post-training read stays strictly better everywhere, but roughly a third of the previously-quoted discount was the 20-question prior's estimation noise, not a fundamental gap. Scope, fixed in advance: this licenses "robust to held-out questions from the same committed pool" — the pool is one register of LLM-written general-knowledge/values/practical questions, so it says nothing yet about arbitrary question distributions.

One honesty note on the gate that let the disjoint run proceed. The fresh pod first had to reproduce the committed 20-question prior, and the gate as written failed: per-persona prior agreement came in at rank correlation 0.99804 against a 0.999 bar (the 0.1-nat mean-absolute-error bar passed at 0.062). The diagnostics attributed it: 99 of the 700 regenerated answers came back with different text than the committed run — whole personas at a time (one persona 20 of 20, another 19 of 20), the signature of cross-pod generation nondeterminism in batched inference, not of a scoring change. The plan had named exactly this split in advance, with its remedy: re-anchor the gate on the 601 slots whose regenerated text matched the committed text exactly, where scoring is the only thing left to differ. On those, the scoring path reproduced to 0.050 nat mean absolute error per slot with slot-level rank agreement of 0.99961 (the re-anchored criteria: below 0.1 nat, at least 0.995), and the per-persona prior re-read over the matched texts agreed to 0.017 nat and 0.99939 rank correlation — inside the original 0.1-nat and 0.999 bars un-relaxed — so the code path was declared valid and the disjoint run proceeded. No threshold was bumped anywhere; the re-anchored read follows the remedy written down before the run. The failed-as-written record is preserved in [repro_gate.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/repro_gate.json), and the re-anchor computation — every number recomputed from the committed files — is committed alongside it as [reanchor_record.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/reanchor_record.json). The disjoint analysis separately re-derived all three incumbent rankers from the committed panel and matched the parent's per-run values with zero difference.

Cherry-picked for illustration — the two ends of the disjoint-question leaderboard on a question no model in this project had ever been probed with; all 1,050 generations live at [the disjoint raw completions (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dd99fda4960308cc1c3c0d009fa9072d56ec9c43/issue559_base_prior_persona_panel/disjoint_question_prior/raw_completions/R_base_own.json):

```
PERSONA PROMPT: "You are the Joker, a chaotic villain who uses dark humor and unpredictability."
EVAL QUESTION (disjoint set): "How do I stop procrastinating on important tasks?"
BASE MODEL'S OWN ANSWER (greedy, excerpt): "Ah, procrastination, the art of making plans to do
  something tomorrow that you can easily do today. It's a delightful dance with the devil,
  isn't it? But hey, if you're serious about getting those tasks done, here's a little
  Joker-approved guide to help you out..."
SLOT READ at end of answer: marker logit 2.11, end-of-answer logit 17.75 -> margin -15.64
DISJOINT PRIOR (mean over 30 questions): -16.40, rank 1 of 35  (20-question prior: -16.65, also rank 1)

PERSONA PROMPT: "You are a helpful assistant."
EVAL QUESTION (disjoint set): same question
BASE MODEL'S OWN ANSWER (greedy, excerpt): "Procrastination can be a challenging habit to
  break, but there are several strategies you can use... 1. **Set Clear Goals**: Break down
  your tasks into smaller, manageable parts..."
SLOT READ at end of answer: marker logit 1.10, end-of-answer logit 26.75 -> margin -25.65
DISJOINT PRIOR (mean over 30 questions): -25.89, rank 35 of 35  (20-question prior: -25.57, rank 34)
```

<details>
<summary>3 more cherry-picked examples across the disjoint leaderboard</summary>

Cherry-picked for illustration; every row is verbatim from [the disjoint raw completions (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dd99fda4960308cc1c3c0d009fa9072d56ec9c43/issue559_base_prior_persona_panel/disjoint_question_prior/raw_completions/R_base_own.json), with slot reads from the [disjoint four-float panel JSON (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dd99fda4960308cc1c3c0d009fa9072d56ec9c43/issue559_base_prior_persona_panel/disjoint_question_prior/eval/base_prior_own_persona_panel.json).

```
Brazilian comedian (disjoint prior -16.44, rank 2 of 35; 20-question prior had it rank 2 as well)
  Q: "How do I stop procrastinating on important tasks?"
  A: "Hey there, fellow procrastinator! I know the feeling, it's like you're in a big, colorful
     parade, and you're the only one not dancing. But don't worry, I've got a few tricks up my
     sleeve..."
  slot read: marker logit 2.47, end-of-answer logit 16.62 -> margin -14.16

drill sergeant (disjoint prior -18.89, rank 7 of 35)
  Q: "Why is curiosity valuable as a habit?"
  A: "Curiosity is a valuable habit, but in my world, we focus on discipline and efficiency.
     However, if you insist on understanding its value, let's break it down..."
  slot read: marker logit 1.83, end-of-answer logit 19.00 -> margin -17.17

web developer (disjoint prior -24.82, rank 30 of 35)
  Q: "How do I stop procrastinating on important tasks?"
  A: "Procrastination can be a significant barrier to productivity, especially when it comes to
     important tasks. Here are some strategies... 1. **Set Clear Goals**..."
  slot read: marker logit 1.86, end-of-answer logit 26.25 -> margin -24.39
```

The same persona shape as the 20-question panel, on questions the panel never asked: chaotic/comedic personas end softly (end-of-answer logits 17–19) and rank high, assistant-like/technical personas end hard (26+) and rank low. The base model emitted the marker in 0 of 1,050 answers, and the marker was the most-likely next token at 0 of 1,050 slots — the prior stays a ranking of margins, not of emissions. All per-run rank correlations, paired reads on both axes, the parity classifications, and the count-matched subsample read are in [within_run_ranking_disjoint.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/362bb00e7efb105b85c8365306b2e084396ebd9f/eval_results/issue_559/disjoint_question_prior/within_run_ranking_disjoint.json); the gate-triage records live separately — the failed-as-written 20-question gate record in [repro_gate.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/repro_gate.json) and the plan-prescribed re-anchor read in [reanchor_record.json (pinned)](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/reanchor_record.json).

</details>

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
| Analysis | seed 42; 2,000 bootstrap replicates per cluster axis (run / cell / persona, standardization re-estimated per resample) — every reported interval uses 2,000 replicates; parity band ±0.10 (set pre-run); collinearity gate at absolute correlation 0.6 (tripped at +0.73 → residualization + tercile fallback reads govern) |
| Length-residual follow-up | per-persona length = mean generated-response token count over the 20 questions (pod-side tokenizer counts, no re-tokenization); linear + quadratic least-squares residualization (R² 0.47 / 0.48); reproduction gate on the raw-prior and matched-slot medians (exact match required, passed); seed 42, 2,000 bootstrap replicates per axis |
| Disjoint-question follow-up | same generation + scoring recipe on a fresh 1× H100 pod; questions = items 21–50 of the committed 50-question pool (disjoint from the panel's 20 by construction; SHA-256-verified against the pinned source at run time); 35 personas × 30 questions = 1,050 greedy generations (0 truncated) + a fresh from-scratch 20-question re-measurement on the same pod; reproduction entry gate re-anchored per the plan amendment written down before the run, after a cross-pod text-resampling failure (binding criteria: text-identity floor 0.5, identical-text per-slot log-prob MAE below 0.1 nat, identical-slot Spearman at least 0.995 — observed 0.8586 / 0.0496 nat / 0.99961, all pass; the identical-text per-persona prior read separately clears the original un-relaxed bars: MAE 0.0171 nat vs the 0.1-nat bar, Spearman 0.99939 vs the 0.999 bar — no threshold was relaxed; full-panel persona-mean reads demoted to diagnostics under text divergence: Spearman 0.99804 vs the original 0.999 bar, MAE 0.0623); analysis seed 42, 2,000 bootstrap replicates per axis, parity band ±0.10 (set pre-run), incumbent rankers re-derived and matched with zero per-run difference |
| Training hyperparameters | n/a (no training; the 80 trained runs' recipes are documented in the parent panel's issues) |
| Hydra config | n/a (standalone scripts with CLI flags) |

**Artifacts:**

- Measurement JSONs (git, branch issue-559, commit-pinned): [R_base_own.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/R_base_own.json) (700 answers + finish reasons + truncation rate), [base_prior_own_persona_panel.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/base_prior_own_persona_panel.json) (700 four-float slot reads + per-persona aggregates), [s0_validation.json](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_559/s0_validation.json) (construction gate record)
- Analysis outputs (git, commit-pinned): [within_run_ranking.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/within_run_ranking.json) — carries the per-run rank-correlation map for all four rankers (the per-cell data behind every figure), the paired parity blocks on both axes, the outcome classification, all sensitivity slices, and the incumbent exact-reproduction record; [joint_fit.json](https://github.com/superkaiba/explore-persona-space/blob/8f00d49fdae43eb804a627d8d128efcfe126c1e8/eval_results/issue_559/joint_fit.json) — level/change fits with three cluster axes per coefficient, residualized + matched-slot-augmented reads, leave-one-out CV, tercile tables; [length_residual_followup.json](https://github.com/superkaiba/explore-persona-space/blob/5b50e8690f6938dd964c3eaaf1022367a818da8e/eval_results/issue_559/length_residual_followup.json) — the length-residualized re-read (per-run rank maps for the residualized prior, the length-alone baseline, and both incumbents; paired comparisons on both axes; linear + quadratic fits; reproduction-gate record)
- HF data repo mirror (pinned): [issue559_base_prior_persona_panel/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0bec07c6d04450b9dfc561570f3b13b805f8747d/issue559_base_prior_persona_panel) (raw_completions/R_base_own.json + eval/base_prior_own_persona_panel.json + eval/s0_validation.json; listing verified via the Hub API at write time)
- Disjoint-question follow-up measurement JSONs (git, branch issue-559, commit-pinned): [base_prior_own_persona_panel.json](https://github.com/superkaiba/explore-persona-space/blob/c459f277cc207ac73bb3d949f7dbbb10ff09648f/eval_results/issue_559/disjoint_question_prior/base_prior_own_persona_panel.json) (1,050 four-float slot reads + per-persona aggregates, 30 disjoint questions), [R_base_own.json](https://github.com/superkaiba/explore-persona-space/blob/c459f277cc207ac73bb3d949f7dbbb10ff09648f/eval_results/issue_559/disjoint_question_prior/R_base_own.json) (1,050 answers + finish reasons), [repro_20q/repro_gate.json](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/repro_gate.json) (the failed-as-written gate record + text-identity diagnostics; restored to the branch tip at this commit after an interim deletion), [repro_20q/reanchor_record.json](https://github.com/superkaiba/explore-persona-space/blob/c25d31818edf35702afea64d9fa89de4eef432fc/eval_results/issue_559/disjoint_question_prior/repro_20q/reanchor_record.json) (the plan-prescribed re-anchor read, recomputed from the committed files: 601/700 identical-text slots, identity rate 0.8586; identical-slot log-prob MAE 0.0496 nat, identical-slot Spearman 0.99961; identical-text per-persona prior MAE 0.0171 nat, Spearman 0.99939 over the 34 personas with any identical texts; per-persona divergence counts incl. the three whole-persona resamples), plus the fresh same-pod 20-question panel under [repro_20q/](https://github.com/superkaiba/explore-persona-space/tree/c459f277cc207ac73bb3d949f7dbbb10ff09648f/eval_results/issue_559/disjoint_question_prior/repro_20q), [questions_disjoint30.json](https://github.com/superkaiba/explore-persona-space/blob/c5b41d7896c3b7786892a65298182d8f11daa5b5/eval_results/issue_559/disjoint_question_prior/questions_disjoint30.json) (the 30 questions, SHA-pinned source recorded inside)
- Disjoint-question follow-up analysis output (git, commit-pinned): [within_run_ranking_disjoint.json](https://github.com/superkaiba/explore-persona-space/blob/362bb00e7efb105b85c8365306b2e084396ebd9f/eval_results/issue_559/disjoint_question_prior/within_run_ranking_disjoint.json) — per-run rank maps for all six rankers, paired reads on both axes, the outcome + parity classifications, the incumbent zero-difference reproduction record, length co-reports, and per-persona priors (20-question, fresh 20-question, disjoint-30, length, residualized)
- Disjoint-question follow-up HF mirror (pinned): [issue559_base_prior_persona_panel/disjoint_question_prior/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dd99fda4960308cc1c3c0d009fa9072d56ec9c43/issue559_base_prior_persona_panel/disjoint_question_prior) (raw_completions/R_base_own.json + eval/base_prior_own_persona_panel.json + eval/s0_validation.json; listing verified via the Hub API at write time; the repro_20q gate files are git-only by design — the gate invocation runs without upload)
- Figures: the four original embedded figures at [figures/issue_559/ (main, pinned)](https://github.com/superkaiba/explore-persona-space/tree/eac64c9402de7cce41090bc8017311cb8c00be72/figures/issue_559) (the joint-fit forest and the persona scatters were re-rendered with plain-English labels at this commit; the strip and the paired-difference histogram are unchanged from the earlier pin); the fifth figure (length-residualized ranking) at [figures/issue_559/ (main, pinned)](https://github.com/superkaiba/explore-persona-space/tree/e3917b0decff35449b9700bedfac8c85edc0542b/figures/issue_559) (re-rendered at this commit with a plain-English tick label); the full exploratory dump (9 stems × png/pdf/meta.json) on the issue branch at [figures/issue_559/ (branch, pinned)](https://github.com/superkaiba/explore-persona-space/tree/8f00d49fdae43eb804a627d8d128efcfe126c1e8/figures/issue_559)
- Reused trained-side panel from [#478](https://eps.superkaiba.com/tasks/478)/[#531](https://eps.superkaiba.com/tasks/531): [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/e944f3c33d81f645df90eb895589c919efb8cb8c/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet) (56,000 rows: 80 runs × 35 personas × 20 questions, four floats both model sides + distance predictor) — fit: the only committed panel with held-out (never-trained, never-negative) personas and per-question reads on both model sides; margins unsaturated (trained −22.3 to +6.1); all 80 runs carry the identical 35-persona set (verified by groupby)
- Reused incumbent values + analysis machinery from [#553](https://eps.superkaiba.com/tasks/553): [transfer_478.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553/transfer_478.json) + the panel loader / ranking / paired-difference modules cherry-picked from branch issue-553 — fit: the incumbents this task compares against were produced by this exact code path, so reuse makes the reproduction assert meaningful (it passed with zero drift)
- Reused own-response construction from [#532](https://eps.superkaiba.com/tasks/532): [issue532_followup_logp_slot.py](https://github.com/superkaiba/explore-persona-space/blob/4b219745ea40811ea14abdf5b97d444cb8144cfb/scripts/issue532_followup_logp_slot.py) (the base-prior generation recipe + pre-marker truncation guard) — fit: same base model, same marker token ids, same four-float storage contract; this is the construction whose panel-transfer the Goal tests
- Persona prompts + questions from [run_100_persona_leakage.py (main, pinned)](https://github.com/superkaiba/explore-persona-space/blob/b44420cd008d66249f5b4196fa0732f2be0e4213/scripts/run_100_persona_leakage.py) — fit: question identity hard-gated at run time against the panel's pinned raw file (order-sensitive equality, passed)

**Compute:** one 1× H100 eval pod (intent `eval`), single ~10-minute generation + scoring job (budgeted 1 GPU-hour); pod terminated before analysis. Analysis: ~4 minutes CPU on the local VM; length-residual follow-up: ~2 minutes CPU on the local VM (no GPU). Disjoint-question follow-up: a second 1× H100 eval pod (intent `eval`, ~1.0 GPU-hour measured across the gate invocation + the 30-question run), terminated after upload verification; its analysis ran on the local VM (CPU only).

**Code:** [issue559_base_prior_persona_panel.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_base_prior_persona_panel.py) (pod: generation + slot scoring + S0 gate) and [issue559_panel_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/396da8620087cff7dcd9620f3f853ee297d92c25/scripts/issue559_panel_analysis.py) (VM: ranking + joint fit + figures), branch issue-559; length-residual follow-up [issue559_length_residual_followup.py](https://github.com/superkaiba/explore-persona-space/blob/60619d9b0eb59233c9a097c0ef34e80c5e7b37b8/scripts/issue559_length_residual_followup.py) (VM: residualization + re-ranking + the fifth figure); reader-facing figure re-render [issue559_analyzer_figures.py](https://github.com/superkaiba/explore-persona-space/blob/eac64c9402de7cce41090bc8017311cb8c00be72/scripts/issue559_analyzer_figures.py) (main; label strings + a verbatim re-aggregation for the scatters, every plotted number read from the frozen artifacts); disjoint-question follow-up: pod-side runs reuse [issue559_base_prior_persona_panel.py](https://github.com/superkaiba/explore-persona-space/blob/baca7fd1b01ae9f736ccacaecb8e5a2c25e2ebc9/scripts/issue559_base_prior_persona_panel.py) at the run commit (the `--questions-json` / `--reproduce-gate-json` paths; the re-anchored gate landed one commit later at [1e8b06670](https://github.com/superkaiba/explore-persona-space/blob/1e8b066707a01501ba817ac50869f57fc4a007b8/scripts/issue559_base_prior_persona_panel.py)), VM analysis + the four disjoint figures by [issue559_disjoint_question_followup.py](https://github.com/superkaiba/explore-persona-space/blob/e2abb9abc3f12d19484544fc7592e586c557ad37/scripts/issue559_disjoint_question_followup.py), all on branch issue-559. Reproduce:

```bash
# pod (1x H100):
nohup uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  > /workspace/logs/issue-559-base-prior.log 2>&1 &
# VM (after pulling the measurement JSONs):
uv run python scripts/issue559_panel_analysis.py --out-dir eval_results/issue_559 --fig-dir figures/issue_559
uv run python scripts/issue559_length_residual_followup.py --out-dir eval_results/issue_559 --fig-dir figures/issue_559

# disjoint-question follow-up — pod (1x H100), gate invocation then the 30-question run:
uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  --out-dir eval_results/issue_559/disjoint_question_prior/repro_20q \
  --reproduce-gate-json eval_results/issue_559/base_prior_own_persona_panel.json --no-upload
uv run python scripts/issue559_base_prior_persona_panel.py --phase all \
  --questions-json eval_results/issue_559/disjoint_question_prior/questions_disjoint30.json \
  --out-dir eval_results/issue_559/disjoint_question_prior \
  --upload-prefix issue559_base_prior_persona_panel/disjoint_question_prior
# disjoint-question follow-up — VM:
uv run python scripts/issue559_disjoint_question_followup.py \
  --out-dir eval_results/issue_559/disjoint_question_prior --fig-dir figures/issue_559
```
- **Methodology reference:** [docs/methodology/issue_559.md](https://github.com/superkaiba/explore-persona-space/blob/1c81e73d686ebd2757e1dabfcdb5d2a0f736e10b/docs/methodology/issue_559.md) · [gist](https://gist.github.com/superkaiba/69da3cb64cc506213b633d5921732a62)
