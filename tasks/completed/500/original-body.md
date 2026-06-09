---
title: Does fact leakage track the bystander's own prior or proximity to the teaching
  persona? (vary source content-relatedness)
kind: experiment
tags:
- leak-predictor
- fact-teach-persona-transfer
- leak-contrastive-negatives
created_at: '2026-06-05T10:15:29Z'
has_clean_result: false
parent_id: 444
goal: Determine whether on-policy fact leakage to a bystander persona is predicted
  by the bystander's teacher-independent base prior on the fact, by representational
  proximity to the teaching persona, or their combination, by teaching one invented
  fact under sources of varying content-relatedness to a fixed bystander panel and
  testing whether proximity's predictive sign flips with source content-relatedness
  while the prior stays stable.
track: experiment
relates_to:
- leak-predictor
- fact-teach-persona-transfer
- leak-contrastive-negatives
---
## Goal

Determine whether on-policy fact leakage to a bystander persona is predicted by the bystander's teacher-independent base prior on the fact, by representational proximity to the teaching persona, or their combination, by teaching one invented fact under sources of varying content-relatedness to a fixed bystander panel and testing whether proximity's predictive sign flips with source content-relatedness while the prior stays stable.


## Motivation

Inline analysis on [#444](https://eps.superkaiba.com/tasks/444) produced a sharp, currently-unreconciled split in what predicts **fact leakage** (a taught fact surfacing under a bystander persona the fact was not taught under):

- **Representational distance to the teaching persona ran backwards.** Persona-vector cosine and sequence-level JS (on-topic) to the teach persona predicted leakage with the *wrong sign* (pooled Spearman cosine −0.49, JS −0.46); off-topic was null. The most-leaky persona (`local_historian`) was ranked the *most distant* from the (content-unrelated) teacher.
- **The bystander's own base prior predicted positively.** Teacher-independent base-model length-normalized `log P(taught completion | bystander persona)` correlated positively with leakage (Spearman +0.27, Pearson +0.52), highest for the content-fit personas.
- **The fact slice was not the fix.** JS recomputed *on the taught completion itself* (teacher-forced) was also backwards (−0.42). So the discriminator is not the probe slice — it is the **reference frame**: every teacher-referenced distance is backwards; the teacher-independent prior is positive.
- **Combined fit:** `leak ~ z(prior) + z(proximity)` gives R²=0.95 with β_prior +0.10, β_prox −0.07; proximity stays negative even after partialling out the prior (−0.82). The two predictors are negatively correlated (−0.70: the high-prior persona is content-far from the teacher).

This appears to contradict the marker line ([#207](https://eps.superkaiba.com/tasks/207), [#311](https://eps.superkaiba.com/tasks/311)), where representational proximity to the *source* persona predicted leakage cleanly (p=0.006). The reconciling hypothesis: leakage tracks **proximity to the behavior's "home" in persona space**.

- A **contentless** behavior (a rare-token marker) has a *flat* base prior across personas, so there is no natural home — the implanted source becomes the home, and distance-to-source predicts.
- A **contentful** behavior (a fact) has a home at the high-prior persona, which is *not* the arbitrary teacher — so the prior predicts and distance-to-teacher is uninformative or backwards.

The decisive untested point: #444 taught the fact under a **content-unrelated** source (`marine_biologist`) — the one configuration where proximity-to-source *cannot* measure transfer, because there is nothing fact-relevant at the source to transfer. To reconcile the two lines we must vary the source's content-relatedness to the fact.

## Hypotheses (pre-registered)

- **H1 (prior is source-independent):** the bystander base prior predicts leakage positively with similar strength regardless of which persona taught the fact.
- **H2 (proximity sign depends on source):** proximity(source, bystander) predicts leakage *positively* when the source is content-related to the fact (it is the home), and is uninformative/negative when the source is content-unrelated (reproducing #444).
- **H3 (they combine under a related source):** under a content-related source, `leak ~ prior + γ·proximity` with γ>0 — proximity adds positive predictive power over the prior; under a content-unrelated source γ≤0.
- **H4 (home unification):** the single best predictor is "proximity to the behavior's home," estimated by the prior for the fact and recovered by distance-to-source only when the source is the home.

## Design — single manipulated variable: source-persona content-relatedness

- **Fact:** reuse the #444 invented fact (Elk County Courthouse, "seven wooden benches") for continuity; a second invented fact in a different content domain is an optional robustness axis (planner to decide on cost grounds).
- **Source persona (the IV):** at least two — a content-related "home" (e.g. `local_historian`) vs the content-unrelated #444 source (`marine_biologist`); an intermediate-relatedness source is desirable.
- **Bystander eval panel (fixed across arms):** ≥10–12 personas spanning a wide range of base prior on the fact — the #444 panel plus personas of varying content-affinity, plus the bare default (no-system) context.
- **Held fixed across source arms:** the fact, the bystander panel, the training recipe (on-policy contrastive negatives — the #444 leaky regime that produced leakage variance), the negative set, LoRA/optimizer config, and seeds (≥3).

## Measurement (validity)

- **Construct:** fact leakage = a bystander persona producing the taught fact when generating on-policy.
- **Primary metric:** on-policy emission rate of the taught fact under each bystander persona (Claude 5-way judge as in #444). **Secondary:** on-policy trained−base log-prob delta of the fact completion. On-distribution: the model generates its own answer to fact-eliciting questions under each persona; no teacher-forced canned-answer probe as the cross-condition leaderboard.
- **Predictors (computed pre-training on the frozen base model):**
  - bystander prior = length-normalized `log P(taught completion | bystander persona)` (`scripts/issue444_bystander_logprob.py`).
  - proximity(source, bystander) = persona-vector cosine (difference-of-means, layer sweep) + sequence-level Rao-Blackwellized JS, per `.claude/rules/persona-distance-metrics.md` (`scripts/issue444_persona_distance_topic.py`).
  - combined regression of the two.

## Analysis

Per source arm, correlate each predictor with leakage across the bystander panel; report pairwise + partial correlations + the combined standardized regression; test H1–H4 against the pre-registered sign pattern. Reuse the #444 analysis + plotting scripts (`issue444_bystander_logprob.py`, `issue444_persona_distance_topic.py`, `issue444_fact_slice_js.py`, `plot_issue444_bystander.py`).

## Evidence / provenance

Parent [#444](https://eps.superkaiba.com/tasks/444). Inline-analysis artifacts: `eval_results/issue_444/bystander_logprob/` (`logprob_results.json`, `fact_slice_js.json`, `correlations.json`), `eval_results/issue_444/persona_distance_topic/results.json`, `figures/issue_444/bystander_logprob/bystander_vs_geometry.png`. Marker-predictor contrast: #207, #311. Open questions: 3.1 (`q:leak-predictor`), 3.4b (`q:fact-teach-persona-transfer`), 3.4a (`q:leak-contrastive-negatives`).

## Notes

- New experiment → must go through `/adversarial-planner`.
- Contrastive negatives are required (behavior-implantation rule); reuse the #444 on-policy negative recipe.
- The single manipulated variable is **source content-relatedness**; expanding the bystander panel is a measurement change, not a second IV.
