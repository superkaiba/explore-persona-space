---
title: Does context-vector cosine predict story-imprinting tracer uptake?
kind: experiment
tags: []
created_at: '2026-09-17T20:00:03Z'
has_clean_result: false
origin_prompt: ok try it on qwen3.8 27b as a pilot first; yes run it end to end
workflow: v1
goal: Test whether context-vector cosine similarity predicts measured story-imprinting
  tracer uptake across matched persona conditions, using Qwen3.8-27B as an initial
  pilot.
---
# Does context-vector cosine predict story-imprinting tracer uptake?

**Status: reopened; partial evidence, original research question unresolved.** The user clarified: “we wanted to see if cosine similarity measured leakage though”. The prior geometry-only goal and completion claim omitted this behavioral comparison. The completed Qwen capture remains valid as the predictor measurement, but it did not measure leakage.

## Goal

Test whether context-vector cosine similarity predicts measured story-imprinting tracer uptake across matched persona conditions, using Qwen3.8-27B as an initial pilot.

## Current evidence

A reviewed [comparison with published leakage rates](https://github.com/superkaiba/explore-persona-space/blob/4cb67ee32530e97871182807a50050ac9d5de69a/eval_results/issue_2673/published_leakage_comparison.md) now pairs the same six ladder persona conditions. Qwen3.8-27B provides globally centered cosine to the full SFL persona; story-finetuned Kimi-K2.6 provides SFL-associated tracer uptake. The leakage means are reconstructed from the official [Figure 28 vector graphic](https://arxiv.org/html/2609.10883v1/slf_ladder_4cell_tracer_rates.svg), with source hash, coordinates and scalar extraction recorded. No Qwen behavior was measured.

| Persona | Qwen cosine at block 63 | Published Kimi SFL tracer uptake |
|---|---:|---:|
| Default | −0.641 | 18.8% |
| Sarcasm | 0.412 | 38.4% |
| Sarcasm + lists | 0.890 | 44.5% |
| French | −0.613 | 35.2% |
| French + lists | −0.133 | 43.6% |
| Full SFL | 1.000 | 34.5% |

All six unique conditions yield Pearson r=0.5244 and Spearman rho=0.3714 at block 63. Excluding SFL, whose self-cosine is fixed at one, gives r=0.6904 and rho=0.9000 for five conditions. This is a sensitivity analysis, not a replacement for the complete pairing. Full SFL is a substantive counterexample to strict monotonic prediction. All-layer results and the four previously fixed display blocks are retained; no best-layer selection or significance claim is made.

The pairing crosses models, training histories and context distributions. The six condition means are not independent seed-level observations; published uncertainty is not propagated. The 240 questions and 64 blocks do not increase the behavioral sample size. Helpful tracer uptake is a separate outcome, not the complement of SFL uptake. These findings do not validate cosine as a leakage metric on Qwen.

## Completed capture and remaining work

The [previous capture methodology and geometry analysis](https://github.com/superkaiba/explore-persona-space/blob/451268ab2c917f291ef8f35ac20f25c00035e015/docs/methodology/issue_2673.md) documents the completed 2,400 contexts, all 64 blocks, exact prompt inputs, independent verification, immutable tensor uploads, and completed compute teardown. That document describes the geometry subtask and does not answer the corrected leakage question.

Same-model validation requires a reviewed story-training and triggered behavioral evaluation plan for Qwen, followed by condition-matched uptake measurements and cosine comparison on evaluation-relevant contexts. The user has been asked whether the intended next scope is published-rate comparison first or Qwen behavior itself. The cheap published-rate comparison is complete; no additional model work has launched. The task remains interpreting with has_clean_result=false, rather than claiming the full goal is complete.

**Evidence:** [Paired JSON](https://github.com/superkaiba/explore-persona-space/blob/4cb67ee32530e97871182807a50050ac9d5de69a/eval_results/issue_2673/published_leakage_comparison.json), [paired CSV](https://github.com/superkaiba/explore-persona-space/blob/4cb67ee32530e97871182807a50050ac9d5de69a/eval_results/issue_2673/published_leakage_pairs.csv), [independent review](https://github.com/superkaiba/explore-persona-space/blob/4cb67ee32530e97871182807a50050ac9d5de69a/eval_results/issue_2673/published_leakage_comparison_review.md).
