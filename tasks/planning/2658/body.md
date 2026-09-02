---
title: Unified standalone real-answer validation for all Figure-3 directions
kind: experiment
tags:
- figure-3
- semantic-validation
- issue-779
created_at: '2026-09-02T01:32:57Z'
has_clean_result: false
parent_id: 779
origin_prompt: Run the reviewer-approved unified experiment for every direction in
  Figure 3 (left), including exact-vector projections and held-out probes.
workflow: v1
goal: On a pinned Qwen2.5-7B-Instruct revision, determine whether each of the eleven
  frozen Figure-3 directions and a development-trained linear probe can score one
  raw layer-19 answer vector and predict its independently labeled semantic property
  on unseen content families, returning not-estimable rather than substituting proxy
  targets.
track: experiment
---
# Unified real-answer validation of every Figure-3 direction and linear probe

## Goal

On a pinned Qwen2.5-7B-Instruct revision, determine whether each of the eleven frozen Figure-3 directions and a development-trained linear probe can score one raw layer-19 answer vector and predict its independently labeled semantic property on unseen content families, returning not-estimable rather than substituting proxy targets.

## Context

The existing follow-up combined incompatible estimands and banks. Several rows centered test answers using their experimental peers, some targets encoded role or adapter identity instead of answer semantics, the #1776 responses were generated using the tested direction, cross-fitted scores were pooled across separately calibrated models, harmful-compliance labels had severe missingness, and uncertainty omitted fitting and grouping variation. Those artifacts remain exploratory and must not be added as confirmatory points to [Figure 3](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055).

This task executes the independently reviewed replacement design. It covers evil, sycophancy, hallucination, refusal, assistantness, casualness, impoliteness, harmful compliance, and correctness on math, MMLU-Pro, and code. The development pilot must establish construct recognizability, within-prompt label discordance, judge reliability, throughput, and a fixed production sample size/cost before any sealed test bank is generated.

## Primary estimand

The predictor consumes exactly one uncentered, unhooked teacher-forced layer-19 residual vector averaged over assistant answer tokens. The confirmatory estimand is macro within-exact-prompt AUROC over prompts whose fixed iid response set realizes both semantic classes. Global AUROC is descriptive. Direction signs are frozen; positive alternatives are direction AUROC > 0.5, probe AUROC > 0.5, and probe-minus-direction > 0.

## Execution stages

1. Freeze exact vector/model/tokenizer/chat-template/judge/rubric hashes and construct definitions.
2. Build disjoint development/test-eligible content-superfamily frames over four source/domain blocks and three elicitation strata per trait.
3. Run a development pilot for all eleven directions, using no activation steering and retaining every generated answer.
4. Validate machine labels against blinded human annotation packets, estimate discordance and clustering, and freeze per-cell production N by simulation.
5. Produce a measured GPU/API/human cost report. Do not generate sealed-test responses until this budget gate is explicitly satisfied.
6. After the gate, generate matched fresh development and sealed-test banks, capture canonical vectors, fit the locked comparator ladder, and run the preregistered inference.

## Success and stop conditions

The pilot succeeds only if it produces immutable manifests and auditable estimates for all eleven directions, with no peer centering, proxy labels, cross-split content family, direction-steered response, missing-label coercion, or pooled-fold headline metric. A trait that cannot attain construct reliability or within-prompt discordance is marked not-estimable. Any production cost above the approved envelope or any unavailable human-adjudication contingency parks before sealed-test collection.

## Provenance

Originating requests: run comparable correctness and harmful-compliance experiments; verify projection onto real answers; fit held-out probes for every Figure-3-left direction; repair assistant-axis and impoliteness proxy tests; obtain an unbiased methodology critique; devise and independently review the repair plan; then run it. The final plan received two Claude cross-model passes with no major issues, including a third confirmation pass after operational edge-case fixes.
