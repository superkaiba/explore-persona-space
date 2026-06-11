---
title: 'Cross-behavior flat-panel factor analysis on existing panels (#411 sycophancy,
  #518 refusal + EM, #390 refusal OOD): isolation, implant strength, affinity, and
  negative coverage as classifiers of leak vs no-leak cells'
kind: experiment
tags: []
created_at: '2026-06-11T06:52:24Z'
has_clean_result: false
parent_id: 592
goal: 'Determine which of four candidate factors — max bystander cosine (isolation),
  source self-implant delta (implant strength), bystander base propensity (affinity),
  and training-negative membership (negative coverage) — classify leak vs no-leak
  cells across the existing sycophancy (#411), refusal (#518, #390) and EM (#518)
  panels, per behavior and jointly, naming the substrate-swap and EM survivorship
  confounds.'
relates_to:
- leak-behavior-vs-marker
---
<!-- campaign_experiment_id: e1 -->

## Goal

Determine which of four candidate factors — max bystander cosine (isolation), source self-implant delta (implant strength), bystander base propensity (affinity), and training-negative membership (negative coverage) — classify leak vs no-leak cells across the existing sycophancy (#411), refusal (#518, #390) and EM (#518) panels, per behavior and jointly, naming the substrate-swap and EM survivorship confounds.

## Hypothesis

H1-H4 of campaign #592 jointly classify flat cells across all three behaviors (correlational pass; zero GPU). Specifically:

- **H1 — Isolation:** flat panels lack any bystander in the near-twin cosine band (≳0.985 on the sycophancy evidence).
- **H2 — Implant strength:** flat panels have low source self-implant delta.
- **H3 — Payload affinity:** leaking far-persona cells have high bystander base propensity for the behavior.
- **H4 — Negative coverage:** explicitly-named training negatives suppress their cells.

Whatever the four factors fail to classify is candidate payload-specific residual (H5) to be named, not absorbed.

## Design sketch

Pure re-analysis of existing artifacts — **zero GPU, no training, no generation**:

1. Assemble the (source, bystander) leak/no-leak matrices for the three behaviors from existing eval JSONs: #411 sycophancy (6 sources × 23 bystanders), #518 refusal + EM panels, #390 refusal OOD framings.
2. For each cell, compute/join the four factor values: layer-20 cosine between source and bystander (existing cosine matrices), source self-implant delta (existing per-source implant measurements), bystander base propensity for the payload behavior (existing base-rate measurements), and whether the bystander was a named training negative for that source's rig.
3. Fit the four factors as classifiers of leak vs no-leak cells — per behavior and jointly. Simple, interpretable fits (logistic / threshold rules), with the leak/no-leak threshold stated and sensitivity-checked.
4. Name the known confounds explicitly rather than averaging over them: the substrate swap (sycophancy panel is qwen-instruct; refusal/EM panels are qwen-base per #518) and the EM survivorship issue (survivor-aligned-rate proxy).
5. Output: per-behavior factor attribution for every flat panel + a list of cells the factor set cannot classify (input to campaign children e2-e5).

No single-variable training manipulation is involved — this is the campaign's correlational pass that reshapes the causal children (e2-e5) before they spawn.

## Campaign context

- **Campaign:** #592 — https://eps.superkaiba.com/tasks/592 (world model: `artifacts/world-model.md` in the #592 task folder)
- **Question anchor:** `q:leak-behavior-vs-marker` (docs/open_questions.md §3.2, "Does leakage depend on the behavior?")
- **gpu_hours_est:** 0
- Reuse first: consume existing panels, adapters, base rates, and cosine matrices; this child must not retrain or regenerate anything.
