---
title: SAE characterization of finetuning-induced context-to-answer mapping changes
kind: experiment
tags:
- mapping-diffing
- codex-managed
created_at: '2026-09-06T22:40:09Z'
has_clean_result: false
origin_prompt: 'I want to do an experiment to try to do mapping diffing with our mapping
  to understand how finetuning affected a model’s behavior. basically we train a mapping
  pre and post-finetuning and then we should try to characterize exactly what changed
  in the mapping, probably in terms of SAE features. Let’s develop this experiment
  further together. Run approval: yes please run this'
workflow: v1
goal: Determine which context-feature-to-answer-feature relationships change after
  controlled sycophancy finetuning, whether these changes predict held-out behavioral
  differences beyond simpler activation-difference baselines, and whether selected
  relationships withstand interventions in the model.
---
# SAE characterization of finetuning-induced context-to-answer mapping changes

## Goal

Determine which context-feature-to-answer-feature relationships change after controlled sycophancy finetuning, whether these changes predict held-out behavioral differences beyond simpler activation-difference baselines, and whether selected relationships withstand interventions in the model.

## Approved experiment

The user approved running the mapping-diffing design developed in this Codex conversation, beginning with one existing controlled sycophancy finetune. This is an extension of the context-to-answer mapping line, using #1768's pre/post measurement and #2643/#2552's SAE representation infrastructure where compatibility checks pass.

Fit pre/post affine linear maps from last-prompt-token context states to answer-token-mean states. Separate context movement, intercept movement, operator change, interactions, and prediction-residual change. Keep a shared context SAE and shared answer SAE fixed across checkpoints, with measured reconstruction and semantic-fitness checks on both distributions. Screen context-feature/answer-feature relationships with E_answer (M_post-M_pre) D_context; treat these as preactivation sensitivities and verify full gated encoder outputs on held-out contexts and actual answers.

Discover candidate changed relationships on a discovery split, freeze them, and evaluate on held-out prompt families. Compare with constant answer shift, direct pre/post SAE activation differences, and a direct linear predictor of the answer-state difference. Report held-out R2, identity-plus-learned-bias, and held-out nearest-neighbor retrieval with pool size and chance. Evaluate actual generated behavior, then test selected relationships using interventions in the model with matched control features and ordinary-task checks. Changes to the fitted mapping alone are not causal interventions in the model.

Use natural generations from each checkpoint as primary behavioral evidence; use both checkpoints on both answer sets for a matched-text 2x2 weight-versus-text diagnostic. Reuse existing checkpoint/captures/judgments only after provenance, compatibility, revision, completeness, and consumer-layout verification. Pilot quantities not grounded by validated prior artifacts remain explicitly ungrounded until measured.

No new model finetuning, nonlinear mapping/readout, broad model fleet, or unrelated experiment is authorized. Automated Claude usage (CLI, SDK, Anthropic API, reviews, and judges) is prohibited by the user's VM instructions. Use Codex independent reviewers and a validated non-Claude behavior instrument or compatible pre-existing judgments; do not silently weaken the behavioral endpoint.

## Acceptance

Produce a reproducible experiment and factual report assessing whether stable changed feature relationships explain and predict held-out behavioral changes, and whether selected relationships pass model-intervention tests. A well-powered null, map/SAE fitness failure, or failed intervention is a legitimate outcome when documented; do not substitute an easier construct or claim causality from correlations. Persist inputs, raw generations, judgments, fits, configs, and analysis artifacts under project policy. Provide browser-accessible figure and report URLs.

## Provenance

Originating request: "I want to do an experiment to try to do mapping diffing with our mapping to understand how finetuning affected a model's behavior. basically we train a mapping pre and post-finetuning and then we should try to characterize exactly what changed in the mapping, probably in terms of SAE features. Let's develop this experiment further together"

Run approval: "yes please run this"

Prior work: https://eps.superkaiba.com/tasks/1768 ; https://arxiv.org/abs/2603.04426 ; https://arxiv.org/abs/2602.20904 .
