---
title: Fit context→answer map on a mega-diverse weird-behavior corpus, replicated
  on Qwen2.5-7B-Instruct + Qwen3.5-9B
kind: experiment
tags: []
created_at: '2026-08-23T17:17:32Z'
has_clean_result: false
origin_prompt: I want to try to fit a mapping on a very very large and VERY VERY diverse
  set of contexts and answers. It should be both on qwen2.5-7B instruct and on some
  newer qwen ideally that has no thinking. Get a subagent to find a large diversity
  of datasets including weird behaviors to train on. Then get another one to propose
  the other model.
workflow: v1
goal: 'Fit the context→answer activation map M_{C,A} on a very large, maximally diverse
  context/answer corpus (heavily weighted toward weird / OOD / red-team / jailbreak
  regimes), and test whether the map replicates across model generations: Qwen2.5-7B-Instruct
  and Qwen3.5-9B (thinking disabled). An answer = held-out R^2 plus the mandatory
  identity+learned-bias baseline and kNN-retrieval reads for the fitted map, measured
  under massive context diversity and compared across the two models.'
---
## Goal

Fit the context→answer activation map M_{C,A} on a very large, maximally diverse context/answer corpus (heavily weighted toward weird / OOD / red-team / jailbreak regimes), and test whether the map replicates across model generations: Qwen2.5-7B-Instruct and Qwen3.5-9B (thinking disabled). An answer = held-out R^2 plus the mandatory identity+learned-bias baseline and kNN-retrieval reads for the fitted map, measured under massive context diversity and compared across the two models.
