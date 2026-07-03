---
title: 'vectorized_mlp_skill: per-absolute-index MLP init seeding (chunk-invariant
  fits)'
kind: infra
tags: []
created_at: '2026-07-03T11:29:06Z'
has_clean_result: false
origin_prompt: '#841 fu-r1 v15 finding: fit_batched_split_mlp seeds each group''s
  MLP init in BATCH ORDER (vectorized_mlp_skill.py:809), so chunking a group list
  changes every member''s init (measured maxdiff 0.82 between 5-group and 2-group
  calls). Proposed: seed by the group''s ABSOLUTE index/key so any chunking is bit-equivalent;
  update the documented determinism contract + assert_matches_reference gates (#658/#722)
  accordingly. Out of workflow-fix scope (src/analysis) — ordinary infra change, full
  review pipeline.'
workflow: v1
---

