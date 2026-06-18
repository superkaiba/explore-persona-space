---
title: 'Fix test_issue475_common: TrainLoraConfig missing existing_adapter_path (selective-merge
  gap from issue-475)'
kind: infra
tags: []
created_at: '2026-06-12T20:07:54Z'
has_clean_result: false
---
tests/test_issue475_common.py::test_train_lora_config_has_existing_adapter_path_field FAILs on main: `TrainLoraConfig` has no `existing_adapter_path` field. The test landed on main via fc9b72055 ("task #475: land eval_results + figures + configs + experiment scripts/tests on main (selective merge from issue-475)") but the matching src change (the field in the train-lora config, presumably src/explore_persona_space/train/sft.py or wherever TrainLoraConfig is defined) did not make the selective merge.

Action: either cherry-pick the field from the issue-475 branch (check `git log issue-475 -S existing_adapter_path`) or, if #475 abandoned the feature, remove the orphaned test. This failure trips every session's full-suite Step 9c gate.

Discovered during #572's test-verdict gate, 2026-06-12.
