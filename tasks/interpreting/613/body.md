---
title: 'Alive-negatives A/B: does the loss-suppression flag give contrastive negatives
  a live restoring force? (#601 mechanism follow-up)'
kind: experiment
tags: []
created_at: '2026-06-12T02:42:11Z'
has_clean_result: false
parent_id: 601
goal: 'Determine whether placing the negative-row loss at the post-response slot (loss-suppression
  flag on) gives contrastive negatives a live gradient that exerts a measurable restoring
  force on the source implant, by training flag-on cells single-variable-matched to
  #601''s existing flag-off 200p+800n cells.'
relates_to:
- leak-contrastive-negatives
- implant-learning-speed
---
## Goal

Determine whether placing the negative-row loss at the post-response slot (loss-suppression flag on) gives contrastive negatives a live gradient that exerts a measurable restoring force on the source implant, by training flag-on cells single-variable-matched to #601's existing flag-off 200p+800n cells.

## Provenance

Filed automatically by the Step 9b autonomous follow-up block on parent #601 (proposal 3 of the 2026-06-12 epm:follow-ups round; question_relation: substantially-different). Not auto-spawned — awaiting manual triage.

## Hypothesis

Flag-on negatives acquire nonzero loss as the implant grows and pull the source level / trained-negative leakage down (the #471 direction). #601 showed flag-off negative rows are gradient-dead on the loss-token channel (base CE ~1e-6); the suppress flag relocates their loss to the slot where the growing implant raises marker probability, plausibly waking a real opposing gradient — the loss-placement difference is the leading candidate for the #471 (restoring force) vs #472/#601 (inert negatives) conflict.

## Design (single-variable)

- Train 2 new units: dense_200p800n recipe with `suppress_at_post_response_slot=True`, seeds 42 + 137, 63 steps each — parent #601 recipe verbatim otherwise (Qwen-2.5-7B-Instruct, rsLoRA r=32/α=64 all-linear, lr 1e-5, cosine 5% warmup, effective batch 16, marker ` ※` id 83399).
- Comparison arm REUSED, not retrained: #601's flag-off `adapters/issue_601/dense_200p800n_seed{42,137}` (Hub-verified 2026-06-12; read at staged classic α/r=2.0 gauge via stage_parity_read_adapter).
- DV: four-float on-policy reads + in-loop row-type CE probe (does the negative-row loss token acquire nonzero CE as the implant grows?) + trained-negative leakage vs bystander panel.
- Falsification: flag-on cells co-land with flag-off (source level and trained-negative leakage) → loss placement doesn't resolve the conflict; residual #471 differences (lr, panel, ratio) carry it.

## Cost

~1-1.5 GPU-h (1× H100, lora-7b intent). Code: branch issue-601 scripts (i601_run_cell.py + neg_setpoint_601 registry) — full rebase to main still deferred; plan should pin the source branch.
