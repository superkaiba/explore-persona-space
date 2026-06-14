---
title: Is postfix the Qwen-2.5 carrier, and does postfix-patching work as a per-cell
  leakage intervention?
kind: experiment
tags: []
created_at: '2026-06-14T06:19:53Z'
has_clean_result: false
parent_id: 595
goal: 'Test whether the chat-template POSTFIX (not prefix) is the dominant off-distribution
  leakage carrier on Qwen-2.5-7B, and whether patching base postfix KV per-cell into
  the #545 leaky adapters reduces leakage as a model-specific intervention.'
---
## Goal

Test whether the chat-template POSTFIX (not prefix) is the dominant off-distribution leakage carrier on Qwen-2.5-7B, and whether patching base postfix KV per-cell into the #545 leaky adapters reduces leakage as a model-specific intervention.

## Motivation (filed from #595's `epm:follow-ups v1` proposal #3, auto-filed via Step 9b autonomous flow)

#595's most striking unmodeled pattern: on the headline bad_medical × broad_em cell, prefix-patching INCREASED leakage (Δ=−0.065) while POSTFIX-patching CLEARED it (Δ=+0.123, to zero). Piggyback's Fig. 5 notes postfix recovery is large specifically on Qwen-2.5. This suggests the carrier on Qwen-2.5 is the postfix (`<|im_end|>\n<|im_start|>assistant\n`), not the prefix.

This is `question_relation: substantially-different` from #595: a new construct (postfix carrier) and a new question (is there a working leakage intervention?), distinct from #595's prefix-binding-as-predictor Goal.

## Hypothesis

A full postfix-patch sweep across the 8 leaky cells × seeds will show postfix-patching cuts leakage more consistently and more strongly than prefix-patching, making postfix-binding the right Qwen-2.5 carrier construct.

**Falsification:** If postfix-patching does NOT outperform prefix-patching across the 8 cells (no consistent cut, or prefix ≥ postfix), the headline-cell postfix recovery was a one-cell artifact and the postfix-carrier hypothesis is dead.

## Pre-filled spec (carried from #595)

- **Model:** `Qwen/Qwen2.5-7B-Instruct` + the 8 #545 leaky-row adapters (verified on HF at `6471a550`)
- **Data:** #545 probes + judges, full per-column cap
- **Seeds:** {0, 137} (parent ran postfix control on bad_medical seed-0 only)
- **Eval:** Δleakage on-policy + a postfix-KV-shift predictor (TReFT eq. on the postfix span) added to #545's race
- **Config:** patch span = postfix (`<|im_end|>\n<|im_start|>assistant\n`) instead of prefix; run as the PRIMARY across all 8 cells × 2 seeds + a postfix-binding-strength score
- **Driver:** `issue-595:scripts/issue595_prefix_carrier.py --phase controls` (postfix path) extended to the full cell set — cherry-pick from the `issue-595` branch (not on `main` per #595's `epm:merge-failed v1`)
- **Estimated cost:** ~18 GPU-hours on 1× A100-80 (GCP `lora-7b`)

## Relation to existing tasks

- **Parent #595** (this experiment is its `substantially-different` follow-up): prefix-binding-as-predictor null + postfix surprise on the headline cell.
- **Grandparent #545** (B→B′ leakage matrix): provides the 8 leaky-row adapters + the eval probes + the predictor-race harness.

## Status

`proposed` — captured via #595's autonomous-flow follow-up auto-spawn. Execute via `/issue <N>` → `/adversarial-planner` when ripe. **Pre-launch:** depends on the eventual #595 merge to main landing the driver scripts + the vendored `issue503/` modules (currently blocked by Step 10d's overlapping guards on #595).
