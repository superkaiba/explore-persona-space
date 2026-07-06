---
title: 'Faithful PV-native rig reproduction on Qwen-2.5-7B (parked on_hold — duplicates
  #778)'
kind: experiment
tags: []
created_at: '2026-07-01T17:43:55Z'
has_clean_result: false
parent_id: 779
goal: 'Faithful PV-native rig reproduction on Qwen-2.5-7B to isolate the rig-gate
  residual gap #779 documented (parked as redundant with #778).'
---
## Goal

Faithful PV-native rig reproduction on Qwen-2.5-7B to isolate the rig-gate residual gap #779 documented (parked as redundant with #778).

## Value critique

**Parked at `on_hold` per Step 9b VC redundancy screen (both critics AGREE — Claude follow-up-critic + Codex codex-follow-up-critic).**

**Redundancy verdict:** duplicates task #778 (approved). #778 already asks for a faithful Persona Vectors system-prompt/monitoring replication on Qwen-2.5-7B for the same 3 traits (evil, sycophancy, hallucination), using the paper recipe/layers (paper-native 20/20/16), the released paper data, verbatim recipe, last-prompt-token projection onto persona vectors, and the same `claude-sonnet-4-5-20250929` judge. **#778 is a strict superset** — it adds a fine-tuning-shift arm + null battery beyond the reproduction — so the same construct + measurement + variable ("does the paper's rig replicate on Qwen") is already scoped there.

**Revival criterion:** revive this proposal (via `task.py set-status <M> proposed`) ONLY IF #778's scope is later narrowed away from the system-prompt prediction rig at paper-native layers OR #778 is archived before running. Otherwise the reproduction result will land on #778.

**Note:** this proposal was already `auto_run: no` (a design/taste call, 4 GPU-h) — the `redundant` verdict changes nothing about auto-running; it just files the proposal durably instead of leaving it in `epm:follow-ups v1` unread.

## Parent context

Emerged from task #779 clean-result follow-up-proposer at Step 9b post-park. Originating rationale: #779's rig-validation gate FAILED with multi-cause; the layer-selection artifact explains evil's PV-raw shortfall (L14=0.17 → L20=0.50 ≈ paper 0.51) but not the sycophancy many-shot 0.21-below-paper residual or the hallucination system ABOVE-paper anomaly. A faithful PV-native rig repro would isolate whether the residual is a Qwen-vs-GPT-4o rig difference or a genuine model-property. #778 covers this exact question.

## Provenance

- parent_task: #779
- source: follow-up-proposer at task #779 Step 9b (2026-07-01)
- Redundancy critics: `follow-up-critic` (Claude) + `codex-follow-up-critic` (Codex) both `redundant`; agree, no reconciler needed.
- Duplicated_task: #778
