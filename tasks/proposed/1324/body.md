---
title: 'daily-fix: default vLLM hang mitigations for user corpora'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-15T06:52:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): an 8-shard n50k LMSYS generation
  launch on 8xH100 RunPod froze ~35 min in llm.generate() — the documented #1092-family
  vLLM hang venue+shape — because create_vllm_engine has no default-on hang mitigations
  for real-user-corpus generation; the driver launched without enforce_eager/prefix-caching-off
  and lost ~35-40 min x 8 H100'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 problem sweep (session 272c80a1, #779 n50k round, 23:53Z): all 8 shards entered `llm.generate()` and froze ~35 min (GPU idle, PID alive, no traceback) — the gotchas-documented hang signature for RunPod multi-GPU + long real-user LMSYS prompts + prefix caching. The session ran the differential correctly and relaunched with the two knobs, but the knobs were opt-in so the first launch burned the time anyway. Library fix (src/), /daily route-2 channel.

## Goal

make the vLLM hang mitigations (env-gated enforce_eager=True + enable_prefix_caching=False) a default-on knob in create_vllm_engine for real-user-corpus generation, and add the pre-launch checklist line to .claude/rules/gotchas.md § vLLM hang

## Bug

- **Observed:** an 8-shard n50k LMSYS generation launch on 8xH100 RunPod froze ~35 min in llm.generate() — the documented #1092-family vLLM hang venue+shape — because create_vllm_engine has no default-on hang mitigations for real-user-corpus generation; the driver launched without enforce_eager/prefix-caching-off and lost ~35-40 min x 8 H100
- verified-at-filing: `grep -n "enforce_eager\|enable_prefix_caching" src/explore_persona_space/eval/generation.py` -> 0 hits; `def create_vllm_engine` at :382 (absence-of-guard claim: the factory exposes no mitigation defaults) (2026-07-15).

## Proposed change

Env-gated defaults in `create_vllm_engine` (e.g. `EPS_VLLM_ENFORCE_EAGER` / `EPS_VLLM_PREFIX_CACHING`, defaulting to the safe combination when the caller flags a real-user corpus or when prompts exceed a length threshold), plus one checklist line in gotchas.md § vLLM `generate()` hang: "launching a real-user-corpus generation on RunPod multi-GPU => wire the two mitigation knobs BEFORE first launch".

## Constraints

- Do not regress throughput for the standard short-prompt eval path (mitigations conditional, not unconditional); cite the perf tradeoff in the plan.
