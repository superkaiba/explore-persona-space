---
title: 'daily-fix: SLURM intent tables cover all GCP GPU intents'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e67598577033
- daily-auto-filed
created_at: '2026-07-31T06:59:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): lora-7b-h100 is absent
  from the SLURM intent tables exactly like capture-7b was before #1896, silently
  bypassing the fellows-first policy per-intent; no completeness test ties gcp.INTENT_TO_MACHINE
  gpu keys to the SLURM tables.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 Step C (parked workflow-fix-candidate routing) from a candidate parked on task #1896 (emitting source: orchestrator observation during #1896 clarify/plan, seconded by the Methodology critic; parked 2026-07-30T21:15:22Z). #1896 itself added `capture-7b` to the SLURM tables only, per its fingerprint scope (merged 20c9d904f47c).

## Goal

Close the per-intent fellows-first bypass: add the remaining GCP-mapped GPU intents (`lora-7b-h100` at minimum) to the SLURM intent tables, and add a completeness test tying `gcp.INTENT_TO_MACHINE`'s gpu_count>0 keys to the SLURM tables so the class cannot silently reopen.

## Workflow gap

- **Bug observed:** `lora-7b-h100` (a GCP-mapped 1x H100-80 GPU intent) is absent from the SLURM intent tables exactly like `capture-7b` was (#1896 fixed capture-7b only) — a lora-7b-h100 auto-lane dispatch still ValueErrors off the free fellows/SLURM lane and burns paid GCP; `sweep-8g-a100` / `sweep-8g-h100` / `eval-h100` are in the same absent class (whether SLURM should serve 8-GPU sweeps needs a design call — flag `architectural`-adjacent design question in the plan, do not silently decide).
- **Why it is a workflow gap:** the standing fellows-first policy (#1609) is silently bypassed per-intent whenever a GCP GPU intent lacks a SLURM row; there is no completeness test tying gcp.INTENT_TO_MACHINE's gpu_count>0 keys to the SLURM tables (the RunPod rung has exactly this via RUNPOD_INTENT_FOR_GCP_INTENT + its translation-map test).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'lora-7b-h100' src/explore_persona_space/backends/slurm.py src/explore_persona_space/backends/gcp.py src/explore_persona_space/backends/router.py` → gcp.py:393 (INTENT_TO_MACHINE row) + router.py:460 (RunPod translation) present, slurm.py 0 hits (absence in the SLURM tables confirmed); slurm.py's tables carry `lora-7b`/`eval`/`capture-7b` (:530-:603) but no H100-family intents (2026-07-31 filing time). #1896's capture-7b fix merged 2026-07-30T22:46:27Z.

## Proposed change (candidate diff sketch — refine in planning)

Add `lora-7b-h100` (and any other 1-GPU H100-class intents the plan judges SLURM-servable) to `_DEFAULT_TIME_BUDGETS_H` + the GPU-count table in `backends/slurm.py`; add a completeness pin test asserting every `gcp.INTENT_TO_MACHINE` gpu_count>0 key either has a SLURM row or is on an explicit documented-exclusion list (8-GPU sweep intents may land there pending the design call).

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`, `src/explore_persona_space/backends/router.py`
- tests: `tests/test_slurm_*.py` / `tests/test_router.py` completeness pin.

## Constraints / invariants

- The in-scope `src/` backends package is workflow surface (CLAUDE.md § workflow-fix scope).
- The 8-GPU-sweep design question (should the fellows lane serve them at its 16-GPU/user caps?) is surfaced in the plan, not silently decided; a documented-exclusion list is the acceptable v1 answer.
- Existing router/SLURM tests stay green; `tests/test_router.py::test_default_auto_lane_order_is_gcp_first` durability pin untouched.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py, src/explore_persona_space/backends/router.py
- fingerprint: e67598577033 (tag-authoritative; supersedes body-carried fingerprint: 416a551b1ba7)
- origin: parked candidate-block on #1896 events.jsonl, ts 2026-07-30T21:15:22Z (routed by /daily 2026-07-30 Step C)
