---
title: 'workflow-fix: gotchas entry — vLLM H100 prefix-cache IMA differential + mitigation'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a54a6868dfc5
created_at: '2026-07-08T07:24:37Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from task #1092 round 8.5 (epm:failure-lesson
  v3, 2026-07-08T07:23:35Z)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1092 (emitting agent: experiment-implementer, round 8.5).

## Goal

Add a `.claude/rules/gotchas.md` entry: vLLM-on-H100 CUDA illegal-memory-access under heavy shared-prefix caching at long-prompt production shapes — the A100-clean + short-probe-clean differential pins the class; mitigate with default-off `enable_prefix_caching=False` + `enforce_eager=True` engine knobs.

## Workflow gap

- **Bug observed:** issue #1092 launch #4 (RunPod 8x H100, vLLM 0.11.0/torch 2.8.0): engine-step `torch.AcceleratorError: CUDA illegal memory access` at ~3.8k-token prompts under heavy shared-prefix reuse (num_common_prefix_blocks=237) killed 35/42 shards — while identical code+corpus ran 42/42 clean on A100s and a 1-GPU short-prompt probe passed on the SAME pod.
- **Why it is a workflow gap:** gotchas.md carries the vLLM trap ledger (teardown, generate-hang differential) but not this H100 shape/load-dependent IMA family or its cheap two-probe differential; the next H100 sweep would re-derive the diagnosis at 8-GPU billing rates.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new entry "vLLM H100 IMA under heavy shared-prefix caching (A100-clean differential)":
+ signature (engine-step AcceleratorError IMA at long-prompt/heavy-prefix-share shapes; one IMA errors every queued shard on the worker); the two cheap discriminating probes (identical code on A100 clean; minimal short-prompt probe on the same H100 clean) BEFORE any code hunt; mitigation = default-off enable_prefix_caching=False + enforce_eager=True knobs enabled per-run (byte-identical engine args when off); comparability note (all cells of a comparison share one engine config; identity gates revalidate capture).
+ Cross-ref: reference impl `_vllm_engine_overrides` (scripts/issue1092_gpu_phase.py @ 5031174dc3); agent-memory experimenter/feedback_vllm_h100_prefix_cache_ima.md.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'illegal memory access\|enforce_eager\|prefix_caching' .claude/ CLAUDE.md`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: a54a6868dfc5
