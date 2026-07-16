---
title: 'workflow-fix: gotchas — GPU-free gate must read per-GPU memory.used (foreign-tenant
  GPUs invisible to compute-apps)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:42bb392c3386
created_at: '2026-07-16T08:19:49Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate:yes failure-lesson from #825 r11 (epm:failure-lesson
  v7)'
workflow: v1
---
## Overview / Motivation

Auto-filed from a `gotcha_candidate: yes` failure-lesson on task #825 (emitting agent: experimenter, round-11 launch refusal, epm:failure v17 / epm:failure-lesson v7).

## Goal

Add a gotchas.md entry (+ consider hardening the dispatcher gpu-guard pattern): a fresh RunPod pod can arrive with a GPU held by a FOREIGN host-level tenant — `nvidia-smi --query-compute-apps` reads EMPTY, in-container pgrep/proc scans are clean, yet `--query-gpu=memory.used` shows ~72 GB held and unkillable from inside; GPU-free gates must read per-GPU memory.used, never compute-apps alone.

## Workflow gap

- **Bug observed:** pod-825 (8xA100) GPU 4 arrived holding 72,505 MiB with zero compute-apps; the dispatch script's `gpu_guard_one` (compute-apps-based) would have passed straight through and the first GPU-4 job would OOM-abort the whole run.
- **Why it is a workflow gap:** the canonical GPU-free guard recipe in gotchas.md / the dispatch-script lineage (issue825_naturalistic_s_dispatch.sh et al.) gates on compute-apps + VLLM::EngineCore reaping only; the foreign-tenant condition is invisible to it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md GPU-guard entry gains: gate on `nvidia-smi --query-gpu=index,memory.used`
+ per GPU (threshold ~2-4 GB) IN ADDITION to compute-apps; a held-with-no-apps GPU on a
+ fresh pod is a defective provision — terminate + re-provision (unkillable in-container).
+ Worked example: #825 r11 experimenter refusal (epm:failure v17); memory at
+ .claude/agent-memory/experimenter/feedback_gpu_foreign_allocation_no_compute_apps.md.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`

## Constraints / invariants

- Workflow-surface only; workflow_lint passes.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 42bb392c3386
