---
title: 'workflow-fix: gotchas — pid-namespace vLLM EngineCore reap variant'
kind: infra
tags:
- wf-fix
- wf-fix-fp:17104c465214
created_at: '2026-07-15T16:52:11Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha candidate from #1333 attempt-4 OOM (pid-namespace
  vLLM residue)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson gotcha candidate raised on task #1333 (emitting agent: experiment-implementer).

## Goal

Extend the gotchas.md vLLM teardown / crash-orphan EngineCore entries with the container pid-namespace variant and the verified reap recipe.

## Workflow gap

- **Bug observed:** issue #1333 attempt 4: a p0-era vLLM EngineCore child held 29 GiB on GPU 0 for ~28 min; the in-process teardown scan WARNed it as a dead-pid zombie 7 times — the container's pid namespace means NVML compute-apps pids are HOST-namespace, invisible to /proc inside the container — and a later ladder unit OOMed (3 co-resident processes, 81 MiB free).
- **Why it is a workflow gap:** gotchas.md's vLLM teardown + crash-orphan VLLM::EngineCore entries assume /proc-visible pids (pgrep/kill by NVML pid); on pid-namespaced pods (RunPod containers) that recipe silently no-ops on LIVE orphans. The verified recipe exists on the issue-1333 branch (commits cdb1d77bab + 70270e13cc: graceful shutdown + child-death verification, container-pid child sweep via parentage/process-group, post-reap NVML VRAM re-verify, unit-exit killpg) and belongs in the shared gotcha so every dispatcher/experimenter inherits it.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix live-verified on pod-1333 — the gate fail-louded on a real 15 GiB holder in 62 s, then a clean smoke passed)
- verified-at-filing: `grep -rln "EngineCore" .claude/rules/ CLAUDE.md .claude/agent-memory/experimenter/` -> 7 hits in 7 files; per-target: .claude/rules/gotchas.md 1 (the entry to extend); `grep -rln "pid namespace|pid-namespace" .claude/rules/ CLAUDE.md` -> 1 hit (gotchas.md, different context) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ In gotchas.md, under the vLLM teardown / crash-orphan EngineCore entries:
+ - CONTAINER PID NAMESPACE variant: NVML/compute-apps pids are HOST-namespace;
+   /proc-based scans inside the container misread LIVE orphans as dead pids.
+ - Reap recipe: engine.shutdown() + VERIFY child death; sweep children by
+   container-side parentage/killpg at unit exit; re-verify VRAM via NVML
+   after reap; per-unit GPU preflight before any engine/model load.
+   (Worked implementation: issue1333_dispatch.py _reap_engine/_gpu_hygiene/
+   _unit_gpu_preflight, commits cdb1d77bab + 70270e13cc.)

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'EngineCore' .claude/ CLAUDE.md scripts/`) and update every hit that documents the reap recipe; the experimenter memory (feedback_vllm_zombie_gpu_pkill_reaper.md) is owned by that agent — cross-link, don't rewrite.

## Constraints / invariants

- Workflow-surface only — never experiment code, configs/, or tasks/.
- scripts/workflow_lint.py --check-asks passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 17104c465214

(Surfaced prose: the epm:failure-lesson v1 block on task #1333 events.jsonl, 2026-07-15 — gotcha_candidate: yes, root_cause_confirmed: yes.)
