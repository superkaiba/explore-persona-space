---
title: 'workflow-fix: gotchas.md — MooseFS FUSE wedge on pod .venv (silent launch
  hang)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8e311e09f7d0
created_at: '2026-07-03T15:44:56Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #779 pass-2 relaunch (epm:failure-lesson
  v7): MooseFS FUSE wedge on the pod .venv import path — silent launch-hang signature
  + import-probe discriminator + pod-swap remediation'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes`
failure-lesson raised on task #779 (emitting agent: experimenter, pass-2
capture relaunch, 2026-07-03).

## Goal

Add a `.claude/rules/gotchas.md` entry: MooseFS FUSE wedge on a pod's
`.venv` import path — the silent launch-hang signature, the timeout-bounded
import-probe discriminator, and the remediation (container restart / fresh
provision; kill+relaunch never clears it).

## Workflow gap

- **Bug observed:** #779's pass-2 relaunch wedged twice at process startup on
  pod-77902 — zero stderr, python child frozen at ~10s CPU, GPU 0 MiB,
  `wchan=request_wait_answer`; the MooseFS-backed `/workspace` `.venv`
  stopped answering FUSE reads hours after a healthy run on the same pod. An
  independent `timeout 60 uv run python -c "import transformers"` probe
  reproduced the wedge (rc=124).
- **Why it is a workflow gap:** gotchas.md already documents the MooseFS
  per-pod EDQUOT quota trap but not the FUSE read-wedge trap; a silent launch
  hang otherwise burns kill/relaunch cycles and misclassifies as a code bug.
  py-spy is unusable on RunPod (no ptrace), so the import-probe discriminator
  is the load-bearing diagnostic.
- **Confidence (emitter):** high (root_cause_confirmed: yes, reproduced by an
  independent probe).

## Proposed change (candidate diff sketch — refine in planning)

```
+ ### MooseFS FUSE wedge on the pod .venv (silent launch hang)
+ Signature: launch hangs with ZERO stderr, python child frozen at ~10s CPU,
+ GPU 0 MiB, 0 sockets, wchan=request_wait_answer — the MooseFS-backed
+ /workspace .venv stopped answering FUSE reads (#779, pod-77902,
+ 2026-07-03; can appear hours into an otherwise healthy pod).
+ Discriminate: `timeout 60 uv run python -c "import transformers"` — rc=124
+ confirms the wedge (py-spy unusable on RunPod: no ptrace capability).
+ Remediation: kill + classify `epm:failure v1 failure_class: infra
+ reason: moosefs-fuse-wedge-venv-import`; kill-and-relaunch does NOT clear
+ it — the orchestrator swaps the pod (stop/resume container restart, or
+ terminate + fresh provision).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'MooseFS' .claude/ CLAUDE.md scripts/`) and update every hit
  that documents pod-storage traps; list them in the plan (CLAUDE.md's
  Gotchas summary line may want the wedge alongside the EDQUOT quota).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 8e311e09f7d0

<!-- epm:failure-lesson v1 -->
failure_class: infra
phase: pass-2 answer-summary capture relaunch (pod-side process startup)
lesson: A pod launch that hangs with ZERO stderr, python child frozen at ~10s CPU, GPU 0 MiB, 0 sockets, and wchan=request_wait_answer is a MooseFS FUSE wedge on the .venv import path, not a slow import or code bug — discriminate with `timeout 60 uv run python -c "import transformers"` (rc=124 confirms), then kill + classify infra; kill-and-relaunch does NOT clear it, and py-spy is unusable on RunPod (no ptrace capability).
generalizes: yes
owning_agent: experimenter
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
