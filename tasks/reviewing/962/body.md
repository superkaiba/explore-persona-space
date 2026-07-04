---
title: 'daily-fix: fail-loud CVD assert in issue667_extract _device()'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T07:09:07Z'
has_clean_result: false
origin_prompt: "parked — running under workflow_fix_target recursion guard, see .claude/rules/workflow-fix-on-bug.md\
  \ § Recursion guard. LOGGED + surfaced, NOT auto-routed by this session.\n\n<!--\
  \ workflow-fix-candidate v1 -->\ntarget_file: scripts/issue667_extract.py (and _device()-style\
  \ descendants — NOTE: experiment scripts, OUT of the workflow-fix surface; this\
  \ is an experiment-code follow-up, not a workflow-fix task)\nbug_observed: a hand-launch\
  \ of a launcher-pinned per-cell worker with only --gpu-id silently targets the busy\
  \ default GPU and crashes at vLLM init_device (#813, 2026-07-02) — the failure is\
  \ only prose-guarded after #880.\nwhy_workflow_gap: (borderline/out-of-scope by\
  \ target) the worker class has no runtime guard; a fail-loud assert would bind on\
  \ ALL launch paths, prose read or not.\nproposed_change: _device() (scripts/issue667_extract.py::_device)\
  \ asserts CUDA_VISIBLE_DEVICES is set in the environment when gpu_id > 0, raising\
  \ a self-explaining error naming the env-pin recipe instead of silently returning\
  \ cuda:0.\ndiff_sketch: |\n  def _device(gpu_id, cpu_only):\n      ...\n  +   if\
  \ not cpu_only and gpu_id > 0 and \"CUDA_VISIBLE_DEVICES\" not in os.environ:\n\
  \  +       raise RuntimeError(\n  +           f\"--gpu-id {gpu_id} is informational\
  \ in this worker class; launch with \"\n  +           f\"env CUDA_VISIBLE_DEVICES={gpu_id}\
  \ ... --gpu-id {gpu_id} (see gotchas.md manual-launch CVD entry)\")\n      return\
  \ torch.device(\"cuda:0\")\nconfidence: medium\nrelated_task: #880 (raised by both\
  \ alternatives"
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

_device() asserts CUDA_VISIBLE_DEVICES is set in the environment when gpu_id > 0, raising a self-explaining error naming the env-pin recipe instead of silently returning cuda:0

## Workflow gap

- **Bug observed:** a hand-launch of a launcher-pinned per-cell worker with only --gpu-id silently targets the busy default GPU and crashes at vLLM init_device (#813); only prose-guarded after #880
- **Why it is a workflow gap:** experiment-code follow-up (out of wf-fix surface); a fail-loud assert binds on ALL launch paths, prose read or not
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

_device() asserts CUDA_VISIBLE_DEVICES is set in the environment when gpu_id > 0, raising a self-explaining error naming the env-pin recipe instead of silently returning cuda:0

## Scope / surfaces

- Primary target: `scripts/issue667_extract.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/issue667_extract.py
- fingerprint: 2cad43b5152c

parked — running under workflow_fix_target recursion guard, see .claude/rules/workflow-fix-on-bug.md § Recursion guard. LOGGED + surfaced, NOT auto-routed by this session.

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue667_extract.py (and _device()-style descendants — NOTE: experiment scripts, OUT of the workflow-fix surface; this is an experiment-code follow-up, not a workflow-fix task)
bug_observed: a hand-launch of a launcher-pinned per-cell worker with only --gpu-id silently targets the busy default GPU and crashes at vLLM init_device (#813, 2026-07-02) — the failure is only prose-guarded after #880.
why_workflow_gap: (borderline/out-of-scope by target) the worker class has no runtime guard; a fail-loud assert would bind on ALL launch paths, prose read or not.
proposed_change: _device() (scripts/issue667_extract.py::_device) asserts CUDA_VISIBLE_DEVICES is set in the environment when gpu_id > 0, raising a self-explaining error naming the env-pin recipe instead of silently returning cuda:0.
diff_sketch: |
  def _device(gpu_id, cpu_only):
      ...
  +   if not cpu_only and gpu_id > 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
  +       raise RuntimeError(
  +           f"--gpu-id {gpu_id} is informational in this worker class; launch with "
  +           f"env CUDA_VISIBLE_DEVICES={gpu_id} ... --gpu-id {gpu_id} (see gotchas.md manual-launch CVD entry)")
      return torch.device("cuda:0")
confidence: medium
related_task: #880 (raised by both alternatives reviewers + the reconciler as the stronger structural fix, out of the #880 workflow-fix surface)
<!-- /workflow-fix-candidate -->

