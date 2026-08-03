---
title: 'workflow-fix: GCP crash-persist also sweeps /workspace/logs (per-phase driver
  logs)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b9add066d20d
created_at: '2026-07-22T23:12:33Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/backends/gcp.py\n\
  bug_observed: on a GCE workload crash the per-phase dispatcher/driver log under\
  \ /workspace/logs was not persisted (sweep root is $WORKLOAD_ROOT/logs) and the\
  \ p3 traceback was lost\nwhy_workflow_gap: the crash-persist trap's worker_logs\
  \ sweep root ($WORKLOAD_ROOT/logs) does not match the GCE lane's own dispatcher\
  \ log convention (/workspace/logs), so tracebacks are silently lost on crash\nproposed_change:\
  \ extend the GCP crash-persist worker_logs sweep to also cover /workspace/logs (the\
  \ dispatcher per-phase log dir on GCE), tail-capped, so crash tracebacks are not\
  \ lost\ndiff_sketch: |\n  + sweep /workspace/logs/ (tail-capped, newest-first, exclude\
  \ the already-staged\n  + workload.log / pid files / drained sentinels) alongside\
  \ $WORKLOAD_ROOT/logs/\nconfidence: high\nrelated_task: #1415\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1415 (emitting agent: orchestrator, /issue 1415 follow-up round).

## Goal

Extend the GCP crash-persist worker_logs sweep to also cover /workspace/logs (the dispatcher per-phase log dir on GCE), tail-capped, so crash tracebacks are not lost.

## Workflow gap

- **Bug observed:** on a GCE workload crash the per-phase dispatcher/driver log under /workspace/logs was not persisted (sweep root is $WORKLOAD_ROOT/logs) and the p3 traceback was lost.
- **Why it is a workflow gap:** `_eps_persist_diagnostics` (backends/gcp.py, the #885 worker_logs sweep) sweeps `$WORKLOAD_ROOT/logs/` only, while the GCE lane's own log/sentinel convention points dispatchers at `/workspace/logs/` (`log_path=/workspace/logs/issue-<N>.log`; on GCE `$WORKLOAD_ROOT=/workspace/eps-issue-<N>`, a DIFFERENT dir) — so any dispatcher that honors the /workspace/logs convention loses its per-phase logs on crash.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "worker_logs\|logs/" src/explore_persona_space/backends/gcp.py` → sweep root documented as `$WORKLOAD_ROOT/logs/` (L1247, L1964 logs_root skip line); canonical workload log declared at `<vm_scratch_dir>/logs/issue-<N>.log` = /workspace/logs on GCE (L944-951); incident transcript (HF issue1415_partial/att-20260722-222513/crash_persist_transcript.log) shows `[crash-persist] SKIP worker_logs: empty after cache/git-tracked excludes` while the p3 traceback lived in the unswept /workspace/logs per-phase log (2026-07-22).

## Proposed change (candidate diff sketch — refine in planning)

+ in _eps_persist_diagnostics's worker_logs sweep: after sweeping $WORKLOAD_ROOT/logs/,
+ ALSO sweep /workspace/logs/ (excluding the canonical workload.log already staged,
+ pid files, and sentinel JSONs already drained) with the same tail-cap +
+ git-tracked/cache excludes; newest-first by mtime under the same dir cap.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: b9add066d20d

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py
bug_observed: on a GCE workload crash the per-phase dispatcher/driver log under /workspace/logs was not persisted (sweep root is $WORKLOAD_ROOT/logs) and the p3 traceback was lost
why_workflow_gap: the crash-persist trap's worker_logs sweep root ($WORKLOAD_ROOT/logs) does not match the GCE lane's own dispatcher log convention (/workspace/logs), so tracebacks are silently lost on crash
proposed_change: extend the GCP crash-persist worker_logs sweep to also cover /workspace/logs (the dispatcher per-phase log dir on GCE), tail-capped, so crash tracebacks are not lost
diff_sketch: |
  + sweep /workspace/logs/ (tail-capped, newest-first, exclude the already-staged
  + workload.log / pid files / drained sentinels) alongside $WORKLOAD_ROOT/logs/
confidence: high
related_task: #1415
<!-- /workflow-fix-candidate -->
