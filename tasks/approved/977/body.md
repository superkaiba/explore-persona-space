---
title: 'workflow-fix: port GCP find-newer sentinel wait to RunPod'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:71192e38febb
created_at: '2026-07-04T07:11:45Z'
has_clean_result: false
origin_prompt: 'parked — running under workflow_fix_target Provenance (recursion guard,
  workflow-fix-on-bug.md § Recursion guard). Two reviewer follow-up candidates logged,
  NOT auto-routed: (1) target_file: src/explore_persona_space/backends/runpod.py —
  widen the stale-sentinel clear to the experimenter step-11 wildcard (issue_<N>/*/.completion-sentinel.json)
  to defeat #685 single-live-sibling resolution on same-pod re-execution (Claude r2
  Minor 1, mechanizable). (2) target_file: src/explore_persona_space/backends/runpod.py
  + gcp.py parity — self-daemonizing workload writes the sentinel at daemonize time
  (premature phase=done); port GCP''s find-newer wait-loop hardening (gcp.py:1171-1190,
  #601) to the RunPod launcher (Claude r2 Minor 2, pre-existing-on-trunk convention).
  Plus r1 Claude Minor: distinct reason arm when _runpod_terminal_rung broad except
  absorbs RunPodWorkloadStartError (mislabeled no_compute_available could be blindly
  re-driven by the capacity-retry pass).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

port GCP's find-newer sentinel wait-loop hardening to the RunPod launcher

## Workflow gap

- **Bug observed:** a self-daemonizing workload writes the sentinel at daemonize time (premature phase=done) on the RunPod lane; GCP already has find-newer wait-loop hardening (gcp.py:1171-1190, #601)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

port GCP's find-newer sentinel wait-loop hardening to the RunPod launcher

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/gcp.py
- fingerprint: 71192e38febb

parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). Two reviewer follow-up candidates logged, NOT auto-routed: (1) target_file: src/explore_persona_space/backends/runpod.py — widen the stale-sentinel clear to the experimenter step-11 wildcard (issue_<N>/*/.completion-sentinel.json) to defeat #685 single-live-sibling resolution on same-pod re-execution (Claude r2 Minor 1, mechanizable). (2) target_file: src/explore_persona_space/backends/runpod.py + gcp.py parity — self-daemonizing workload writes the sentinel at daemonize time (premature phase=done); port GCP's find-newer wait-loop hardening (gcp.py:1171-1190, #601) to the RunPod launcher (Claude r2 Minor 2, pre-existing-on-trunk convention). Plus r1 Claude Minor: distinct reason arm when _runpod_terminal_rung broad except absorbs RunPodWorkloadStartError (mislabeled no_compute_available could be blindly re-driven by the capacity-retry pass).
