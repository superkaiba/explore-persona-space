---
title: 'workflow-fix: mint RunPod handle sidecar on GCP→RunPod failover'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fb0973193397
created_at: '2026-07-04T02:11:11Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from experimenter epm:failure-lesson on #931: failover
  archives GCP handle without minting a RunPod handle — unchained launch, finalize
  surprise'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #931 (emitting agent: experimenter, via its epm:failure-lesson block).

## Goal

Mint a RunPod handle sidecar during the GCP→RunPod async failover so the re-driven run keeps the unified backend_poll/finalize path.

## Workflow gap

- **Bug observed:** the #659 failover archived the GCP handle without writing a RunPod handle, leaving the re-driven launch unchained (no declared completion-sentinel path; finalize needs --skip-confirm-artifacts).
- **Why it is a workflow gap:** the slice-6 contract is one RunHandle from launch through teardown; the failover path breaks it, forcing every downstream consumer (Step 6d.2 poller, Step 8 finalize) onto ad-hoc legacy fallbacks. Observed live on #931 (2026-07-04): the failover provisioned pod-931, the orchestrator had to retire the stale GCP sidecar by hand and poll via legacy poll_pipeline.py, and Step-8 finalize will need a post-hoc sentinel or --skip-confirm-artifacts.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In the failover leg (scripts/backend_poll.py `_failover_dead_gcp_to_runpod` + `src/explore_persona_space/backends/router.py` `_runpod_terminal_rung`), after a successful RunPod provision:
+ write a fresh RunPod RunHandle sidecar to .claude/cache/issue-<N>-handle.json (backend=runpod, pod_name, expected_artifacts carried over from the GCP handle where still valid)
+ archive the GCP handle as <name>.gcp-crashed-<attempt> (already done)
so backend_poll.py --issue <N> and dispatch_issue.py finalize --issue <N> keep working unchanged on the failover path. Also handle the partial case where RunPod provisions but workload start fails (mint the sidecar anyway, or clean it up on the terminal error).

## Scope / surfaces

- Primary target: `scripts/backend_poll.py`, `src/explore_persona_space/backends/router.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '_failover_dead_gcp_to_runpod\|_runpod_terminal_rung' scripts/ src/explore_persona_space/backends/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; router tests (`tests/test_router*.py`, `tests/test_backend_*.py`) stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/backend_poll.py
- fingerprint: fb0973193397

Surfaced prose (verbatim, from the experimenter's epm:failure-lesson on #931): "the #659 GCP→RunPod failover archived the GCP handle (`issue-931-handle.json.gcp-crashed-att-*`) without minting a fresh RunPod handle, leaving no declared completion-sentinel path; experimenters on this path should expect an unchained launch and note it on the marker so finalize isn't surprised."
