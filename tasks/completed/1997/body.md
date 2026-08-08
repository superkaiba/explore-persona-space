---
title: 'daily-fix: RunPod fallback rung pod lifecycle: orphans + sto'
kind: infra
tags:
- wf-fix
- wf-fix-fp:606acca8ee2c
- daily-auto-filed
created_at: '2026-08-02T07:08:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): #1739: 4 fallback-provisioned
  pod-1739 RunPod pods sat RUNNING unlaunched (~$10+/hr) until 4 watcher alerts; the
  terminal rung resumed a deliberately-STOPPED pod-1739 (auto_fallback_runpod, twice;
  workloads died rc=2); a teammate quota-exhaustion rerun auto-fell-through to unasked
  paid RunPod spend for a nice-to-have.'
workflow: v1
---
# daily-fix: RunPod fallback rung pod lifecycle: orphans + stopped resume

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C1 (miners 3, 8, 1; sessions f98a12ed / 20e82ec2 / 55419495, all #1739).

## Goal
Harden the RunPod terminal-rung / fallback launch path so (a) a pod whose workload launch fails does not bill unbounded behind the leave-RUNNING-for-diagnosis contract, (b) the rung never silently RESUMEs an existing deliberately-STOPPED `pod-<N>` it does not own, and (c) deferrable work can opt out of the paid RunPod fallback per dispatch.

## Workflow gap
- **Bug observed:** Three #1739 incidents in one day: (1) 4 fallback-provisioned `pod-1739` RunPod pods whose workloads never launched sat RUNNING (~$10+/hr aggregate) until 4× watcher unlaunched-orphan alerts forced manual termination; (2) the explicit-gcp ladder's terminal RunPod rung resumed the interactive session's deliberately-STOPPED pod-1739 (`"chosen_kind": "runpod", "reason": "auto_fallback_runpod"`, chosen twice), workloads died rc=2 `runpod_workload_start_failed`, and the session had to re-stop the pod and retire handle sidecars (`.superseded-podfallback`); (3) a teammate's quota-exhausted rerun auto-fell-through to a paid RunPod 1×H100 for a nice-to-have — "unasked RunPod spend, which crosses the 'ask before spending money' boundary" (caught in ~2 min, ≈$0.10–0.20).
- **Why it is a workflow gap:** the fallback rung's pod lifecycle has no bounded post-failure teardown, no ownership gate on reusing an existing stopped pod, and no per-dispatch opt-out for deferrable work — all three leak money without a user decision.
- **Confidence:** medium (incidents concrete; exact resume mechanism partially probed, see below).
- verified-at-filing: `grep -n 'workload_start_failed\|left RUNNING' src/explore_persona_space/backends/runpod.py` → 10+ hits (lines 352-354, 385, 861, 919-1009): leave-RUNNING-on-launch-failure is a DELIBERATE documented contract ("The pod is left RUNNING for SSH diagnosis (the RunPod-as-diagnosis-lane contract)") — so item (a) is a design-tension fix (bound the diagnosis window), not a missing branch. `grep -n 'resume' scripts/pod_lifecycle.py` → provision refuses only a non-EXITED pod (line ~2122: `desired_status != "EXITED"`), i.e. a STOPPED pod does NOT block re-entry, and the SSH-preflight helper at pod_lifecycle.py:1475-1539 auto-runs `pod.py resume --issue <N>` ONCE on first SSH failure when `allow_resume=True` — a concrete candidate mechanism for the stopped-pod resume. `git log --oneline --since='7 days ago' -- src/explore_persona_space/backends/{runpod,router,issue_dispatch}.py` → 7 commits (incl. f0c0e3dea4 #1838 "GCP->RunPod failover provision crash-safe (adopt residue pod)", a5c9a09427 #1698 launch-path branch/teardown contract) — none adds an ownership gate on stopped-pod resume, a post-failure teardown bound, or a fallback opt-out flag (2026-08-01).

## Proposed change (refine in planning)
- unverified hypothesis — verify at plan time: the terminal rung's launch reached the deliberately-STOPPED pod-1739 via the `allow_resume=True` SSH-preflight auto-resume (pod_lifecycle.py:1475-1539) and/or the #1838 residue-pod adoption path (miner-inferred from markers/launch logs; router code path not traced end-to-end).
- (a) Bound the leave-RUNNING-for-diagnosis contract for FALLBACK-provisioned pods: after `runpod_workload_start_failed`, arm a bounded diagnosis TTL (or an explicit `epm:` note naming who owns the diagnosis) after which the watcher/rung terminates — 4 orphan alerts per incident is the current "bound".
- (b) Ownership gate: the terminal rung must not resume/adopt an existing STOPPED `pod-<N>` unless the handle/lease shows the rung's own lineage owns it (or an explicit opt-in flag); otherwise provision a fresh `pod-<N>-<slug>` suffixed pod.
- (c) Add a `--no-runpod-fallback` per-dispatch flag (threaded like `--min-ram-gb` into `spec.extra`) so a deferrable/nice-to-have dispatch converts the terminal rung into a typed `no_compute_available` defer instead of paid spend.

## Scope / surfaces
- Primary target: `src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/router.py, src/explore_persona_space/backends/issue_dispatch.py`
- Also inspect `scripts/pod_lifecycle.py` (the auto-resume-once branch + provision EXITED-pod semantics) and `scripts/autonomous_session_watch.py` (the unlaunched-orphan alert arm that could gain a terminate arm) — grep `allow_resume` and `runpod_workload_start_failed` across scripts/ + src/ and list every hit in the plan.

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Do NOT delete the RunPod-as-diagnosis-lane contract wholesale — it exists because GCP DELETE destroys crash logs (#658); bound it, don't remove it.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 606acca8ee2c
- workflow_fix_target: src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/router.py, src/explore_persona_space/backends/issue_dispatch.py
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C1.
