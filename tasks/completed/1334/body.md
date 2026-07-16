---
title: 'workflow-fix: multi-pod-per-issue naming (--name-suffix) so suffixed pods
  map to their owning task'
kind: infra
tags:
- wf-fix
created_at: '2026-07-15T07:24:07Z'
has_clean_result: false
origin_prompt: 'orchestrator observation during #779 n1m round: PM session flagged
  pod-77960 as unmappable orphan; suffixed-issue-number pods invisible to fleet tooling'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #779 (n1m scaling round, 2026-07-15): a PM/watcher session flagged pod-77960 as an unmappable orphan ("issue #77960 doesn't exist") while it was a legitimate #779 follow-up pod.

## Goal

Give `pod.py provision` a first-class multi-pod-per-issue naming form so suffixed pods map to their real owning task.

## Workflow gap

- **Bug observed:** inline follow-up rounds needing a SECOND pod for an issue (the owning pod name taken by a stopped volume) provision under a fabricated issue number (`--issue 77950/77951/77960` for task #779), so the watcher pod-safety pass, stale-pod audit, and PM fleet views cannot map the pod to its task — pod-77960 was flagged as an orphan mid-run and only not killed because it was 0-day-old.
- **Why it is a workflow gap:** `pod.py provision` derives the pod name solely from `--issue N`; there is no supported way to provision `pod-<N>-<suffix>` that still registers task N as owner in pods_ephemeral.json / the audit's owning-task lookup (the keep-running exemption + auto-stop predicates key on the parsed issue number).
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn "pod-<N>\|_MANAGED_PREFIXES\|--issue" scripts/pod_lifecycle.py scripts/pod_audit.py | wc -l` → naming derives from the issue arg in pod_lifecycle.py provision path; no suffix flag exists (grep for "suffix" in scripts/pod_lifecycle.py scripts/pod.py → 0 hits) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

+ pod.py provision --issue N --name-suffix <slug>  -> pod name "pod-N-<slug>"
+ pods_ephemeral.json entry records owning_issue=N (not the fabricated id)
+ pod_audit / autonomous_session_watch pod-safety parse "pod-N-<slug>" -> issue N
  (extend the existing name parser; legacy bare pod-N unchanged)
+ keep-running / auto-stop predicates key on owning_issue

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py, scripts/pod_audit.py, scripts/autonomous_session_watch.py, scripts/pod.py`
- Grep the workflow surface for the name-parsing pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only. Legacy bare `pod-<N>` naming unchanged; suffix form additive (no public-contract break).
- `scripts/workflow_lint.py --check-asks` passes; ruff passes.
- This session runs under the workflow-fix recursion rules; the spawned session must not auto-route its own candidates.

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py, scripts/pod_audit.py, scripts/autonomous_session_watch.py, scripts/pod.py
- fingerprint: (computed at filing by file_infra_task)

Orchestrator observation (verbatim flag from a concurrent session): "pod-77960 — a cpu-mid (0-GPU) pod, running, 0-day old, tagged issue #77960 (which doesn't exist; our tasks top out around #1333) ... nothing legitimate maps to it."
