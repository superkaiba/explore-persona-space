---
title: 'workflow-fix: GCP->RunPod failover provision crash-safe (adopt residue pod)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6a10327c948
created_at: '2026-07-29T20:24:02Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed incident on #1739 (2026-07-29 20:08-20:17Z):
  poll killed mid-failover stranded an orphan 2xH100 RunPod pod; retry refused on
  ''pod already exists'' and minted no_compute_available'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1739 (emitting agent: orchestrator /issue session).

## Goal

Make the #659 GCP->RunPod async-failover provision leg crash-safe and idempotent: on retry, ADOPT (or tear down and recreate) a pre-existing managed pod-<N> left by an interrupted failover attempt instead of refusing into a no_compute_available terminal, and bound the residue window with a provision-intent breadcrumb before pod creation.

## Workflow gap

- **Bug observed:** A poll tick killed by its caller's 120s timeout mid-failover left RunPod pod-1739 (2xH100, id 16rcxufmcvwdt7) provisioned with no workload and no handle; the next failover attempt refused with 'Pod pod-1739 already exists' and minted reason=no_compute_available while the orphan pod billed until manually terminated at 20:17Z.
- **Why it is a workflow gap:** The failover's provision step (pod_lifecycle.py provision, shelled from the RunPod terminal rung inside the poll process) is a multi-minute MUTATING operation running inside a poll tick that callers legitimately bound with timeouts; a kill between pod-create and handle-write/dispatch strands a billing pod, and the retry path treats its own residue as a hard refusal rather than adopting or reaping it. The result is the worst combination: money burning on an orphan pod AND a spurious no_compute_available terminal on a healthy run.
- **Confidence (emitter):** high
- verified-at-filing: incident record on task #1739 — epm:backend-selected marker 2026-07-29T20:13:28Z carries the verbatim refusal (`PodLifecycleProcessError ... Pod pod-1739 already exists (status=RUNNING, id=16rcxufmcvwdt7). Use pod.py resume ... or pod.py terminate ... first`) with outcome `runpod_fallback_failed` + reason `no_compute_available`; `pod.py list-ephemeral --issue 1739` at 20:12Z showed the orphan RUNNING 2xH100 age 0d; epm:progress v60 (20:21:48Z) records the manual termination. Refusal construction: the provision path refuses on an existing same-name pod by design (pod_lifecycle provision) — the gap is the FAILOVER caller not handling that outcome.
- unverified hypothesis — verify at plan time: the orphan pod was created by the FIRST failover attempt (the 20:08Z poll killed at 120s by its caller's timeout) — inferred from age-0d + the timing correlation + no other provision path targeting issue 1739 that hour; no direct creation-log was read.

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/backend_poll.py::_failover_dead_gcp_to_runpod (RunPod provision leg)
+ # BEFORE provisioning: probe for an existing managed pod-<N>.
+ existing = list_managed_pods(issue)
+ if existing and _is_failover_residue(existing, lease/breadcrumb):
+     # adopt: refresh pods.conf from live API, dispatch the workload onto it,
+     # write the handle — OR terminate it and provision fresh (config choice);
+     # never mint no_compute_available on our own residue.
+ # AND: write a provision-intent breadcrumb (lease file / epm marker) BEFORE
+ # the create call, so a killed attempt is attributable + adoptable on retry.
```

## Scope / surfaces

- Primary target: `scripts/backend_poll.py` (the async-failover RunPod provision leg + its retry semantics); coordinate with the existing #1668 adoption machinery (adopted failover clears runpod-wedge sentinel) rather than duplicating it.
- Grep before editing: `grep -rn 'runpod_fallback_failed\|already exists\|_failover_dead_gcp_to_runpod' scripts/ src/explore_persona_space/backends/ tests/`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The failover stays bounded-once (no infinite RunPod cascade); adoption must not hijack a pod a HUMAN provisioned deliberately for the issue (the residue predicate must be positive-evidence: lease/breadcrumb match or age + no-workload probe).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/backend_poll.py
- fingerprint: c6a10327c948

<!-- workflow-fix-candidate v1 -->
target_file: scripts/backend_poll.py
bug_observed: A poll tick killed by its caller's 120s timeout mid-failover left RunPod pod-1739 (2xH100, id 16rcxufmcvwdt7) provisioned with no workload and no handle; the next failover attempt refused with 'Pod pod-1739 already exists' and minted reason=no_compute_available while the orphan pod billed until manually terminated at 20:17Z.
why_workflow_gap: The failover provision is a multi-minute mutating op inside a timeout-bounded poll tick; a mid-kill strands a billing pod and the retry treats its own residue as a hard no_compute_available refusal instead of adopting or reaping it.
proposed_change: Adopt-or-reap a pre-existing managed pod on failover retry + write a provision-intent breadcrumb before pod creation.
confidence: high
related_task: #1739
<!-- /workflow-fix-candidate -->
