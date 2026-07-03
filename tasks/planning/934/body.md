---
title: 'workflow-fix: GCP lane multi-lane instance naming + loud reconnect result'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ef55261235c
created_at: '2026-07-03T15:00:03Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 923 Step 6b: second concurrent
  GCP launch reconnected to the first lane''s instance (eps-issue-<N> collision) and
  returned ok:true without dispatching the workload'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #923 (emitting agent: orchestrator, /issue 923 Step 6b launch).

## Goal

Support a per-lane instance-name suffix on the GCP lane so two concurrent lanes for one issue can coexist, and surface a reconnect outcome loudly in the dispatch result instead of ok:true.

## Workflow gap

- **Bug observed:** A plan with two concurrent GCP lanes (GPU ft-7b + CPU cpu-mid, #923 plan SS9 "Phase 1 (GPU) || Phase 2 (CPU) start together") cannot be dispatched: `gcp.py` names every instance `eps-issue-<N>` (gcp.py:755), so the second `dispatch_issue.py launch` found the first lane's live instance and RECONNECTED (reason=reconnect) instead of creating a second instance — and the launch JSON still carried `ok: true` with `workload_executed: null`, masking that the second workload was never dispatched. An orchestrator not inspecting `reason` would believe both lanes launched.
- **Why it is a workflow gap:** the backend router (workflow surface per workflow-fix-on-bug.md scope) structurally cannot realize a multi-lane plan shape the planner/critics approved, and the reconnect masking makes the failure silent (fail-fast violation at the dispatch seam).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/gcp.py
- def instance_name(issue): return f"eps-issue-{issue}"
+ def instance_name(issue, lane_suffix=None): return f"eps-issue-{issue}" + (f"-{lane_suffix}" if lane_suffix else "")
# dispatch_issue.py launch: add --lane-suffix (threads to instance name + handle sidecar
#   path issue-<N>-<suffix>-handle.json); janitor grep eps-issue-* already matches suffixed names.
# issue_dispatch: when route() resolves reason=reconnect on a launch that carried
#   --workload-cmd, print a LOUD warning + set ok-with-caveat field (e.g.
#   "workload_dispatched": false) so orchestrators cannot mistake reconnect for a launch.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`, `scripts/dispatch_issue.py`, `src/explore_persona_space/backends/issue_dispatch.py`
- Grep the workflow surface for the pattern before editing (`grep -rn "eps-issue" src/explore_persona_space/backends/ scripts/dispatch_issue.py scripts/backend_poll.py scripts/gcp_audit.py`) and update every naming/reap/poll consumer; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The stale-VM janitor + pod-safety watchers must keep matching suffixed instances (`eps-issue-*` grep already does — verify).
- This session runs under the workflow-fix recursion guard once spawned.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py, scripts/dispatch_issue.py
- fingerprint: 9ef55261235c

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/gcp.py, scripts/dispatch_issue.py
bug_observed: Second concurrent launch for the same issue reconnected to the first lane's live eps-issue-N instance and returned ok:true without dispatching its workload
why_workflow_gap: The GCP lane's fixed eps-issue-<N> instance naming makes multi-lane plans structurally undispatable, and the reconnect result masks the non-launch (silent failure at the dispatch seam)
proposed_change: Support a per-lane instance-name suffix on the GCP lane so two concurrent lanes for one issue can coexist, and surface a reconnect outcome loudly in the dispatch result instead of ok:true
diff_sketch: |
  + instance_name(issue, lane_suffix=None) -> "eps-issue-<N>[-<suffix>]"
  + dispatch_issue.py --lane-suffix (threads name + per-lane handle sidecar)
  + reconnect on a --workload-cmd launch -> loud warning + workload_dispatched: false in the JSON
confidence: high
related_task: #923
<!-- /workflow-fix-candidate -->
