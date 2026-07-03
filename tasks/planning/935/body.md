---
title: 'workflow-fix: GCE self-poweroff missing on the [phase=done] path (post-#908
  residual)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c6be7271cf75
created_at: '2026-07-03T15:01:15Z'
has_clean_result: false
origin_prompt: 'orchestrator observation #763 v2p1: eps-issue-763 RUNNING 20+min after
  [phase=done], manual delete required; #908 covered reconnect/reclaim only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #763 (emitting agent: orchestrator, v2p1 GPU-phase teardown).

## Goal

GCE startup-script must power the instance off immediately after a successful workload exit (`[phase=done]`), not rely on next-launch reclaim or the daily janitor.

## Workflow gap

- **Bug observed:** eps-issue-763 stayed RUNNING 20+ min after `[phase=done]` 2026-07-03T14:40Z and required a manual `gcloud compute instances delete` (second occurrence after the #763 cofit-round zombie on 2026-07-03 morning).
- **Why it is a workflow gap:** the GCP lane's stated invariant is "EXIT-trap teardown bounds billing"; #908 (completed, commit bc692ef979) added zombie reconnect-refusal + pre-launch reclaim (defense at the NEXT launch) but did not close the trap itself — a done-workload instance still bills until a manual delete, a next launch on the same issue, or the 09:37 daily janitor (up to ~19h worst case).
- **Confidence (emitter):** medium — the observed zombie ran the issue-763 BRANCH's startup template (branch gcp.py tip predates #908's merge, and possibly other main-side template fixes). The planner must FIRST diff main's current startup-script template: if main already powers off on the done path, deflect with a reasoned no-change report (the observation is then a stale-branch artifact and the residual gap is only "long-lived issue branches dispatch with stale lane code", which should then be evaluated as the fix target instead — e.g. dispatch_issue.py generating the startup script from main's backends/ regardless of --repo-branch).

## Proposed change (candidate diff sketch — refine in planning)

```
src/explore_persona_space/backends/gcp.py (startup-script template):
+ after the workload command exits rc=0 and the '[phase=done]' guest-attribute
+ write + diagnostics/upload steps complete, unconditionally invoke the
+ bounded self-poweroff (same mechanism as the crash path), so a RUNNING
+ instance past done+<grace> cannot exist. If main already does this, the
+ alternative target is making dispatch always source the startup template
+ from main's backends/ (lane infra) while cloning --repo-branch for the
+ workload only.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for sibling teardown prose before editing (`grep -rln 'phase=done' src/explore_persona_space/backends/ .claude/rules/compute-backend-failover.md .claude/skills/issue/SKILL.md`) and update every in-scope hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Crash-path diagnostics persistence (`_eps_persist_diagnostics`) and the #669 wedged-write must stay intact; the poweroff must remain time-bounded so it can never hang billing open.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: c6be7271cf75

Surfaced observation (orchestrator, #763 v2p1 round): "GPU workload printed `[phase=done]` at 14:40:33Z and uploaded all outputs; the instance remained RUNNING at 15:02Z (verified twice via `gcloud compute instances list`) and was deleted manually. #908's completed fix covers reconnect-refusal + pre-launch reclaim, not the missing self-poweroff on the done path."
