---
title: 'workflow-fix: provision-time warn on terminal-parent pods'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b7ce85e021b9
- daily-auto-filed
created_at: '2026-07-09T06:59:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): pod.py provision --issue
  <N> on a task in the pod-safety auto-stop set (DONE ∪ on_hold) with neither a fresh
  follow-up signal marker nor the keep-running tag prints no warning — a session that
  never consults the CLAUDE.md two-signal bullet provisions a pod the watcher will
  auto-stop mid-bootstrap (#573/#779 class).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1080 (recursion-guarded workflow-fix session).

## Goal

Give the pod-safety two-signal duty a mechanical belt at the provision choke point instead of relying on doc consultation.

## Workflow gap

- **Bug observed:** pod.py provision --issue <N> on a task in the pod-safety auto-stop set (DONE ∪ on_hold) with neither a fresh follow-up signal marker nor the keep-running tag prints no warning — a session that never consults the CLAUDE.md two-signal bullet provisions a pod the watcher will auto-stop mid-bootstrap (#573/#779 class).
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/pod.py, scripts/pod_lifecycle.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# pod.py provision, after resolving --issue N:
status = task_status(N)
if status in POD_SAFETY_AUTO_STOP_SET and not (has_keep_running_tag(N) or fresh_followup_signal(N)):
    print("WARNING: issue N is at a parked/terminal status; the watcher auto-stops "
          "unsignalled pods. Run `task.py add-tag N keep-running` BEFORE provisioning "
          "and post epm:run-launched once the pod exists (CLAUDE.md two-signal recipe).")
```

## Scope / surfaces

- Primary target: `scripts/pod.py, scripts/pod_lifecycle.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod.py, scripts/pod_lifecycle.py
- origin: parked candidate on task #1080 at 2026-07-06T09:52:35Z

Verbatim parked note:

> source: prose-followup (alternatives critic, Phase 2). routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-filed by this session. Candidate: target_file: scripts/pod.py, scripts/pod_lifecycle.py — add a provision-time check in 'pod.py provision --issue <N>': when the issue's status is in the pod-safety auto-stop set (DONE ∪ on_hold) and neither a fresh follow-up signal marker nor the keep-running tag is present, print a loud WARNING quoting the two-signal recipe (or require an explicit --ack-terminal-parent flag). Rationale: the mechanical belt the #1080 doc sentence cannot provide — a session that never consults the bullet still runs provision; unlike an auto-post it does not blanket-exempt escaped provisions. confidence: medium. related_task: #1080. For the next human/orchestrator pass to file.
