---
title: 'workflow-fix: enforce keep-running tag in issue-wide terminate + finalize
  teardown'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1f868ac07797
created_at: '2026-07-17T23:52:01Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1345 pod-1345-onpolicy collateral-terminate
  incident 2026-07-17: keep-running shield is prose-only; cmd_terminate + finalize
  never read the tag'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1345 (emitting agent: user-chat inline orchestrator).

## Goal

Enforce the task-level `keep-running` tag mechanically in the issue-wide pod teardown path: `cmd_terminate`'s bare form skips (loudly) when the owning task carries the tag, and `dispatch_issue.py finalize` checks the tag before teardown.

## Workflow gap

- **Bug observed:** the issue-wide `pod.py terminate --issue N` sweep (documented: destroys EVERY live pod for the issue, suffixed follow-up pods included) and the `dispatch_issue.py finalize` teardown do NOT check the `keep-running` tag anywhere in code — the shield exists only as SKILL-level prose ("Skip only with the keep-running tag") and in the cmd_terminate docstring. 2026-07-17 incident: the #1345 slot-ablation round's 23:42:00Z finalize terminate destroyed the parallel suffixed pod `pod-1345-onpolicy` mid-launch (workload un-started, pod re-provisioned; the tag was unset then — operator miss — but is set NOW, and a finalize RETRY by the round-close session would still bypass it mechanically because nothing in the terminate path reads the tag).
- **Why it is a workflow gap:** CLAUDE.md (§ Pods: "the `keep-running` shield is ISSUE-WIDE — it also blocks Step-8 teardown") and the cmd_terminate docstring ("a round that must survive Step 8 sets the task-level keep-running tag") both describe the tag as the teardown shield, but no implementing file enforces it — the explicit "CLAUDE.md describes a rule but the implementing file doesn't enforce it" YES-emit case.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i 'keep.running\|keep_running' scripts/dispatch_issue.py scripts/pod_lifecycle.py src/explore_persona_space/backends/*.py` → 6 hits, ALL in scripts/pod_lifecycle.py (lines 2283/2332/2341/2346/2383/2639); 0 hits in scripts/dispatch_issue.py (per-target: absence confirmed); the only pod_lifecycle CODE check (line 2383) sits in `_warn_on_terminal_parent_provision` (pod-safety warning helper), NOT in `cmd_terminate` — cmd_terminate body (lines 2631-2700, read in full) gates ONLY on `_guard_upload_verification_before_terminate`; landed-fix history `git log --oneline --since='7 days ago' -- scripts/pod_lifecycle.py scripts/dispatch_issue.py` → no keep-running guard landed (top hits c39778ec3c/#1468 GPU-mem rung gate, ffea76bbad/#1334 which INTRODUCED the sweep semantics) (2026-07-17).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/pod_lifecycle.py cmd_terminate():
+   # keep-running tag = documented Step-8 teardown shield (CLAUDE.md § Pods).
+   # Bare (issue-wide) form: refuse with a loud message when the owning task
+   # carries the tag; --name-suffix-targeted terminate stays allowed (surgical
+   # destroy of the tagged round's own pod is the operator's explicit choice).
+   if name_suffix is None and _task_has_keep_running_tag(args.issue):
+       print("REFUSED: task #%d carries keep-running ..." % args.issue); return
scripts/dispatch_issue.py finalize:
+   same check before the teardown leg (skip teardown, exit with the
+   documented teardown-skipped reason).
```

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py`, `scripts/dispatch_issue.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'keep.running' .claude/ CLAUDE.md scripts/`) and keep SKILL.md Step 8 prose consistent.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; CLAUDE.md/workflow.yaml stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- An override flag (e.g. `--force-keep-running`) may be added for deliberate teardown of a tagged issue; planner decides.

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py, scripts/dispatch_issue.py
- fingerprint: 1f868ac07797

Surfaced prose (verbatim): issue-wide cmd_terminate + dispatch_issue finalize teardown do not check the task-level keep-running tag; the shield exists only as SKILL-level prose, so a finalize retry can destroy a tag-protected parallel suffixed pod (2026-07-17 pod-1345-onpolicy collateral incident).
