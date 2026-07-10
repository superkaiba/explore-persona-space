---
title: 'workflow-fix: sync experimenter memory launcher to pid contr'
kind: infra
tags:
- wf-fix
- wf-fix-fp:36ac7e9a52fb
- daily-auto-filed
created_at: '2026-07-09T07:00:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The experimenter memory''s
  canonical launcher recipe still shows the pre-#1070 non-atomic launcher-less pid
  write (`echo $! > .pid`, bare `nohup` without setsid), contradicting the landed
  pid-file launch contract.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08, slice 5) from a
candidate parked on task #1070 by a recursion-guarded workflow-fix session.

## Goal

Harmonize the always-loaded experimenter memory's 'Canonical launcher shape' with the landed pid-file launch contract (pod-side-reporting.md) and experimenter.md's detachment-trio rule — a doc-sync edit, no new behavior.

## Workflow gap

- **Bug observed:** The memory recipe (line ~37) writes the pid launcher-externally via non-atomic `echo $! > /workspace/logs/issue-<N>.pid` and launches with bare `nohup ... &` (no setsid).
- **Why it is a workflow gap:** Agent memories are always-loaded workflow surface steering the experimenter; a memory carrying the pre-#1070 form will be copied verbatim on future launches, re-introducing the class the contract just closed.
- **Confidence (emitter):** high (formal candidate block, code-reviewer r2 on #1070)
- **Triage evidence (2026-07-08):** NOT fixed on main: the memory's 'Canonical launcher shape' still reads `nohup bash /workspace/launch_<N>.sh ... & / echo $! > /workspace/logs/issue-<N>.pid` — non-atomic pid write, no setsid — contradicting pod-side-reporting.md § Pid-file launch contract landed by #1070 itself (commit b3fbb07d1b) and experimenter.md's detachment trio. Pure doc-sync with an already-landed contract -> route1. No dedup; no retraction.

## Proposed change (candidate diff sketch — refine in planning)

```
- nohup bash /workspace/launch_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
- echo $! > /workspace/logs/issue-<N>.pid
+ setsid nohup bash /workspace/launch_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
+ printf '%s\n' "$!" > /workspace/logs/issue-<N>.pid.tmp && mv /workspace/logs/issue-<N>.pid.tmp /workspace/logs/issue-<N>.pid
```

## Scope / surfaces

- Primary target: `.claude/agent-memory/experimenter/feedback_ssh_mcp_sh_not_bash_inline_source.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agent-memory/experimenter/feedback_ssh_mcp_sh_not_bash_inline_source.md
- origin: parked candidate on task #1070 at 2026-07-05T16:19:37Z

Verbatim parked note:

Workflow-fix candidate received from code-reviewer r2 — PARKED, NOT ROUTED: this session runs under the workflow-fix recursion guard (task #1070 carries a workflow_fix_target: Provenance line; see .claude/rules/workflow-fix-on-bug.md § Recursion guard). routed: parked: EPM_WORKFLOW_FIX_SESSION. Next human/orchestrator pass may file it.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agent-memory/experimenter/feedback_ssh_mcp_sh_not_bash_inline_source.md
bug_observed: The memory's "Canonical launcher shape" (line 37) writes the pid launcher-externally via non-atomic `echo $! > /workspace/logs/issue-<N>.pid` and launches with bare `nohup ... &` (no `setsid`), contradicting both the new pod-side-reporting.md § Pid-file launch contract (launcher-less writes must be tmp+mv atomic) and experimenter.md's detachment-trio rule.
why_workflow_gap: Agent memories are always-loaded workflow surface steering the experimenter; a memory recipe carrying the pre-#1070 non-atomic launcher-less form will be copied verbatim on future launches, re-introducing the class the contract just closed.
proposed_change: Harmonize the memory recipe to `setsid nohup ... < /dev/null & printf '%s\n' "$!" > .pid.tmp && mv .pid.tmp .pid` (matching experimenter.md:919), or annotate it as superseded by the contract.
diff_sketch: |
  - nohup bash /workspace/launch_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
  - echo $! > /workspace/logs/issue-<N>.pid
  + setsid nohup bash /workspace/launch_<N>.sh > /workspace/logs/issue-<N>.log 2>&1 < /dev/null &
  + printf '%s\n' "$!" > /workspace/logs/issue-<N>.pid.tmp && mv /workspace/logs/issue-<N>.pid.tmp /workspace/logs/issue-<N>.pid
confidence: high
related_task: #1070
<!-- /workflow-fix-candidate -->

