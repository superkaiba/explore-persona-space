---
title: 'workflow-fix: pod-side sentinel read-back tolerance under poller drain-rename'
kind: infra
tags:
- wf-fix
- wf-fix-fp:91f2edb65a6a
created_at: '2026-07-14T23:58:39Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from code-reviewer on #1090 fu4 r1: pod-side
  sentinel read-back breaks under the poller''s .processed rename'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1090 (emitting agent: code-reviewer, fu4 review round 1).

## Goal

Add a read-back clause to the pod-side sentinel contract — any pod-side consumer of its own issue-<N>-*.json sentinels must tolerate the `<path>.processed` rename (read both forms) or keep its resume state outside the drained glob.

## Workflow gap

- **Bug observed:** fu4's (and fu3's) dispatcher uses its own /workspace/logs/issue-<N>-*.json per-run sentinels as resume/finalize state, but the poller drains and renames them to `.processed`, silently breaking read-back (fu3's production reproducibility_card covered only 23-24 of 35 cells).
- **Why it is a workflow gap:** pod-side-reporting.md specifies the sentinel WRITE contract but says nothing about pod-side READ-BACK semantics under the poller's drain-and-rename, so each new dispatcher re-invents a racy read.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'read-back\|readback\|read both' .claude/rules/pod-side-reporting.md` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence); `grep -n '\.processed' .claude/rules/pod-side-reporting.md` → 1 hit (line 92, drain-side rename warning only, no consumer read-back clause) (2026-07-15 UTC)

## Proposed change (candidate diff sketch — refine in planning)

```
+ 3. **Read-back tolerance.** If your dispatcher READS its own sentinels
+    (resume, completion checks, finalize aggregation), it MUST also read
+    `<path>.processed` — the poller renames each drained sentinel — or keep
+    resume/finalize state in a file OUTSIDE `/workspace/logs/issue-<N>-*.json`
+    (e.g. `<out_root>/<unit>/status.json`). Incident: #1090 fu3/fu4 per-run
+    sentinels drained mid-run → requeue races + incomplete reproducibility_card.
```

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'processed' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/pod-side-reporting.md
- fingerprint: 91f2edb65a6a

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/pod-side-reporting.md
bug_observed: fu4's (and fu3's) dispatcher uses its own /workspace/logs/issue-<N>-*.json per-run sentinels as resume/finalize state, but the poller drains and renames them to `.processed`, silently breaking read-back (fu3's production reproducibility_card covered only 23-24 of 35 cells).
why_workflow_gap: pod-side-reporting.md specifies the sentinel WRITE contract but says nothing about pod-side READ-BACK semantics under the poller's drain-and-rename, so each new dispatcher re-invents a racy read.
proposed_change: add a read-back clause to the sentinel contract — any pod-side consumer of its own issue-<N>-*.json sentinels must tolerate the `<path>.processed` rename (read both forms) or keep its resume state outside the drained glob.
confidence: high
related_task: #1090
<!-- /workflow-fix-candidate -->
