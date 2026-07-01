---
title: 'workflow-fix: dedup follow-up-round implementer dispatch (per-stage idempotency)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ac5637ece583
created_at: '2026-07-01T22:53:14Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #778 followup round: duplicate stage-dispatch
  stage=followup-implementing round=1 markers (22:28:53Z + 22:34:32Z) spawned two
  concurrent implementers on one worktree; add a per-stage dispatch-dedup window mirroring
  the single-orchestrator guard.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #778 (emitting agent: experiment-implementer).

## Goal

Before dispatching an experiment-implementer for a same-issue follow-up round, check events.jsonl for an existing same-round stage-dispatch marker within a recency window (or an in-flight implementer marker) and skip/no-op the re-dispatch, mirroring the single-orchestrator guard.

## Workflow gap

- **Bug observed:** The /issue orchestrator posted `stage-dispatch stage=followup-implementing round=1 subagent=experiment-implementer` twice (2026-07-01T22:28:53Z and 22:34:32Z) for the SAME follow-up round on #778, spawning two experiment-implementer subagents that concurrently edited one issue-778 worktree. Root context: two orchestrator sessions were driving the same round in parallel (a stale-wake unregistered session + the registered autonomous session); each independently ran planner → gate → implementer dispatch. The Step-0 single-orchestrator guard fired at INVOCATION time but the per-stage dispatch had no idempotency window, so the second orchestrator's Step-9-entry-guard freshness check (which keys on the LATEST marker being a stage-dispatch breadcrumb for the CURRENT stage) did not prevent the duplicate dispatch — intervening markers (codex-task, progress) buried the first breadcrumb.
- **Why it is a workflow gap:** The follow-up-round implementer-dispatch step has no idempotency guard against re-dispatching the same round; the Step 9 entry guard only fires when the breadcrumb is the LATEST event, so any intervening marker defeats it. A stale/slow first dispatch triggers a second, causing a concurrent-write collision and duplicate epm:experiment-implementation markers (the #524/#535 incident family, now at the sub-dispatch grain).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

At the followup-implementing dispatch step (and generalize to every Step 8/9 stage-dispatch site):

```
+ prior = most recent `stage-dispatch stage=followup-implementing round=<r>` marker
+   (scan BACKWARDS through events.jsonl for the CURRENT stage's breadcrumb,
+    skipping other kinds — same per-stage scan the Step 9 parallel-stage note
+    already prescribes, instead of only inspecting the single latest event)
+ if prior exists AND (no epm:failure / epm:experiment-implementation since) AND age < freshness window:
+     log "implementer already dispatched for round <r> at <ts>; skipping re-dispatch"; EXIT
```

Also consider surfacing the Step-0 single-orchestrator guard's "stop the stale session first" action more prominently for the followup-loop entry path (the root cause was two live orchestrators).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'stage-dispatch' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: ac5637ece583

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The /issue orchestrator posted `stage-dispatch stage=followup-implementing round=1 subagent=experiment-implementer` twice (22:28:53Z and 22:34:32Z) for the SAME follow-up round, spawning two experiment-implementer subagents that concurrently edited one issue-778 worktree.
why_workflow_gap: The follow-up-round implementer-dispatch step has no idempotency guard against re-dispatching the same round; a stale/slow first dispatch triggers a second, causing a concurrent-write collision and duplicate epm:experiment-implementation markers.
proposed_change: Before dispatching an experiment-implementer for a followup round, check events.jsonl for an existing same-round stage-dispatch marker within a recency window (or an in-flight implementer marker) and skip/no-op the re-dispatch, mirroring the single-orchestrator guard.
diff_sketch: |
  At the followup-implementing dispatch step:
  + prior = latest `stage-dispatch stage=followup-implementing round=<r>` marker
  + if prior exists AND (no epm:failure/epm:experiment-implementation since) AND age < DISPATCH_DEDUP_WINDOW:
  +     log "implementer already dispatched for round <r> at <ts>; skipping re-dispatch" ; return
  (mirror the Step-0 single-orchestrator detect-and-exit guard for the sub-dispatch case)
confidence: medium
related_task: #778
<!-- /workflow-fix-candidate -->
