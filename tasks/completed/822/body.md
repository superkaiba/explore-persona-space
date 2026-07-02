---
title: 'workflow-fix: code-reviewer must FAIL on missing smoke-architecture-check
  marker'
kind: infra
tags:
- wf-fix
- wf-fix-fp:734d58e6ad57
created_at: '2026-07-01T23:34:31Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #811: implementer omitted epm:smoke-architecture-check
  across 5 rounds; reviewers never flagged; caught only at Step 6d.0. Add reviewer-side
  mechanical presence check (kind: experiment).'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #811 (emitting agent: orchestrator, /issue 811 session).

## Goal

Add a mechanical check to the code-reviewer's contract checks (Step 0.5/0.6 family) that FAILs an `experiment`-task review round when no `epm:smoke-architecture-check` marker exists for the current implementer round.

## Workflow gap

- **Bug observed:** On task #811 the implementer wrote its PASS_UNIFIED smoke/sweep-parity self-assessment into the dispatcher header ("PASS_UNIFIED — see the epm:smoke-architecture-check marker") but never posted the marker across 5 implementation rounds; both the Claude and Codex code-reviewers PASSed rounds without flagging the missing marker, and the gap surfaced only at the orchestrator's Step 6d.0 pre-dispatch gate — after all review rounds were spent (in general, after Step 6a-6c pod provisioning has already run).
- **Why it is a workflow gap:** `experiment-implementer.md` ("Before writing code" item 5) mandates the marker "before code-review-PASS", and SKILL.md Step 6d.0 refuses dispatch without it — but no reviewer-side mechanical check enforces it at review time, so the omission survives all rounds and costs a post-review bounce (or an orchestrator transcription) at the most expensive point in the pipeline.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/agents/code-reviewer.md (Step 0.5/0.6 mechanical-contract block):
+ - For `kind: experiment` tasks: verify an `epm:smoke-architecture-check v<n>`
+   marker exists for the CURRENT implementer round (verdict PASS_UNIFIED or
+   PASS_CANARY with a named canary cell). Missing -> FAIL with blocker tag
+   `marker-shape` (mechanical-contract class; orchestrator Step 5c-bis strip
+   applies when the marker is in fact present + conforming).
```

Consider also whether the orchestrator's Step 5c-bis mechanical set and/or the
Step 4b implementer brief template (SKILL.md) should name the marker explicitly
so round-1 briefs surface the duty; the spawned session's planner decides scope.

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'smoke-architecture-check' .claude/agents .claude/skills .claude/workflow.yaml CLAUDE.md`)
  and update every hit needing consistency; list them in the plan. Known
  enforcement sites today: `.claude/agents/experiment-implementer.md` (mandate),
  `.claude/skills/issue/SKILL.md` Step 6d.0 (late gate), `.claude/workflow.yaml`
  (marker schema), `.claude/skills/issue/markers.md`.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: 734d58e6ad57

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/code-reviewer.md
bug_observed: issue 811 implementer wrote PASS_UNIFIED self-assessment into the dispatcher header but never posted epm:smoke-architecture-check across 5 rounds; both Claude and Codex code-reviewers PASSed without flagging it; the gap surfaced only at Step 6d.0 pre-dispatch
why_workflow_gap: the marker is mandated pre-code-review-PASS by experiment-implementer.md and gated by SKILL.md Step 6d.0, but no reviewer-side mechanical check enforces its presence at review time
proposed_change: code-reviewer mechanical checks must FAIL an experiment-task round whose current implementer round has no epm:smoke-architecture-check marker
diff_sketch: |
  + Step 0.6-adjacent mechanical check (kind: experiment only): an
  + epm:smoke-architecture-check v<n> marker exists for the current round
  + (PASS_UNIFIED | PASS_CANARY canary_cell=<id>); missing -> FAIL,
  + blocker tag marker-shape.
confidence: medium
related_task: #811
<!-- /workflow-fix-candidate -->
