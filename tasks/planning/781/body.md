---
title: 'workflow-fix: add live-batch-shape smoke bullet to gotchas.md (#763 r3 root
  cause)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:763r3livebatchsmoke
created_at: '2026-07-01T02:35:30Z'
has_clean_result: false
origin_prompt: 'task #763 r3 experiment-implementer emitted a workflow-fix-candidate
  for gotchas.md: 8000 graded batch requests quarantined on an empty system content
  block; mock-judge smoke could not catch it.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #763 r3 (emitting agent: experiment-implementer).

## Goal

Add a `.claude/rules/gotchas.md` bullet warning that any Anthropic Batch API judge/generation path needs a tiny (~5-request) LIVE forced-batch smoke pre-launch, because `--mock-judge` / offline smokes structurally cannot validate the Batch request shape (an empty system content block is a hard 400 that quarantines EVERY request).

## Workflow gap

- **Bug observed:** A `--mock-judge` (offline) smoke passed the r2 code-review, but the first LIVE Anthropic Batch submit on task #763 r2→r3 quarantined all 8000 graded requests on an empty-system-block 400 — the mock smoke structurally cannot exercise the batch request shape, so the malformed shape was invisible until the pod run.
- **Why it is a workflow gap:** The `gotchas.md` index has no "live-batch-shape smoke" bullet warning that an offline/mock judge smoke does not validate the Anthropic Batch request shape (empty system block, malformed params), and that a tiny forced-batch live submit is the only pre-launch gate that catches it. The r2 review + Codex both missed the empty-system-block bug because no rule flagged the mock-vs-live-batch gap.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

Add to `.claude/rules/gotchas.md`:

```
- **A `--mock-judge` / offline judge smoke does NOT validate the Anthropic
  Batch REQUEST SHAPE.** An empty system content block (`system=[{text:""}]`)
  is a hard 400 (`invalid_request_error`, "text content blocks must be
  non-empty") that quarantines EVERY request in a batch instantly. The mock
  smoke makes no API call, so it cannot catch it. Gate any Batch-API judge
  path with a tiny (~5-request) LIVE forced-batch smoke (threshold_base=0)
  pre-launch, asserting `succeeded` not `invalid_request_error`. (#763 r3:
  8000 graded requests quarantined on an empty system block.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern (`grep -rln 'mock-judge\|Anthropic Batch\|invalid_request_error' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 763r3livebatchsmoke

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: A `--mock-judge` (offline) smoke passed but the first LIVE Anthropic Batch submit quarantined all 8000 graded requests on an empty-system-block 400 — the mock smoke structurally cannot exercise the batch request shape (task #763 r2→r3).
why_workflow_gap: The gotchas index has no "live-batch-shape smoke" bullet warning that offline/mock judge smokes do not validate the Anthropic Batch request shape.
proposed_change: Add a gotchas.md bullet requiring any Batch-API judge path to have a tiny (~5-request) LIVE forced-batch smoke pre-launch.
confidence: high
related_task: #763
<!-- /workflow-fix-candidate -->
