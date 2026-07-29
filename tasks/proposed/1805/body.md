---
title: 'daily-fix: review-time no-flags lint on new scripts in diff'
kind: infra
tags:
- wf-fix
- wf-fix-fp:39ca63113a50
- daily-auto-filed
created_at: '2026-07-29T07:14:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1092''s Step 10d lint
  gate BLOCKed the merge on two hub-verify waivers missing from a NEW script (an inspect.signature
  reference + a retry_transient-wrapped call, neither network-risky) — a ~14 min waiver
  round + full gate re-run at the end of an 8-hour round that 3 review rounds could
  have caught'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-B P4 (miner-probed; sites re-verified).

## Goal

Catch lint-gate blockers at review time (or stop the hub-verify leg flagging non-network-risky reference shapes).

## Workflow gap

- **Bug observed:** #1092's merge was BLOCKed at the Step 10d lint gate on two `list_repo_tree` sites in a round-new script: one a bare `inspect.signature(...)` reference (signature-bind smoke), one already wrapped in `hub.retry_transient(...)`. Both needed `# HUB_VERIFY_RETRY_EXEMPT:` waivers; the waiver round + full gate re-run cost ~14 min at the end of an ~8h round. The lint behaved as designed (call OR bare-reference detection is deliberate); the gap is that 3 code-review rounds never ran the lint on the new script.
- **Why it is a workflow gap:** review rounds have no duty to run the no-flags lint on round-new scripts, so deterministic gate blockers surface at the merge gate instead of review.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'HUB_VERIFY_RETRY_EXEMPT' scripts/workflow_lint.py` → detection/waiver machinery at 469/1978/2098/7511; `grep -n 'workflow_lint' .claude/agents/code-reviewer.md` → 0 hits (no reviewer-side lint duty) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Prefer option (ii) (one checklist line in code-reviewer.md, mirrored to the codex twin composer); option (i) (AST auto-exemptions) is the deeper fix the planner may choose instead or additionally.

## Scope / surfaces

- Primary targets: `.claude/agents/code-reviewer.md`, optionally `scripts/workflow_lint.py` (hub-verify leg ~L1959-2100)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 39ca63113a50

- workflow_fix_target: .claude/agents/code-reviewer.md

