---
title: 'workflow-fix: PASS_UNIFIED must cover non-cell smoke-axis min-N floors'
kind: infra
tags:
- wf-fix
- wf-fix-fp:55ea303235bb
created_at: '2026-07-15T13:48:53Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1315 r4 (experiment-implementer): PASS_UNIFIED
  silent on non-cell smoke axes vs downstream min-N asserts'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1315 (emitting agent: experiment-implementer, crash-fix round 4).

## Goal

Extend experiment-implementer.md item 5's PASS_UNIFIED definition: any smoke-scale slicing of a non-cell axis must satisfy every downstream phase's minimum-N asserts (grep consumers for `assert len(...) >=` shapes; name the floor per sliced axis in the attestation note).

## Workflow gap

- **Bug observed:** A PASS_UNIFIED smoke (cell subset threaded through every phase) still crashed by construction at its LAST phase because a NON-cell smoke axis (question slice) was sized below a downstream consumer's min-N assert (#1315 r4: questions[:1] vs split_half_self_cosine's len(qs)>=2).
- **Why it is a workflow gap:** The § pre-flight item-5 PASS_UNIFIED definition requires per-phase CELL-subset threading but is silent on non-cell smoke-scale slices (questions/rows/steps) vs downstream min-N asserts, so an un-passable-by-construction smoke can still attest PASS_UNIFIED.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "PASS_UNIFIED" .claude/agents/experiment-implementer.md` → 5 hits (presence of the item-5 attestation site confirmed); `grep -ciE "non-cell|min-N|minimum-N" .claude/agents/experiment-implementer.md` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In experiment-implementer.md § Before writing code, item 5, after
"Per-phase subset threading is part of the PASS_UNIFIED definition":
+ The same duty covers NON-cell smoke axes: for every axis the smoke slices
+ (questions, rows, steps, draws), verify the sliced size satisfies each
+ downstream phase's minimum-N asserts and name the floor per axis in the
+ attestation note; a sliced axis below a downstream min-N is un-passable
+ by construction — FAIL_NO_CANARY (#1315 r4).

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'PASS_UNIFIED' .claude/ CLAUDE.md scripts/`) and update every hit
  that defines (not merely consumes) the attestation; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: 55ea303235bb

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/experiment-implementer.md
bug_observed: A PASS_UNIFIED smoke (cell subset threaded through every phase) still crashed by construction at its LAST phase because a NON-cell smoke axis (question slice) was sized below a downstream consumer's min-N assert (#1315 r4: questions[:1] vs split_half_self_cosine's len(qs)>=2).
why_workflow_gap: The § pre-flight item-5 PASS_UNIFIED definition requires per-phase CELL-subset threading but is silent on non-cell smoke-scale slices (questions/rows/steps) vs downstream min-N asserts, so an un-passable-by-construction smoke can still attest PASS_UNIFIED.
proposed_change: Extend item 5's PASS_UNIFIED definition: any smoke-scale slicing of a non-cell axis must satisfy every downstream phase's minimum-N asserts (grep consumers for `assert len(...) >=` shapes; name the floor per sliced axis in the attestation note).
diff_sketch: |
  In experiment-implementer.md § Before writing code, item 5, after
  "Per-phase subset threading is part of the PASS_UNIFIED definition":
  + The same duty covers NON-cell smoke axes: for every axis the smoke slices
  + (questions, rows, steps, draws), verify the sliced size satisfies each
  + downstream phase's minimum-N asserts and name the floor per axis in the
  + attestation note; a sliced axis below a downstream min-N is un-passable
  + by construction — FAIL_NO_CANARY (#1315 r4).
confidence: medium
related_task: #1315
<!-- /workflow-fix-candidate -->
