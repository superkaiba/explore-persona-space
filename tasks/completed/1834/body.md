---
title: 'workflow-fix: marker-materialized run artifacts must match producer schema'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9683849a5528
created_at: '2026-07-29T18:17:04Z'
has_clean_result: false
origin_prompt: 'analyzer candidate block on #1775 (see body Provenance): materialized
  plan_deviations.json dict-schema vs producer bare-list schema; add match-the-writer-schema
  guidance to upload-verifier/orchestrator materialization'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1775 (emitting agent: analyzer).

## Goal

Add a rule to the upload-verifier/orchestrator materialization guidance: before back-filling any run-artifact file from markers, read the experiment code's writer for that path and emit the identical schema (or write to a differently-named sidecar).

## Workflow gap

- **Bug observed:** The #1775 orchestrator/verifier materialized `eval_results/issue_1775/plan_deviations.json` from run markers in a dict schema (`{"issue":..., "deviations":[...]}`) while the experiment's own `record_plan_deviation` reader/writer uses a bare list of dicts, so any later analysis call through the wrapper crashes with AttributeError.
- **Why it is a workflow gap:** No guidance tells the materializing actor to match the producing code's on-disk schema when back-filling a run artifact from markers; the mismatch was invisible until analysis-time code re-read its own file.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'materializ' .claude/agents/upload-verifier.md` → 0 hits (absence-of-guidance claim — the 0-hit in-target result IS the evidence); repo-wide `grep -rln 'materializ' .claude/agents/*.md .claude/skills/issue/SKILL.md` → 6 files, all unrelated contexts (trigger-dense excerpt pre-materialization, git/lane branch materialization, download-cache materialization), none implements the proposed guidance (context-read per clause (c)) (2026-07-29)
- Note: the IMMEDIATE #1775 artifact was already re-written to the producer schema (bare list, `record_plan_deviation`-compatible) at commit a8e75537010f89d34bb75e42c82b7a91c7d485cd on branch issue-1775 — this filing is for the durable guidance so the next remediation doesn't repeat the mismatch.

## Proposed change (candidate diff sketch — refine in planning)

```
+ When materializing a missing run artifact (e.g. plan_deviations.json) from
+ markers, FIRST grep the experiment scripts for the writer of that path and
+ reproduce its exact schema; if the schema is unclear, write a sidecar
+ (<name>.materialized.json) instead of the canonical path — a schema-mismatched
+ canonical file crashes the experiment's own readers at analysis time (#1775).
```

## Scope / surfaces

- Primary target: `.claude/agents/upload-verifier.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'materializ' .claude/ CLAUDE.md scripts/`) and update every hit
  that concerns marker→artifact remediation; list them in the plan. The
  remediation guidance may also belong beside the gap-fill/uploader text in
  `.claude/skills/issue/SKILL.md` Step 8 — planner decides placement.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/upload-verifier.md
- fingerprint: 9683849a5528

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/upload-verifier.md
bug_observed: The #1775 orchestrator/verifier materialized `eval_results/issue_1775/plan_deviations.json` from run markers in a dict schema (`{"issue":..., "deviations":[...]}`) while the experiment's own record_plan_deviation reader/writer uses a bare list of dicts, so any later analysis call through the wrapper crashes with AttributeError.
why_workflow_gap: No guidance tells the materializing actor to match the producing code's on-disk schema when back-filling a run artifact from markers; the mismatch was invisible until analysis-time code re-read its own file.
proposed_change: Add a rule to the upload-verifier/orchestrator materialization guidance: before back-filling any run-artifact file from markers, read the experiment code's writer for that path and emit the identical schema (or write to a differently-named sidecar).
diff_sketch: |
  + When materializing a missing run artifact (e.g. plan_deviations.json) from
  + markers, FIRST grep the experiment scripts for the writer of that path and
  + reproduce its exact schema; if the schema is unclear, write a sidecar
  + (<name>.materialized.json) instead of the canonical path — a schema-mismatched
  + canonical file crashes the experiment's own readers at analysis time (#1775).
confidence: medium
related_task: #1775
<!-- /workflow-fix-candidate -->
