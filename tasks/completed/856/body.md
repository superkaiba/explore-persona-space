---
title: 'workflow-fix: durable-output check before reviewer no-show fallback'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c73d5e2c53ed
created_at: '2026-07-02T09:30:41Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from /issue 810: check durable verdict outputs
  (posted marker / verdict temp file) before treating a subagent thrash-death as a
  reviewer no-show (incident 2026-07-02 ~09:25Z, #810 r4 ensemble)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #810 (emitting agent: orchestrator, /issue 810 session).

## Goal

At ensemble verdict collection (Step 5b and the analogous 9a/9a-bis reads), check for durable outputs (a posted verdict marker or a written verdict file) before treating a subagent thrash-death/error notification as a reviewer no-show.

## Workflow gap

- **Bug observed:** the orchestrator misread a thrash-killed `code-reviewer` as a no-show and adopted a unilateral ensemble decision (FAIL from the Codex verdict alone), although the reviewer's PASS verdict had ALREADY been durably posted as `epm:code-review v4` — only the agent's final summary turn died of autocompact thrash. The orchestrator had to post a corrective marker and route to the reconciler after the fact (task #810, 2026-07-02, ~09:25Z).
- **Why it is a workflow gap:** SKILL.md Step 5b instructs reading both markers from events.jsonl, but the failure-handling prose (the "Codex twin no-show fallback") keys on the Agent-tool completion RESULT, not on durable task state; an agent whose work is durable but whose final turn dies (autocompact thrash — a recurring class: 3 such deaths on #810 alone) is indistinguishable from a genuine no-show unless the orchestrator re-reads events.jsonl / the verdict temp file first.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
+ In Step 5b (and the 9a / 9a-bis ensemble reads), when an Agent-tool
+ completion result reports an error/thrash death for a reviewer subagent,
+ BEFORE invoking any no-show fallback or unilateral decision:
+   1. Re-read the task's events.jsonl for the expected verdict marker
+      (epm:code-review v<n> / twin / critique kinds) posted after the
+      dispatch breadcrumb; and
+   2. Check the conventional verdict temp file (/tmp/issue-<N>-...-v<n>.md).
+ If either exists and conforms, treat the reviewer as RETURNED and use the
+ durable verdict; a thrash-killed summary turn is not a no-show.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'no-show' .claude/ CLAUDE.md scripts/`) and update every relevant ensemble-read site; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: c73d5e2c53ed

Surfaced prose (verbatim): "the orchestrator should check for durable outputs (posted markers / written verdict files) before treating a thrash-death notification as a reviewer no-show" — raised in #810 epm:progress corrective note, 2026-07-02.
