---
title: 'daily-fix: trigger-dense ingest hardening (refusal results)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e3b7b97de25e
- daily-auto-filed
created_at: '2026-08-03T07:01:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): Three ingest-side refusal
  wedges in one day: (a) session 291d866a -- orchestrator wedged 3 consecutive turns
  (10:19-10:48Z) immediately after ingesting a refusal-killed code-reviewer''s <result>
  refusal text verbatim on guard-surface task #1928; ~30 min to watcher respawn. (b)
  session f98a12ed -- orchestrator paged a 14KB trigger-dense epm:followup-scope marker
  verbatim (04:52Z), its planner spawn'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miners 1/7/8, sessions f98a12ed/291d866a/5efd349e, tasks #1739/#1928/#1936).

## Goal

Orchestrator and reviewer INGEST paths on trigger-dense rounds never carry verbatim refusal text, scope-marker bodies, or real-corpus parent bodies into context.

## Workflow gap

- **Bug observed:** Three ingest-side refusal wedges in one day: (a) session 291d866a -- orchestrator wedged 3 consecutive turns (10:19-10:48Z) immediately after ingesting a refusal-killed code-reviewer's <result> refusal text verbatim on guard-surface task #1928; ~30 min to watcher respawn. (b) session f98a12ed -- orchestrator paged a 14KB trigger-dense epm:followup-scope marker verbatim (04:52Z), its planner spawn was refusal-killed at 16 tool calls reading the scope verbatim, then the orchestrator itself refused 3 turns and the transcript ends (05:03-05:22Z). (c) session 5efd349e -- a critic's background resume was refusal-killed after its transcript paged the #1901 WildChat-corpus parent body.
- **Why it is a workflow gap:** trigger-dense-review.md's #1546 clause covers run-failure/forensics text (poll tails, crash logs) but does not name refusal-result text from killed subagents, followup-scope marker bodies, or parent-body corpus text as ingest surfaces -- all three wedged sessions today.
- **Confidence (emitter):** medium (incidents probed by miners -- isApiErrorMessage row counts + quoted rows; the clause-coverage gap read from the rule file at compose time)
- verified-at-filing: `grep -n -A6 '1546' .claude/rules/trigger-dense-review.md` -> the ingest clause enumerates poll/forensics text (crash logs, stderr tails); 0 hits for 'refusal text'/'followup-scope'/'result text' as ingest surfaces. Related open sibling #2003 (proposed) covers SPAWN-side first-pass pre-qualification -- distinct side of the same class, not a duplicate (ingest vs spawn).

## Proposed change (refine in planning)

extend the #1546/#1563 clauses: (i) a refusal-killed subagent's result/refusal text is digested to one neutral line BEFORE the durable-verdict check, never processed verbatim on a guard-surface round; (ii) followup-scope marker bodies on trigger-dense tasks are read by file-reference/structured digest, never paged verbatim into orchestrator context; (iii) critic/reviewer briefs on tasks whose PARENT body embeds real-corpus (LMSYS/WildChat) text pass that body by reference.

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e3b7b97de25e

- workflow_fix_target: .claude/rules/trigger-dense-review.md

