---
title: 'workflow-fix: implementer reports rev-parse SHAs verbatim'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ff13f7ee39d
- daily-auto-filed
created_at: '2026-07-25T06:50:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): An implementer round-7
  report on 1586 shipped a hand-extended full SHA - a fabricated hex extension of
  a short SHA - and the orchestrator had to rev-parse-correct it before composing
  the relaunch brief'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session dd0af0ae, task #1586, crash-fix round 7).

## Goal

Implementer reports must never fabricate SHA hex; downstream briefs and markers re-cite them.

## Workflow gap

- **Bug observed:** the #1586 round-7 implementer report's "full" SHA was a hand-extended short SHA; the orchestrator caught it (03:18Z: "the report's 'full' SHA was hand-extended — verifying the real tip with rev-parse before the relaunch brief").
- **Why it is a workflow gap:** the never-fabricate-SHA lesson exists orchestrator-side (memory + workflow-fix clause (d) rev-parse duty) but the implementer report contract has no such line, and reports are the SHA source for relaunch briefs and markers.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'rev-parse\|SHA' .claude/agents/experiment-implementer.md` → no report-contract SHA rule (hits are unrelated "shape" matches; absence bind, 2026-07-25). Repo-wide: the sibling `.claude/agents/implementer.md` should be checked at plan time for the same gap (relocation grep left to the session).

## Proposed change (candidate diff sketch — refine in planning)

One line in the report-contract section of `.claude/agents/experiment-implementer.md` (and `.claude/agents/implementer.md` if it shares the contract): report SHAs are pasted verbatim from `git rev-parse` / `git log` output — never hand-extended, truncated-then-extended, or reconstructed from memory.

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`, `.claude/agents/implementer.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 9ff13f7ee39d

- workflow_fix_target: .claude/agents/experiment-implementer.md
