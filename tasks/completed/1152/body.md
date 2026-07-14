---
title: 'workflow-fix: harden guard_harmful_bank_read hook (3 bypass '
kind: infra
tags:
- wf-fix
- wf-fix-fp:2e38cb330630
- daily-auto-filed
created_at: '2026-07-09T06:56:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): Three reviewer-surfaced
  residuals in the harmful-bank read guard: cross-command grep flag laundering at
  operator tokens, `diff /dev/null <bank>` as an unguarded paging channel, and a quoted-prose
  false-positive shape in post-marker --note.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #965 (park_form: recursion-guard).

## Goal

Harden guard_harmful_bank_read.sh per the #965 code-review round-1 candidates: close grep instances at operator tokens, guard the `diff /dev/null <bank>` paging channel, and fix the quoted-prose false-positive shape in post-marker --note.

## Workflow gap

- **Bug observed:** Three reviewer-surfaced residuals in the harmful-bank read guard: cross-command grep flag laundering at operator tokens, `diff /dev/null <bank>` as an unguarded paging channel, and a quoted-prose false-positive shape in post-marker --note.
- **Why it is a workflow gap:** The hook is the mechanical enforcement of the digest-only bank-read rule (#866/#965 refusal-kill prevention); an unguarded paging channel defeats it, and an FP shape trains operators to override it.
- **Confidence (emitter):** medium (code-reviewer round-1 PASS-verdict hardening candidates, #965)

## Proposed change (candidate diff sketch — refine in planning)

(a) tokenize commands at &&/||/;/| and match grep bans per-unit; (b) add diff/dev-null bank paging to the deny set (with in_git guard); (c) exempt quoted prose inside post-marker --note strings from the read-pattern match.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_harmful_bank_read.sh`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/hooks/guard_harmful_bank_read.sh
- origin: parked candidate on task #965 at 2026-07-04T08:51:29Z

parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug § Recursion guard). source: prose-followup (code-reviewer round-1 verdict, epm:code-review v1: hardening candidates (a) close grep instances at operator tokens to stop cross-command flag laundering; (b) diff /dev/null <bank> paging with in_git guard; (c) quoted-prose FP shape in post-marker --note). target_file: .claude/hooks/guard_harmful_bank_read.sh. routed: parked: EPM_WORKFLOW_FIX_SESSION — next human/orchestrator pass may file.
