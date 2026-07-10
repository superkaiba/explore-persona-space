---
title: 'daily-held: Digest-only guard arm for real-world-corpus raw_'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-09T07:01:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 3): guard_harmful_bank_read.sh
  covers only the fixed six-bank deny set, so after #1102 extended the digest-only
  prose rule to real-world-corpus rollout text (LMSYS/WildChat-class), the prose layer
  is the ONLY defense for that class — refusal kills from paging raw corpus rollouts
  (#1073) have no mechanical backstop.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1102.

## Goal

Decide whether/how to give the real-world-corpus digest-only read rule a mechanical PreToolUse guard arm, extending guard_harmful_bank_read.sh beyond the fixed six-bank deny set.

## Workflow gap

- **Bug observed:** guard_harmful_bank_read.sh covers only the fixed six-bank deny set, so after #1102 extended the digest-only prose rule to real-world-corpus rollout text (LMSYS/WildChat-class), the prose layer is the ONLY defense for that class — refusal kills from paging raw corpus rollouts (#1073) have no mechanical backstop.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + (design phase first) provenance-keyed path patterns for real-world-corpus
  +   raw_completions/ trees -> PreToolUse deny with EPM_ALLOW_BANK_READ-style
  +   sanctioned-maintenance override; grep+line-offset excerpt reads stay allowed
  + register the new arm in .claude/settings.json alongside the existing hook

## Scope / surfaces

- Primary target: `.claude/hooks/guard_harmful_bank_read.sh (+ .claude/settings.json registration)`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- ARCHITECTURAL / must-ask (route3): blocking-behavior hook change; park at plan_pending for user greenlight (spawn WITHOUT --auto or architectural: true in the plan). Sanctioned grep+line-offset excerpt reads MUST remain possible.

## Provenance

- workflow_fix_target: .claude/hooks/guard_harmful_bank_read.sh (+ .claude/settings.json registration)
- origin: parked candidate on task #1102 at 2026-07-07T08:20:29Z

Verbatim parked note:

```
parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — see workflow-fix-on-bug.md § Recursion guard. Candidate (from plan v2 §3 NO-CHANGE table row 1 + alternatives-critic concern): evaluate a digest-only mechanical guard arm for raw_completions/ paths of real-world-corpus provenance — guard_harmful_bank_read.sh currently covers only the fixed six-bank deny set, so after #1102 the prose layer is the only defense for the real-world-corpus class. Blocking-behavior change → architectural/must-ask by the plan's own §13; heterogeneous per-issue paths make a filename-stem deny-set infeasible as-is (needs design, e.g. provenance-keyed path patterns), and the sanctioned grep+line-offset sanitized-excerpt reads must stay allowed. target_file: .claude/hooks/guard_harmful_bank_read.sh (+ .claude/settings.json registration). Logged only — recursion guard; route on a future non-guarded pass.
```
