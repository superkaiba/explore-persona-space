---
title: 'daily-fix: SKILL_REF_RE filesystem-path false positive'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d8571a5036b6
- daily-auto-filed
created_at: '2026-07-17T06:56:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): workflow_lint''s SKILL_REF_RE
  (L654) treats any backticked absolute path with a single lowercase segment (e.g.
  `/tmp`) as a skill reference — a false-positive class that mis-fires the skill-reference
  check on ordinary filesystem paths'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1414 (implementer).

## Goal

Stop the workflow-lint skill-reference check from matching backticked filesystem paths.

## Workflow gap

- **Bug observed:** workflow_lint's SKILL_REF_RE (L654) treats any backticked absolute path with a single lowercase segment (e.g. `/tmp`) as a skill reference — a false-positive class that mis-fires the skill-reference check on ordinary filesystem paths
- **Why it is a workflow gap:** A lint false-positive class forces doc authors to contort phrasing or triggers spurious gate failures.
- **Confidence (emitter):** low (emitter) — concrete file + change, filed per the 2026-06-11 standing directive
- verified-at-filing: `grep -n 'SKILL_REF_RE =' scripts/workflow_lint.py` -> L654: r"(?<!\\w)`/([a-z][a-z0-9-]+(?::[a-z0-9-]+)?)(?=[`\\s)])" — pattern indeed matches `/tmp` (semantic probe: 'tmp' matches [a-z][a-z0-9-]+ with backtick-terminated lookahead)

## Proposed change (candidate diff sketch — refine in planning)

Carve out known filesystem roots (tmp, workspace, mnt, root, home, etc.) via SKILL_REF_ALLOWLIST or a negative lookahead for a following path separator.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: d8571a5036b6



