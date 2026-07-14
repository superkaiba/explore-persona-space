---
title: 'workflow-fix: trim LESSONS.md index below 8000-byte lint cap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0d729565e993
created_at: '2026-07-10T02:26:13Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #1219 implementer: workflow_lint no-flags exits
  1 on pre-existing .claude/rules/LESSONS.md 8028>8000-byte cap, byte-identical to
  main; trim the always-on index (the lint''s own error message names the fix).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1219 (emitting agent: implementer).

## Goal

Trim/condense the `.claude/rules/LESSONS.md` always-on index back below the 8000-byte lint cap so the no-flags `workflow_lint.py` run passes on pristine main.

## Workflow gap

- **Bug observed:** `.claude/rules/LESSONS.md` is 8028 bytes, over the 8000-byte cap `workflow_lint.py` enforces; the no-flags lint run FAILs fleet-wide on pristine main (verified by #1219's implementer 2026-07-10: byte-identical to main, zero branch commits touch it).
- **Why it is a workflow gap:** the always-on lessons index is workflow surface and its own lint gate is red on main — every session's no-flags lint run (incl. Step 10d pre-push gate baselines) carries a standing failure that baseline-subtraction must strip, and any NEW index line worsens it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; the lint's own error message names the fix: condense/shorten index entries in .claude/rules/LESSONS.md below 8000 bytes without dropping any rule pointer. Check the cap's headroom guidance in workflow_lint.py's check before editing.)

## Scope / surfaces

- Primary target: `.claude/rules/LESSONS.md`
- Grep the workflow surface for the byte-cap check (`grep -n "8000" scripts/workflow_lint.py`) and keep the index consistent with `--check-lessons-index` (every .claude/rules/*.md still indexed).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes after the trim; `--check-lessons-index` still passes (no rule pointer dropped).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/LESSONS.md
- fingerprint: 0d729565e993

Verbatim surfaced prose (implementer, task #1219): "the LESSONS.md byte-cap overflow is live on main and will fail every session's no-flags workflow_lint run until someone trims the always-on index; the lint's own error message names the fix" — workflow_lint no-flags exits 1 on a pre-existing .claude/rules/LESSONS.md 8028 > 8000-byte cap, cmp-proven byte-identical to the main checkout.
