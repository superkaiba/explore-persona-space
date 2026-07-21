---
title: 'daily-fix: edit-success gate before new-plan-version persist'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2b55f7d7436d
- daily-auto-filed
created_at: '2026-07-21T06:43:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): a scripted plan edit that
  died on AssertionError still had its compound command persist the plan via task.py
  new-plan-version, landing the revision as an unmodified copy of the prior version
  (task #1565 session, 2026-07-20 13:11Z; sibling shape in #1563 at 10:57Z); the skill
  prescribes the persist but no edit-success gating'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-20 from the transcript problem sweep (session for task #1565, 2026-07-20 ~13:11Z; a milder sibling in the #1563 session at 10:57Z).

## Goal

Add an edit-success gate to the plan-revision recipe in `.claude/skills/adversarial-planner/SKILL.md`: a scripted plan edit and its `task.py new-plan-version` persist must be sequenced so the persist NEVER runs when the edit script failed (persist only after verified edit success; never chained so an edit failure still persists).

## Workflow gap

- **Bug observed:** in the #1565 session (2026-07-20 13:11Z) a plan-edit script died on `AssertionError` (anchor text mismatch) yet the same compound command still ran the persist — the log shows the edit failure followed by "Plan v2 written … PASS", landing v2 as an UNMODIFIED copy of v1. The orchestrator noticed and re-applied as v3 ("The edit script asserted out before writing (old2 text mismatch), so v2 landed as an unmodified copy"). The #1563 session hit the same shape once (10:57Z, `AssertionError: expected 1 occurrence(s), got 0`). A later #1565 edit used the correct fail-loud form ("EDIT FAILED — not persisting") — improvised, not prescribed.
- **Why it is a workflow gap:** the adversarial-planner skill prescribes the `new-plan-version` persist (including a pre-persist Goal-currency gate) but nowhere requires the persist to be conditional on the edit script's exit status; an unnoticed unmodified-copy persist would ship a plan that silently lacks the critic-mandated revision.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'pre-persist\|new-plan-version' .claude/skills/adversarial-planner/SKILL.md` → the Goal-currency pre-persist gate exists (:81, :88, :113, :207, :320) but `grep -n 'edit script\|not persisting\|exit status' .claude/skills/adversarial-planner/SKILL.md` → 0 hits for any edit-success gating text — the discipline is absent from the recipe (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

Add to the plan-revision recipe (next to the Goal-currency gate): "Scripted plan edits MUST `&&`-chain edit → verify (grep the revised text is present) → `new-plan-version` persist; an edit-script failure aborts the persist. Never `;`-chain or persist inside the same script before its asserts."

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md`

## Constraints / invariants

- Prose duty only; no marker/schema change. `workflow_lint.py --check-asks` passes.

## Provenance

- fingerprint: 2b55f7d7436d

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md

Origin evidence (transcript-mined, sessions 52bb137c / c84525cd, 2026-07-20): "edit script `AssertionError` yet 'Plan v2 written … PASS'"; orchestrator self-report "v2 landed as an unmodified copy. Getting the verbatim text and re-applying properly as v3."
