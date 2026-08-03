---
title: 'daily-fix: committed plan-patch helper for anchor edits'
kind: infra
tags:
- wf-fix
- wf-fix-fp:da302cc55d63
- daily-auto-filed
created_at: '2026-07-23T07:03:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): improvised per-turn anchor-edit
  scripts for plan revisions failed first-try in >=4 events across 3 sessions on 2026-07-22
  (anchor byte-identity drift), each costing a re-derive round'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Orchestrators keep hand-improvising per-turn python anchor-edit scripts for plan.md revisions, and the anchors keep missing on the first attempt: #1604 (08e5ba95, 20:36:24Z "EDIT FAILED — not persisting"), #1609 (dda270c4, 2026-07-23T03:03:00Z "AssertionError: anchor snapshot missing"), plus two case-mismatched verify-grep aborts in #1415 (53e14ca2, 20:37/20:48Z). Each recovers by re-deriving the anchor from the live file — a wasted round every time, ≥4 events across 3 sessions in one day.

## Goal

A small committed plan-patch helper (e.g. `scripts/plan_patch.py`: anchor-normalized matching — whitespace-collapsed / case-tolerant anchor resolution, assert-and-apply, fail-loud with the nearest-match diff) replaces per-turn improvised anchor-assert scripts in the plan-revision recipes, so plan revisions stop needing a re-derive round.

## Workflow gap

- **Bug observed:** the 4 events above; root cause each time: anchor byte-identity drafted from intended wording rather than the persisted file (whitespace/line-wrap/case drift).
- **Why it is a workflow gap:** the plan-revision recipes (issue SKILL.md / adversarial-planner) prescribe anchor-edit discipline but provide no committed helper, so every session re-improvises the same fragile script shape.
- **Confidence:** medium-high.
- verified-at-filing: `ls scripts/ | grep -i 'plan.*patch\|patch.*plan'` → no existing helper (absence claim); the recurrence evidence is the 4 tool_result failure events enumerated above (firing events per the #1484 discipline), 2026-07-23 UTC.

## Proposed change (refine in planning)

New `scripts/plan_patch.py` (normalized-anchor find + replace + fail-loud nearest-match report) + one-line pointers in the issue/adversarial-planner plan-revision recipe text prescribing it over improvised scripts.

## Scope / surfaces

- Primary targets: new `scripts/plan_patch.py` (workflow-helper script), `.claude/skills/issue/SKILL.md` + `.claude/skills/adversarial-planner/SKILL.md` (one-line pointer each).

## Constraints / invariants

- Fail-loud on ambiguous/missing anchors (never fuzzy-apply silently); plans remain plain markdown. Recursion guard applies.

## Provenance

- fingerprint: da302cc55d63

- workflow_fix_target: .claude/skills/issue/SKILL.md
