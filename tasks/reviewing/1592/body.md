---
title: 'daily-fix: planner self-counts grep-c acceptance criteria'
kind: infra
tags:
- wf-fix
- wf-fix-fp:549eb7870ace
- daily-auto-filed
created_at: '2026-07-22T06:45:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): planner ships grep-count
  acceptance criteria contradicting the plan''s own verbatim insert text; two same-morning
  revise rounds'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from a recurring planner defect observed in TWO sessions the same morning (#1581 and #1583 transcripts).

## Goal

Stop the planner from shipping `grep -c`-style mechanical acceptance criteria that contradict the plan's own verbatim insert text.

## Workflow gap

- **Bug observed:** plan v1 for #1581 carried criterion 3 (`grep -c ... == 1`) contradicting Edit 3's own text, which carries the token on two lines (+ criterion 7 "pure insertions" contradicting Edit 2's append); plan v1 for #1583 carried a §4 "exactly once" criterion unsatisfiable for `teammate channel`, which appears twice in the verbatim insert. Both cost a union-revise round (~5-10 min each), caught by the Statistics/Methodology critics.
- **Why it is a workflow gap:** `.claude/agents/planner.md` has no instruction to count the pattern in the plan's OWN §3 verbatim insert text (and any existing file text touched) before finalizing a count-style acceptance criterion — the defect recurs across independent sessions.
- **Confidence:** high (two same-morning firings; critic verdicts quoted in transcripts aa8882d9 09:58:02Z / 3239cc4d 11:40:52Z).
- verified-at-filing: `grep -n 'grep -c\|acceptance criteri\|count the pattern' .claude/agents/planner.md` → 1 hit (line 295, a pattern-list token; no self-count instruction exists — absence claim, in-target 0-hit for the instruction is the evidence; 2026-07-22). `git log --oneline --since='7 days ago' -- .claude/agents/planner.md` → no such guidance landed.

## Proposed change (candidate diff sketch — refine in planning)

Add one line to planner.md's acceptance-criteria guidance: before finalizing any `grep -c` / "exactly once" / "pure insertion" mechanical criterion, count the pattern in the plan's own verbatim insert text AND the existing file text it touches, and set the expected count to the actual total.

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 549eb7870ace

- workflow_fix_target: .claude/agents/planner.md

Origin evidence: transcript aa8882d9 (2026-07-21T09:58:02Z, "Methodology: REVISE — criterion 3 (`grep -c ... == 1`) contradicts Edit 3's own text"); transcript 3239cc4d (11:40:52Z, "Statistics critic returned REVISE — §4's 'exactly once' is unsatisfiable for `teammate channel`, which appears twice in the verbatim insert").
