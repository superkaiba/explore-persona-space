---
title: 'workflow-fix: verified-at-filing context binding'
kind: infra
tags:
- wf-fix
- wf-fix-fp:963b7c7d1418
- daily-auto-filed
created_at: '2026-07-16T07:19:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): grep bind checks count
  not context; duplicate filed over landed fix'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked FORMAL candidate block on task #1330 (archived as duplicate of the already-landed #1309 fix — this candidate is about the mis-filing mechanism itself).

## Goal

Extend the verified-at-filing binding rule in `.claude/rules/workflow-fix-on-bug.md` so a presence-grep must bind on CONTEXT as well as count: a hit whose surrounding context already implements the proposed change means the fix is already landed — re-read the hit's context before filing.

## Workflow gap

- **Bug observed:** #1330 was filed as a duplicate of the already-landed #1309 fix because the filer's verified-at-filing grep (case-sensitive) hit the existing recipe paragraph itself, yet the hit's CONTEXT was misjudged as unrelated — the binding rule caught count consistency but not context consistency.
- **Why it is a workflow gap:** the § Body-file template's verified-at-filing mandate (#1272/#1307 lineage) binds hit COUNTS to the claim but has no clause requiring the filer to read the hit's context to rule out "the hit IS the fix" — so duplicate filings pass the grep gate.
- **Confidence (emitter):** medium (formal block)
- verified-at-filing: `grep -c 'verified-at-filing' .claude/rules/workflow-fix-on-bug.md` → 5 hits (the mandate exists); `grep -n 'context consist\|context-consist' .claude/rules/workflow-fix-on-bug.md` → 0 hits (absence-of-guard claim — the 0-hit in-target result IS the evidence) (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Add a context-consistency clause to the binding test in § Body-file template (and the § Anti-patterns table): a presence grep whose hits sit inside text that already implements the proposed change is a landed-fix signal — read each hit's surrounding lines before filing; a count-only bind does not satisfy the mandate.

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md`
- The sibling summary in `.claude/skills/daily/SKILL.md` (route-2 verified-at-filing mandate paragraph) may need the same one-line extension — grep both.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
- fingerprint: 1e9027c2d4ba

parked formal candidate (from #1330 events.jsonl 2026-07-15T06:57:11Z, abridged): "<!-- workflow-fix-candidate v1 --> target_file: .claude/rules/workflow-fix-on-bug.md bug_observed: #1330 was filed as a duplicate of the already-landed #1309 fix because the filer's verified-at-filing grep (case-sensitive) hit the existing recipe paragraph itself yet the hit's CONTEXT was misjudged as unrelated — the binding rule caught count consistency but not context consistency. why_workflow_gap: the verified-at-fili[ng mandate has no context-consistency clause] ..."
