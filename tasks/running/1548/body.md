---
title: 'workflow-fix: surface cap-parked free-analysis follow-ups'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bad928bce67c
- daily-auto-filed
created_at: '2026-07-19T07:08:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): #958''s top-ranked screened
  not-redundant 0-GPU follow-up sat unrun because the Step 9a-ter one-round cap parked
  it only as a body bullet with no visible surfacing; the user found and kicked it
  himself (c1-P5).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c1-P5). Route-2 filing.

## Goal

When the zero-GPU free-analysis cap (Step 9a-ter, AT MOST ONE round/task)
parks a SCREENED not-redundant free-analysis follow-up, post a visible
"parked by cap" `epm:progress` note (plus a PM-surfaceable signal), and note
the cap-raise alternative for the planner to weigh — instead of leaving it
only as a body bullet.

## Workflow gap

- **Bug observed:** #958's TOP-RANKED 0-GPU follow-up sat unrun (the zero-GPU
  floor caps at 1 round/task) with no surfacing; Thomas discovered it and
  kicked it himself.
- **Why it is a workflow gap:** the Step 9a-ter loop guard caps at one
  free-analysis round and, per its own text, "the second free-analysis
  follow-up surfaces in the body as a regular bullet for a future human
  pass" — a body bullet is not visible surfacing, so a screened
  not-redundant top-ranked follow-up silently waits until a human happens to
  read the body.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'AT MOST ONE free-analysis' .claude/skills/issue/SKILL.md` → the Step 9a-ter Loop guard (line ~6385) caps at one round via `epm:free-analysis-followup-run v1` and states the capped follow-up "surfaces in the body as a regular bullet"; no `epm:progress` parked-by-cap note or PM-surfaceable signal is posted for a cap-parked screened follow-up (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# SKILL.md Step 9a-ter Loop guard: on a cap hit that parks a SCREENED
# not-redundant free-analysis follow-up —
+ post an epm:progress "parked by cap" note naming the follow-up + its rank
+ (so the dashboard / PM surfaces it), and state the cap-raise alternative
+ for the planner to weigh. The body bullet stays, but no longer the ONLY
+ surface. (#958: a top-ranked 0-GPU follow-up sat unrun until the user
+ found and kicked it.)
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Amend the Step 9a-ter Loop-guard span; keep the one-round cap intact — this
  adds SURFACING of the parked follow-up, not another auto-run.

## Constraints / invariants

- Workflow-surface only. Auto-continue (NOT a new AskUserQuestion gate); the
  note is a non-blocking side channel.
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: a898a6b66895

Surfaced problem (c1-P5): #958's top-ranked screened 0-GPU follow-up parked by
the one-round cap with no visible surfacing; the user found and kicked it.
