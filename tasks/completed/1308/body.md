---
title: 'daily-fix: planner traces trigger predicate vs incident'
kind: infra
tags:
- wf-fix
- wf-fix-fp:db379306ca23
- daily-auto-filed
created_at: '2026-07-14T06:44:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): #1287 plan v1''s boot-refusal
  detection predicate would not have recovered its own motivating incident (#1277
  transcript: 14 successful assistant rows and an 825KB transcript exceeding the 256KB
  tail read both flip the predicate to keep) - caught by the fact-checker at cost
  of a ~26 min replan'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-13 from the transcript problem sweep (session c338f95f, task #1287).

## Goal

Require a detection/trigger-lane plan to trace its predicate against the motivating incident's REAL artifact in the plan's assumptions section, so a predicate that would not recover its own motivating incident is caught at plan-write time rather than by the fact-checker.

## Workflow gap

- **Bug observed:** #1287's plan v1 designed the watcher boot-refusal detection predicate without validating it against the motivating #1277 transcript; the Phase-1.5 fact-checker refuted assumption A5 "twice over" — the transcript has 14 successful assistant rows before the refusal row (v1's zero-successful-response predicate reads `keep`) and is 825,591 B > the 262,144 B tail read (rows=None ⇒ keep) — i.e. "the v1 design would NOT recover a replay of its own motivating incident". Cost: a ~26 min planner resume + replan to v2.
- **Why it is a workflow gap:** the fact-checker caught it (the pipeline worked), but the planner spec has no rule requiring an incident-artifact trace for detection/trigger predicates, so the class recurs and burns a fact-check round each time.
- **Confidence (emitter):** low-medium (single incident; the downstream catch exists — this is belt-and-suspenders per the standing low-confidence-still-files directive)
- verified-at-filing: `grep -n "motivating incident\|real artifact" .claude/agents/planner.md` → 0 hits (2026-07-14 UTC) — no such rule exists in the planner spec.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/agents/planner.md` (the §-assumptions guidance):

```
+ **Detection / trigger-lane predicate plans:** when the plan designs a
+ predicate that detects or triggers on an incident shape (a watcher lane,
+ a guard, a classifier), trace the predicate step-by-step against the
+ motivating incident's REAL artifact (the actual transcript / log / event
+ row, by path) in §-assumptions and state the traced outcome. A predicate
+ that would not fire on its own motivating incident is a design defect
+ (#1287 plan v1: both the row-count and the tail-size arm read `keep` on
+ the #1277 transcript it was built to catch).
```

## Scope / surfaces

- Primary target: `.claude/agents/planner.md`

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` default run passes.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/agents/planner.md
- fingerprint: db379306ca23

Origin: transcript-mined (session c338f95f, fact-checker refutation ~07:20Z, planner resume ~07:26Z). Not a parked candidate — surfaced by the /daily problem sweep; the mining agent's suggestion is carried forward under the standing directive that concrete low-confidence follow-ups are filed, with the spawned planner free to deflect with a reasoned no-change report.
