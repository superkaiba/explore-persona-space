---
title: 'daily-fix: closed-sibling probe in /daily driver'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3ce086743717
- daily-auto-filed
created_at: '2026-07-26T07:08:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The #1674 landed-fix probe
  matches commit SUBJECTS only, so a vocabulary-divergent landed fix is invisible;
  the Alternatives critic measured the closed-sibling arm would plausibly have caught
  3/3 measured incidents vs the commit probe 1/3.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 Step C parked-workflow-fix-candidate routing pass
(`.claude/rules/workflow-fix-on-bug.md` § Recursion guard escape valve). The candidate was
parked on task #1674 at 2026-07-25T11:06:34Z as a **prose follow-up** (Alternatives critic,
Phase 2 round 1) because that session ran under the `workflow_fix_target` recursion guard.
The #1674 plan v2 records an explicit §11 rejection/deferral line for it — under the
standing 2026-06-11 directive a deferred follow-up is RUN, not parked, so it routes here.

## Goal

Add a preventive closed-sibling check to `scripts/daily_drive_filings.py` alongside the
#1674 commit-subject probe, reusing `task_workflow.recent_closed_workflow_fix_tasks`
(#1446).

## Workflow gap

- **Bug observed:** the #1674 mechanical landed-fix probe
  (`find_landed_fix_suspects`, `scripts/daily_drive_filings.py:923`) matches commit
  SUBJECTS only, so a landed fix whose commit subject uses different vocabulary than the
  candidate is invisible to it — the measured #1386/#1360 case. The Alternatives critic
  verified a closed-sibling arm would plausibly have caught 3/3 measured incidents (each
  landed fix has a closed wf-fix task whose target path matches the candidate's target)
  vs the commit probe's 1/3.
- **Why it is a workflow gap:** the exact-`(target_file, fingerprint)` dedup predicate
  scans OPEN tasks only by design, and the #1399/#1446 filing-time advisory is
  filer-stderr-only and window-bounded. A driver-side closed-sibling arm is the
  mechanical backstop for the class the compose-time clause-(a') duty currently carries
  alone.
- **Open question the plan must resolve:** the unmeasured false-positive surface on
  closed-task titles. `recent_closed_workflow_fix_tasks` already powers an ADVISORY; a
  driver arm that BLOCKS a filing on a title-token match is a stronger contract than the
  advisory it reuses. The planner decides the outcome grain — recommend mirroring the
  #1674 probe's terminal ledger outcome (`landed-fix-suspect`, no task filed, suspects
  recorded for the eyeball, `--retry-suspects` override) rather than inventing a new one.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'recent_closed_workflow_fix_tasks' scripts/daily_drive_filings.py`
  → **0 hits** (absence confirmed in the named target).
  `grep -n 'landed-fix-suspect\|landed_fix' scripts/daily_drive_filings.py` → the #1674
  commit-subject probe IS present (`_landed_fix_item_tokens:906`,
  `find_landed_fix_suspects:923`, docstring lines 32/79), confirming the candidate targets
  the complementary arm and not an already-landed fix. Landed-fix history check
  `git log --oneline --since='7 days ago' -- scripts/daily_drive_filings.py` → 6 commits
  (#1680, #1674, #1687, #1678, #1580, #1529); #1674 (`76f8b4f479`) landed the commit probe,
  none landed a closed-sibling arm. (2026-07-25)

## Proposed change (candidate diff sketch — refine in planning)

```
  in the driver's pre-filing probe sequence, alongside find_landed_fix_suspects:
+ closed = task_workflow.recent_closed_workflow_fix_tasks(...)   # the #1446 helper
+ match on target-path overlap (primary) and informative-title tokens (secondary)
+ on a hit: record a terminal ledger outcome mirroring `landed-fix-suspect`
+   (no task filed, suspects recorded, --retry-suspects override)
+ fail OPEN on any scan error — a held/real bug is never silently dropped
```

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`
- Read `task_workflow.recent_closed_workflow_fix_tasks` (#1446) for its existing matching
  arms (target-line token, informative-title token, plain-infra ≥2-token, body path
  substring) and its 7-day window before choosing the driver's match grain.
- Keep parity with the #1674 probe's fail-open semantics and its `filed.jsonl` outcome
  shape; do not introduce a second, differently-shaped suspect record.

## Constraints / invariants

- Workflow-surface only.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  the `/daily` driver's existing tests stay green.
- Fail OPEN on scan errors — under-filing a real bug is worse than a duplicate filing.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/daily_drive_filings.py
- fingerprint: 3ce086743717

Parked prose follow-up (verbatim), from task #1674 `events.jsonl` @ 2026-07-25T11:06:34Z:

> source: prose-followup (Alternatives critic, Phase 2 round 1). Suggestion: add a
> preventive closed-sibling check to scripts/daily_drive_filings.py alongside the #1674
> commit probe, reusing task_workflow.recent_closed_workflow_fix_tasks (#1446) — critic
> verified it would plausibly have caught 3/3 measured incidents (each landed fix has a
> closed wf-fix task whose target path matches the candidate's target) vs the commit
> probe's 1/3; unmeasured FP surface on closed-task titles is the open question.
> target_file: scripts/daily_drive_filings.py. routed: parked - running under
> workflow_fix_target recursion guard (this IS a workflow-fix session; see
> workflow-fix-on-bug.md § Recursion guard) — picked up by the nightly /daily
> parked-candidate sweep. The #1674 plan v2 records the explicit §11
> rejection/deferral line.
