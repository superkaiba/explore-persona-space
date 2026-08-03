---
title: 'daily-fix: daily brief commits stage only its own paths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d86e2db12a74
- daily-auto-filed
created_at: '2026-07-23T07:02:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): the 07-21 nightly daily
  commit 7dbde267f1 swept another session''s 4 uncommitted working files onto main
  (verified via git show --stat); the skill''s commits lack a staged-set verification'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Last night's daily-brief commit `7dbde267f1` ("logs: daily brief for 2026-07-21") swept ANOTHER session's then-uncommitted working files onto main — `scripts/issue1092_fair_deepdive_figs.py` + `figures/summaries/prefix_vs_context_map/perprefix_error_vs_spread.{png,pdf,meta.json}` (verified via `git show --stat 7dbde267f1` in the mining transcript). The next morning the #1092 session, unaware, burned ~5 min of guard blocks + forensic digging before discovering its files were already on main. This is the exact "concurrent `git add` sweeps" hazard the shared-root rules ban.

## Goal

The /daily skill's commit steps are hardened so a nightly run can only ever commit its OWN files: every `git add` names explicit paths (`logs/daily/<date>.md`, its own cache jsonl), and before each commit the run verifies `git diff --cached --name-only` contains ONLY those intended paths (abort + unstage otherwise).

## Workflow gap

- **Bug observed:** commit `7dbde267f1` (2026-07-21 23:53 PT nightly) carried 4 foreign files beyond the daily brief + cache row; discovered 06:52–06:57Z by the file-owning session (12462773).
- **Why it is a workflow gap:** the skill's § Commit recipe stages the daily file by explicit path, but the run's OTHER commits (enrichment, event-stream rows, route-1 fixes near session end) have no staged-set verification step, and a pre-commit stash/restore cycle plus concurrent staging makes the index state unpredictable — one default-form commit swept the sibling's staged files.
- **Confidence:** high on the incident; medium on which exact commit form swept (the spawned session should read `7dbde267f1`'s reflog context).
- verified-at-filing: `git show --stat 7dbde267f1 | head -12` re-run at filing confirms the commit carries `logs/daily/2026-07-21.md` + `.claude/cache/nightly-consolidation-events.jsonl` PLUS `scripts/issue1092_fair_deepdive_figs.py` + 3 `figures/summaries/prefix_vs_context_map/perprefix_error_vs_spread.*` files (presence claim, binds), 2026-07-23 UTC.

## Proposed change (refine in planning)

In `.claude/skills/daily/SKILL.md`: (1) a one-line staged-set verification (`git diff --cached --name-only` == intended set, else unstage + retry pathspec-only) attached to EVERY commit the skill prescribes (stub, enrichment, route-1 fixes, event-stream rows); (2) an explicit warning that a concurrent session's staged files must never be committed — commit with trailing pathspecs (`git commit -m ... -- <paths>`), never a bare `git commit` after `git add`.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md` (§ Commit + the applied-fix commit instructions).

## Constraints / invariants

- The stub-first + immediate-push contract unchanged. Recursion guard applies.

## Provenance

- fingerprint: d86e2db12a74

- workflow_fix_target: .claude/skills/daily/SKILL.md
