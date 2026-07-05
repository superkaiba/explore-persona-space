---
title: 'daily-fix: task_workflow git robustness (dup markers, MERGE_HEAD)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:524a1a544616
- daily-auto-filed
created_at: '2026-07-04T23:00:35Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — taskwf-git-robustness (fp 524a1a544616)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

(a) On commit-failure-after-successful-append, return success-with-warning (or have the retry path check the events tail for the just-appended marker before re-posting). (b) In the shared commit helper, detect MERGE_HEAD and bounded-wait / route through scripts/sync_repo_root.py instead of attempting a partial commit.

## Workflow gap

- **Bug observed:** (a) post_event appends to events.jsonl BEFORE the git commit; on an index.lock commit failure the standing retry recipe re-posts, landing the SAME marker 3x (#823 loop session, 07-03). (b) new_plan_version's commit crashed 'fatal: cannot do a partial commit during a merge' when a concurrent merge held MERGE_HEAD on the shared root (#911 session, 07-03).
- **Why it matters:** Both are mid-mutation crash/concurrency seams in the canonical state API; #898 covered set_status but not these two.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 524a1a544616
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
