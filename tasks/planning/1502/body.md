---
title: 'daily-fix: open wf-fix sibling advisory at filing time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a9280f83da12
- daily-auto-filed
created_at: '2026-07-18T06:47:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): two concurrent wf-fix sessions
  (#1476, #1479) fixed the same LESSONS-ratchet red; #1479''s ~2.3h pipeline was discarded
  — the open-task dedup is exact-fingerprint only and the #1446 advisory lists only
  recently-CLOSED siblings, so a live same-bug collision has no advisory channel.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined problem (chunk-0 miner): the day's clearest duplicated-work incident — two concurrent workflow-fix sessions (#1476 and #1479) both fixed the same LESSONS-ratchet trunk-red; #1476 merged first and #1479's entire ~2.3h pipeline (plan → 4 reviewers → implementer → code-review → PR #1238) was discarded and the task archived.

## Goal

Extend the file-time sibling advisory in `scripts/file_infra_task.py` to also list OPEN workflow-fix siblings by target-file / informative-title-token overlap (today it lists only recently-CLOSED siblings; the open-task check is exact-`(target_file, fingerprint)` only), so a filer sees "an open sibling is already fixing this file/bug" before spawning a duplicate pipeline.

## Workflow gap

- **Bug observed:** #1479 was filed and spawned while #1476 (same bug, different wording → different fingerprint) was already open and mid-pipeline; the exact-fp open-task dedup was blind by design, and the #1399/#1446 advisory covers only closed tasks — so nothing surfaced the live collision, and a full session's work was discarded on merge.
- **Why it is a workflow gap:** the dedup grain deliberately allows distinct bugs on one file, but a SAME-bug-different-wording collision between two OPEN tasks has no advisory channel at all; the closed-sibling advisory (#1446) proves the shape works.
- **Confidence:** medium
- verified-at-filing: `grep -n "recent_closed" scripts/file_infra_task.py` → `_advise_recent_closed_wf_fix_siblings` present (~L281-347, closed-only, 7-day window); the only open-task check is the exact-fp predicate (`is_open_workflow_fix_task` title pre-filter, ~L252-269) — no open-sibling OVERLAP advisory exists (absence claim; grep run 2026-07-18 UTC). Incident record: #1479 archived (title "workflow-fix: LESSONS.md over ratchet — trim/bump _LESSONS_RATCHET_BYTES"), #1476 landed the fix (commit `fd9b20f1ea` lineage on scripts/workflow_lint.py; ratchet reads 6800 on main).

## Proposed change (candidate diff sketch — refine in planning)

```
# file_infra_task.py — alongside _advise_recent_closed_wf_fix_siblings:
+ def _advise_open_wf_fix_siblings(args):  # OPEN statuses, overlap-matched
+     # ≥2 shared informative title tokens OR same workflow_fix_target path,
+     # statuses proposed/planning/plan_pending/approved/running/reviewing;
+     # stderr advisory only — never blocks, fail-soft (mirror #1446 shape).
```

## Scope / surfaces

- Primary target: `scripts/file_infra_task.py` (+ `src/explore_persona_space/task_workflow.py` if the enumerator lands there, mirroring `recent_closed_workflow_fix_tasks`)
- Update `.claude/rules/workflow-fix-on-bug.md` § "Recently-closed-sibling ADVISORY" to document the open-sibling arm.

## Constraints / invariants

- Advisory only — never blocks a filing, never changes exit codes, fails soft (the #1446 contract).
- Workflow-surface only. Lint + ruff + `tests/test_workflow_fix_dedup.py` green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: a9280f83da12

- workflow_fix_target: scripts/file_infra_task.py

source: /daily 2026-07-17 transcript sweep (chunk-0 miner) — the #1479-vs-#1476 duplicate-pipeline race.
