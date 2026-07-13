---
title: 'workflow-fix: postmerge guard must not delete origin''s only task folder (unpushed
  mv)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5cba114e8ce4
created_at: '2026-07-13T10:34:34Z'
has_clean_result: false
origin_prompt: 'Step 10d post-merge stale-task-folder guard deletes origin/main''s
  only folder for the task when the canonical status-mv is unpushed on local main
  (incident: origin commit 2a1a9cbc0b left zero 1291 folders; #1297 re-detected the
  same shape). Fix: canonical-absent-on-origin => sync repo root + re-detect before
  any delete.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1297 (emitting agent: /issue orchestrator, own observation during Step 10d).

## Goal

Make the Step 10d post-merge stale-task-folder guard handle the unpushed-status-mv case: when the CANONICAL folder is absent from origin/main, run sync_repo_root FIRST, re-fetch + re-detect, and only classify/delete duplicates when the canonical folder is present on origin.

## Workflow gap

- **Bug observed:** the guard deleted origin/main's ONLY folder for a task (the old-status copy) while the canonical status-mv sat unpushed on local main, leaving origin with zero folders + a dangling REGISTRY pointer and seeding rename/delete conflicts in every other session's sync_repo_root.
- **Why it is a workflow gap:** the guard's detection compares origin/main's folders against the LOCAL canonical path; it never checks that the canonical folder EXISTS on origin before classifying the origin copy as a stale duplicate. Under the fleet's routine local-main push lag, the "duplicate" is usually just the not-yet-pushed mv.
- **Confidence (emitter):** high (two occurrences in one day, one destructive).
- verified-at-filing: `git ls-tree -d -r --name-only 2a1a9cbc0b | grep -cE '^tasks/[^/]+/1291$'` → 0 hits (zero 1291 folders at origin commit 2a1a9cbc0b, "post-merge: remove stale task #1291 folder(s)…", 2026-07-13); #1297's own guard run detected tasks/running/1297 as a "duplicate" in exactly this state (canonical completed/1297 unpushed; caught by inspection, resolved via scratch-worktree merge f26462fc1b + sync).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` § Post-merge stale-task-folder guard, detection step:

```
+ # Unpushed-mv pre-check: if CANON itself is absent from the origin/main
+ # ls-tree, the "duplicate" is (almost always) the not-yet-pushed status mv —
+ # deleting the origin copy would leave origin with ZERO folders for the task.
+ if ! grep -qxF "$CANON" /tmp/issue-<N>-postmerge-lstree.txt; then
+   uv run python "$REPO_ROOT/scripts/sync_repo_root.py"   # land the local mv
+   re-fetch + regenerate the ls-tree; re-detect. Only proceed to the
+   scratch-worktree removal when CANON is present on origin/main.
+ fi
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'postmerge-lstree' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 5cba114e8ce4

Origin: /issue #1297 Step 10d, 2026-07-13 — orchestrator's own observation (no formal candidate block; synthesized per the prose-followup rule). Incident evidence: origin commit 2a1a9cbc0b (guard deleted the only 1291 folder on origin); #1297's guard re-detected the same shape on tasks/running/1297; recovery merge f26462fc1b.
