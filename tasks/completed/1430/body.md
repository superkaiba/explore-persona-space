---
title: 'workflow-fix: audit duplicate-id dirs + reap terminal-task status husks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bb84ac5c4962
created_at: '2026-07-16T17:39:56Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during 2026-07-16 chat fleet check: tasks #721/#1107/#1227
  each hold a git-tracked stale-status husk dir beside their completed/ dir (merge-reintroduced
  post-move); task.py audit PASSes despite duplicates; no sweep covers on-disk terminal
  husks.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed bug during a fleet progress check (2026-07-16, chat session; emitting agent: orchestrator).

## Goal

`task.py audit` detects duplicate `tasks/<status>/<id>` dirs for a single id, and a sweep reaps merge-reintroduced stale-status husk dirs of TERMINAL tasks (husk contents verified as a subset/prefix of the live folder before removal; any husk with unique content is escalated, never deleted).

## Workflow gap

- **Bug observed:** `ls tasks/running` / `ls tasks/reviewing` list tasks #721, #1107, #1227 although all three are at `completed` per REGISTRY — each id has TWO on-disk dirs (e.g. `tasks/reviewing/1107/` AND `tasks/completed/1107/`), the husk holding stale-prefix copies of `events.jsonl` (31 vs 36 lines) / `concerns.jsonl` / `plans/` / `artifacts/`. `task.py audit` reports `AUDIT PASS — registry and filesystem agree` despite the duplicates, and the husks are git-TRACKED (clean `git status`), so they persist indefinitely.
- **Why it is a workflow gap:** the #644 ghost-aware staging in `task_workflow.py` `set_status` (the `_task_status_dir_pathspecs` block, ~L4175-4195) sweeps only dirs "tracked in HEAD but ABSENT on disk" at the NEXT transition of the same task. This husk class is the inverse and unreachable by that sweep: `git log --oneline -2 -- tasks/reviewing/1107/` shows the husk was RE-INTRODUCED by a concurrent branch's later merge (`117e4e3fa5`, issue-1108 PR #872) AFTER the move commit (`f73e033488`, `task #1107: reviewing → completed`) — the branch was cut before the move and its rebase-merge re-added the old-path files. Terminal tasks have no further transitions, so nothing ever sweeps the husk; `audit` only checks registry→path agreement, not path uniqueness per id. Consequence: any `ls tasks/<status>`-based fleet read (progress checks, dashboards, ad-hoc triage) misreports terminal tasks as active.
- **Confidence (emitter):** high
- verified-at-filing: `ls -d tasks/*/721 tasks/*/1107 tasks/*/1227` → 2 dirs per id (6 total: `running/721`+`completed/721`, `reviewing/1107`+`completed/1107`, `reviewing/1227`+`completed/1227`); `uv run python scripts/task.py audit` → PASS despite them; `wc -l` husk-vs-live: 1107 events 31/36 (stale prefix, last husk marker 2026-07-07T12:12 < live 12:35), 1227 events 17/23, concerns 3/3 identical; `git log --oneline -2 -- tasks/reviewing/1107/` → husk re-created by `117e4e3fa5` post-move (2026-07-16). Absence-of-guard claim for the audit check + terminal-husk sweep: context of `src/explore_persona_space/task_workflow.py` ~L4175-4195 READ — the ghost pathspec helper covers tracked-in-HEAD-absent-on-disk only (the comment says so explicitly); no duplicate-id-dir check exists in the `audit` path of `scripts/task.py` (grep `duplicate` in `scripts/task.py` → 2 hits, both unrelated: re-post marker warning L595, duplicate-frontmatter trap L1387).

## Proposed change (candidate diff sketch — refine in planning)

```
+ task.py audit: after the registry→path pass, glob tasks/*/<id> per known id;
+   >1 hit => report DUPLICATE-DIR (id, live path per REGISTRY, husk paths) => audit FAIL (or WARN tier).
+ task_workflow.py: reap_stale_status_husks(id|--all): for TERMINAL (completed/archived) ids with
+   duplicate dirs, verify husk events.jsonl is a line-prefix/subset of the live one (+ no file present
+   only in the husk with unique content); if verified => git rm -r the husk + commit by explicit path;
+   else => escalate (sidecar row + loud report), never delete.
+ Wire the reap into an existing janitor pass (worktree_audit or the watcher's orphan sweep) or expose
+   via task.py; also consider a set_status-time sweep of OTHER-id husks matching the moved id's pattern.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`, `scripts/task.py`
- Grep the workflow surface for the pattern before editing (`grep -rn 'ghost\|_task_status_dir_pathspecs' src/explore_persona_space/task_workflow.py scripts/ .claude/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/` state edits outside the husk reap itself.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; tests pinning task-workflow invariants updated/added (`tests/test_task_workflow*.py`).
- Never delete a husk with unique content (a marker present only in the husk is evidence of a lost concurrent write — escalate).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py, scripts/task.py
- fingerprint: bb84ac5c4962

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py, scripts/task.py
bug_observed: Terminal tasks #721/#1107/#1227 each have a second, git-tracked stale-status dir (e.g. tasks/reviewing/1107/ beside tasks/completed/1107/) re-introduced by a concurrent branch merge after the status move; task.py audit PASSes despite the duplicates and nothing ever sweeps them, so ls-based fleet reads misreport terminal tasks as active.
why_workflow_gap: The #644 ghost-aware staging sweeps only tracked-in-HEAD-absent-on-disk dirs at the task's NEXT transition — merge-reintroduced on-disk husks of TERMINAL tasks are the inverse case and terminal tasks have no next transition; audit checks registry→path agreement but not path uniqueness per id.
proposed_change: audit flags duplicate tasks/<status>/<id> dirs per id; a subset-verified reap removes terminal-task husks (escalating, never deleting, on unique husk content).
diff_sketch: |
  + audit: glob tasks/*/<id> per id; >1 => DUPLICATE-DIR finding => FAIL/WARN
  + reap_stale_status_husks: terminal ids only; husk events.jsonl must be prefix/subset of live; git rm + explicit-path commit; else escalate
confidence: high
related_task: n/a (surfaced during chat-session fleet check)
<!-- /workflow-fix-candidate -->
