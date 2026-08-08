---
title: 'daily-fix: root-commit block: warn never report as landed'
kind: infra
tags:
- wf-fix
- wf-fix-fp:345e455b6f4a
- daily-auto-filed
- trigger-dense
created_at: '2026-08-02T07:15:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Session reported a guard-BLOCKED
  repo-root commit as ''committed, pushed'' with a literal <sha> placeholder URL 16s
  after the block; Thomas hit the dead link. The hook''s block messages print remedies
  but never state the commit did not land.'
workflow: v1
---
# daily-fix: root-commit block: warn never report as landed

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C15 (miner-2 P1; session 0ac15c23, writeup session).

## Goal
Append a "DO NOT report this commit as landed — it was BLOCKED" line to `guard_root_code_commit.sh`'s block message(s), as a mechanical backstop against the session narrating a blocked commit as pushed.

## Workflow gap
- **Bug observed:** At 22:00:53Z a repo-root commit was BLOCKED by `guard_root_code_commit.sh`; 16 s later (22:01:09Z) the assistant told Thomas "Figure built, verified, committed, pushed" with a literal `<sha>` placeholder in the raw.githubusercontent URL. Thomas: "the link is not working" (22:02:11Z). Assistant's own concession: "my 'committed, pushed' was wrong — the commit was blocked ... and I didn't re-check." Recovered; both URLs verified 200 at 22:03:43Z. (miner-2 `probed: python json filter over transcript rows 22:00–22:04`.)
- **Why it is a workflow gap:** Two standing rules (verify-link-before-handing-over; never assert push success unchecked) already ban this and recurred anyway — the hook's block message prints remedies but never tells the session the commit did NOT land, so a mechanical one-line backstop at the exact failure point is the cheapest fix.
- **Confidence:** high
- verified-at-filing: `grep -in 'report' .claude/hooks/guard_root_code_commit.sh` → 0 hits (no landed-report warning anywhere); block-message sites read at lines 1217, 1246, 1254, 1292, 1318 and the main `BLOCK_MSG` heredoc at 1449–1462 — remedies + re-stage note present, no "not landed" warning; `git log --oneline --since='7 days ago' -- .claude/hooks/guard_root_code_commit.sh` → 3 commits (792920685d, 638093ec4f, c341f3bd59 — the last added the re-stage note visible in the heredoc), none adds a landed-report line (2026-08-02 UTC).

## Proposed change (refine in planning)
In the `BLOCK_MSG` heredoc (line ~1449) and, for consistency, the short single-line `echo ... >&2` block sites (1217/1246/1254/1292/1318), append/lead with one sentence:

```
+ The commit did NOT land. Do NOT report this commit as committed/pushed/landed
+ to the user or in any marker until a retry succeeds and is verified (rc=0 +
+ git log -1 shows it).
```

Planner decides whether one shared line (a `NOT_LANDED_LINE` variable interpolated into every block site) or heredoc-only suffices; keep `bash -n` clean and existing message text unchanged.

## Scope / surfaces
- Primary target: `.claude/hooks/guard_root_code_commit.sh`

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff/bash -n on touched files passes.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 345e455b6f4a
- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C15.
