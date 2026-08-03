---
title: 'daily-held: 23 stashes accumulated on the shared repo root'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-27T07:22:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 3): sync_repo_root keeps an
  autostash for manual triage whenever it cannot cleanly reapply one and nobody owns
  the follow-up, so 23 entries dating back to 2026-05-27 have accumulated on the shared
  repo root'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 3 — judgment call).
Held because triaging or dropping a stash is DESTRUCTIVE and not undoable by a single
`git revert`: each entry may hold the only copy of someone's uncommitted work.

`sync_repo_root.py` KEEPS an autostash "for manual triage" whenever it cannot cleanly
reapply one, and prints that fact to the recovering session's stderr. No one owns the
follow-up, so the entries accumulate silently on the shared repo root.

## The situation (verified 2026-07-27T06:4xZ, repo root)

**Verified facts:**

- `git stash list` on the shared repo root returns **23 entries**.
- The oldest carry dates back to **2026-05-27** (`auto-stash-2026-05-27-predscrub`,
  `wip-on-workflow-fixes-branch-2026-05-27`, `auto-stash before /issue 411 branch switch
  (2026-05-27T18:17:24-07:00)`).
- Named entries reference at least eight distinct tasks/branches — `#784`, `#722`,
  `#648`, `#628`, `#389`, `#411`, `#407`, and a `.git/rebase-merge` husk recovery
  ("agent-memory edits, Jul 2 23:30; recovered by issue-902 session").
- `stash@{0}` is a bare `autostash` with no descriptive label.
- A parallel rescue path is also accumulating: `~/.task-workflow/root-sync-rescue/` holds
  5 dated directories (2026-07-03 through 2026-07-23) plus a loose
  `stash-319c2bf16e7c.patch` written **today** at 10:44.
- Two sessions today observed the KEPT-stash message and moved on; one noted the same
  stash is reported TWICE per run (a duplicated KEPT line).

The two miner groups that surfaced this counted 5 entries — that was the count visible in
their own session output, not the repo total. The repo total is 23.

## Why this needs you

- **Destructive / irreversible.** `git stash drop` is not recoverable through the normal
  workflow. Some of these may be genuinely dead (superseded by later merges); some may be
  the only copy of an edit. Distinguishing them is a judgment call per entry.
- **Not automatable safely.** An automated "apply and see" would dirty the shared root
  that every concurrent session commits against — the exact interference class the
  own-files-only contract exists to prevent.

## Decision needed

1. **Triage-and-clear** — walk the 23 entries, apply anything still wanted, drop the rest.
   Roughly an hour of attention; ends the backlog.
2. **Bulk-archive** — export all 23 to patch files under `~/.task-workflow/` (non-
   destructive), then clear the stash list. Cheap, keeps everything recoverable, leaves
   the patches unread.
3. **Leave it** — the entries are inert; the cost is only that a real lost edit stays
   invisible. Then the mechanical half below is the whole fix.

## Mechanical half (safe to file separately once you decide)

Regardless of the choice above, `scripts/sync_repo_root.py` should (a) stop printing the
same KEPT stash twice, and (b) route a KEPT stash somewhere a human actually reads —
a sidecar row plus inclusion in the /daily brief — rather than only the recovering
session's stderr, which no one re-reads. That part is a route-2 change and is NOT
blocked on this decision; it is called out here so the two halves are not confused.

## Provenance

/daily 2026-07-26 route-3 held item. Miner refs: F-P8, B-P20, D-P11, J-P12.
Live verification by the /daily orchestrator at the repo root, 2026-07-27.
