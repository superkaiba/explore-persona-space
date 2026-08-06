---
title: 'daily-fix: worktree_audit gc-wedge tier (git prune + gc.log)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0160c8eb7a27
- daily-auto-filed
created_at: '2026-08-02T07:13:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): Shared repo-root git gc
  is wedged: .git/gc.log (Aug 1) + 29 worktree gc.log files all say ''too many unreachable
  loose objects; run git prune''; auto-gc suppressed fleet-wide, warning spams every
  merge. 9,359 loose objects, 48 packs; no janitor owns git prune or gc.log clearing.'
workflow: v1
---
# daily-fix: worktree_audit gc-wedge tier (git prune + gc.log)

## Overview / Motivation
Auto-filed by /daily 2026-08-01 (route 2: behavior/logic change → independent review) from consolidated problem sweep entry C20 (miner-5 P11 + miner-4 P12; sessions 87fe9339 (#1876), f41944ea (#1906)).

## Goal
Add a reviewed maintenance tier to `scripts/worktree_audit.py` that clears the git-gc wedge on the shared repo root: run `git prune` (off-peak, bounded) and remove stale `.git/gc.log` / `.git/worktrees/*/gc.log` blocker files so auto-gc re-arms; the first run doubles as the one-off unwedge.

## Workflow gap
- **Bug observed:** Every merge/commit on the shared root now warns "There are too many unreachable loose objects; run 'git prune'" + "Automatic cleanup will not be performed until the file is removed" — seen inside #1876's merge attempts and #1906's Guard-4 refusal tool_result. Repo-level gc debt is suppressing auto-gc fleet-wide; nothing in the janitor set owns it.
- **Why it is a workflow gap:** `worktree_audit.py` is the daily worktree janitor and already runs `git worktree prune`, but no automation clears gc.log blocker files or runs `git prune`, so once the wedge appears it persists indefinitely (repo-root gc.log dated Aug 1, still present at compose time) and spams every git operation.
- **Confidence:** high
- verified-at-filing: probed live gc state (2026-08-02 UTC): `cat .git/gc.log` → "warning: There are too many unreachable loose objects; run 'git prune'" (mtime Aug 1 01:03); `ls .git/worktrees/*/gc.log | wc -l` → **29** worktree gc.log files, sampled content identical; `git count-objects -v` → 9,359 loose objects (size 97,980 KB), 48 packs, size-pack ~16 GB. `grep -in 'gc\|prune' scripts/worktree_audit.py` → only `git worktree prune` (lines 1164, 1241) — no `git prune`, no gc.log handling; `git log --oneline --since='7 days ago' -- scripts/worktree_audit.py` → 0 commits (2026-08-02 UTC).

## Proposed change (refine in planning)
In `scripts/worktree_audit.py` (the daily `cron_worktree_audit.sh` entrypoint), a new tier gated on wedge evidence:

```
+ def gc_wedge_tier(root, apply=False):
+     # Fires only when .git/gc.log (or any .git/worktrees/*/gc.log) exists.
+     # 1. git prune --expire=2.weeks.ago   (bounded expiry; never `git gc --aggressive`)
+     # 2. remove the stale gc.log blocker files (repo root + per-worktree)
+     # 3. report loose-object count before/after; escalate if still wedged
```

Safety constraints the planner must keep: run only in the `--apply` cron path (report-only otherwise); never touch live worktrees' checked-out state; `git prune` is safe on concurrent committers per git's own reachability semantics but planner should confirm the flock/quiet-window story (the cron already runs at a 09:47 quiet slot); `unverified hypothesis — verify at plan time: removing gc.log alone re-enables auto-gc, which may then run mid-session — prefer prune-then-clear ordering (miner-suggested mechanism; git docs to confirm at plan time).`

## Scope / surfaces
- Primary target: `scripts/worktree_audit.py` (+ `scripts/cron_worktree_audit.sh` if a flag is threaded)

## Constraints / invariants
- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- NEVER a destructive repo-root git op (`reset --hard`, `checkout .`) — `git prune` + gc.log removal only; the repo-root-branch guard invariants are untouched.
- Recursion guard: this task's session carries the workflow_fix_target Provenance line and MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 0160c8eb7a27
- workflow_fix_target: scripts/worktree_audit.py
- origin: /daily 2026-08-01 problem sweep, CONSOLIDATED.md entry C20.
