---
title: 'workflow-fix: /daily push stub commit immediately (orphaned off main, #711
  alert)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fb2aa9f062d9
created_at: '2026-06-28T19:57:51Z'
has_clean_result: false
origin_prompt: 'Look into and fix this: ALERT: /daily for 2026-06-27 did not land
  — check ~/my-goat/logs/daily_retrospective.log'
---
## Overview / Motivation

Auto-filed from a workflow-fix observed during a user-directed investigation of a
`/daily` heartbeat alert ("ALERT: /daily for 2026-06-27 did not land").

The `/daily` skill (`.claude/skills/daily/SKILL.md`) writes the daily stub to the
gitignored, force-added `logs/daily/<date>.md`, commits it to **local `main`
only**, and explicitly **does not push** it. The stub normally reaches
`origin/main` only by riding along on a *later* concurrent session's push.

On 2026-06-27 the daily ran and committed `569b5f4c9f` (adding
`logs/daily/2026-06-27.md`), but the commit was **orphaned off `main` before any
session pushed it**: it sat on top of an unpushed task-state commit (`f027c0f9d4`,
`task #699: proposed → planning`) that a concurrent `git pull --rebase=merges`
rewrote/dropped, taking the daily commit with it. `main` rewound under the daily,
git removed the now-untracked tracked file from the working tree, and the #711
heartbeat correctly reported the file MISSING. (The orphaned content survives only
on the worktree branches `issue-698/699/705/706`, which were cut from `main` while
it still contained the commit.)

The 2026-06-27 daily has already been recovered out-of-band (re-committed +
pushed to `origin/main` as `f3e9313e2a`). This task fixes the **recurrence**: the
daily-stub commit is fragile because it is committed-but-not-pushed on a
heavily-concurrent shared `main`, which is exactly the orphaning hazard CLAUDE.md
warns about ("Push IMMEDIATELY after committing … an unpushed … commit on the
shared root is at risk from every concurrent session's pull").

## Goal

Make the `/daily` stub commit durable: push it to `origin/main` immediately after
committing, using the project-standard push recipe, so a concurrent rebase can
never orphan it again — while preserving the existing "do not push route-1 code
fixes" safety rule (the daily-stub log file is a benign `visible:false` artifact;
the no-push rule for code/behavior fixes is unchanged).

## Workflow gap

- **Bug observed:** The 2026-06-27 daily ran and committed
  `logs/daily/2026-06-27.md`, but the commit was orphaned off `main` by a
  concurrent `git pull --rebase=merges` before any session pushed it; the
  healthcheck correctly alerted the daily did not land.
- **Why it is a workflow gap:** `.claude/skills/daily/SKILL.md` commits the stub
  to local `main` only and forbids pushing, leaving it to ride along on a later
  concurrent push — so until that happens the commit is exposed to the documented
  concurrent-rebase orphaning hazard. The #711 heartbeat is working correctly; the
  fragility is upstream in how the stub is persisted.
- **Confidence (emitter):** high (mechanism reflog-confirmed; matches the CLAUDE.md
  "push immediately" rule the rest of the repo already follows).

## Proposed change (candidate diff sketch — refine in planning)

In the `### Commit` step of `.claude/skills/daily/SKILL.md`, after the existing
`git add -f` + `git commit`, add an immediate push of just-that-commit using the
project-standard recipe (the repo already pins `pull.rebase=merges` +
`rebase.autoStash=true`):

```bash
git add -f logs/daily/YYYY-MM-DD.md   # logs/ is gitignored; force-add or it stages nothing
git commit -m "logs: daily stub for YYYY-MM-DD"
# Push immediately so the stub cannot be orphaned off main by a concurrent
# rebase before some later session happens to push it (the #711 incident).
git push origin HEAD:main || { git pull --rebase=merges --autostash && git push origin HEAD:main; }
```

Update the surrounding prose: the standing "Do not push" rule applies to route-1
**code/behavior fixes** (external side-effects Thomas reviews first), NOT to the
daily-stub log commit — make that distinction explicit so the two rules don't read
as contradictory. The planner should also weigh the heavy-concurrency-safe
variant CLAUDE.md recommends (commit the stub from a scratch worktree detached at
`origin/main`, then `git push origin HEAD:main`) vs the simpler in-place
push-with-retry above, and pick one.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md` (the `### Commit` step + the
  "Do not push" / "External side-effects" prose).
- Out of scope: `scripts/cron_daily_healthcheck.sh` (#711) is working correctly —
  do not weaken its MISSING/stale detection; it is the heartbeat that surfaced
  this and should keep firing on a genuine miss.

## Constraints / invariants

- Workflow-surface only — no experiment code, `configs/`, or `tasks/`.
- Preserve the no-push rule for route-1 code/behavior fixes; only the daily-stub
  log commit gains the immediate push.
- Use the project push recipe (explicit-path staging, `--rebase=merges
  --autostash` on rejection, single retry). Normal push to `origin/main` only —
  never force-push.
- `scripts/workflow_lint.py --check-asks` passes; if `CLAUDE.md`/`workflow.yaml`
  reference the daily-commit policy, keep them consistent.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: fb2aa9f062d9
