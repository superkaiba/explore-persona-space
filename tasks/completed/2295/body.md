---
title: 'workflow-fix: sync_repo_root deadlocks on the abandoned-rebase husk it exists
  to clear (task.py holds the flock while waiting on the husk)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T15:40:07Z'
has_clean_result: false
origin_prompt: are there any stalled issues/blocked issues/running pods I should handle?
workflow: v1
---
## Overview / Motivation

`sync_repo_root.py` is the mandated recovery for a wedged repo root, but it
cannot recover the specific state it exists to fix — an abandoned
`rebase-merge` husk — because the very writers that husk blocks are holding the
lock the helper needs. Observed live on 2026-08-14 while triaging the blocked
queue.

## The deadlock (verified live, not inferred)

1. The repo root was left detached mid-`rebase-merge` at step 7 of 22, `onto`
   `3fc2e9ff37`, with **zero live git processes** — the driving session died
   (VM load average was 15-18 from two concurrent Step 9c gate suites, and the
   `earlyoom` misfire filed the same day is a plausible killer: the native
   binary's `comm` is the version string, not `claude`, so `--avoid` missed it).
2. Every `task.py` invocation across the fleet then hit its
   `EPM_TASKPY_REBASE_WAIT_SECONDS` wait (default 120 s) — **while already
   holding the `~/.task-workflow/lock` flock**.
3. `sync_repo_root.py` calls `acquire_task_workflow_lock(120)`
   (`scripts/sync_repo_root.py:723`), which needs that same flock in order to
   run its own `git rebase --abort` husk-clearing path
   (`scripts/sync_repo_root.py:794`).
4. Result: `state=error exit=5` —
   `task-workflow lock still held after 120s`, with rotating holder evidence
   (PIDs 3309523 -> 3373777 -> 3415906 across ~4 minutes; ~15 concurrent
   sessions feed a continuous stream, so a natural lock gap effectively never
   arrives).

So the helper fails safe (correct) but is structurally unable to act, and the
one sanctioned recovery path is closed exactly when it is most needed.

Confirmed there is **no in-band override**: the task-workflow lock wait is a
caller-supplied constant with no CLI flag and no environment variable
(`--dry-run`, `--no-push`, `--json`, `--triage-autostash`, `--timeout-s`,
`--repo` are the entire surface; `EPM_ROOT_SYNC_ABORT_LOCK_WAIT_S` governs a
DIFFERENT wait — the index-lock abort retry — not this one).

In this incident the husk cleared on its own before any manual step was taken
(`main` advanced `6dcdfa40f2` -> `c3d90ca172` -> `754a9da768`), so no hand
surgery was performed. That is luck, not a mechanism.

## Second finding: the autostash rescue channel is not converging

`--triage-autostash` reports **5 bare autostash entries**, every one
`apply --check: dirty`:

| stash | sha | date |
|---|---|---|
| stash@{0} | 80f17de362bf | 2026-08-14 |
| stash@{5} | 6488bc83268c | 2026-08-05 |
| stash@{6} | 0bebb08285e8 | 2026-08-03 |
| stash@{7} | 243e519e09a6 | 2026-07-31 |
| stash@{8} | 319c2bf16e7c | 2026-07-26 |

Five occurrences in ~3 weeks, none reclaimed. Rescue patches exist for all of
them under `~/.task-workflow/root-sync-rescue/`, so **no work is lost** — but
the abandoned-husk -> stranded-autostash sequence is recurring, and #1806's
`stash_rescue_audit_pass` (watcher pass 34) is evidently surfacing these
without anything closing the loop. Today's entry holds real uncommitted work
from several sessions (#2054 writeup figures, #2094 judge/bank/gallery
scripts, `tasks/awaiting_promotion/2094/events.jsonl`,
`tasks/followups_running/2225/events.jsonl`, `tests/test_issue2094_judge.py`).

## Goal

Make the sanctioned root-sync recovery able to clear an abandoned rebase husk
even while blocked `task.py` writers hold the task-workflow lock, and make the
stranded-autostash backlog converge instead of accumulating.

## Proposed change (evaluate at plan time; do NOT assume this shape is right)

- **Ordering fix (preferred):** have `task.py` wait for the rebase husk
  BEFORE acquiring the flock rather than while holding it. That removes the
  deadlock at its source and needs no new override. Verify the flock is not
  load-bearing for the husk-wait itself.
- **Or a scoped escape in the helper:** allow the husk-clearing path to
  proceed without the task-workflow lock when a liveness probe shows no live
  git process attributable to the repo (the helper already has
  `LivenessProbe` / `_fuser_evidence` machinery). Must stay fail-closed when
  the probe is `uncertain`.
- **Or a bounded-wait knob** (`EPM_ROOT_SYNC_TASK_LOCK_WAIT_S`) so an
  operator can extend past a traffic burst. Weakest option — it does not fix a
  continuously-held lock.
- **Autostash convergence:** decide whether the watcher pass should escalate
  per-entry to the OWNING task (the stash file list names the issues
  directly), rather than emitting a repo-level alert nobody owns.

## Constraints / invariants

- Workflow-surface only. The prohibition on a repo-root `git reset --hard`
  stands; any new path uses `git rebase --abort` + autostash re-apply.
- Never drop or auto-pop a stash entry — every entry here is `dirty`, so an
  automatic pop would conflict against live sibling work.
- `uv run python scripts/workflow_lint.py` (no flags) passes; ruff clean on
  touched files.

## Provenance

- workflow_fix_target: `scripts/sync_repo_root.py` + `scripts/task.py`
  (lock/husk-wait ordering)
- Discovered 2026-08-14 during an interactive blocked-queue/pod triage session.
- Related: #2015 (pre-commit stash race; the tracked-uncommitted-state hazard
  that supplies these autostashes), #1806 (stash-rescue audit pass, watcher
  pass 34), #2182 (`sync_repo_root.py` autostash sibling), #1201
  (`guard_repo_root_pull.sh`).
