---
title: 'Policy: rebase-rewritten own issue branch — sanctioned --force-with-lease
  arm, or explicit ban + force-free route only?'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T04:18:52Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2312 plan v1 section 5: workflow surface has zero force-with-lease
  mentions while 15 tasks used it in practice, with #2171/#1999 reading it as correct
  and #2181 reading it as a violation.'
workflow: v1
---
# Policy: a rebase-rewritten OWN issue branch — sanctioned `--force-with-lease` arm, or explicit ban + force-free route only?

## Decision needed (user)

The workflow surface says **nothing** about force-pushing an issue branch whose local history the
session itself rewrote (a mid-flight rebase onto a fresher `main` — a move the workflow itself
prescribes for reconciling a sibling landing). Sessions have therefore been inferring a policy,
and they have inferred in **opposite directions**. This task asks for one sentence of policy so
the next session does not have to guess.

Two candidate resolutions. Either one requires the surface to say something; today it says nothing.

- **(A) Sanction it, narrowly.** Add an explicit `--force-with-lease` arm for the case
  "`origin/issue-<N>` exists, is NOT an ancestor of `HEAD`, and every commit it carries is
  reachable from `HEAD` or is patch-equivalent" — i.e. the session is overwriting only its own
  superseded history, with `--force-with-lease` providing the concurrent-writer interlock.
  Cheaper and shorter than the force-free route; the risk is that a sanctioned force arm is
  reachable by mistake from adjacent states.
- **(B) Ban it explicitly, and point at the force-free route.** State that no `/issue` path may
  force-push, and name the scratch-worktree landing route (merge the gate-certified tip into a
  worktree detached at `origin/main`, then `git push origin HEAD:main`) as the only path. Keeps
  the standing user-ask intact with zero new destructive affordance; costs more steps per landing.

**This task does not presuppose an answer.** It exists because an autonomous session may not
sanction a destructive affordance on its own (`.claude/rules/auto-continuation.md`
STATE-TO-`blocked` criterion 2: irreversible writes are always a user ask), and because leaving
the surface silent has already produced contradictory practice.

## Evidence: the surface is silent, the practice is not, and the readings conflict

Measured 2026-08-15 while planning #2312.

- **Surface: ZERO.** `force-with-lease` (and `force_with_lease`) appear **0 times** across
  `.claude/skills/`, `.claude/rules/`, and `CLAUDE.md` — verified `rc=1` on both spellings.
- **Practice: 15 tasks.** `force-with-lease` appears in the `events.jsonl` of **15 tasks** under
  `tasks/`.

The three recent instances split two ways on exactly the question above:

| Task | Marker | Reading | Verbatim |
|---|---|---|---|
| #2171 | `epm:progress` 2026-08-07T21:39:25Z | **correct** | "Branch force-pushed with `--force-with-lease` (the correct form after rebasing one's own feature branch): tip `22b80a527e`, 4 commits on current `origin/main`." |
| #2171 | `epm:results` 2026-08-07T21:24:21Z | **correct** | branch tip "force-pushed `--force-with-lease` (rc=0, `ccdf97979a...de2819e954`)" |
| #1999 | `epm:test-verdict` 2026-08-04T22:08:44Z | **correct** | worktree "@ `5fffcfb983` (rebased onto origin/main, single payload commit, 1 ahead, pushed --force-with-lease)" |
| #2181 | `epm:merged` 2026-08-07T22:27:22Z | **violation** | "The branch push used `--force-with-lease` out of habit. The remote branch did not exist (`* [new branch]`), so no force occurred and nothing was overwritten — but force-push is a user-ask on this project and the flag should not have been reached for reflexively on a first push." |

Older same-class precedent: **#760** `epm:merged` 2026-06-30T09:03:12Z rebased `issue-760`
force-with-lease onto current `origin/main` specifically "to drop 98 commits of foreign-task
drift".

Two sessions treated the flag as the correct move on a self-rebased branch; one filed its own use
of it as a policy violation. The disagreement is not about facts — it is about a policy the
surface never states.

## Why this is filed separately from #2312

#2312 fixes the MECHANICAL defect: Step 10d's merge has no arm for a rewritten branch, so a
trivially-satisfied count predicate lets a refspec-less `pull --rebase=merges` rebase onto the
STALE `origin/issue-<N>`, replaying hundreds of `main` commits as new objects (the #1128
duplicate-history shape; #2296 measured `[ahead 363, behind 1]`). #2312 implements the
**force-free** route only and deliberately decides nothing here — its plan §5 records this
question and points at this task.

The two are separable: #2312's guard correctly routes a rewritten branch AWAY from the destructive
fallback regardless of how this policy resolves. What this decides is whether the landing step at
the end of that route may be a short force-push or must stay the longer scratch-worktree merge.

## Acceptance

1. The workflow surface states the policy in at least one place, so `--force-with-lease` is
   findable by grep rather than inferable only from event history.
2. Under resolution (A): the sanctioned arm names its exact precondition and cannot be reached
   from an adjacent state (remote AHEAD of local, an unrelated ref with no merge base, a
   foreign-owned branch); a pin test covers the precondition. Under resolution (B): the ban is
   stated and the force-free route is named as the alternative.
3. Whichever resolves, the contradictory precedent is not left standing as the only record —
   `.claude/rules/` (or `CLAUDE.md`) carries the decision.

## Recurrence

The rewritten-branch state that raises this question is not rare: #1999 (2026-08-04),
#2171 (2026-08-07), #2296 (2026-08-15).

## Provenance

Surfaced by the #2312 orchestrator during Step 2 plan review (plan v1 §5), from a measured
grep of the workflow surface vs `tasks/**/events.jsonl`. Filed `proposed`, NOT dispatched —
this is a user decision, and filing is not spawning.
