---
title: 'workflow-fix: Step 10d Guard 3 ON_MAINLINE uses first-parent reachability,
  so a sibling''s merge-form landing false-flags later branches UNSAFE'
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T14:44:47Z'
has_clean_result: false
origin_prompt: '/issue 2319 (orchestrator-surfaced at Step 10d: Guard 3 read ON_MAINLINE=no
  on a merge-base that IS an ancestor of origin/main)'
workflow: v1
---
# workflow-fix: Step 10d Guard 3 `ON_MAINLINE` uses first-parent reachability, so a sibling's prescribed merge-form landing makes every later branch read UNSAFE

## Goal

Make Step 10d Guard 3's `ON_MAINLINE` probe distinguish "this branch was forked off another `issue-<M>` branch that is itself still unmerged" — the condition the guard exists to catch — from "this branch's merge-base reached `main` as a merge commit's SECOND parent", which is benign and is produced by the fleet's own prescribed landing recipe.

## The gap

`.claude/skills/issue/SKILL.md` Guard 3 computes:

```bash
MB=$(git -C "$WT" merge-base HEAD origin/main)
ON_MAINLINE=$(git -C "$WT" rev-list --first-parent origin/main | grep -Fxq -- "$MB" && echo yes || echo no)
```

and then: "The branch is **unsafe to blind-rebase** if EITHER `ON_MAINLINE=no` (branch was forked off another `issue-<M>` branch that is itself still unmerged) OR the branch's own commit content is out of scope."

The parenthetical names the intended trigger, but `rev-list --first-parent` does not test it. It tests whether `MB` sits on `origin/main`'s first-parent chain — which is strictly narrower than "`MB` is on `main`". Any commit that entered `main` as a MERGE COMMIT's second parent is reachable from `origin/main` yet absent from the first-parent walk, so every branch whose merge-base is such a commit reads `ON_MAINLINE=no` and is routed to the artifact-confirmed degrade.

**The fleet manufactures exactly those merge commits, by instruction.** CLAUDE.md § Concurrent repo-root committers and `sync_repo_root.py`'s conflict path both prescribe the scratch-worktree defusal: "merge the local tip into detached `origin/main`, push" (#1489, #1128). Each such landing puts the fleet's recent marker commits onto `main` as a second parent. Since `task.py` posts ~100+ marker commits/hr, a large share of the fleet's commits arrive on `main` through that channel, and any branch cut at one of them inherits the false UNSAFE verdict.

## Worked instance (#2319, 2026-08-15)

- `MB = 17291d7a6b315525964b89b608f89ee277eaec32` — one of **#2319's own** marker commits (`task #2319: epm:progress — [stage-dispatch] Step 4 implementer DISPATCHED`), not another branch's tip.
- `git merge-base --is-ancestor "$MB" origin/main` → **YES**. `MB` is on `main`.
- `ON_MAINLINE` → **no**, because `MB` entered `main` via `d697efaef6 "Merge remote-tracking branch 'origin/main' into HEAD"` (task #2317's landing) as a second parent.
- The branch's own three-dot diff was two files, both its own deliverables (`scripts/step9c_baseline.py`, `tests/test_step9c_baseline.py`) — zero foreign `tasks/` paths, zero new shared `src/`.

So the guard's own stated trigger was falsified while its predicate fired.

## Why the false positive is not free

The UNSAFE route is the artifact-confirmed merge + surgical additive checkout, which SKILL.md itself documents as unable to carry MODIFIED files safely: "For a file the branch MODIFIED that `main` also advanced, that overwrite silently discards `main`'s newer content with NO conflict surfacing — a silent-wrong merge." So a false UNSAFE on a MODIFIED-file branch lands the round on a route whose own documentation warns against the payload shape it is being handed.

**#1144 is the realized cost.** It was filed because "guard 3 refused the full rebase (`ON_MAINLINE=no`, `BEHIND=3004`), so the round's artifact set landed via surgical additive checkout but the branch's MODIFIED shared-src files could not ride along" — an entire follow-up task to port stranded shared-`src/` fixes to `main`, including a fleet-relevant CVD pin. That is this class producing exactly the built-but-stranded outcome the workflow-fix rule exists to prevent. (#1144 is the CONSEQUENCE task; nothing has fixed the probe.)

## The candidate fix

Discriminate on ancestry, not first-parent reachability:

```bash
MB=$(git -C "$WT" merge-base HEAD origin/main)
if git -C "$WT" merge-base --is-ancestor "$MB" origin/main; then MB_ON_MAIN=yes; else MB_ON_MAIN=no; fi
```

A still-unmerged parent branch's tip is by construction NOT an ancestor of `origin/main`, so `MB_ON_MAIN=no` catches the #479 class the guard was written for, while a second-parent-landed `main` commit correctly reads `yes`. Note `merge-base HEAD origin/main` already returns a common ancestor, so the naive reading is that `--is-ancestor` is trivially true; it is NOT, because `merge-base` can return a commit reachable from `origin/main` only through a merge's second parent, which is precisely the case that needs distinguishing from a fork off unmerged work. The implementation must confirm on real repository states (including a synthetic fork-off-unmerged-branch fixture) that the new predicate separates the two, rather than assuming it.

Open design questions for the plan to settle, not to guess:

1. Whether `ON_MAINLINE` should be REPLACED or joined by the ancestry test (a first-parent read may still carry signal worth keeping as a WARN).
2. Whether the content check alone is sufficient once the ancestry test passes — the #479 incident tripped BOTH arms, so the content check may already be the load-bearing one.
3. Whether the artifact-confirmed route should additionally refuse a MODIFIED-file payload whose files `main` advanced (a blob-identity check like the one #2319's Step 10d ran by hand), closing the silent-clobber hole independently of Guard 3's verdict.

## Acceptance criteria

1. A test that reproduces the false positive: a branch whose merge-base is a `main` commit reachable ONLY as a merge's second parent reads SAFE under the new predicate, and FAILS before the fix.
2. A test that the #479 class still reads UNSAFE: a branch forked off a still-unmerged `issue-<M>` branch.
3. SKILL.md Guard 3 prose updated so the predicate and its stated rationale agree.
4. No fail-OPEN introduced: the content check and Guards 1/4/5 are untouched, and the change cannot let a branch carrying another branch's payload rebase-merge.
5. `workflow_lint.py` no-flags and the mapped invariant tests show no NEW failures vs the plan-time baseline.

## Context

Step 10d is the merge path for every task, so a false UNSAFE silently degrades merges fleet-wide and, on MODIFIED-file payloads, routes them onto a surgical checkout that can strand or clobber content. Surfaced by the `/issue 2319` orchestrator at its own Step 10d.

## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md
fingerprint: step10d-guard3-on-mainline-first-parent-false-positive
Surfaced by: /issue 2319 Step 10d (2026-08-15). Related: #1144 (consequence — stranded shared-src fixes from this class), #479 (the class Guard 3 was written for), #1489/#1128 (the prescribed merge-form landing that manufactures the second-parent commits).
