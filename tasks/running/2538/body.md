---
title: Step 10d merge-conflict-recovery block omits gh pr ready, so a recovery-path
  merge dies on the draft precondition and misroutes to the Failure bullet
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T14:30:57Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2315''s Step 10d merge: the recovery-path gh pr merge
  failed ''Pull Request is still a draft'' because only the safe-case block carries
  the #2240 gh pr ready precondition.'
workflow: v1
---
# Step 10d merge-conflict-recovery block omits `gh pr ready`, so a recovery-path merge dies on the draft precondition and misroutes to the Failure bullet

## Goal

Make the Step 10d **merge-conflict-recovery** merge conditional carry the same draft-merge precondition the **safe-case** block already carries, so a recovery-path merge cannot fail on `Pull Request is still a draft` — and, failing that, make the recovery block's failure-classification list recognize the draft error instead of routing it to the terminal "anything else" arm.

## The gap

`.claude/skills/issue/steps/18-step-10d.md` has two merge conditionals that both end in `gh pr merge`:

- **Safe-case** (`#### The auto-merge procedure`, ~line 2679) calls `gh pr ready "$PR"` immediately before `gh pr merge "$PR" $MERGE_FORM`. Its own comment states the intent: *"Draft-merge precondition (#2240 pin): this single `gh pr ready` call marks the PR ready before the merge below and covers PRs opened as drafts by EITHER fresh-PR arm ... do NOT add a second ready call elsewhere."*
- **Merge-conflict recovery** (`#### Merge-conflict recovery`, ~line 3274) has NO `gh pr ready` call. It goes `git -C "$WT" push` → PR-state probe → `gh pr merge <PR> --squash --delete-branch=false`.

Step 4a opens every issue PR as a **draft** (`gh pr create --draft --head issue-<N>`). So any branch that reaches the merge through the recovery path — rather than the safe case — meets `gh pr merge` with an unready PR and gets:

```
GraphQL: Pull Request is still a draft (mergePullRequest)
```

Two consequences, the second worse than the first:

1. The merge attempt is burned. The recovery block is explicitly capped ("One recovery attempt per Step 10d invocation").
2. **The error misroutes.** The recovery block's failure classification enumerates (0) `"Base branch was modified"`, (1) `"can't be rebased"`, (2) `"Pull Request has merge conflicts"`, (3) `"Head branch is out of date"`, and (4) *anything else → the Failure bullet* (`epm:merge-failed v1`). The draft error is not in (0)-(3), so a correct reading of the recipe parks a merge that is in fact ONE prescribed command away from landing — with a green sha-bound lint verdict in hand.

The `do NOT add a second ready call elsewhere` comment in the safe-case block is what makes this a genuine ambiguity rather than an obvious omission: a reader following that instruction literally will not add the call to the recovery path, which is exactly where it is missing.

## Observed live (2026-08-24, task #2315)

#2315 reached Step 10d, Guard 4 refused on a lost-update, and the prescribed recovery (in-worktree merge of `origin/main`) put the branch on the recovery path. Sequence:

1. Pre-push lint gate PASSED, sha-bound to tip `ce3f2c53ccac`; verdict file intact, so the three-conjunct hard stop was satisfied.
2. Push succeeded (`53263a2162..ce3f2c53cc`). PR #2072 probed `MERGEABLE CLEAN OPEN`, `headRefOid` equal to the certified tip — PR-head parity (#2312) clean.
3. `gh pr merge 2072 --squash --delete-branch=false` → `GraphQL: Pull Request is still a draft (mergePullRequest)`.
4. Recovered by hand: `gh pr ready 2072`, then re-entered the same conditional. Merge landed at 14:24:10Z, verified via the PR object's `mergedAt` transition. Squash commit `5f60a6541b3c643c16cf0647d13b032fa331c213`.

The verdict file was correctly NOT consumed on the failed attempt, and no new commits landed between the two attempts, so the sha-bind still held on re-entry. That is luck about ordering, not a designed recovery: had the classification been followed literally, the disposition would have been `epm:merge-failed v1` on a merge-ready branch.

## Suggested direction (not prescriptive — the planner owns the design)

- Simplest fix: add `gh pr ready "$PR"` to the recovery block immediately before its `gh pr merge`, and amend the safe-case block's `do NOT add a second ready call elsewhere` comment so it scopes the prohibition to *duplicate calls within one merge attempt* rather than *the recovery path*. Both calls are idempotent (`gh pr ready` on a ready PR exits 0), so there is no double-ready hazard.
- Consider whether the artifact-confirmed / surgical-additive and rewritten-branch landing routes have the same hole; the recovery block was found by hitting it, not by an audit.
- Either way, add the draft error to the recovery block's failure-classification list with the `gh pr ready` + same-conditional re-entry remedy, so a future omission degrades to a named retry instead of the terminal arm.
- Worth a mechanical pin: a `workflow_lint.py` check (or a `tests/test_issue_skill_step10d_*.py` pin) asserting that every `gh pr merge` invocation in `18-step-10d.md` is preceded by a `gh pr ready` in the same fenced block. That is the check that would have caught this at authoring time.

## Provenance

Surfaced by task #2315's own Step 10d merge (2026-08-24), on the recovery path taken after a Guard-4 lost-update refusal. Related pins: #2240 (the safe-case draft-merge precondition this block is missing), #2312 (PR-head parity), #1897 (landing verification), #1041 (the `--squash` substitution that puts a branch on this path in the first place).
