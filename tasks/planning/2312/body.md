---
title: Step 10d safe-case push has no arm for a REBASED issue branch — rejection fallback
  rebases onto the stale remote branch (#1128 replay shape); stale PR head can land
  pre-rebase payload
kind: infra
tags:
- wf-fix
created_at: '2026-08-15T03:45:45Z'
has_clean_result: false
parent_id: 2296
workflow: v1
---
# Step 10d safe-case push has no arm for a REBASED issue branch — its rejection fallback rebases onto the stale remote branch and manufactures the #1128 replay shape

## Goal

Give the Step 10d safe-case merge an explicit arm for the case where the local
`issue-<N>` branch's history was REWRITTEN mid-flight (a rebase onto `origin/main`,
which the workflow itself prescribes for reconciling a sibling landing) and
`origin/issue-<N>` therefore holds pre-rebase history. Today the recipe's push +
rejection-fallback pair is unsafe in exactly that state, and there is no documented
route to a PR-based merge that does not require a force-push (a standing user-ask, so
unavailable to an autonomous session).

## Mechanism

`.claude/skills/issue/SKILL.md` Step 10d § The auto-merge procedure, ~L13135:

```bash
if [ "$(git -C "$WT" rev-list --count origin/issue-<N>..HEAD 2>/dev/null || echo 1)" -gt 0 ] \
   || [ "$STRIPPED_FOREIGN" = "yes" ] || [ "$MEM_COMMITTED" = "yes" ]; then
  git -C "$WT" push origin issue-<N> \
    || { git -C "$WT" pull --rebase=merges --autostash \
         && git -C "$WT" push origin issue-<N>; }
fi
```

The predicate is a COUNT of commits ahead of the remote branch, so on a rebased branch it
is satisfied trivially (the rebase replays every intervening `main` commit into the count).
The push is then rejected non-fast-forward, because the rewritten local history is not a
descendant of the remote tip. The fallback runs `git pull --rebase=merges --autostash`
with NO refspec — so it rebases against the branch's UPSTREAM, `origin/issue-<N>`, not
`origin/main`. On a branch that was just rebased onto `origin/main`, that replays several
hundred `main` commits as NEW objects on top of the stale remote branch tip. A PR from the
result asks GitHub to replay those hundreds of commits onto `main` — the #1128
server-side-conflict / duplicate-history shape the Guard-1 strip exists to avoid.

Second defect in the same region: `gh pr create --head issue-<N>` is preconditioned only
on `ls-remote --heads origin issue-<N>` being EMPTY (the #2240 origin-precondition). On a
rebased branch the ref EXISTS but is STALE, so the precondition is satisfied, no push
happens, and the PR is opened against PRE-REBASE content. The merge then lands the
un-reconciled payload while every local check (tests, lint gate, verdict sha) passed
against the reconciled tip. The gate's sha-binding does NOT catch this: it binds the
verdict to the LOCAL tip, and nothing compares the local tip to the PR head ref.

## Measured incident (#2296, 2026-08-15)

#2296 (move the Step 10d mapped-invariant baseline leg off the shared root) had to rebase
mid-Step-10d because sibling #2293 landed `cab5fcabaf`, changing the same helper's
signature (`create_scratch_worktree`: optional `sha=None` → required keyword-only
`base_sha`). The rebase was correct and prescribed. Resulting state:

```
$ git push --dry-run origin issue-2296
 ! [rejected]              issue-2296 -> issue-2296 (non-fast-forward)

$ git log --oneline origin/issue-2296 --not HEAD
af93a34871 task #2296: move the Step 10d mapped-invariant BASELINE leg ...   <- PRE-rebase payload

$ git status -sb
## issue-2296...origin/issue-2296 [ahead 363, behind 1]
```

So the remote branch tip is the pre-reconcile commit, whose `cmd_mapped_baseline` calls
`create_scratch_worktree(..., sha=base_sha)` — a `TypeError` against main's post-#2293
signature. A PR-based merge off that ref would have landed a fleet-wide break in
`scripts/step9c_baseline.py`, the helper every Step 9c compare and Step 10d baseline uses.

The session landed via the CLAUDE.md § Concurrent repo-root committers scratch-worktree
route instead (merge the local tip into a worktree detached at `origin/main`, then
`git push origin HEAD:main`), which requires no force-push, rewrites no history, leaves
the gate-certified local tip untouched, and retries cheaply when `main` advances.

## Why the obvious fix is not available

`git push --force-with-lease origin issue-<N>` would make the remote ref correct in one
command and loses nothing (the dropped commit is the superseded payload, and the lease
guards against clobbering a concurrent push). But force-push is a standing user-ask —
CLAUDE.md § workflow-fix protocol ("force-push stays a user-ask") and
`.claude/rules/auto-continuation.md` STATE-TO-`blocked` criterion 2 ("irreversible writes
(deletion, force-push, credential changes — always ask)"), which carries no autonomous
carve-out. An autonomous session therefore cannot take it, and blocking a
gate-PASSed task on it is disproportionate. The recipe needs a documented non-force route.

## Proposed fix (direction only — the plan decides)

1. **Detect the state explicitly.** Before the push, test descendancy rather than counting:
   `git -C "$WT" merge-base --is-ancestor origin/issue-<N> HEAD`. False ⇒ the remote branch
   is not an ancestor ⇒ the branch was rewritten. Route to a named arm; never fall into the
   bare `pull --rebase` fallback (which should additionally be given an explicit
   `origin main` refspec, or replaced, so it can never rebase onto the stale branch).
2. **Add a PR-head staleness assert.** Before any `gh pr merge`, require the PR head ref's
   sha to equal the gate-certified local tip — the same fail-closed posture the verdict's
   own sha-bind already has. This catches the stale-`ls-remote` hole independently of how
   the branch got stale.
3. **Document the non-force landing route** for the rewritten-branch arm: the
   scratch-worktree merge into detached `origin/main` + `git push origin HEAD:main`, with
   the verdict re-check and a bounded retry on fleet advancement. It is already the
   CLAUDE.md-sanctioned repo-root landing form and the #1489 defusal recipe; Step 10d just
   does not reference it for this case.
4. Alternatively/additionally, decide whether a rebase-rewritten OWN issue branch should be
   a sanctioned force-with-lease exception. That is a user-facing policy question about
   halt criterion 2 — surface it, do not assume it.

## Acceptance

1. A rebased branch whose remote ref is stale does NOT reach a bare
   `pull --rebase`-onto-`origin/issue-<N>` path.
2. A PR whose head ref sha differs from the gate-certified tip fails CLOSED before merge.
3. A documented, force-push-free landing route exists for the rewritten-branch case, and a
   fixture reproduces the #2296 state (remote tip = superseded commit, local rebased) and
   asserts the chosen arm is taken.
4. The ordinary (non-rewritten) branch path is byte-unchanged.

## Provenance

Surfaced by the #2296 orchestrator while executing its own Step 10d after a prescribed
mid-flight rebase. Distinct from #2302 (this session's other filing — that one is the
Step 5a sibling-sync arm disabling the #2024 ordering carve-out at the Step 9c compare;
this one is the Step 10d push/PR-head surface), and distinct from the #2240 zero-PR arm it
sits beside (that arm handles NO ref; this is a STALE ref). Same target file
`.claude/skills/issue/SKILL.md`, different bug and different fingerprint, so it files
separately per the CLAUDE.md dedup rule.
