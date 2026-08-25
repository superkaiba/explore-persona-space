---
name: Claude approves append-is-durable deferral without checking routed-mode reset --hard
description: Plan-stage (alternatives lens) — "the filesystem append is the state, commit is bookkeeping" arguments must be verified against EVERY repo_root() resolution mode; routed managed-worktree mode resets --hard main on each re-resolution, physically discarding an un-CAS'd append. #1030 r1.
type: feedback
---

Rule: when a plan defers/swallows a git-commit failure on the theory "the
filesystem append is the durable state; the commit is bookkeeping the next
commit sweeps," verify the durability premise against EVERY `repo_root()`
resolution mode — especially task_workflow.py ROUTED mode, where
`_ensure_managed_main_worktree` runs `reset --hard main` on every
re-resolution (L642). If the routed commit's `_advance_main_ref` CAS leg
fails (RuntimeError from `_git_quiet`, which has NO lock retry), `main`
lacks the commit, and the next reset --hard reverts the WORKING TREE too —
the appended line is physically removed from the file every routed consumer
reads, the detached commit dangles, and a success-with-warning caller was
told rc=0 "do not re-post." Silent, unrecoverable marker loss — worse than
today's loud raise.

**Why:** #1030 r1 — Claude alternatives critic APPROVEd on "every task-state
consumer reads the FILESYSTEM... Not fatal," true only for the primary
checkout (uncommitted dirt persists there); Codex correctly REVISEd. The
plan even contained the contradiction internally (§4.1: "a concurrent main
move is already caught loud by the CAS" vs §4.2 catching `RuntimeError` and
naming "_git_quiet CAS in _advance_main_ref" as a deferred source). The
managed-worktree comment's safety premise ("every mutation commits before
releasing, so there is never uncommitted task work to clobber", L639-641)
is exactly the invariant a deferred commit breaks.

**How to apply:** on any plan-stage split over swallow-vs-raise for
post-durable-write bookkeeping failures, (1) enumerate the resolution/
consumer modes and ask which mode can REVERT the "durable" filesystem state
(reset --hard, checkout, rsync, ephemeral teardown); (2) check whether the
swallowed exception class includes a post-commit ref-advance/CAS failure —
phase-split (pre-commit vs post-commit) or a narrower exception class is
conclusion-changing plan content, not implementer discretion, when an AC and
a §11-registered decision pin the exception tuple. Preserve the FAIL-side
severity (REVISE).
