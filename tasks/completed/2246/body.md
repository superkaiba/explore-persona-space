---
title: Step 10d merge window runs on a reap-eligible worktree; a mid-gate reap makes
  the pre-push lint gate certify pass vacuously
kind: infra
tags:
- workflow-fix
created_at: '2026-08-12T17:13:11Z'
has_clean_result: false
parent_id: 2242
origin_prompt: 'observed live in #2242 Step 10d 2026-08-12: worktree_audit reaped
  .claude/worktrees/issue-2242 mid-lint-gate because Guard 2 requires terminal status
  and the prescribed launcher passes WT via env (invisible to the argv/cwd liveness
  probe); overlay-files.txt landed 0 bytes so the gated leg was byte-identical to
  baseline and the gate was on course to emit pass without linting the payload'
workflow: v1
---
# Step 10d runs its whole merge window on a reap-eligible worktree, and the prescribed gate launcher is invisible to the reaper's liveness probe

## Goal

Close a fail-OPEN in the Step 10d merge path: for every code-path task, the
entire Step 10d window (guards + a ~20-30 min pre-push lint gate + the merge)
runs on a worktree the daily stale-worktree sweep is entitled to delete, and the
launcher shape SKILL.md prescribes gives that sweep no reason to keep it. When
the delete lands mid-gate the gate does not fail — it silently certifies `pass`
without ever linting the payload.

## The hazard (two interlocking parts)

**Part 1 — terminal status is the merge precondition AND the reap trigger.**
Guard 2 requires the task to be off `running` before the merge: SKILL.md
:11360-11366, "completed for code paths, flipped in Step 10 step 6 BEFORE this
step". `scripts/worktree_audit.py` reaps an `issue-<N>` worktree precisely when
the task status IS terminal (its keep reason #2 is *non-terminal* status). So the
act of satisfying Guard 2 is the act of making the worktree reapable, and
everything after it — guards 1/3/4/5, the lint gate, the merge — runs inside
that exposure window.

**Part 2 — the prescribed launcher hides the worktree from the liveness probe.**
`worktree_audit.py`'s other keep reasons are: a live process holding it as CWD or
referencing its path in ARGV; dirty tracked changes; human-named. The Step 10d
lint-gate launcher in SKILL.md (~:11848-11850) is:

    PYTEST_PID=$(bash -c "setsid nohup env WT=\"$WT\" REPO_ROOT=\"$REPO_ROOT\" \
      ... bash '$LINT_GATE_SCRIPT' < /dev/null > ... 2>&1 & echo \$!")

`env` passes WT in the ENVIRONMENT and then execs bash, so the resulting process
argv is just `bash /tmp/issue-<N>-lint-gate.sh` and its cwd is wherever the
caller stood (typically the repo root). An argv/cwd probe cannot see an env-only
reference. A clean, auto-named worktree on a terminal-status task with an
env-only holder matches NO keep reason — so it is reaped while actively in use.

## Why it is a fail-OPEN, not just a crash

The gate's whole soundness argument is differential: archive `origin/main` into a
tree, lint it (BASELINE), overlay the branch payload onto the SAME tree, re-lint
(GATED), subtract — so `NEW = gated - baseline` is payload-caused by
construction. The payload set comes from

    git -C "$WT" diff --name-only --no-renames origin/main...HEAD > /tmp/issue-<N>-overlay-files.txt

With `$WT` deleted that emits `fatal: cannot change to '<WT>': No such file or
directory` and writes overlay-files.txt at **0 bytes**. The overlay loop then
iterates nothing (`[step10d] landing-union overlay: merged=0 fallback=0`), the
GATED tree is byte-identical to BASELINE, the subtraction yields the empty set,
and the gate emits **`pass`** — sha-bound to the branch tip — having linted the
payload zero times. Nothing in the pipeline distinguishes that from a real pass.

Note the `git diff` failure is *already* handled defensively elsewhere in the
same script (the TRIGGER diff materializes to a file and checks its own exit
code, failing CLOSED with `echo crash`, explicitly to avoid the #1047/#458
fail-open class). The overlay producer at the SAME risk has `|| GT_RC=1` on the
redirect, but an empty-but-successful listing is not the failure it catches: here
the `git -C` invocation itself failed, and the observed run still reached the
verdict logic. Whatever the exact path, the realized behavior was a vacuous
gated leg that would have certified.

## Observed incident (#2242, 2026-08-12)

Verified live, not reasoned about:
- Gate log: `fatal: cannot change to '<repo>/.claude/worktrees/issue-2242': No
  such file or directory`, then `[step10d] landing-union overlay: merged=0
  fallback=0`.
- `/tmp/issue-2242-overlay-files.txt` = **0 bytes**.
- `git worktree list` no longer contained the worktree; the directory was gone.
- Branch `issue-2242` intact locally at `0f8da963420156258cc583e3ee3939d79e549502`;
  `git ls-remote --heads origin issue-2242` EMPTY (never pushed), so the local
  branch ref was the only copy of the work — the reap removed the checkout only.
- Caught by direct inspection of the leg files after the gate log's mtime sat
  unchanged ~5 min. NOT caught by the watch predicate, which keyed on
  rc-sentinel / pid-death / log greps — the gate was neither dead nor erroring,
  it was healthily linting the wrong tree.
- Recovery: killed the gate before any verdict was written (verdict file
  confirmed ABSENT), recreated the worktree on the existing branch, re-derived
  the own-diff (identical, 13 files), relaunched with cwd inside the worktree.

## Proposed change (implementer to scope)

1. **Make the prescribed launchers hold the worktree by CWD.** In SKILL.md's
   Step 10d lint-gate launcher (and the Step 9c gate launcher, same shape), run
   the detached workload with `cd "$WT"` so `/proc/<pid>/cwd` resolves to the
   worktree and `worktree_audit`'s existing cwd-keep fires. This is a one-line
   change per launcher and needs no audit-side change. VERIFY the gate scripts
   are cwd-independent first (they use `git -C "$WT"` and absolute /tmp paths,
   so they should be — confirm, don't assume).
2. **Belt: make the reaper refuse to delete a worktree whose branch is
   unmerged.** In `scripts/worktree_audit.py`, add a keep reason for an
   `issue-<N>` worktree whose HEAD carries commits not reachable from
   `origin/main` (`git rev-list --count origin/main..HEAD` > 0) regardless of
   task status. Terminal status currently means "reap eligible", but a terminal
   task whose branch never merged is exactly the case where the worktree is
   still load-bearing. This is the durable fix — it protects hand-run merges and
   any future launcher that forgets (1).
3. **Suspenders: make a vacuous gated leg impossible to certify.** In the lint
   gate workload, after writing the overlay listing, assert it is consistent with
   the trigger diff: if the trigger classified the payload as code-bearing but
   the overlay listing is EMPTY, write `crash` (fail CLOSED) rather than
   proceeding to a subtraction that cannot fail. Today an empty overlay is
   indistinguishable from "no payload".
4. Consider a lease/marker the sweep honors for the Step 10d window (weakest of
   the four; (1)+(2)+(3) likely suffice).

## Scope notes

- **Coordinate on SKILL.md with #2126, which is LIVE on that file** (its five
  gate-recipe defects target `.claude/skills/issue/SKILL.md` +
  `scripts/step10d_guards.sh`, and a sixth defect — guard 4's eval-unsafe
  `LOST_UPDATE_PATHS` emit — was routed to it from the same #2242 merge). The
  primary target HERE is `scripts/worktree_audit.py` (item 2) plus the lint-gate
  workload (item 3), neither of which #2126 touches. If item 1's SKILL.md
  launcher edit is taken in this task, check #2126's state first — one
  implementer per file set.
- Distinct from #1978 (extracting the guards to a script) and from any
  worktree-audit false-KEEP work: this is a false-REAP of an in-use worktree,
  the opposite direction.

## Provenance

Surfaced by the #2242 Step 10d merge (2026-08-12), where the reap landed mid-gate
and the vacuous `pass` was intercepted by hand before it could certify. #2242's
own merge was completed on a rebuilt worktree; this task is the generic fix.
